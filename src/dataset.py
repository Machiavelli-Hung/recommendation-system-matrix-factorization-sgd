from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import torch


IndexTriple = Tuple[int, int, float]  # (user_idx, item_idx, rating)


def _try_open_csv(path: str) -> Iterable[dict]:
    """
    Open CSV with delimiter detection and encoding fallback.
    Returns an iterator of dict rows (csv.DictReader).
    """
    # Try utf-8 first, fallback to latin-1 if needed
    for enc in ("utf-8", "latin-1"):
        try:
            f = open(path, "r", encoding=enc, newline="")
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
                has_header = csv.Sniffer().has_header(sample)
            except Exception:
                dialect = csv.excel
                has_header = True
            reader = csv.DictReader(f, dialect=dialect) if has_header else csv.DictReader(f)
            # Prime one read to validate; then reconstruct iterable
            first = next(reader)
            def gen():
                yield first
                for row in reader:
                    yield row
            return gen()
        except StopIteration:
            f.close()
            return []
        except Exception:
            try:
                f.close()
            except Exception:
                pass
            continue
    raise ValueError(f"Could not open CSV: {path}")


def _find_ratings_csv_in_dir(data_dir: str) -> Optional[str]:
    candidates = [
        "Ratings.csv", "ratings.csv", "ratings_clean.csv",
        "ratings_small.csv", "interactions.csv"
    ]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path
    # Fallback: first csv containing 'rating' column
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(data_dir, fname)
        try:
            rows = _try_open_csv(path)
            for row in rows:
                cols_lower = {k.lower() for k in row.keys()}
                if any(c in cols_lower for c in ("rating", "book-rating", "score")):
                    return path
                break
        except Exception:
            continue
    return None


def _normalize_field(row: dict, keys: List[str], default: Optional[str] = None) -> Optional[str]:
    for k in keys:
        if k in row:
            return row[k]
    # case-insensitive lookup
    lower_map = {k.lower(): k for k in row.keys()}
    for k in keys:
        if k.lower() in lower_map:
            return row[lower_map[k.lower()]]
    return default


@dataclass
class RatingsDataset:
    """
    Loads ratings from either:
      - a single ratings CSV (columns like [userId, itemId, rating]), or
      - a data directory containing Book-Crossing style CSVs (Users.csv, Books.csv, Ratings.csv)

    Builds:
      - dense ratings matrix M (torch.float32)
      - Ω: list of (i, j, r) for all observed (non-zero) entries
    """

    csv_path_or_dir: str
    # Optional limits to prevent huge dense M causing OOM
    max_users: Optional[int] = None
    max_items: Optional[int] = None
    max_ratings: Optional[int] = None
    # Optional normalization to [0,1]
    normalize_ratings: bool = False
    rating_min: Optional[float] = None
    rating_max: Optional[float] = None

    def __post_init__(self) -> None:
        self.user_id_to_index: Dict[str, int] = {}
        self.item_id_to_index: Dict[str, int] = {}
        self.index_to_user_id: List[str] = []
        self.index_to_item_id: List[str] = []
        self.original_min: Optional[float] = None
        self.original_max: Optional[float] = None

        if os.path.isdir(self.csv_path_or_dir):
            ratings_path = _find_ratings_csv_in_dir(self.csv_path_or_dir)
            if ratings_path is None:
                raise ValueError(f"Could not find ratings CSV inside directory: {self.csv_path_or_dir}")
            rows_src = _try_open_csv(ratings_path)
        else:
            ratings_path = self.csv_path_or_dir
            rows_src = _try_open_csv(ratings_path)

        # Try to map common header variants:
        # - Users: userId, UserID, User-ID
        # - Items: itemId, ItemID, bookId, BookID, ISBN
        # - Rating: rating, Rating, Book-Rating, score
        rows_all: List[Tuple[str, str, float]] = []
        for row in rows_src:
            user_raw = _normalize_field(row, ["userId", "UserID", "User-ID", "user_id"])
            item_raw = _normalize_field(row, ["itemId", "ItemID", "item_id", "ISBN", "bookId", "BookID"])
            rating_raw = _normalize_field(row, ["rating", "Rating", "Book-Rating", "book_rating", "score"])
            if user_raw is None or item_raw is None or rating_raw is None:
                # Skip rows that don't have required fields
                continue
            user_id = str(user_raw).strip()
            item_id = str(item_raw).strip()
            try:
                rating = float(str(rating_raw).strip())
            except ValueError:
                # Non-numeric ratings are skipped
                continue
            # Keep only observed ratings (non-zero). This matches Ω definition.
            if rating == 0.0:
                continue
            rows_all.append((user_id, item_id, rating))

        if len(rows_all) == 0:
            raise ValueError(f"No valid ratings found in {ratings_path}")

        # Compute original min/max before filtering for normalization (if requested)
        if self.normalize_ratings:
            if self.rating_min is not None and self.rating_max is not None:
                rmin, rmax = float(self.rating_min), float(self.rating_max)
            else:
                # Auto-detect from data
                rmin = min(r for (_, _, r) in rows_all)
                rmax = max(r for (_, _, r) in rows_all)
            if rmax <= rmin:
                rmax = rmin + 1.0
            self.original_min = rmin
            self.original_max = rmax

        # Optional: subsample to limit size before building dense matrix
        rows_filtered: List[Tuple[str, str, float]] = rows_all
        # 1) Limit by max_ratings (truncate after shuffling for randomness)
        if self.max_ratings is not None and self.max_ratings > 0 and len(rows_filtered) > self.max_ratings:
            rng = random.Random(2025)
            rng.shuffle(rows_filtered)
            rows_filtered = rows_filtered[: self.max_ratings]
        # 2) Popularity filter for users/items to keep dense block small
        if self.max_users is not None or self.max_items is not None:
            from collections import Counter
            user_cnt = Counter(u for (u, _, _) in rows_filtered)
            item_cnt = Counter(i for (_, i, _) in rows_filtered)
            if self.max_users is not None and self.max_users > 0:
                top_users = {u for u, _ in user_cnt.most_common(self.max_users)}
            else:
                top_users = set(user_cnt.keys())
            if self.max_items is not None and self.max_items > 0:
                top_items = {i for i, _ in item_cnt.most_common(self.max_items)}
            else:
                top_items = set(item_cnt.keys())
            rows_filtered = [(u, it, r) for (u, it, r) in rows_filtered if u in top_users and it in top_items]
            # If still too many after filtering by popularity and max_ratings wasn't used, we can truncate again
            if self.max_ratings is not None and self.max_ratings > 0 and len(rows_filtered) > self.max_ratings:
                rows_filtered = rows_filtered[: self.max_ratings]

        # Apply normalization if requested: r' = (r - rmin)/(rmax - rmin)
        if self.normalize_ratings and self.original_min is not None and self.original_max is not None:
            scale = self.original_max - self.original_min
            def scale_r(r: float) -> float:
                return (r - self.original_min) / scale
            rows = [(u, it, scale_r(r)) for (u, it, r) in rows_filtered]
        else:
            rows = rows_filtered

        # Map IDs to contiguous indices
        for user_id, item_id, _ in rows:
            if user_id not in self.user_id_to_index:
                self.user_id_to_index[user_id] = len(self.index_to_user_id)
                self.index_to_user_id.append(user_id)
            if item_id not in self.item_id_to_index:
                self.item_id_to_index[item_id] = len(self.index_to_item_id)
                self.index_to_item_id.append(item_id)

        self.num_users = len(self.index_to_user_id)
        self.num_items = len(self.index_to_item_id)

        # Build dense M and Ω
        self.M = torch.zeros((self.num_users, self.num_items), dtype=torch.float32)
        self.omega: List[IndexTriple] = []
        for user_id, item_id, rating in rows:
            ui = self.user_id_to_index[user_id]
            ii = self.item_id_to_index[item_id]
            self.M[ui, ii] = rating
            self.omega.append((ui, ii, rating))

    def train_val_test_split(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, List[IndexTriple]]:
        assert 0 <= val_ratio < 1 and 0 <= test_ratio < 1 and val_ratio + test_ratio < 1
        rng = random.Random(seed)
        indices = list(range(len(self.omega)))
        rng.shuffle(indices)

        n = len(indices)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

        train = [self.omega[i] for i in train_idx]
        val = [self.omega[i] for i in val_idx]
        test = [self.omega[i] for i in test_idx]
        return {"train": train, "val": val, "test": test}

    @staticmethod
    def sample_minibatches(
        omega: List[IndexTriple],
        batch_size: int,
        seed: int | None = None,
    ) -> List[List[IndexTriple]]:
        rng = random.Random(seed)
        shuffled = omega[:]  # copy
        rng.shuffle(shuffled)
        batches: List[List[IndexTriple]] = []
        for start in range(0, len(shuffled), batch_size):
            batches.append(shuffled[start : start + batch_size])
        return batches


