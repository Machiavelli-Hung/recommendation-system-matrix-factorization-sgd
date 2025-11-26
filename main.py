import argparse
import os
import torch

from src.dataset import RatingsDataset
from src.model import MFModel
from src.train import train_sgd
from src.utils import evaluate_rmse, save_checkpoint, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matrix Factorization via SGD (PyTorch)")
    parser.add_argument("--data_dir", type=str, default="matrix_factorization/src/data",
                        help="Directory containing Users.csv, Books.csv, Ratings.csv (Book-Crossing style)")
    parser.add_argument("--data_csv", type=str, default="",
                        help="Optional path to a single ratings CSV with columns [userId,itemId,rating]")
    parser.add_argument("--max_users", type=int, default=10000, help="Limit number of users (most active). 0 = no limit")
    parser.add_argument("--max_items", type=int, default=10000, help="Limit number of items (most popular). 0 = no limit")
    parser.add_argument("--max_ratings", type=int, default=1000000, help="Limit number of rating rows. 0 = no limit")
    parser.add_argument("--normalize_ratings", action="store_true", help="Scale ratings to [0,1] before training")
    parser.add_argument("--rating_min", type=float, default=None, help="Custom min rating for normalization (optional)")
    parser.add_argument("--rating_max", type=float, default=None, help="Custom max rating for normalization (optional)")
    parser.add_argument("--k", type=int, default=20, help="Latent dimension")
    parser.add_argument("--alpha", type=float, default=1e-2, help="Learning rate for SGD")
    parser.add_argument("--lambda_", type=float, default=1e-3, help="L2 regularization strength λ")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4096, help="Minibatch size from Ω")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio from Ω")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio from Ω")
    parser.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Optional path to load a checkpoint before training/eval")
    parser.add_argument("--save_dir", type=str, default="matrix_factorization/checkpoints",
                        help="Directory to save trained checkpoints")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    # Load dataset and build Ω splits
    data_source = args.data_csv if args.data_csv else args.data_dir
    ds = RatingsDataset(
        data_source,
        max_users=None if args.max_users == 0 else args.max_users,
        max_items=None if args.max_items == 0 else args.max_items,
        max_ratings=None if args.max_ratings == 0 else args.max_ratings,
        normalize_ratings=bool(args.normalize_ratings),
        rating_min=args.rating_min,
        rating_max=args.rating_max,
    )
    num_users, num_items = ds.num_users, ds.num_items

    splits = ds.train_val_test_split(val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    omega_train = splits["train"]
    omega_val = splits["val"]
    omega_test = splits["test"]

    #### See data
    print("omega_train length =", len(omega_train))
    print("omega_val length   =", len(omega_val))
    print("omega_test length  =", len(omega_test))

    print("\nSample 10 train rows:")
    u, i, r = omega_train[0]
    print("user_id:", u)
    print("item_id:", i)
    print("rating:", r)
    for row in omega_train[:100]:
        print(row)
    print("\nSample 10 val rows:")
    for row in omega_val[:100]:
        print(row)
    print("\nSample 10 test rows:")
    for row in omega_test[:100]:
        print(row)
     
    ####

    ###check point 1 :return

    ####
    # Initialize model
    model = MFModel(num_users=num_users, num_items=num_items, k=args.k, device=device)

    # Optionally load checkpoint
    if args.checkpoint and os.path.isfile(args.checkpoint):
        load_checkpoint(args.checkpoint, model)

    # Train
    train_sgd(
        model=model,
        omega_train=omega_train,
        omega_val=omega_val,
        ratings_matrix=ds.M,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        lambda_=args.lambda_,
        device=device,
        verbose=True,
    )

    # Evaluate on test set
    test_rmse = evaluate_rmse(model, omega_test, ds.M, device=device)
    print(f"Test RMSE: {test_rmse:.6f}")

    # Save checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "mf_model.pt")
    save_checkpoint(ckpt_path, model, ds.user_id_to_index, ds.item_id_to_index)
    print(f"Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()


