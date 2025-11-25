python -m matrix*factorization.main --data_dir matrix_factorization/src/data --k 32 --alpha 0.01 --lambda* 1e-3 --epochs 5 --batch_size 8192 --device auto

python -m matrix*factorization.main \
 --ratings_csv /home/hung/Hung/src/src/bai-tap-tren-lop/recommandation-system<Matrix_Factorization>/matrix_factorization/src/data/Ratings.csv \
 --k 40 \
 --epochs 25 \
 --batch_size 2048 \
 --alpha 5e-3 \
 --lambda* 1e-3 \
 --max_users 20000 \
 --max_items 20000 \
 --max_ratings 200000 \
 --val_ratio 0.1 \
 --test_ratio 0.1 \
 --device auto \
 --seed 42 \
 --save_dir /home/hung/Hung/src/src/bai-tap-tren-lop/recommandation-system<Matrix_Factorization>/matrix_factorization/checkpoints

python -m matrix*factorization.main --ratings_csv '/home/hung/Hung/src/src/bai-tap-tren-lop/recommandation-system<Matrix-Factorization>/matrix_factorization/src/data/Ratings.csv' --k 40 --epochs 1 --batch_size 2048 --alpha 5e-3 --lambda* 1e-3 --max_users 200 --max_items 200 --max_ratings 5000 --val_ratio 0.1 --test_ratio 0.1 --device auto --seed 42 --save_dir '/home/hung/Hung/src/src/bai-tap-tren-lop/recommandation-system<Matrix-Factorization>/matrix_factorization/checkpoints'

##### final

python main.py --data*csv '/home/hung/Hung/src/src/bai-tap-tren-lop/recommandation-system/src/data/Ratings.csv' --k 40 --epochs 1 --batch_size 2048 --alpha 5e-3 --lambda* 1e-3 --max_users 200 --max_items 200 --max_ratings 5000 --val_ratio 0.1 --test_ratio 0.1 --device auto --seed 42 --save_dir '/home/hung/Hung/src/src/bai-tap-tren-lop/recommandation-system/src/checkpoints'
