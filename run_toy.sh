python train_toy.py --dataset scurve     --x_dim 3 --z_dim 5 --q_dim 64 --g_dim 64 --d_dim 64 --epochs 10000 --learning_rate 2e-4 --scheduler 1 --iter_gq 1 --iter_d 20 --lambda_mmd 1.0 --lambda_rank 0.01 --device cuda

python train_toy.py --dataset swissroll  --x_dim 2 --z_dim 5 --q_dim 64 --g_dim 64 --d_dim 64 --epochs  5000 --learning_rate 2e-4 --scheduler 1 --iter_gq 5 --iter_d 20 --lambda_mmd 1.0 --lambda_rank 0.01 --device cuda

python train_toy.py --dataset hyperplane --x_dim 5 --z_dim 7 --q_dim 64 --g_dim 64 --d_dim 64 --epochs 10000 --learning_rate 2e-4 --scheduler 1 --iter_gq 1 --iter_d 20 --lambda_mmd 1.0 --lambda_rank 0.01 --device cuda
