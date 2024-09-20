# Digit 1
python3 train_mnist.py --dataset MNIST --digit 1 --z_dim_min 1 --z_dim_max 16 --structure_dim 64 --batch_size 256 --batch_size_eval 512 --epochs 50000 --learning_rate 1e-4 --scheduler 1 --iter_gq 1 --iter_d 5 --lambda_mmd 1.0 --lambda_gp 5.0 --lambda_rank 0.002 --device cuda
sleep 2m

# Digit 2
python3 train_mnist.py --dataset MNIST --digit 2 --z_dim_min 1 --z_dim_max 16 --structure_dim 64 --batch_size 256 --batch_size_eval 512 --epochs 50000 --learning_rate 1e-4 --scheduler 1 --iter_gq 1 --iter_d 5 --lambda_mmd 1.0 --lambda_gp 5.0 --lambda_rank 0.002 --device cuda
sleep 2m

# All digits
python3 train_mnist.py --dataset MNIST --z_dim_min 1 --z_dim_max 20 --structure_dim 64 --batch_size 256 --batch_size_eval 512 --epochs 50000 --learning_rate 1e-4 --scheduler 1 --iter_gq 3 --iter_d 10 --lambda_mmd 1.0 --lambda_gp 5.0 --lambda_rank 0.0005 --device cuda
