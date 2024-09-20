# LWGAN
python3 train_celeba.py --data_dir ./data/celeba/ --z_dim_min 1 --z_dim_max 128 --structure_dim 64 --batch_size 128 --batch_size_eval 512 --epochs 100000 --learning_rate 1e-4 --weight_decay 0.0 --scheduler 1 --iter_gq 5 --iter_d 10 --lambda_mmd 1.0 --lambda_gp 5.0 --lambda_rank 0.002 --device cuda
sleep 30m

# WGAN
python3 train_celeba_wgan.py --data_dir ./data/celeba/ --z_dim_min 1 --z_dim_max 128 --structure_dim 64 --batch_size 128 --batch_size_eval 512 --epochs 100000 --learning_rate 2e-4 --weight_decay 0.0 --scheduler 1 --iter_gq 1 --iter_d 10 --lambda_gp 20.0 --device cuda
sleep 30m

# WAE
python3 train_celeba_wae.py --data_dir ./data/celeba/ --z_dim_min 1 --z_dim_max 128 --structure_dim 64 --batch_size 128 --batch_size_eval 512 --epochs 100000 --learning_rate 5e-4 --weight_decay 0.0 --scheduler 1 --iter_gq 1 --iter_d 10 --lambda_mmd 100.0 --device cuda
sleep 30m

# CycleGAN
python3 train_celeba_cyclegan.py --data_dir ./data/celeba/ --z_dim_min 1 --z_dim_max 128 --structure_dim 64 --batch_size 128 --batch_size_eval 512 --epochs 10000 --learning_rate 1e-4 --weight_decay 0.0 --scheduler 1 --iter_gq 5 --iter_d 15 --lambda_cycle 10.0 --device cuda
