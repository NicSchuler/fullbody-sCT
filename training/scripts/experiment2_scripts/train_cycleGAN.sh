CUDA_VISIBLE_DEVICES=4 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_cyclegan_abdomen_sep_first_layer \
--model cycle_gan  \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--netG unet_256_sep_first_layer \
--netD basic \
--n_epochs 100 \
--n_epochs_decay 0 \
--print_freq 100 \
--save_epoch_freq 1





CUDA_VISIBLE_DEVICES=7 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/32p99/7materialized_splits_BodyRegion/AB/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_cyclegan_abdomen_32p99 \
--model cycle_gan \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--netG unet_256 \
--netD basic \
--n_epochs 100 \
--n_epochs_decay 0 \
--print_freq 100 \
--save_epoch_freq 1




CUDA_VISIBLE_DEVICES=3 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/33nyul/7materialized_splits_BodyRegion/AB/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_cyclegan_abdomen_33nyul \
--model cycle_gan \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--netG unet_256 \
--netD basic \
--n_epochs 100 \
--n_epochs_decay 0 \
--print_freq 100 \
--save_epoch_freq 1



CUDA_VISIBLE_DEVICES=6 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_03LIC/7materialized_splits_BodyRegion/AB/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_cyclegan_abdomen_34normalized_n4_03LIC \
--model cycle_gan \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--netG unet_256 \
--netD basic \
--n_epochs 100 \
--n_epochs_decay 0 \
--print_freq 100 \
--save_epoch_freq 1



CUDA_VISIBLE_DEVICES=5 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_08LIC/7materialized_splits_BodyRegion/AB/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_cyclegan_abdomen_34normalized_n4_08LIC \
--model cycle_gan \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--netG unet_256 \
--netD basic \
--n_epochs 100 \
--n_epochs_decay 0 \
--print_freq 100 \
--save_epoch_freq 1




CUDA_VISIBLE_DEVICES=5 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_centerspecific_03LIC/7materialized_splits_BodyRegion/AB/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_cyclegan_abdomen_34normalized_n4_centerspecific_03LIC \
--model cycle_gan \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--netG unet_256 \
--netD basic \
--n_epochs 100 \
--n_epochs_decay 0 \
--print_freq 100 \
--save_epoch_freq 1


CUDA_VISIBLE_DEVICES=3 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_centerspecific_08LIC/7materialized_splits_BodyRegion/AB/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_cyclegan_abdomen_34normalized_n4_centerspecific_08LIC \
--model cycle_gan \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--netG unet_256 \
--netD basic \
--n_epochs 100 \
--n_epochs_decay 0 \
--print_freq 100 \
--save_epoch_freq 1