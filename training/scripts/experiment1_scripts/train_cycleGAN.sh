CUDA_VISIBLE_DEVICES=6 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cyclegan_abdomen \
--model cycle_gan \
--input_nc 1 \
--output_nc 1 \
--batch_size 32 \
--netG unet_256 \
--netD basic



CUDA_VISIBLE_DEVICES=5 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/brain/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cyclegan_brain \
--model cycle_gan \
--input_nc 1 \
--output_nc 1 \
--batch_size 32 \
--netG unet_256 \
--netD basic



CUDA_VISIBLE_DEVICES=4 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/HN/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cyclegan_head_neck \
--model cycle_gan \
--input_nc 1 \
--output_nc 1 \
--batch_size 32 \
--netG unet_256 \
--netD basic


CUDA_VISIBLE_DEVICES=3 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/pelvis/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cyclegan_pelvis \
--model cycle_gan \
--input_nc 1 \
--output_nc 1 \
--batch_size 32 \
--netG unet_256 \
--netD basic




CUDA_VISIBLE_DEVICES=2 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/TH/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cyclegan_thorax \
--model cycle_gan \
--input_nc 1 \
--output_nc 1 \
--batch_size 32 \
--netG unet_256 \
--netD basic






CUDA_VISIBLE_DEVICES=5 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splits_31baseline/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cyclegan_allregions_final \
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



CUDA_VISIBLE_DEVICES=7 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/brain/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cyclegan_brain_final \
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


CUDA_VISIBLE_DEVICES=7 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/TH/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cyclegan_thorax_final \
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




CUDA_VISIBLE_DEVICES=7 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/pelvis/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cyclegan_pelvis_final \
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




CUDA_VISIBLE_DEVICES=6 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/HN/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cyclegan_head_neck_final \
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





#TODO
CUDA_VISIBLE_DEVICES=7 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cyclegan_abdomen_final \
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




CUDA_VISIBLE_DEVICES=1 python train.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/pelvis/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cyclegan_pelvis_final \
--model cycle_gan \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--netG unet_256 \
--netD basic \
--continue_train
