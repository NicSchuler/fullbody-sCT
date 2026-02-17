CUDA_VISIBLE_DEVICES=1 python train.py \
--dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/pix2pix/AB \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 3_experiment_pix2pix_abdomen_swin_v2_low_lr \
--model pix2pix \
--direction AtoB \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--save_epoch_freq 1 \
--netG swin_v2_unet_256 \
--lr 0.00002

CUDA_VISIBLE_DEVICES=4 python train.py \
--dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/pix2pix/AB \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 3_experiment_pix2pix_abdomen_swin_v2_frozen_low_lr \
--model pix2pix \
--direction AtoB \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--save_epoch_freq 1 \
--netG swin_v2_unet_256_encoder_frozen_except_first \
--lr 0.00002

CUDA_VISIBLE_DEVICES=5 python train.py \
--dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/pix2pix/AB \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 3_experiment_pix2pix_abdomen_resnet \
--model pix2pix \
--direction AtoB \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--save_epoch_freq 1 \
--netG resnet_9blocks