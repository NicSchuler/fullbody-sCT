CUDA_VISIBLE_DEVICES=5 python train_pure_swinv2.py \
    --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/pix2pix/AB \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 3_experiment_pure_abdomen_swin_v2_low_lr \
    --input_nc 1 \
    --output_nc 1 \
    --n_epochs 100 \
    --n_epochs_decay 0 \
    --lr 0.00002 \
    --save_epoch_freq 1

CUDA_VISIBLE_DEVICES=4 python train_pure_swinv2.py \
    --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/pix2pix/AB \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 3_experiment_pure_abdomen_swin_v2_frozen_low_lr \
    --input_nc 1 \
    --output_nc 1 \
    --n_epochs 100 \
    --n_epochs_decay 0 \
    --lr 0.00002 \
    --save_epoch_freq 1 \
    --freeze_encoder_except_first