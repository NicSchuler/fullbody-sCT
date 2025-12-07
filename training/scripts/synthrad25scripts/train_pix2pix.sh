CUDA_VISIBLE_DEVICES=0 python train.py \
--dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/HN/pix2pix/AB \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name pix2pix_synthrad_head_neck \
--model pix2pix \
--direction AtoB \
--input_nc 1 \
--output_nc 1 \
--batch_size 96