CUDA_VISIBLE_DEVICES=7 python train.py \
--dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/pix2pix/AB \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name pix2pix_synthrad_abdomen_final \
--model pix2pix \
--direction AtoB \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--print_freq 100 \
--save_epoch_freq 1





CUDA_VISIBLE_DEVICES=7 python train.py \
--dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/brain/pix2pix/AB \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name pix2pix_synthrad_brain_final \
--model pix2pix \
--direction AtoB \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--print_freq 100 \
--save_epoch_freq 1






CUDA_VISIBLE_DEVICES=7 python train.py \
--dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/HN/pix2pix/AB \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name pix2pix_synthrad_headneck_final \
--model pix2pix \
--direction AtoB \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--print_freq 100 \
--save_epoch_freq 1





CUDA_VISIBLE_DEVICES=6 python train.py \
--dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/pelvis/pix2pix/AB \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name pix2pix_synthrad_pelvis_final \
--model pix2pix \
--direction AtoB \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--print_freq 100 \
--save_epoch_freq 1





CUDA_VISIBLE_DEVICES=6 python train.py \
--dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/TH/pix2pix/AB \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name pix2pix_synthrad_thorax_final \
--model pix2pix \
--direction AtoB \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--print_freq 100 \
--save_epoch_freq 1



CUDA_VISIBLE_DEVICES=7 python train.py \
--dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splits_31baseline/pix2pix/AB \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name pix2pix_synthrad_allregion_final \
--model pix2pix \
--direction AtoB \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--save_epoch_freq 1