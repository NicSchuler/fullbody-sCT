CUDA_VISIBLE_DEVICES=0 python train.py \
--dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/pix2pix/AB \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_pix2pix_synthrad_abdomen_sep_input_layers \
--model pix2pix \
--direction AtoB \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--preprocess None \
--netG unet_256_sep_first_layer \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--save_epoch_freq 1






CUDA_VISIBLE_DEVICES=7 python train.py \
--dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/32p99/7materialized_splits_BodyRegion/AB/pix2pix/AB \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_pix2pix_synthrad_abdomen_32p99 \
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




CUDA_VISIBLE_DEVICES=0 python train.py \
--dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/33nyul/7materialized_splits_BodyRegion/AB/pix2pix/AB \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_pix2pix_synthrad_abdomen_33nyul \
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