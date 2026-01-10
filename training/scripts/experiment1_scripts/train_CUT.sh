CUDA_VISIBLE_DEVICES=5 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_THUF \
--model cut \
--CUT_mode CUT \
--input_nc 1 \
--output_nc 1 \
--batch_size 200 \
--netG unet_256\
--netD basic


##ensure that you execute the script in the traing_cut folder!!! ## 
cd /fullbody-sCT/training_cut/

CUDA_VISIBLE_DEVICES=7 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_synthrad_abdomen \
--model cut \
--CUT_mode CUT \
--input_nc 1 \
--output_nc 1 \
--batch_size 200 







CUDA_VISIBLE_DEVICES=1 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splits_31baseline/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_synthrad_allregions2 \
--model cut \
--CUT_mode CUT \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--preprocess None \
--print_freq 1000 \
--lambda_NCE 1 \
--lambda_GAN 2 \
--lr 0.0001 \
--gan_mode nonsaturating \
--continue_train \
--epoch latest \
--epoch_count 11

##epoch 15 11056 sec



CUDA_VISIBLE_DEVICES=3 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/brain/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_synthrad_brain \
--model cut \
--CUT_mode CUT \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--n_epochs 50 \
--n_epochs_decay 50 \
--no_html \
--preprocess none \
--print_freq 1000 \
--lambda_NCE 1 \
--lambda_GAN 2 \
--lr 0.0001 \
--pool_size 40 \
--gan_mode nonsaturating \
--num_patches 64 \





CUDA_VISIBLE_DEVICES=7 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/HN/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_synthrad_HN \
--model cut \
--CUT_mode CUT \
--output_nc 1 \
--input_nc 1 \
--batch_size 1 \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--preprocess None \
--print_freq 1000 \
--gan_mode nonsaturating \
--display_id -1 \
--continue_train \
--epoch latest \
--epoch_count 14



CUDA_VISIBLE_DEVICES=1 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/brain/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_synthrad_brain \
--model cut \
--CUT_mode CUT \
--output_nc 1 \
--input_nc 1 \
--batch_size 1 \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--preprocess None \
--print_freq 1000 \
--gan_mode nonsaturating \
--display_id -1 \
--continue_train \
--epoch latest \
--epoch_count 10



CUDA_VISIBLE_DEVICES=1 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/brain/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_synthrad_brain \
--model cut \
--CUT_mode CUT \
--output_nc 1 \
--input_nc 1 \
--batch_size 1 \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--preprocess None \
--print_freq 1000 \
--gan_mode nonsaturating \
--display_id -1 \
--continue_train \
--epoch latest \
--epoch_count 10



CUDA_VISIBLE_DEVICES=1 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_synthrad_AB \
--model cut \
--CUT_mode CUT \
--output_nc 1 \
--input_nc 1 \
--batch_size 1 \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--preprocess None \
--print_freq 1000 \
--gan_mode nonsaturating \
--display_id -1


CUDA_VISIBLE_DEVICES=7 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/pelvis/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_synthrad_pelvis \
--model cut \
--CUT_mode CUT \
--output_nc 1 \
--input_nc 1 \
--batch_size 1 \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--preprocess None \
--print_freq 1000 \
--gan_mode nonsaturating \
--display_id -1



CUDA_VISIBLE_DEVICES=4 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splits_31baseline/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_synthrad_allregions_final \
--model cut \
--CUT_mode CUT \
--output_nc 1 \
--input_nc 1 \
--batch_size 1 \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--preprocess None \
--print_freq 100 \
--save_epoch_freq 1


CUDA_VISIBLE_DEVICES=1 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/brain/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_synthrad_brain_final \
--model cut \
--CUT_mode CUT \
--output_nc 1 \
--input_nc 1 \
--batch_size 1 \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--preprocess None \
--print_freq 100 \
--save_epoch_freq 1



CUDA_VISIBLE_DEVICES=1 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/HN/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_synthrad_HN_final \
--model cut \
--CUT_mode CUT \
--output_nc 1 \
--input_nc 1 \
--batch_size 1 \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--preprocess None \
--print_freq 100 \
--save_epoch_freq 1



CUDA_VISIBLE_DEVICES=2 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/HN/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_synthrad_HN_final \
--model cut \
--CUT_mode CUT \
--output_nc 1 \
--input_nc 1 \
--batch_size 1 \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--preprocess None \
--print_freq 100 \
--save_epoch_freq 1 \
--continue_train \
--epoch latest \
--epoch_count 92





CUDA_VISIBLE_DEVICES=1 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/pelvis/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_synthrad_pelvis_final \
--model cut \
--CUT_mode CUT \
--output_nc 1 \
--input_nc 1 \
--batch_size 1 \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--preprocess None \
--print_freq 100 \
--save_epoch_freq 1



CUDA_VISIBLE_DEVICES=2 python train.py \
  --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/pelvis/cyclegan/train' \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --name cut_synthrad_pelvis_final \
  --model cut \
  --CUT_mode CUT \
  --output_nc 1 \
  --input_nc 1 \
  --batch_size 1 \
  --n_epochs 100 \
  --n_epochs_decay 0 \
  --no_html \
  --preprocess None \
  --print_freq 100 \
  --save_epoch_freq 1 \
  --continue_train \
  --epoch latest \
  --epoch_count 87




##TODO

CUDA_VISIBLE_DEVICES=6 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_synthrad_abdomen_final \
--model cut \
--CUT_mode CUT \
--output_nc 1 \
--input_nc 1 \
--batch_size 1 \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--preprocess None \
--print_freq 100 \
--save_epoch_freq 1


CUDA_VISIBLE_DEVICES=6 python train.py \
--dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/TH/cyclegan/train' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name cut_synthrad_TH_final \
--model cut \
--CUT_mode CUT \
--output_nc 1 \
--input_nc 1 \
--batch_size 1 \
--n_epochs 100 \
--n_epochs_decay 0 \
--no_html \
--preprocess None \
--print_freq 100 \
--save_epoch_freq 1