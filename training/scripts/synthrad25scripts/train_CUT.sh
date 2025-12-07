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