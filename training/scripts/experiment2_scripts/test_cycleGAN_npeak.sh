cd ../..

CUDA_VISIBLE_DEVICES=6 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_03LIC/7materialized_splits_BodyRegion/AB/cyclegan/test' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_cyclegan_abdomen_34normalized_n4_03LIC \
--model cycle_gan \
--phase test \
--input_nc 1 \
--output_nc 1 \
--netG unet_256 \
--netD basic \
--results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
--preprocess None \
--epoch 50


CUDA_VISIBLE_DEVICES=6 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_08LIC/7materialized_splits_BodyRegion/AB/cyclegan/test' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_cyclegan_abdomen_34normalized_n4_08LIC \
--model cycle_gan \
--phase test \
--input_nc 1 \
--output_nc 1 \
--netG unet_256 \
--netD basic \
--results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
--preprocess None \
--epoch 50


CUDA_VISIBLE_DEVICES=6 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_centerspecific_03LIC/7materialized_splits_BodyRegion/AB/cyclegan/test' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_cyclegan_abdomen_34normalized_n4_centerspecific_03LIC \
--model cycle_gan \
--phase test \
--input_nc 1 \
--output_nc 1 \
--netG unet_256 \
--netD basic \
--results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
--preprocess None \
--epoch 50


CUDA_VISIBLE_DEVICES=6 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_centerspecific_08LIC/7materialized_splits_BodyRegion/AB/cyclegan/test' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_cyclegan_abdomen_34normalized_n4_centerspecific_08LIC \
--model cycle_gan \
--phase test \
--input_nc 1 \
--output_nc 1 \
--netG unet_256 \
--netD basic \
--results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
--preprocess None \
--epoch 50
