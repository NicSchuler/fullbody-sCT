cd ../..


CUDA_VISIBLE_DEVICES=7 python test_synth.py  \
    --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/test' \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 2_experiment_cyclegan_abdomen_sep_first_layer \
    --model cycle_gan  \
    --phase test \
    --input_nc 1 \
    --output_nc 1 \
    --netG unet_256_sep_first_layer \
    --netD basic \
    --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
    --preprocess None \
    --epoch 50





CUDA_VISIBLE_DEVICES=7 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/32p99/7materialized_splits_BodyRegion/AB/cyclegan/test' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_cyclegan_abdomen_32p99 \
--model cycle_gan \
--input_nc 1 \
--output_nc 1 \
--netG unet_256 \
--netD basic \
--results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
--preprocess None \
--epoch 50




CUDA_VISIBLE_DEVICES=7 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/33nyul/7materialized_splits_BodyRegion/AB/cyclegan/test' \
--checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
--name 2_experiment_cyclegan_abdomen_33nyul \
--model cycle_gan \
--input_nc 1 \
--output_nc 1 \
--netG unet_256 \
--netD basic \
--results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
--preprocess None \
--epoch 50