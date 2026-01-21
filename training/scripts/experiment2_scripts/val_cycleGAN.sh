CUDA_VISIBLE_DEVICES=2 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/val \
  --name 2_experiment_cyclegan_abdomen_sep_first_layer \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --batch_size 1 \
  --preprocess None \
  --netG unet_256_sep_first_layer \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch all








CUDA_VISIBLE_DEVICES=2 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/val \
  --name 2_experiment_cyclegan_abdomen_32p99 \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --batch_size 1 \
  --preprocess None \
  --netG unet_256 \
  --netD basic \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch all




CUDA_VISIBLE_DEVICES=2 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/val \
  --name 2_experiment_cyclegan_abdomen_33nyul \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --batch_size 1 \
  --preprocess None \
  --netG unet_256 \
  --netD basic \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch all