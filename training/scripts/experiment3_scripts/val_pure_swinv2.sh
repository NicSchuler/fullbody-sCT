# validation script of pix2pix still works as it only needs/loads the generator
CUDA_VISIBLE_DEVICES=6 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/pix2pix/AB \
  --name 3_experiment_pure_abdomen_swin_v2_low_lr \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --netG swin_v2_unet_256 \
  --epoch all 

  CUDA_VISIBLE_DEVICES=6 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/pix2pix/AB \
  --name 3_experiment_pure_abdomen_swin_v2_frozen_low_lr \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --netG swin_v2_unet_256 \
  --epoch all 