CUDA_VISIBLE_DEVICES=0 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/val \
  --name cyclegan_abdomen \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --num_test 1000 \
  --netG unet_256 \
  --netD basic \
  --epoch all


CUDA_VISIBLE_DEVICES=0 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/brain/cyclegan/val \
  --name cyclegan_brain \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --num_test 1000 \
  --netG unet_256 \
  --netD basic \
  --epoch all



CUDA_VISIBLE_DEVICES=0 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/HN/cyclegan/val \
  --name cyclegan_head_neck \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --num_test 1000 \
  --netG unet_256 \
  --netD basic \
  --epoch all


CUDA_VISIBLE_DEVICES=0 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/pelvis/cyclegan/val \
  --name cyclegan_pelvis \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --num_test 1000 \
  --netG unet_256 \
  --netD basic \
  --epoch all


CUDA_VISIBLE_DEVICES=2 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/TH/cyclegan/val \
  --name cyclegan_thorax \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --num_test 1000 \
  --netG unet_256 \
  --netD basic \
  --epoch all


  CUDA_VISIBLE_DEVICES=2 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splits_31baseline/cyclegan/val \
  --name cyclegan_allregions \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --num_test 50 \
  --netG unet_256 \
  --netD basic \
  --epoch all



  #### FINAL MODELS
CUDA_VISIBLE_DEVICES=1 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splits_31baseline/cyclegan/val \
  --name cyclegan_allregions_final \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --netG unet_256 \
  --netD basic \
  --epoch all


  CUDA_VISIBLE_DEVICES=1 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/val \
  --name cyclegan_abdomen_final \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --netG unet_256 \
  --netD basic \
  --epoch all



  CUDA_VISIBLE_DEVICES=1 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/brain/cyclegan/val \
  --name cyclegan_brain_final \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --netG unet_256 \
  --netD basic \
  --epoch all



  CUDA_VISIBLE_DEVICES=1 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/HN/cyclegan/val \
  --name cyclegan_head_neck_final \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --netG unet_256 \
  --netD basic \
  --epoch all




CUDA_VISIBLE_DEVICES=2 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/pelvis/cyclegan/val \
  --name cyclegan_pelvis_final \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --netG unet_256 \
  --netD basic \
  --epoch all


  CUDA_VISIBLE_DEVICES=1 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/TH/cyclegan/val \
  --name cyclegan_thorax_final \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --netG unet_256 \
  --netD basic \
  --epoch all