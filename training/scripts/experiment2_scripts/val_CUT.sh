CUDA_VISIBLE_DEVICES=2 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/val \
  --name 2_experiment_cut_synthrad_abdomen_sep_first_layer \
  --model cut \
  --CUT_mode CUT \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --preprocess None \
  --netG resnet_9blocks_sep_first_layer \
  --epoch all


  CUDA_VISIBLE_DEVICES=7 python validate_epochs_synth.py \
  --phase val \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/val \
  --name 2_experiment_cut_synthrad_abdomen_32p99 \
  --model cut \
  --CUT_mode CUT \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --preprocess None \
  --epoch all