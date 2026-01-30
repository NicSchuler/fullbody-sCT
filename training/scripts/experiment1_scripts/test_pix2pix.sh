
cd ../..

#test abdomen
CUDA_VISIBLE_DEVICES=3 python test_synth.py \
  --phase test \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/pix2pix/AB \
  --name pix2pix_synthrad_abdomen_final \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results/ \
  --epoch 50


  CUDA_VISIBLE_DEVICES=3 python test_synth.py \
  --phase test \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/brain/pix2pix/AB \
  --name pix2pix_synthrad_brain_final \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch 50

CUDA_VISIBLE_DEVICES=3 python test_synth.py \
  --phase test \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/HN/pix2pix/AB \
  --name pix2pix_synthrad_headneck_final \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch 50


  CUDA_VISIBLE_DEVICES=3 python test_synth.py \
  --phase test \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/pelvis/pix2pix/AB \
  --name pix2pix_synthrad_pelvis_final \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch 50

CUDA_VISIBLE_DEVICES=3 python test_synth.py \
  --phase test \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/TH/pix2pix/AB \
  --name pix2pix_synthrad_thorax_final \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch 50

CUDA_VISIBLE_DEVICES=3 python test_synth.py \
  --phase test \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splits_31baseline/pix2pix/AB \
  --name pix2pix_synthrad_allregion_final \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch 50


