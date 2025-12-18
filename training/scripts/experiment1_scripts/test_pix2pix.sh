
#test abdomen
CUDA_VISIBLE_DEVICES=2 python test_synth.py \
  --phase test \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/pix2pix/AB \
  --name pix2pix_synthrad_abdomen \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results






  CUDA_VISIBLE_DEVICES=2 python test_synth.py \
  --phase test \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/brain/pix2pix/AB \
  --name pix2pix_synthrad_brain \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results





    CUDA_VISIBLE_DEVICES=2 python test_synth.py \
  --phase test \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/HN/pix2pix/AB \
  --name pix2pix_synthrad_head_neck \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results






      CUDA_VISIBLE_DEVICES=2 python test_synth.py \
  --phase test \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/pelvis/pix2pix/AB \
  --name pix2pix_synthrad_pelvis \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results



        CUDA_VISIBLE_DEVICES=2 python test_synth.py \
  --phase test \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/TH/pix2pix/AB \
  --name pix2pix_synthrad_thorax \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results







CUDA_VISIBLE_DEVICES=2 python test_synth.py \
  --phase test \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splits_31baseline/pix2pix/AB \
  --name pix2pix_synthrad_allregions \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results








  CUDA_VISIBLE_DEVICES=0 python test_synth.py \
  --phase test \
  --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/pix2pix/AB \
  --name pix2pix_synthrad_abdomen_bs1 \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
  --model pix2pix \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \