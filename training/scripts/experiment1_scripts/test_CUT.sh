
cd ../../../training_cut

CUDA_VISIBLE_DEVICES=3 python test_synth.py \
 --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splits_31baseline/cyclegan/test' \
 --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
 --phase test \
 --name cut_synthrad_allregions_final \
 --model cut \
 --CUT_mode CUT \
 --input_nc 1 \
 --output_nc 1 \
 --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
 --epoch 50


CUDA_VISIBLE_DEVICES=3 python test_synth.py \
 --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/test' \
 --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
 --phase test \
 --name cut_synthrad_abdomen_final \
 --model cut \
 --CUT_mode CUT \
 --input_nc 1 \
 --output_nc 1 \
 --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
 --epoch 50


CUDA_VISIBLE_DEVICES=3 python test_synth.py \
 --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/brain/cyclegan/test' \
 --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
 --phase test \
 --name cut_synthrad_brain_final \
 --model cut \
 --CUT_mode CUT \
 --input_nc 1 \
 --output_nc 1 \
 --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
 --epoch 50


CUDA_VISIBLE_DEVICES=3 python test_synth.py \
 --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/HN/cyclegan/test' \
 --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
 --phase test \
 --name cut_synthrad_HN_final \
 --model cut \
 --CUT_mode CUT \
 --input_nc 1 \
 --output_nc 1 \
 --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
 --epoch 50


CUDA_VISIBLE_DEVICES=3 python test_synth.py \
 --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/pelvis/cyclegan/test' \
 --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
 --phase test \
 --name cut_synthrad_pelvis_final \
 --model cut \
 --CUT_mode CUT \
 --input_nc 1 \
 --output_nc 1 \
 --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
 --epoch 50


CUDA_VISIBLE_DEVICES=3 python test_synth.py \
 --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/TH/cyclegan/test' \
 --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
 --phase test \
 --name cut_synthrad_TH_final \
 --model cut \
 --CUT_mode CUT \
 --input_nc 1 \
 --output_nc 1 \
 --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
 --epoch 50
