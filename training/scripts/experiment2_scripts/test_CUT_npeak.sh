cd ../../../training_cut

CUDA_VISIBLE_DEVICES=4 python test_synth.py \
    --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_03LIC/7materialized_splits_BodyRegion/AB/cyclegan/test' \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 2_experiment_cut_synthrad_abdomen_34normalized_n4_03LIC \
    --phase test \
    --model cut \
    --CUT_mode CUT \
    --output_nc 1 \
    --input_nc 1 \
    --batch_size 1 \
    --preprocess None \
    --epoch 50 \
    --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results


CUDA_VISIBLE_DEVICES=4 python test_synth.py \
    --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_08LIC/7materialized_splits_BodyRegion/AB/cyclegan/test' \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 2_experiment_cut_synthrad_abdomen_34normalized_n4_08LIC \
    --phase test \
    --model cut \
    --CUT_mode CUT \
    --output_nc 1 \
    --input_nc 1 \
    --batch_size 1 \
    --preprocess None \
    --epoch 50 \
    --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results


CUDA_VISIBLE_DEVICES=4 python test_synth.py \
    --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_centerspecific_03LIC/7materialized_splits_BodyRegion/AB/cyclegan/test' \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 2_experiment_cut_synthrad_abdomen_34normalized_n4_centerspecific_03LIC \
    --phase test \
    --model cut \
    --CUT_mode CUT \
    --output_nc 1 \
    --input_nc 1 \
    --batch_size 1 \
    --preprocess None \
    --epoch 50 \
    --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results


CUDA_VISIBLE_DEVICES=4 python test_synth.py \
    --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_centerspecific_08LIC/7materialized_splits_BodyRegion/AB/cyclegan/test' \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 2_experiment_cut_synthrad_abdomen_34normalized_n4_centerspecific_08LIC \
    --phase test \
    --model cut \
    --CUT_mode CUT \
    --output_nc 1 \
    --input_nc 1 \
    --batch_size 1 \
    --preprocess None \
    --epoch 50 \
    --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results
