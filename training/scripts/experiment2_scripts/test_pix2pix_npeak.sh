cd ../..

CUDA_VISIBLE_DEVICES=4 python test_synth.py \
    --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_03LIC/7materialized_splits_BodyRegion/AB/pix2pix/AB \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 2_experiment_pix2pix_synthrad_abdomen_34normalized_n4_03LIC \
    --model pix2pix \
    --phase test \
    --direction AtoB \
    --input_nc 1 \
    --output_nc 1 \
    --preprocess None \
    --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results/ \
    --epoch 50


CUDA_VISIBLE_DEVICES=4 python test_synth.py \
    --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_08LIC/7materialized_splits_BodyRegion/AB/pix2pix/AB \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 2_experiment_pix2pix_synthrad_abdomen_34normalized_n4_08LIC \
    --model pix2pix \
    --phase test \
    --direction AtoB \
    --input_nc 1 \
    --output_nc 1 \
    --preprocess None \
    --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results/ \
    --epoch 50


CUDA_VISIBLE_DEVICES=4 python test_synth.py \
    --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_centerspecific_03LIC/7materialized_splits_BodyRegion/AB/pix2pix/AB \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 2_experiment_pix2pix_synthrad_abdomen_34normalized_n4_centerspecific_03LIC \
    --model pix2pix \
    --phase test \
    --direction AtoB \
    --input_nc 1 \
    --output_nc 1 \
    --preprocess None \
    --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results/ \
    --epoch 50


CUDA_VISIBLE_DEVICES=4 python test_synth.py \
    --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/34normalized_n4_centerspecific_08LIC/7materialized_splits_BodyRegion/AB/pix2pix/AB \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 2_experiment_pix2pix_synthrad_abdomen_34normalized_n4_centerspecific_08LIC \
    --model pix2pix \
    --phase test \
    --direction AtoB \
    --input_nc 1 \
    --output_nc 1 \
    --preprocess None \
    --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results/ \
    --epoch 50
