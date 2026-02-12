cd ../..

CUDA_VISIBLE_DEVICES=4 python test_synth.py \
    --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/pix2pix/AB \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 2_experiment_pix2pix_synthrad_abdomen_sep_input_layers \
    --model pix2pix \
    --phase test \
    --direction AtoB \
    --input_nc 1 \
    --output_nc 1 \
    --preprocess None \
    --netG unet_256_sep_first_layer \
    --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results/ \
    --epoch 50


CUDA_VISIBLE_DEVICES=4 python test_synth.py \
    --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/32p99/7materialized_splits_BodyRegion/AB/pix2pix/AB \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 2_experiment_pix2pix_synthrad_abdomen_32p99 \
    --model pix2pix \
    --phase test \
    --direction AtoB \
    --input_nc 1 \
    --output_nc 1 \
    --preprocess None \
    --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results/ \
    --epoch 50



CUDA_VISIBLE_DEVICES=4 python test_synth.py \
    --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/33nyul/7materialized_splits_BodyRegion/AB/pix2pix/AB \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 2_experiment_pix2pix_synthrad_abdomen_33nyul \
    --model pix2pix \
    --phase test \
    --direction AtoB \
    --input_nc 1 \
    --output_nc 1 \
    --preprocess None \
    --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results/ \
    --epoch 50