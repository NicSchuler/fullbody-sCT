cd ../../../training_cut

CUDA_VISIBLE_DEVICES=4 python test_synth.py \
    --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/test \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 2_experiment_cut_synthrad_abdomen_sep_first_layer \
    --phase test \
    --model cut \
    --CUT_mode CUT \
    --output_nc 1 \
    --input_nc 1 \
    --batch_size 1 \
    --netG resnet_9blocks_sep_first_layer \
    --preprocess None \
    --epoch 50 \
    --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results


CUDA_VISIBLE_DEVICES=6 python test_synth.py \
    --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/32p99/7materialized_splits_BodyRegion/AB/cyclegan/test' \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 2_experiment_cut_synthrad_abdomen_32p99 \
    --phase test \
    --model cut \
    --CUT_mode CUT \
    --output_nc 1 \
    --input_nc 1 \
    --batch_size 1 \
    --preprocess None \
    --epoch 50 \
    --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results




CUDA_VISIBLE_DEVICES=3 python test_synth.py \
    --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/experiment2/33nyul/7materialized_splits_BodyRegion/AB/cyclegan/test' \
    --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints \
    --name 2_experiment_cut_synthrad_abdomen_33nyul \
    --phase test \
    --model cut \
    --CUT_mode CUT \
    --output_nc 1 \
    --input_nc 1 \
    --batch_size 1 \
    --preprocess None \
    --epoch 50 \
    --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results