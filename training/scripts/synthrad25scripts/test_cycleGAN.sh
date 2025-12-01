
#test abdomen
CUDA_VISIBLE_DEVICES=7 python test_synth.py --phase test --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splitsNonNormalizedBodyRegion/AB/pix2pix/AB --name pix2pix_synthrad_abdomen --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints --model pix2pix --direction AtoB --input_nc 1 --output_nc 1 --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results




CUDA_VISIBLE_DEVICES=7 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/flavian_subset/7materialized_splits_31baselineBodyRegion/AB/cyclegan/test' --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints --phase test --name cyclegan_THUF --model cycle_gan --input_nc 1 --output_nc 1 --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results