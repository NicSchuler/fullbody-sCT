CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splitsNonNormalized/pix2pix/AB --name pix2pix_synthrad_whole_data --model pix2pix --direction AtoB --input_nc 1 --output_nc 1


##train abdominal region only
CUDA_VISIBLE_DEVICES=6 python train.py --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splitsNonNormalizedBodyRegion/AB/pix2pix/AB --checkpoints_dir /local/scratch/datasets/FullbodySCT/80checkpoints --name pix2pix_synthrad_abdomen --model pix2pix --direction AtoB --input_nc 1 --output_nc 1


##train pelvic region only
CUDA_VISIBLE_DEVICES=6 python train.py --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splitsNonNormalizedBodyRegion/AB/pix2pix/AB --checkpoints_dir /local/scratch/datasets/FullbodySCT/80checkpoints --name pix2pix_synthrad_pelvic --model pix2pix --direction AtoB --input_nc 1 --output_nc 1


