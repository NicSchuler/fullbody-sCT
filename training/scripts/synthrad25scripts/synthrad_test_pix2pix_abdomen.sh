CUDA_VISIBLE_DEVICES=3 python test_synth.py --phase test --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splitsNonNormalizedBodyRegion/AB/pix2pix/AB --name pix2pix_synthrad_abdomen --checkpoints_dir /home/user/mfrei/projects/fullbody-sCT/training/checkpoints --model pix2pix --direction BtoA --input_nc 1 --output_nc 1



CUDA_VISIBLE_DEVICES=7 python test_synth.py --phase test --dataroot /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splitsNonNormalizedBodyRegion/AB/pix2pix/AB --name pix2pix_synthrad_abdomen --checkpoints_dir /home/user/fthuer/fullbody-sCT/training/checkpoints --model pix2pix --direction AtoB --input_nc 1 --output_nc 1