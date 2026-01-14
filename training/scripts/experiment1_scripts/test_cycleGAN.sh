CUDA_VISIBLE_DEVICES=0 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/test'  \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints  \
  --phase test \
  --name cyclegan_abdomen \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --netG unet_256 \
  --netD basic \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results



  CUDA_VISIBLE_DEVICES=0 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/brain/cyclegan/test'  \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints  \
  --phase test \
  --name cyclegan_brain \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --netG unet_256 \
  --netD basic \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results


  CUDA_VISIBLE_DEVICES=0 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/HN/cyclegan/test'  \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints  \
  --phase test \
  --name cyclegan_head_neck \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --netG unet_256 \
  --netD basic \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results



    CUDA_VISIBLE_DEVICES=0 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/pelvis/cyclegan/test'  \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints  \
  --phase test \
  --name cyclegan_pelvis \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --netG unet_256 \
  --netD basic \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results



    CUDA_VISIBLE_DEVICES=0 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/TH/cyclegan/test'  \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints  \
  --phase test \
  --name cyclegan_thorax \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --netG unet_256 \
  --netD basic \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results


##FULL BODY

CUDA_VISIBLE_DEVICES=2 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splits_31baseline/cyclegan/test'  \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints  \
  --phase test \
  --name cyclegan_allregions \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --netG unet_256 \
  --netD basic \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results


# Fullbody CycleGAN model evaluated per body part (epoch 50)
CUDA_VISIBLE_DEVICES=3 python test_synth.py \
  --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/6materialized_splits_31baseline/cyclegan/test'  \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints  \
  --phase test \
  --name cyclegan_allregions_final \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --netG unet_256 \
  --netD basic \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch 50


CUDA_VISIBLE_DEVICES=3 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/AB/cyclegan/test'  \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints  \
  --phase test \
  --name cyclegan_abdomen_final \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --netG unet_256 \
  --netD basic \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch 50

CUDA_VISIBLE_DEVICES=3 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/brain/cyclegan/test'  \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints  \
  --phase test \
  --name cyclegan_brain_final \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --netG unet_256 \
  --netD basic \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch 50



##TODO
CUDA_VISIBLE_DEVICES=3 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/HN/cyclegan/test'  \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints  \
  --phase test \
  --name cyclegan_head_neck_final \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --netG unet_256 \
  --netD basic \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch 50

CUDA_VISIBLE_DEVICES=3 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/pelvis/cyclegan/test'  \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints  \
  --phase test \
  --name cyclegan_pelvis_final \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --netG unet_256 \
  --netD basic \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch 50

CUDA_VISIBLE_DEVICES=3 python test_synth.py --dataroot '/local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/7materialized_splits_31baselineBodyRegion/TH/cyclegan/test'  \
  --checkpoints_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/8checkpoints  \
  --phase test \
  --name cyclegan_thorax_final \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --netG unet_256 \
  --netD basic \
  --results_dir /local/scratch/datasets/FullbodySCT/Synthrad_combined_preprocessed/100results \
  --epoch 50
