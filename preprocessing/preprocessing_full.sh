#!/bin/bash

# source  .venv/bin/activate
source  ~/miniconda3/etc/profile.d/conda.sh
conda activate preprocessing_env

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "which python:    $(which python)"
echo "Starting on:     $(date)"

mkdir /local/scratch/datasets/FullbodySCT/USZ_Data/nyul_normalized 

python -u CT_MR_preprocessing.py "$@"
python -u resampling.py "$@"
# # # if prefitted nyul model is available, use this
# # nyul-normalize /local/scratch/datasets/FullbodySCT/USZ_Data/before_temp/MR -m /local/scratch/datasets/FullbodySCT/USZ_Data/before_temp/masks/ -o /local/scratch/datasets/FullbodySCT/USZ_Data/nyul_normalized -v --output-max-value 1 --output-min-value 0 --min-percentile 2 --max-percentile 98 -lsh <PATH_TO_REPO>/normalization/nyul_model_params.npy
# else fit new nyul model
nyul-normalize /local/scratch/datasets/FullbodySCT/USZ_Data/before_temp/MR -m /local/scratch/datasets/FullbodySCT/USZ_Data/before_temp/masks/ -o /local/scratch/datasets/FullbodySCT/USZ_Data/nyul_normalized -v --output-max-value 1 --output-min-value 0 --min-percentile 2 --max-percentile 98 
python -u slice_creator.py "$@"

setfacl -R -m u:mfrei:rwx /local/scratch/datasets/FullbodySCT/USZ_Data
setfacl -R -m u:fthuer:rwx /local/scratch/datasets/FullbodySCT/USZ_Data
setfacl -R -m u:nschuler:rwx /local/scratch/datasets/FullbodySCT/USZ_Data

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"