# Preprocessing

This repo is based on [medical-physics-usz/synthetic_CT_generation](https://github.com/medical-physics-usz/synthetic_CT_generation), though modifications have been made. It is used for following steps:

* DICOM to Nifti transformation (CT_MR_preprocessing.py)
* Resampling to canonical voxel size + volume dimensions (resampling.py)
* nyul-normalization (if executed)
* create 2D Nifti slices (slice_creator.py)

## Usage

1. Create environment
```
conda env create --file preprocessing_env.yml
conda activate preprocessing_env
```
If conda is not found, run `source ~/miniconda3/etc/profile.d/conda.sh`.

2. Run the preprocessing
```
./preprocessing_full.sh -mode preprocessing_train -is_rstruct_file_with_tumour True
```

Note: the data in `initial data` should look like this:
```text
  /Pat015_2_SB_KIDr_1a
    /Plan
      /CT
        CT1.3.12.2.1107.5.1.4.65204.30000020073107453621100001809.dcm
        ...
      /MR
        MR2.16.840.1.114493.1.4.228.3.20200731123826703.dcm
        ...
      RTPLAN2.16.840.1.114493.1.4.228.3.20210305171100260.dcm
      RTSTRUCT2.16.840.1.114493.1.4.228.3.20210305171100267.dcm
      RTDOSE2.16.840.1.114493.1.4.228.3.20210305171100273.dcm
    /CT_reg
      CT1.3.12.2.1107.5.1.4.65204.30000020073107453621100001936.dcm
      ...
  /Pat001_1_SB_LIV_1a
    	...
  /Pat002_2_SB_ABD_1a
    	...
```

## Folder Access

The data in the folder `/local/scratch/datasets/FullbodySCT/USZ_Data` should only be accessible to the project group. This can be achieved by running the following 3 lines:

```
setfacl -R -m u:mfrei:rwx /local/scratch/datasets/FullbodySCT/USZ_Data
setfacl -R -m u:fthuer:rwx /local/scratch/datasets/FullbodySCT/USZ_Data
setfacl -R -m u:nschuler:rwx /local/scratch/datasets/FullbodySCT/USZ_Data
```

Note that this gives read/write access to the files and folders recursively, **but it needs to be run after every modification of files.** This is not like sharing a folder in onedrive but it gives access to the files currently in the folder! It will print lots of errors because you cannot give permission to files owned by someone else but it will still work for those files that you can give access to.