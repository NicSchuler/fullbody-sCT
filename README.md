# fullbody-sCT

This repo is a collection of sub-projects that require separate environments. Go into the respective folder to find instructions on usage of the respective sub-project.

Todo 
1. MHA to DICOM or Nifti (perhaps CT_MR_preprocessing.py could be ingnored)
    - MHA to DICOM: 
    - MHA to Nifti: Flavian via MHA_NIFTI_preprocessing.py
2. DICOM files into the folderstructure needed for the script (reprocessing-readme) or anotherway to bring it in CT_MR_preprocessing.py or resampling.py (if directly MHA to Nifti) 
3. In CT_MR_preprocessing.py Masks are constructed, I think we actually don't need that, we should be able to disable it
4. Check if it can run (maybe we also need to create an excel file that correctly specifies the train/test split)

![Fullbody-sCT Workflow](images/Overview.jpg)

USZ Fullbody SCT data is stored in `/local/scratch/datasets/FullbodySCT/USZ_Data` as dcm = DICOM. 
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

SynthRad2023

SynthRad2025
 /local/scratch/datasets/FullbodySCT/SynthRAD2025 
 ```text
    /excel

    /synthRAD2025_Task1_Train
       /Task1
            /AB
                /1ABA005
                    ct.mha
                    mask.mha
                    mr.mha
                /...
                /overviews
                    1ABA005_overview.png
                    ...
                    1_AB_train_parameters.xlsx
            /HN
                /1HNA001
                    ct.mha
                    mask.mha
                    mr.mha
                /...
                /overviews
                    1HNA001_overview.png
                    ...
                    1_HN_train_parameters.xlsx
            /TH
                /1THA001
                    ct.mha
                    mask.mha
                    mr.mha
                /...
                /overviews
                    1THA001_overview.png
                    ...
                    1_TH_train_parameters.xlsx   

    /synthRAD2025_Task1_Train_Nifti
        /excel
            data_CT_MR_TEMP_second_paper.xlsx
        /nifti
            1ABA005
                1ABA005_3D_body.nii
                1ABA005_3D_CT_air_overwrite.nii  
                3D_mask_body.nii
            ...
            1THB226
                1THB226_3D_body.nii
                1THB226_3D_CT_air_overwrite.nii
                3D_mask_body.nii

    /synthRAD2025_Task1_Val_Input
        /Task1
            /AB
                /1ABA002
                    mask.mha
                    mr.mha
                /...
                /overviews
                    1ABA002_overview.png
                    ...
                    1_AB_val_parameters.xlsx
            /HN
                /1HNA002
                    mask.mha
                    mr.mha
                /...
                /overviews
                    1HNA002_overview.png
                    ...
                    1_HN_val_parameters.xlsx
            /TH
                /1THA007
                    mask.mha
                    mr.mha
                /...
                /overviews
                    1THA007_overview.png
                    ...
                    1_TH_val_parameters.xlsx