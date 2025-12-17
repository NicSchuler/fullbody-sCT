import os
import csv
import numpy as np
import nibabel as nib
import pandas as pd

import sys
sys.path.append('../')

from util import get_sorted_files_from_folder

from n_peak_normalization import NPeakNormalizerNew
from util_new import calculate_voxel_spacing_from_affine, dilate_mask

# Constants: the strings and endings in the file names
NIFTI_ENDING = ".nii"
MASK_STRING = "mask_"
BODY_ENDING = "_body"
FAT_ENDING = "_fat"
LIVER_ENDING = "_liver"
BLADDER_ENDING = "_bladder"
actual_voxel_size_x = 1.6304
actual_voxel_size_y = 1.6304
actual_voxel_size_z =3
voxel_spacing_arr=np.array([actual_voxel_size_x,actual_voxel_size_y,actual_voxel_size_z])

min_percentile=2
max_percentile=98
path_excel_resampling = "/srv/beegfs02/scratch/mr_to_ct/data/excel/n_peaks_second_paper_NEW.xlsx"



def main():
    ###############################
    # All parameters and settings #
    ###############################
    df_whole_list = pd.DataFrame()

    is_use_biasfieldcorrection = True # Choose whether biasfield corrected data is used
    is_use_biasfieldcorrection_2x = False # Choose whether two times biasfield corrected data is used
    is_use_new_fat_mask = True # To resolve some issues with non-fat tissue in fat masks, activate this to modify the fat masks. Recommended as True
    fat_mask_dilate_radius = 0.0 #3.0 # Radius for dialation used in removing low intensity regions from fat masks. Applies only if is_use_new_fat_mask

    gradient_radius = 3.0 # The voxel radius used in gradient calculation. 3.0 # In mm

    # Settings for the KDE are now given as a dict
    kde_settings = {
        "discretization_count" : 1024,
        "grid_number" : 10_000, # Number of grid points in KDE. Recommended: at least above 1000, better 10'000 or more
        "bandwith_selector" : 'ISJ', # Bandwidth estimation method for KDE. recommended: "ISJ". possible: "silverman"
    }

    # Settings for the normalizer are now given separately for the gradient threshold step and the peak detection step
    gradient_threshold_settings_liver = {
        "high_gradient_outlier_cutoff_percentage" : 10., # Cut off this fraction of high gradient voxels for high gradient
        # snippet selection
        "n_snippets_min" : 10, # The minimum number of snippets considered. Recommended: 10
        "n_snippets_max" : 100, # The maximum number of snippets considered. Recommended: 100
        "vol_snippets_min" : 1000, # The minimum volume (in mm^3) that each snippet should have. Needs to be adjusted by body
        # region. Recommended in brain: about 1000
        "lam" : .1, #1.0 # Weighting factor lambda, higher means a smaller selected homogeneous region
    }
    gradient_threshold_settings_fat = gradient_threshold_settings_liver.copy() # For liver use same settings
    gradient_threshold_settings_background = gradient_threshold_settings_liver.copy()
    gradient_threshold_settings_background["n_snippets_min"] = 2 # In background mask, use exactly two snippets
    gradient_threshold_settings_background["n_snippets_max"] = 2 # In background mask, use exactly two snippets

    # For liver, detect the largest peak
    peak_detection_settings_liver = {
        "smooth_radius" : 0.0, # Smoothing radius in mm, used to isolate peaks of homogeneous regions better
        "is_use_smooth_peaks" : False, # Decide whether to use histogram peaks in smoothed image
        "peak_prom_fraction" : 0.1, # Decide what prominence of histogram peak is considered a separate region
        "peak_selection_mode" : 'most', # Decide rationale for peak selection in case multiple have been found
    }
    peak_detection_settings_fat = peak_detection_settings_liver.copy()
    peak_detection_settings_fat["peak_selection_mode"] = "right" # In fat mask, use the rightmost peak
    peak_detection_settings_background = peak_detection_settings_liver.copy() # In background use same settings

    # Choose the names of which masks you want to use. Later on, the proper variables get assigned automatically
    mask_name_list = ["background", "liver", "fat"]
    # Choose the settings dict for each mask
    gradient_threshold_settings_list = [gradient_threshold_settings_background, gradient_threshold_settings_liver, gradient_threshold_settings_fat]
    peak_detection_settings_list = [peak_detection_settings_background, peak_detection_settings_liver, peak_detection_settings_fat]
    # Choose the output intensity value for each mask. If "mean" is chosen, automatically use the mean of the whole data set
    goal_intensity_list = ["median", "median", "median"]

    # Determine paths of the input and output data
    base_folder_path = "/srv/beegfs02/scratch/mr_to_ct/data/normalization/before_temp"
    if is_use_biasfieldcorrection_2x:
        # Biasfield corrected data
        mr_image_folder = os.path.join(base_folder_path, "MR_bfc_2x")
        output_folder_path = "/srv/beegfs02/scratch/mr_to_ct/data/normalization/after_npeaks/MR_bfc_2x"
    elif is_use_biasfieldcorrection:
        # Biasfield corrected data
        mr_image_folder = os.path.join(base_folder_path, "MR_bfc")
        output_folder_path = "/srv/beegfs02/scratch/mr_to_ct/data/normalization/after_npeaks/MR_bfc"
    else:
        # Not biasfield corrected data
        mr_image_folder = os.path.join(base_folder_path, "MR")
        output_folder_path = "/srv/beegfs02/scratch/mr_to_ct/data/normalization/after_npeaks_new_params/MR"
    body_mask_folder = os.path.join(base_folder_path, "masks")
    fat_mask_folder = os.path.join(base_folder_path, "masks_fat")
    liver_mask_folder = os.path.join(base_folder_path, "masks_liver")
    bladder_mask_folder = os.path.join(base_folder_path, "masks_water_bladder")

    # Determine path where the plots from the normalization algorithm will be saved.
    plot_folder = "/srv/beegfs02/scratch/mr_to_ct/data/normalization/snippet_norm_images_new_params"

    # Determine the path of the csv file that the peak locations should be written into. if None, don't write csv.
    csv_save_path = os.path.join("/srv/beegfs02/scratch/mr_to_ct/data/normalization", "peak_locations.csv")

    ################
    # Main program #
    ################

    # If save in a csv, clear the existing file and write a header.
    if not csv_save_path is None:
        header = ["file name"] + [f"peak {index+1} ({name}) intensity" for index, name in enumerate(mask_name_list)]

        # Note: newline='' is necessary in Windows so that there is no empty space between rows. But potentially needs to be removed in Linux
        # See: https://stackoverflow.com/questions/3348460/csv-file-written-with-python-has-blank-lines-between-each-row
        with open(csv_save_path, 'w', newline='') as f:
            f.truncate() # Clear existing file

            writer = csv.writer(f, delimiter=";") # Use delimiter ; so that the file is easily readable in excel
            writer.writerow(header)


    normalizer = NPeakNormalizerNew(kde_param_dict=kde_settings, gradient_radius=gradient_radius,
                                    histogram_distance_function=None, plot_folder=plot_folder) # Define normalizer

    # Obtain list of nifti files with the chosen ending in the chosen folder
    image_name_list = get_sorted_files_from_folder(mr_image_folder, f"{BODY_ENDING}{NIFTI_ENDING}")

    inclusion_name_list = None
    if not inclusion_name_list is None:
        image_name_list = [name for name in image_name_list if name in inclusion_name_list]

    peak_list_list = []

    # For each found nifti file, load all the corresponding masks and execute normalization
    for image_name in image_name_list:
        print(f"Processing image {image_name}")

        # Create paths for all possible nifti files
        mr_image_path = os.path.join(mr_image_folder, f"{image_name}{BODY_ENDING}{NIFTI_ENDING}")
        body_mask_path = os.path.join(body_mask_folder, f"{MASK_STRING}{image_name}{BODY_ENDING}{NIFTI_ENDING}")
        fat_mask_path = os.path.join(fat_mask_folder, f"{MASK_STRING}{image_name}{FAT_ENDING}{NIFTI_ENDING}")
        liver_mask_path = os.path.join(liver_mask_folder, f"{MASK_STRING}{image_name}{LIVER_ENDING}{NIFTI_ENDING}")
        bladder_mask_path = os.path.join(bladder_mask_folder, f"{MASK_STRING}{image_name}{BLADDER_ENDING}{NIFTI_ENDING}")
        is_bladder_mask_exists = os.path.exists(bladder_mask_path)

        # Load images from nifti files
        mr_image = nib.load(mr_image_path)
        body_mask_image = nib.load(body_mask_path)
        fat_mask_image = nib.load(fat_mask_path)
        liver_mask_image = nib.load(liver_mask_path)
        if is_bladder_mask_exists:
            bladder_mask_image = nib.load(bladder_mask_path) # If bladder mask exists, load it. Not used right now, but maybe of interest

        # voxel_spacing_arr = calculate_voxel_spacing_from_affine(mr_image.affine)
        # print(f"Gradient radius {gradient_radius}. Voxel spacing {voxel_spacing_arr}. In integers: {np.floor(gradient_radius/voxel_spacing_arr)}")

        # Extract arrays from images
        image_arr = mr_image.get_fdata()
        body_mask_arr = body_mask_image.get_fdata().astype(bool)
        fat_mask_arr = fat_mask_image.get_fdata().astype(bool)
        liver_mask_arr = liver_mask_image.get_fdata().astype(bool)
        if is_bladder_mask_exists:
            bladder_mask_arr = bladder_mask_image.get_fdata().astype(bool)

        #adjust masks if the images are misaligned (new ones that masha sent)
        if body_mask_arr.shape[2] == image_arr.shape[2] + 2:
            body_mask_arr = body_mask_arr[:,:,1:-1]
        if fat_mask_arr.shape[2] == image_arr.shape[2] + 2:
            fat_mask_arr = fat_mask_arr[:,:,1:-1]
        if liver_mask_arr.shape[2] == image_arr.shape[2] + 2:
            liver_mask_arr = liver_mask_arr[:,:,1:-1]

        # determine background mask
        background_mask_arr = image_arr == 0 # Find background voxels (where image is 0)
        background_mask_arr = background_mask_arr & (~liver_mask_arr & ~fat_mask_arr) # Exclude liver and fat mask voxels from background
        #background_mask_arr = ~body_mask_arr

        # If chosen, calculate a modified fat mask that is above the median intensity of liver mask
        if is_use_new_fat_mask:
            # Calculate the median intensity and width of the liver peak
            liver_median = np.nanmedian(image_arr[liver_mask_arr])

            # New way of determining the modified fat mask: We take the median intensity of the liver.
            # Then we look at all the voxels in the image that are below that intensity.
            # Then we also include all voxels that are neighboring to those voxels (within given dilate radius)
            # The voxels selected get removed from the fat mask. That leaves only voxels that are high intensity and border high intensity voxels
            # Performs well in general, but a bit questionable for patients 13 and 85.
            fat_mask_arr = fat_mask_arr & ~dilate_mask(image_arr < liver_median, fat_mask_dilate_radius,
                                                       voxel_spacing_arr=voxel_spacing_arr)


        mask_list = []
        for mask_name in mask_name_list:
            if mask_name == "background":
                mask_list.append(background_mask_arr)
            if mask_name == "liver":
                mask_list.append(liver_mask_arr)
            if mask_name == "fat":
                mask_list.append(fat_mask_arr)

        peak_list = normalizer.calculate_peak_locs(
            image_arr, mask_list, base_mask=~background_mask_arr,
            grad_thresh_settings_list=gradient_threshold_settings_list,
            peak_det_settings_list=peak_detection_settings_list, voxel_spacing_arr=voxel_spacing_arr,
            image_name=image_name, mask_name_list=mask_name_list
        )

        peak_list_list.append(peak_list)

        # Write name and peak locations into a new row of the csv
        if not csv_save_path is None:
            row = [image_name] + peak_list
            with open(csv_save_path, 'a', newline='') as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(row)

    peak_arr = np.array(peak_list_list)
    for index, goal_intensity in enumerate(goal_intensity_list):
        if str(goal_intensity) == "mean":
            goal_intensity_list[index] = np.mean(peak_arr[:, index])
        elif str(goal_intensity) == "median":
            goal_intensity_list[index] = np.median(peak_arr[:, index])

    for image_name, peak_list in zip(image_name_list, peak_list_list):
        print(f"Saving final version of image {image_name}")

        # Create paths for all possible nifti files
        mr_image_path = os.path.join(mr_image_folder, f"{image_name}{BODY_ENDING}{NIFTI_ENDING}")

        # Load images from nifti files
        mr_image = nib.load(mr_image_path)

        # Extract arrays from images
        image_arr = mr_image.get_fdata()

        norm_image_arr = normalizer.normalize(image_arr, goal_intensity_list=goal_intensity_list,
                                              peak_intensity_list=peak_list)
        # Save normalized image as nifti in the output folder. Name, affine and header are the same as the input nifti
        new_image = nib.Nifti1Image(norm_image_arr, mr_image.affine, mr_image.header)



        #apply body mask
        body_contour=body_mask_image.get_fdata()
        masked_image= body_contour*norm_image_arr
        norm_image_arr=masked_image


        #find min max for further outlier handling
        # patients percentile
        min_p = np.percentile(norm_image_arr, min_percentile)
        max_p = np.percentile(norm_image_arr, max_percentile)
        if min_p<0:
            min_p=0
        norm_image_arr[norm_image_arr < min_p] = min_p
        norm_image_arr[norm_image_arr > max_p] = max_p


        pat_int_min= np.min(norm_image_arr)
        pat_int_max = np.max(norm_image_arr)
        print("min max intensities after 2% outlier removal")
        print(pat_int_min, pat_int_max)
        df_whole_list = df_whole_list.append(
            {'file name': image_name, 'min intensity': pat_int_min, 'max intensity': pat_int_max }, ignore_index=True)



        nib.save(new_image, os.path.join(output_folder_path, f"{image_name}{BODY_ENDING}{NIFTI_ENDING}"))

    print("saving final excel")
    df_whole_list.to_excel(path_excel_resampling)


if __name__ == "__main__":
    main()