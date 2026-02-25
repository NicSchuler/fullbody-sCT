import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from util import get_sorted_files_from_folder, transform_image
# from mri_visualization.mri_region_visualizer import DynamicImageVisualization
from util_new import calculate_KDE, calculate_KDE_with_discretized_points

# Constants: the strings and endings in the file names
NIFTI_ENDING = ".nii"
MASK_STRING = "mask_"
BODY_ENDING = "_body"
NYUL_ENDING = "_nyul"
FAT_ENDING = "_fat"
LIVER_ENDING = "_liver"
BLADDER_ENDING = "_bladder"
DEFAULT_FIGSIZE = (15, 12)
### APPLY your values for min max percentile crop on MR
_min_mr = 0
_max_mr = 4095

def main():
    ###############################
    # All parameters and settings #
    ###############################
    is_use_single_intensity_scale = True

    is_use_bfc = False # Choose whether biasfield corrected data is used
    is_use_volume_scaled_heights = False # Choose whether the histogram heights should be multiplied by their voxel count, thus representing the true size of the regions
    is_use_2xbfc = True # Choose whether biasfield corrected data is used
    is_use_volume_scaled_heights = True # Choose whether the histogram heights should be multiplied by their voxel count, thus representing the true size of the regions
    # Choose parameters for the KDE.
    kde_grid = 10000
    kde_bw = "ISJ" # "ISJ", "silverman" # ISJ is generally better, but has lead to some weird effects in Nyul liver. TODO: investigate. Actually it works now, but only for bfc?
    kde_discretization_count = 1024

    # Determine paths of the input and output data
    base_folder_path = "PATH/before_temp"
    base_folder_path_normalized_nyul = "PATH/Nyul_normalized_temp"
    # base_folder_path_normalized_no_norm = "PATH/MR_bfc_no_norms"
    base_folder_path_normalized_npeak = "PATH/after_npeaks/MR_bfc_2x"

    is_use_n_peak = True

    if is_use_2xbfc:
        mr_folder_name = "MR_bfc_2x"
    elif is_use_bfc:
        mr_folder_name = "MR_bfc"
    else:
        mr_folder_name = "MR"

    mr_image_folder = os.path.join(base_folder_path, mr_folder_name)
    mr_image_folder_nyul = os.path.join(base_folder_path_normalized_nyul)
    # mr_image_folder_no_norm = os.path.join(base_folder_path_normalized_no_norm)
    mr_image_folder_npeak = os.path.join(base_folder_path_normalized_npeak)
    body_mask_folder = os.path.join(base_folder_path, "masks")
    fat_mask_folder = os.path.join(base_folder_path, "masks_fat")
    fat_mr_mask_folder = os.path.join(base_folder_path, "masks_fat_mr")
    liver_mask_folder = os.path.join(base_folder_path, "masks_liver")
    bladder_mask_folder = os.path.join(base_folder_path, "masks_water_bladder")

    output_folder = os.path.join("PATH/comparison_graphs/")
    norm1 = "no"
    norm2 = "nyul"
    norm3 = "no_norm" if not is_use_n_peak else "npeak"
    bfcstring = "bfc" if is_use_bfc else "nobfc"
    bfcstring = "2xbfc" if is_use_2xbfc else bfcstring
    bwstring = kde_bw.lower()
    rescale_string = "_rescale" if is_use_volume_scaled_heights else ""
    intensity_string = "_intensitysinglescale" if is_use_single_intensity_scale else ""
    output_file_name = os.path.join(output_folder,
                                    f"kde_comparison_{norm1}_{norm2}_{norm3}_{bfcstring}_{bwstring}{rescale_string}{intensity_string}.png")

    # Choose initial and ending colors for all plots. Recommended initial: (0.4, 0.4, 0.4). Shows nice transition from gray to color
    initial_color = np.array([0.4, 0.4, 0.4])
    end_color_whole = np.array([0.12, 0.46, 0.79])
    end_color_liver = np.array([0.12, 0.52, 0.12])
    end_color_fat = np.array([0.31, 0, 0.44])

    # Choose the cutoff percentiles for the x limits of the plots (only visual effect)
    plot_cutoff_percentile_left = 0.5
    plot_cutoff_percentile_right = 99.5

    ################
    # Main program #
    ################

    # Obtain list of nifti files with the chosen ending in the chosen folder
    image_name_list = get_sorted_files_from_folder(mr_image_folder, f"{BODY_ENDING}{NIFTI_ENDING}")
    # image_name_list = get_sorted_files_from_folder(mr_image_folder, f"{NIFTI_ENDING}")

    # Give list of names of images that should be plotted. If "None" is given, they are all plotted
    #inclusion_name_list = []
    inclusion_name_list = None
    if not inclusion_name_list is None:
        image_name_list = [name for name in image_name_list if name in inclusion_name_list]

    # Set up plots and labels
    fig, axs = plt.subplots(3, 3, figsize=DEFAULT_FIGSIZE)

    axs[0][0].set_title("Body")
    axs[0][1].set_title("Liver")
    axs[0][2].set_title("Fat")

    axs[0][0].set_ylabel("No Normalization")
    axs[1][0].set_ylabel("Nyul")
    if not is_use_n_peak:
        axs[2][0].set_ylabel("Two Peaks")
    else:
        axs[2][0].set_ylabel("N Peaks")

    # Deefine array where the x limits of each subplot will be saved
    xlims = np.zeros((3, 3, 2)) * np.nan

    n_images = len(image_name_list)

    for image_index, image_name in enumerate(image_name_list):
        print(f"Image name: {image_name}")

        # Load and preprocess data...
        mr_image_path = os.path.join(mr_image_folder, f"{image_name}{BODY_ENDING}{NIFTI_ENDING}")
        mr_image_path_nyul = os.path.join(mr_image_folder_nyul, f"{image_name}{BODY_ENDING}{NYUL_ENDING}{NIFTI_ENDING}")
        # mr_image_path_no_norm = os.path.join(mr_image_folder_no_norm, f"{image_name}{BODY_ENDING}{NIFTI_ENDING}")
        mr_image_path_npeak = os.path.join(mr_image_folder_npeak, f"{image_name}{BODY_ENDING}{NIFTI_ENDING}")
        body_mask_path = os.path.join(body_mask_folder, f"{MASK_STRING}{image_name}{BODY_ENDING}{NIFTI_ENDING}")
        fat_mask_path = os.path.join(fat_mask_folder, f"{MASK_STRING}{image_name}{FAT_ENDING}{NIFTI_ENDING}")
        #fat_mr_mask_path = os.path.join(fat_mr_mask_folder, f"{MASK_STRING}{image_name}{FAT_ENDING}{NIFTI_ENDING}")
        liver_mask_path = os.path.join(liver_mask_folder, f"{MASK_STRING}{image_name}{LIVER_ENDING}{NIFTI_ENDING}")
        #bladder_mask_path = os.path.join(bladder_mask_folder, f"{MASK_STRING}{image_name}{BLADDER_ENDING}{NIFTI_ENDING}")
        #is_bladder_mask_exists = os.path.exists(bladder_mask_path)

        mr_image = nib.load(mr_image_path)
        mr_image_nyul = nib.load(mr_image_path_nyul)
        # mr_image_no_norm = nib.load(mr_image_path_no_norm)
        if is_use_n_peak:
            mr_image_npeak = nib.load(mr_image_path_npeak)
        body_mask_image = nib.load(body_mask_path)
        fat_mask_image = nib.load(fat_mask_path)
        #fat_mr_mask_image = nib.load(fat_mr_mask_path)
        liver_mask_image = nib.load(liver_mask_path)
        #if is_bladder_mask_exists:
        #    bladder_mask_image = nib.load(bladder_mask_path)

        image_arr = mr_image.get_fdata()
        sample = (image_arr - _min_mr) / (_max_mr - _min_mr)
        image_arr = sample

        image_arr[image_arr < 0] = 0
        image_arr[image_arr > 1] = 1


        image_arr_nyul = mr_image_nyul.get_fdata()
        image_arr_nyul[image_arr_nyul < 0] = 0
        image_arr_nyul[image_arr_nyul > 1] = 1
        # image_arr_no_norm = mr_image_no_norm.get_fdata()
        if is_use_n_peak:
            image_arr_npeak = mr_image_npeak.get_fdata()
            _min_npeak = 0
            _max_npeak = 855
            sample = (image_arr_npeak - _min_mr) / (_max_mr - _min_mr)
            image_arr_npeak = sample
            image_arr_npeak[image_arr_npeak < 0] = 0
            image_arr_npeak[image_arr_npeak > 1] = 1

        body_mask_arr = body_mask_image.get_fdata().astype(bool)
        fat_mask_arr = fat_mask_image.get_fdata().astype(bool)
        #fat_mr_mask_arr = fat_mr_mask_image.get_fdata().astype(bool)
        liver_mask_arr = liver_mask_image.get_fdata().astype(bool)
        #if is_bladder_mask_exists:
        #    bladder_mask_arr = bladder_mask_image.get_fdata().astype(bool)

        # Go through all combinations of regions and masks to plot the images in the proper subplots
        #adjust masks if the images are misaligned (new ones that masha sent)
        if body_mask_arr.shape[2] == image_arr.shape[2] + 2:
            body_mask_arr = body_mask_arr[:,:,1:-1]
        if fat_mask_arr.shape[2] == image_arr.shape[2] + 2:
            fat_mask_arr = fat_mask_arr[:,:,1:-1]
        if liver_mask_arr.shape[2] == image_arr.shape[2] + 2:
            liver_mask_arr = liver_mask_arr[:,:,1:-1]

        # Adjust fat mask to not contain intensities below median liver intenstiy...
        median_liver_intensity = np.nanmedian(image_arr[liver_mask_arr])
        fat_mask_arr = fat_mask_arr & (image_arr >= median_liver_intensity)

        # Go through all normalization verisions
        image_arr_list =[image_arr, image_arr_nyul, image_arr_npeak]
        # image_arr_list = [image_arr, image_arr_nyul, image_arr_twopeak] if not is_use_n_peak else [image_arr, image_arr_nyul, image_arr_npeak]
        #image_arr_list = [image_arr, image_arr_zscore, image_arr_twopeak] if not is_use_n_peak else [image_arr, image_arr_zscore, image_arr_npeak]
        for normalization_index, intensity_arr in enumerate(image_arr_list):
            # Global intensity scale if chosen
            if is_use_single_intensity_scale:
                # Calculate histogram through KDE, get grid and bw
                body_kde_grid, body_pdf, body_kde_bw = calculate_KDE_with_discretized_points(intensity_arr[body_mask_arr], kde_grid, kde_bw, n_discrete=kde_discretization_count)
                norm_cumsum_pdf = np.cumsum(body_pdf)/np.sum(body_pdf)
                left_boundary = body_kde_grid[np.searchsorted(norm_cumsum_pdf, plot_cutoff_percentile_left/100., 'right')]
                right_boundary = body_kde_grid[np.searchsorted(norm_cumsum_pdf, plot_cutoff_percentile_right/100, 'left')]

            # Go through all combinations of regions and masks to plot the images in the proper subplots
            for region_index, (mask_arr, end_color) in enumerate(zip([body_mask_arr, liver_mask_arr, fat_mask_arr], [end_color_whole, end_color_liver, end_color_fat])):
                color = initial_color + image_index/(n_images-1)*(end_color-initial_color) # Determine color of this line
                if is_use_single_intensity_scale:
                    grid, pdf = calculate_KDE(intensity_arr[mask_arr], body_kde_grid, body_kde_bw) # Calculate histogram through KDE
                else:
                    grid, pdf = calculate_KDE(intensity_arr[mask_arr], kde_grid, kde_bw) # Calculate histogram through KDE
                if is_use_volume_scaled_heights: # If we choose so, show organ histograms scaled according to size
                    pdf *= np.count_nonzero(mask_arr)
                axs[normalization_index][region_index].plot(grid, pdf, color=color) # Plot the line in the appropriate subplot

                if not is_use_single_intensity_scale:
                    norm_cumsum_pdf = np.cumsum(pdf)/np.sum(pdf)
                    left_boundary = grid[np.searchsorted(norm_cumsum_pdf, plot_cutoff_percentile_left/100., 'right')]
                    right_boundary = grid[np.searchsorted(norm_cumsum_pdf, plot_cutoff_percentile_right/100, 'left')]
                xlims[normalization_index][region_index][0] = np.nanmin((xlims[normalization_index][region_index][0], left_boundary))
                xlims[normalization_index][region_index][1] = np.nanmax((xlims[normalization_index][region_index][1], right_boundary))



    # For each subplot, apply the determined limits
    #for region_index, xlim_arr in enumerate(xlims):
            #    for normalization_index, xlim in enumerate(xlim_arr):
            # axs[normalization_index][region_index].set_xlim(xlims[normalization_index][region_index][0], xlims[normalization_index][region_index][1])

    # For each subplot, apply fixed x-axis limits from 0 to 1
    for region_index, xlim_arr in enumerate(xlims):
        for normalization_index, xlim in enumerate(xlim_arr):
            if region_index == 0:  # Assuming body_mask_arr is the first in the list
                axs[normalization_index][region_index].set_xlim(0, 1)  # Fixed x-axis range for body mask
            else:
                axs[normalization_index][region_index].set_xlim(
                    xlims[normalization_index][region_index][0],
                    xlims[normalization_index][region_index][1]
                )  # Dynamic range for other masks

    for normalization_index in range(len(image_arr_list)):
        for region_index in range(3):  # Assuming 3 regions (body, liver, fat)
            axs[normalization_index][region_index].tick_params(axis='both', labelsize=16)  # Set both X and Y tick font size
            axs[normalization_index][region_index].set_yticks([])  # Remove ticks
            axs[normalization_index][region_index].set_ylabel("")  # Remove label
            axs[normalization_index][region_index].set_xlabel("")  # Remove label

    plt.tight_layout() # Make things more compact visually
    plt.show() # Show the finished plot
    plt.savefig(output_file_name)



if __name__ == "__main__":
    main()