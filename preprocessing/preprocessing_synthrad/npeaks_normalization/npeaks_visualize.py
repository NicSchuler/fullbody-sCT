import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap

INTERPOLATION_MODE = "nearest" # "none" # Choose imshow interpolation mode. "nearest" works best for saving PDFs
RASTERIZE_MODE = False # Choose imshow rasterize mode. If "nearest" interpolation is chosen, can leave it off.
IMAGE_FORMAT = ".pdf" # ".png" # Format of the summary plots. PDF is recommended to get vector graphics.

# Default colors:
# Peak: (255, 32, 32)
# Homogeneous, in peak: (255, 32, 128)
# Homogeneous, out of peak: (128, 32, 255)
# Mask: (128, 128, 255)
PEAK_COLOR = np.array((255, 32, 32)) / 255.
HOM_PEAK_COLOR = np.array((255, 32, 128)) / 255.
HOM_NONPEAK_COLOR = np.array((128, 32, 255)) / 255.
MASK_COLOR = np.array((128, 128, 255)) / 255.

PEAK_ALPHA = 0.8
HOM_PEAK_ALPHA = 0.5
HOM_NONPEAK_ALPHA = 0.3
MASK_ALPHA = 0.2

DEFAULT_PLOT_SIZE = (17.92, 10.33) # In inches
DEFAULT_PLOT_DPI = 200.

# Using proper rcParams settings
TITLE_FONT_SIZE = 'large'
LABEL_FONT_SIZE = 'medium'
FONT_SIZE = 10.0
LEGEND_TITLE_FONT_SIZE = None
LEGEND_FONT_SIZE = 'medium'
XTICK_LABEL_SIZE = 'medium'
XTICK_MAJOR_SIZE = 3.5
XTICK_MAJOR_WIDTH = 0.8
XTICK_MINOR_SIZE = 2.0
XTICK_MINOR_WIDTH = 0.6
YTICK_LABEL_SIZE = 'medium'
YTICK_MAJOR_SIZE = 3.5
YTICK_MAJOR_WIDTH = 0.8
YTICK_MINOR_SIZE = 2.0
YTICK_MINOR_WIDTH = 0.6

# For 3d images, specify which of the three orthogonal slices should be shown on the summary plot as local intensity
# change.
LIC_INDEX = 0

def calculate_extent(intensity_arr, voxel_spacing_arr):
    extent_arr = intensity_arr.shape*voxel_spacing_arr
    return extent_arr

def calculate_largest_mask_indices(mask_arr):
    axis_arr = np.arange(mask_arr.ndim)
    most_voxels_index_list = [
        np.argmax(np.count_nonzero(mask_arr, axis=tuple(np.delete(axis_arr, i)))) for i in axis_arr
    ]
    return most_voxels_index_list

def select_ort_slices(intensity_arr, index_arr):
    slice_arr_list = [intensity_arr.take(indices=index, axis=i) for i, index in enumerate(index_arr)]
    return slice_arr_list

class NormVisualizer:


    def __init__(self, plot_folder, plot_size=DEFAULT_PLOT_SIZE, plot_dpi=DEFAULT_PLOT_DPI, color_list=None,
                 alpha_list=None, is_adjust_grayscale_to_view=True, is_compensate_nifti_coords=False):
        # Set up rcParams
        plt.rcParams["axes.labelsize"] = LABEL_FONT_SIZE
        plt.rcParams["axes.titlesize"] = TITLE_FONT_SIZE
        plt.rcParams["font.size"] = FONT_SIZE
        plt.rcParams["legend.title_fontsize"] = LEGEND_TITLE_FONT_SIZE
        plt.rcParams["legend.fontsize"] = LEGEND_FONT_SIZE
        plt.rcParams["xtick.labelsize"] = XTICK_LABEL_SIZE
        plt.rcParams["xtick.major.size"] = XTICK_MAJOR_SIZE
        plt.rcParams["xtick.major.width"] = XTICK_MAJOR_WIDTH
        plt.rcParams["xtick.minor.size"] = XTICK_MINOR_SIZE
        plt.rcParams["xtick.minor.width"] = XTICK_MINOR_WIDTH
        plt.rcParams["ytick.labelsize"] = YTICK_LABEL_SIZE
        plt.rcParams["ytick.major.size"] = YTICK_MAJOR_SIZE
        plt.rcParams["ytick.major.width"] = YTICK_MAJOR_WIDTH
        plt.rcParams["ytick.minor.size"] = YTICK_MINOR_SIZE
        plt.rcParams["ytick.minor.width"] = YTICK_MINOR_WIDTH

        self.plot_folder = plot_folder
        self.plot_size = plot_size
        self.plot_dpi = plot_dpi
        self.color_list = color_list if not color_list is None else [PEAK_COLOR, HOM_PEAK_COLOR, HOM_NONPEAK_COLOR,
                                                                     MASK_COLOR]
        self.alpha_list = alpha_list if not alpha_list is None else [PEAK_ALPHA, HOM_PEAK_ALPHA, HOM_NONPEAK_ALPHA,
                                                                     MASK_ALPHA]
        self.is_adjust_grayscale_to_view = is_adjust_grayscale_to_view
        self.is_compensate_nifti_coords = is_compensate_nifti_coords

    def plot(self, intensity_arr, local_intensity_change_arr, mask_arr, local_intensity_change_threshold,
             histogram_calculation_function, peak_intensity, peak_bounds, lic_hist_string=None, hist_peak_string=None,
             voxel_spacing_arr=None, image_name=None, mask_name=None):
        # Can only plot normalization visualization for images of 2 or 3 dimensions at this point
        if intensity_arr.ndim not in [2, 3]:
            raise ValueError(f"Error: Plotting of normalization results is not implemented for image arrays with "
                             f"dimension {intensity_arr.ndim}"
                             )

        # If voxel spacing is given, set up a proper array and use mm as the axis label
        if voxel_spacing_arr is None:
            is_use_mm = False
            voxel_spacing_arr = np.ones(intensity_arr.ndim)
        else:
            is_use_mm = True

        # Set up the title string
        image_name = "" if image_name is None else image_name
        title_string = f"Normalization Summary for Image {image_name}"

        # Set up the string above the local intensity change histogram
        lic_hist_string = f"Local Intensity Change Threshold: {local_intensity_change_threshold:.1f}" \
            if lic_hist_string is None else lic_hist_string

        # Set up the string above the histogram peak visualization
        hist_peak_string = f"Histogram Peak Intensity: {peak_intensity:.1f}" \
            if hist_peak_string is None else hist_peak_string

        mm_string = " mm" if is_use_mm else ""

        # Calculate necessary histograms, masks and quantities
        hom_mask_arr = mask_arr & (local_intensity_change_arr < local_intensity_change_threshold)
        peak_hom_mask_arr = hom_mask_arr & (intensity_arr > peak_bounds[0]) & (intensity_arr < peak_bounds[1])

        grid_arr, pdf_arr = histogram_calculation_function(intensity_arr[mask_arr])
        hom_grid_arr, hom_pdf_arr = histogram_calculation_function(intensity_arr[hom_mask_arr])
        lic_grid_arr, lic_pdf_arr = histogram_calculation_function(local_intensity_change_arr[mask_arr])

        peak_hom_pdf_arr = hom_pdf_arr.copy()
        nonpeak_hom_pdf_arr = hom_pdf_arr.copy()
        peak_hom_pdf_arr[(hom_grid_arr <= peak_bounds[0]) | (hom_grid_arr >= peak_bounds[1])] = np.nan
        nonpeak_hom_pdf_arr[(hom_grid_arr > peak_bounds[0]) & (hom_grid_arr < peak_bounds[1])] = np.nan

        hom_lic_pdf_arr = lic_pdf_arr.copy()
        inhom_lic_pdf_arr = lic_pdf_arr.copy()
        hom_lic_pdf_arr[lic_grid_arr >= local_intensity_change_threshold] = np.nan
        inhom_lic_pdf_arr[lic_grid_arr < local_intensity_change_threshold] = np.nan

        axis_arr = np.arange(mask_arr.ndim)
        image_extent_arr = calculate_extent(intensity_arr, voxel_spacing_arr)

        # Define the intensity range we consider. Will be overriden in 3 dimensions if self.is_adjust_grayscale_to_view
        norm = Normalize(vmin=np.nanmin(intensity_arr), vmax=np.nanmax(intensity_arr))
        if intensity_arr.ndim == 2:
            center_index_arr = None
            slice_intensity_arr_list = [intensity_arr, local_intensity_change_arr]
            slice_mask_arr_list = [mask_arr, mask_arr]
            slice_hom_mask_arr_list = [hom_mask_arr, hom_mask_arr]
            slice_peak_hom_mask_arr_list = [peak_hom_mask_arr, peak_hom_mask_arr]

            dimension_arr = np.zeros_like(axis_arr)

            extent_1_list = [image_extent_arr[0], image_extent_arr[0]]
            extent_2_list = [image_extent_arr[1], image_extent_arr[1]]
            res_string_list = [" x ".join(f"{voxel_spacing_arr[index]:.2f}{mm_string}" for index in axis_arr)
                               for i in axis_arr]
            index_string_list = ["", ""]
        else:
            # Determine the image slices to be plotted
            center_index_arr = calculate_largest_mask_indices(peak_hom_mask_arr)
            slice_intensity_arr_list = select_ort_slices(intensity_arr, center_index_arr)
            slice_mask_arr_list = select_ort_slices(mask_arr, center_index_arr)
            slice_hom_mask_arr_list = select_ort_slices(hom_mask_arr, center_index_arr)
            slice_peak_hom_mask_arr_list = select_ort_slices(peak_hom_mask_arr, center_index_arr)

            extent_1_list = [
                image_extent_arr[0] if not axis==0 else image_extent_arr[1] for axis in axis_arr
            ]
            extent_2_list = [
                image_extent_arr[2] if not axis==2 else image_extent_arr[1] for axis in axis_arr
            ]

            # Determine plot strings based on data
            res_string_list = [" x ".join(f"{voxel_spacing_arr[index]:.2f}{mm_string}"
                                          for index in np.delete(axis_arr, i))
                               for i in axis_arr]
            index_string_list = [f"Index {center_index_arr[i]},\n" if not center_index_arr is None else ""
                                 for i in axis_arr]

            # Add local intensity change of one perspective as one of the plots
            slice_lic_arr = select_ort_slices(local_intensity_change_arr, center_index_arr)[LIC_INDEX]
            slice_intensity_arr_list.append(slice_lic_arr)
            slice_mask_arr_list.append(slice_mask_arr_list[LIC_INDEX])
            slice_hom_mask_arr_list.append(slice_hom_mask_arr_list[LIC_INDEX])
            slice_peak_hom_mask_arr_list.append(slice_peak_hom_mask_arr_list[LIC_INDEX])
            extent_1_list.append(extent_1_list[LIC_INDEX])
            extent_2_list.append(extent_2_list[LIC_INDEX])
            res_string_list.append(res_string_list[LIC_INDEX])
            index_string_list.append(f"Local Intensity Change\nIndex {center_index_arr[LIC_INDEX]},\n")

            dimension_arr = np.append(axis_arr, LIC_INDEX)

            if self.is_adjust_grayscale_to_view:
                # Take the minimum and maximum intensity from the displayed areas for normalization
                norm = Normalize(vmin=np.nanmin(np.concatenate(slice_intensity_arr_list[:-1], axis=None)),
                                 vmax=np.nanmax(np.concatenate(slice_intensity_arr_list[:-1], axis=None)))


        # Set up plot depending on the amount of dimensions in the intensity array
        # Need 1 plot for the local intensity change histogram, 1 for the histogram (including the peak) and
        # n+1 for the image slices, where n is the number of dimensions. Use one slice through each dimension and one
        # slice of local intensity change as a visualization aid.
        fig, axs = plt.subplots(2, intensity_arr.ndim)
        image_axs = axs[0, :]
        if intensity_arr.ndim == 3:
            # Add bottom left axes as a plot for local intensity change if have 3 dimensions
            image_axs = np.append(image_axs, axs[1, 0])
        lic_hist_ax = axs[1, -2]
        hist_peak_ax = axs[1, -1]
        plt.suptitle(title_string)

        cmap = "gray"
        mask_norm = Normalize(vmin=0.9, vmax=1.0)

        mask_cmap = ListedColormap([np.array((255, 255, 255)) / 255., self.color_list[3]])
        hom_mask_cmap = ListedColormap([np.array((255, 255, 255)) / 255., self.color_list[1]])
        peak_hom_mask_cmap = ListedColormap([np.array((255, 255, 255)) / 255., self.color_list[0]])

        mask_alpha = self.alpha_list[3]
        hom_mask_alpha = self.alpha_list[1]
        peak_hom_mask_alpha = self.alpha_list[0]

        for i, image_ax in enumerate(image_axs):
            extent = [0, extent_1_list[i], 0, extent_2_list[i]]

            # Define the arrays to be plotted. Since the plot axis does not correspond to the data axis, we have to
            # transpose the arrays (and reverse some data points for the z-direction due to nifti weirdness)
            plot_slice_intensity_arr = np.copy(slice_intensity_arr_list[i]).T
            plot_slice_mask_arr = np.copy(slice_mask_arr_list[i]).T
            plot_slice_hom_mask_arr = np.copy(slice_hom_mask_arr_list[i]).T
            plot_slice_peak_hom_mask_arr = np.copy(slice_peak_hom_mask_arr_list[i]).T

            # Reorient plots to compensate for nifti orientation weirdness or imshow settings.
            if dimension_arr[i] == 2 and not self.is_compensate_nifti_coords:
                plot_slice_intensity_arr[:,:] = plot_slice_intensity_arr[::-1,:]
                plot_slice_mask_arr[:,:] = plot_slice_mask_arr[::-1,:]
                plot_slice_hom_mask_arr[:,:] = plot_slice_hom_mask_arr[::-1,:]
                plot_slice_peak_hom_mask_arr[:,:] = plot_slice_peak_hom_mask_arr[::-1,:]
            if dimension_arr[i] == 0 and self.is_compensate_nifti_coords:
                plot_slice_intensity_arr[:,:] = plot_slice_intensity_arr[:,::-1]
                plot_slice_mask_arr[:,:] = plot_slice_mask_arr[:,::-1]
                plot_slice_hom_mask_arr[:,:] = plot_slice_hom_mask_arr[:,::-1]
                plot_slice_peak_hom_mask_arr[:,:] = plot_slice_peak_hom_mask_arr[:,::-1]

            # Do the actual plots
            image_ax.imshow(plot_slice_intensity_arr, extent=extent, cmap=cmap, norm=norm, origin="lower",
                            interpolation=INTERPOLATION_MODE, rasterized=RASTERIZE_MODE)

            image_ax.imshow(plot_slice_mask_arr, extent=extent, cmap=mask_cmap, norm=mask_norm,
                            alpha=mask_alpha*plot_slice_mask_arr, origin="lower", interpolation=INTERPOLATION_MODE,
                            rasterized=RASTERIZE_MODE)
            image_ax.imshow(plot_slice_hom_mask_arr, extent=extent, cmap=hom_mask_cmap, norm=mask_norm,
                            alpha=hom_mask_alpha*plot_slice_hom_mask_arr, origin="lower",
                            interpolation=INTERPOLATION_MODE, rasterized=RASTERIZE_MODE)
            image_ax.imshow(plot_slice_peak_hom_mask_arr, extent=extent, cmap=peak_hom_mask_cmap, norm=mask_norm,
                            alpha=peak_hom_mask_alpha*plot_slice_peak_hom_mask_arr, origin="lower",
                            interpolation=INTERPOLATION_MODE, rasterized=RASTERIZE_MODE)

            image_ax.set_title(f"{index_string_list[i]}Resolution = {res_string_list[i]}")
            image_ax.grid(False)

            if is_use_mm:
                image_ax.set_xlabel("mm")
                image_ax.set_ylabel("mm")

        # Plotting local intensity change histograms
        lic_hist_ax.plot(lic_grid_arr, hom_lic_pdf_arr, color=self.color_list[1], label="homogeneous voxels")
        lic_hist_ax.fill_between(lic_grid_arr, hom_lic_pdf_arr, color=self.color_list[1], alpha=0.5)
        lic_hist_ax.plot(lic_grid_arr, inhom_lic_pdf_arr, color=self.color_list[3], label="inhomogeneous voxels")
        lic_hist_ax.fill_between(lic_grid_arr, inhom_lic_pdf_arr, color=self.color_list[3], alpha=0.5)

        lic_hist_ax.legend()
        lic_hist_ax.set_xlabel("Local Intensity Change")
        lic_hist_ax.set_ylabel("Frequency")
        lic_hist_ax.set_title(lic_hist_string)
        lic_hist_ax.grid(False) # False

        # Plotting homogeneous region histogram and determined histogram peak
        hist_peak_ax.plot(grid_arr, pdf_arr, color=self.color_list[3], label="whole mask")
        hist_peak_ax.plot(hom_grid_arr, nonpeak_hom_pdf_arr, color=self.color_list[1], label="homogeneous voxels")
        hist_peak_ax.fill_between(hom_grid_arr, nonpeak_hom_pdf_arr, color=self.color_list[1], alpha=0.5)
        hist_peak_ax.plot(hom_grid_arr, peak_hom_pdf_arr, color=self.color_list[0], label="voxels in peak")
        hist_peak_ax.fill_between(hom_grid_arr, peak_hom_pdf_arr, color=self.color_list[0], alpha=0.5)

        hist_peak_ax.vlines((peak_intensity), np.nanmin(pdf_arr), np.nanmax(pdf_arr),
                            colors="k", linestyles="--", label="found peak intensity")

        hist_peak_ax.legend()
        hist_peak_ax.set_xlabel("Intensity")
        hist_peak_ax.set_ylabel("Frequency")
        hist_peak_ax.set_title(hist_peak_string)
        hist_peak_ax.grid(False) # False

        if not (image_name is None or mask_name is None):
            save_path = os.path.join(f"{self.plot_folder}", f"{image_name}_mask{mask_name}{IMAGE_FORMAT}")
            plt.rcParams['figure.dpi'] = self.plot_dpi
            fig.set_size_inches(self.plot_size)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()


class BrainSplitVisualizer:


    def __init__(self, plot_folder, plot_size=(800, 600), plot_dpi=100., color_list=None):
        self.plot_folder = plot_folder
        self.plot_size = plot_size
        self.plot_dpi = plot_dpi
        self.color_list = color_list if not color_list is None else ["black", "green", "lime"]

    def visualize_histogram_split(self, grid_arr, pdf_arr, lic_weighted_pdf_arr, split_int, erosion_width,
                                  cutoff_fraction, avg_lic_arr, avg_lic_arr_raw, image_name=None):
        # Rescale the lic weighted pdf so that it can be shown on the same scale as the pdf
        plottable_lic_weighted_pdf_arr = lic_weighted_pdf_arr/np.nansum(lic_weighted_pdf_arr)*np.nansum(pdf_arr)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        plt.title(f"Split intensity: {split_int:.2f}\nSettings: Brain erosion width: {erosion_width},"
                  f"\n edges cutoff fraction {cutoff_fraction}")
        ax1.plot(grid_arr, pdf_arr, color=self.color_list[0], label="total histogram")
        ax1.plot(grid_arr, plottable_lic_weighted_pdf_arr, color=self.color_list[2],
                 label="local intensity change weighted histogram\n(arbitrary scale)")
        # Plot this invisible line to make sure the average local intensity change curve shows up in the legend easily.
        ax1.plot([grid_arr[0]], [pdf_arr[0]], color=self.color_list[1], label="average local intensity change")
        ax1.vlines((split_int), np.amin(pdf_arr), np.amax(pdf_arr), colors=self.color_list[1], linestyles="--",
                   label="found split")

        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Probability Density")
        ax1.legend()
        ax2.plot(grid_arr, avg_lic_arr_raw, color="g", linestyle=":", label="average local intensity change")
        ax2.plot(grid_arr, avg_lic_arr, color="g", label="average local intensity change after exclusion")
        ax2.set_ylabel("Local Intensity Change")

        ax2.set_ylim(0, 1.5*np.nanmax(avg_lic_arr))

        if not image_name is None:
            save_path = os.path.join(f"{self.plot_folder}", f"{image_name}_brainsplit{IMAGE_FORMAT}")
            plt.rcParams['figure.dpi'] = self.plot_dpi
            fig.set_size_inches(self.plot_size)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()