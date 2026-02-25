import numpy as np

import time

import matplotlib.pyplot as plt

from scipy.signal import find_peaks, peak_prominences, peak_widths, convolve
from scipy.ndimage import median_filter, generate_binary_structure, grey_erosion, grey_dilation, maximum_filter, \
    minimum_filter, binary_opening, binary_closing, binary_dilation, binary_erosion, sobel
from KDEpy import FFTKDE


def combine_dict_with_default(new_dict, default_dict):
    new_dict = new_dict.copy() if not new_dict is None else {}
    for key in default_dict:
        new_dict[key] = new_dict.get(key, default_dict[key])
    return new_dict

def obtain_flat_arrays(image_arr, gradient_arr, mask=None, is_ignore_nan=True, is_return_coords=False):
    if mask is None:
        mask = np.ones_like(image_arr).astype(bool) # If no mask specified, take whole image

    if is_ignore_nan:
        mask = mask.copy() # If ignore nans, exclude objects from mask where image or gradient are nan
        mask = mask & (~np.isnan(image_arr)) & (~np.isnan(gradient_arr))

    # Obtain all gradients in mask, sort both intensities and gradients according to ascending gradients
    flat_image_arr = image_arr[mask]
    flat_gradient_arr = gradient_arr[mask]

    gradient_order = np.argsort(flat_gradient_arr)
    flat_image_arr = flat_image_arr[gradient_order]
    flat_gradient_arr = flat_gradient_arr[gradient_order]

    if not is_return_coords:
        return flat_image_arr, flat_gradient_arr
    else:
        flat_coord_arr = np.argwhere(mask)
        flat_coord_arr = flat_coord_arr[gradient_order]
        return flat_image_arr, flat_gradient_arr, flat_coord_arr

def calculate_KDE(array, grid, bw='ISJ', weights=None, is_return_bandwith=False):
    # Workaround because of numerical problems
    threshold = 1e+4
    rescale_factor = 1.
    #print(f"The array is {array}")
    #print(f"The grid is {grid}")
    #print(f"The array shape is {array.shape}")
    #print(f"The array max is {np.nanmax(array)}")
    array_intensity_range = np.nanmax(array)-np.nanmin(array)
    is_rescale = array_intensity_range > threshold
    if is_rescale:
        array = array.copy()
        rescale_factor = array_intensity_range/threshold
        array /= rescale_factor
        if not isinstance(grid, int):
            grid = grid.copy()
            grid /= rescale_factor
        if not isinstance(bw, str):
            bw /= rescale_factor

    #print(f"The rescale factor is {rescale_factor}")
    #print(f"The array max is {np.nanmax(array)}")
    # Suggested for bw: automatic bw selection using Improved Sheather Jones (ISJ)
    # Always use gaussian kernel
    KDE = FFTKDE(kernel='gaussian', bw=bw) # Define KDE
    KDE.fit(array, weights=weights) # Fit data

    # If grid is an integer, a new grid will be created by the evaluate funciton. Otherwise, the existing one is used
    if isinstance(grid, int):
        intensities, pdf_arr = KDE.evaluate(grid) # Evaluate KDE on grid
    else:
        pdf_arr = KDE.evaluate(grid) # Evaluate KDE on grid
        intensities = grid
    bw = KDE.bw

    if is_rescale:
        #array *= rescale_factor
        #if not isinstance(grid, int):
        #    grid *= rescale_factor
        intensities *= rescale_factor
        pdf_arr /= rescale_factor
        bw *= rescale_factor

    if not is_return_bandwith:
        return intensities, pdf_arr
    return intensities, pdf_arr, bw #Return also KDE bandwith

def calculate_KDE_with_discretized_points(arr, kde_grid, kde_bandwith, n_discrete=1024):
    val_min = np.nanmin(arr)
    val_range = np.nanmax(arr)-val_min
    discretized_arr = (arr.copy() - val_min) / (val_range/n_discrete) if not val_range <= 0 else arr.copy() - val_min
    discretized_arr = np.around(discretized_arr)
    discretized_arr = discretized_arr * (val_range/n_discrete) + val_min
    grid, discretized_pdf, bw = calculate_KDE(discretized_arr, kde_grid, bw=kde_bandwith, is_return_bandwith=True)
    pdf = calculate_KDE(discretized_arr, grid, bw=bw) [1]
    return grid, pdf, bw

def calculate_voxel_volume(voxel_spacing_arr=None):
    # Calculate voxel volume.
    # If no voxel spacing given, default to counting the number of voxels. Thus 1 voxel has volume 1
    if voxel_spacing_arr is None:
        return 1.
    else:
        return np.prod(voxel_spacing_arr)

def automatic_snippet_number_determination(n_voxels, voxel_spacing_arr=None, n_min=10, n_max=100, vol_min=1000):
    # Calculate how many voxel a snippet should contain to fulfill the vol_min requirement
    voxel_number_min = vol_min/calculate_voxel_volume(voxel_spacing_arr)
    # Calculate how many snippet should be created. Want at least n_min and at most n_max. If in between, should pick
    # an amount that leaves every snippet with at least vol_min volume.
    return max(n_min, min(n_max, int(n_voxels/voxel_number_min)))

def calculate_ball_kernel(radius):# Define ball as kernel
    int_radius = int(np.floor(radius)) # If radius is lower than an integer, no voxel of that distance is included
    kernel_size = int(2*int_radius+1)
    kernel_3d = np.array([[[(((i-int_radius)**2+(j-int_radius)**2+(k-int_radius)**2)<=radius**2)
                            for k in range(kernel_size)] for j in range(kernel_size)] for i in range(kernel_size)], dtype=np.float64)
    return kernel_3d

def calculate_structuring_element(radius, dim=3, voxel_spacing_arr=None):
    if voxel_spacing_arr is None:
        voxel_spacing_arr = np.ones(dim)
    int_radius_arr = np.floor(radius/voxel_spacing_arr).astype(np.int32)
    #structuring_element_size_arr = 2*int_radius_arr + 1
    coordinate_list = [np.linspace(-int_r, int_r, 2*int_r+1)*d for d, int_r in zip(voxel_spacing_arr, int_radius_arr)]
    grid_arr = np.array(np.meshgrid(*coordinate_list, indexing="ij"))
    dist_arr = np.linalg.norm(grid_arr, axis=0)
    structuring_element = dist_arr <= radius
    return structuring_element

def calculate_ball_kernel_library(radius, dim=3):# Define ball as kernel
    structure = generate_binary_structure(dim, connectivity=int(radius**2))
    return structure

def calculate_local_mean_image(image, radius=1.8, is_ignore_nan=True, voxel_spacing_arr=None):
    int_radius = int(np.floor(radius)) # If radius is lower than an integer, no voxel of that distance is included
    kernel_3d = calculate_structuring_element(radius, dim=image.ndim, voxel_spacing_arr=voxel_spacing_arr)
    #kernel_3d /= np.sum(kernel_3d) # Should not divide here yet so we can retain the count

    if is_ignore_nan:
        # If we ignore nan, have to do the padding manually
        shape_padded = tuple(np.array(image.shape) + 2*int_radius)

        padded_image = np.nan * np.ones(shape_padded)
        padded_image[int_radius:-int_radius, int_radius:-int_radius, int_radius:-int_radius] = image[:,:,:]

        nan_of_image = np.isnan(padded_image)
        nonnan_image = padded_image.copy()
        nonnan_image[nan_of_image] = 0.

        local_mean_image = convolve(nonnan_image, kernel_3d, mode="same")
        local_mean_image[nan_of_image] = np.nan

        nonnan_convolved = convolve((~nan_of_image).astype(np.float64), kernel_3d, mode="same")
        local_mean_image /= nonnan_convolved #Need to normalize by the amount of non nan overlap
        #local_mean_image /= np.sum(kernel_3d) #Need to divide to normalize

        return local_mean_image[int_radius:-int_radius, int_radius:-int_radius, int_radius:-int_radius]
    else:
        kernel_3d /= np.sum(kernel_3d) # Should divide here in order to keep things normalized
        local_mean_image = convolve(image, kernel_3d, mode="same")
        return local_mean_image

def calculate_smooth_image_in_mask(image_arr, mask, filter_radius=1.5, voxel_spacing_arr=None):
    masked_image_arr = image_arr.copy()
    masked_image_arr[~mask] = np.nan
    final_image = np.zeros(mask.shape) * np.nan
    flat_coo_arr = np.argwhere(mask)

    min_indices = np.amin(flat_coo_arr, axis=0).astype(int)
    max_indices = np.amax(flat_coo_arr, axis=0).astype(int) + 1
    # Smooth the reconstructed image with the chosen radius, should reduce noise
    smooth_image_arr = calculate_local_mean_image(masked_image_arr[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1], min_indices[2]:max_indices[2]], filter_radius, is_ignore_nan=True, voxel_spacing_arr=voxel_spacing_arr)
    final_image[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1], min_indices[2]:max_indices[2]] = smooth_image_arr
    return final_image

def calculate_local_median_image(image, radius=1.8, is_ignore_nan=True): # todo
    int_radius = int(np.floor(radius)) # If radius is lower than an integer, no voxel of that distance is included
    kernel_3d = calculate_structuring_element(radius)

def calculate_mask_compactness(mask, radius=1.0, is_normalized=True, voxel_spacing_arr=None): # Calculate average number of neighbors among voxels in a mask
    int_radius = int(np.floor(radius)) # If radius is lower than an integer, no voxel of that distance is included
    kernel_3d = calculate_structuring_element(radius, dim=mask.ndim, voxel_spacing_arr=voxel_spacing_arr)
    kernel_3d[int_radius, int_radius, int_radius] = 0. # Exclude the center element from the kernel, so it only counts neighbors
    max_neighbor_amount = np.count_nonzero(kernel_3d)
    local_neighbor_count = convolve(mask.astype(np.float64), kernel_3d, mode="same")
    mean_neighbors_in_mask = np.mean(local_neighbor_count[mask])
    if not is_normalized:
        return max_neighbor_amount - mean_neighbors_in_mask
    else:
        return 1. - mean_neighbors_in_mask / max_neighbor_amount


def calculate_dice_coeff(mask_1, mask_2):
    return 2*np.count_nonzero(mask_1 & mask_2)/(np.count_nonzero(mask_1) + np.count_nonzero(mask_2))

def dilate_mask(mask, radius=1.8, iterations=1, voxel_spacing_arr=None):
    structure = calculate_structuring_element(radius, dim=mask.ndim, voxel_spacing_arr=voxel_spacing_arr)
    return binary_dilation(mask, structure, iterations=iterations)

def erode_mask(mask, radius=1.8, iterations=1, voxel_spacing_arr=None):
    structure = calculate_structuring_element(radius, dim=mask.ndim, voxel_spacing_arr=voxel_spacing_arr)
    return binary_erosion(mask, structure, iterations=iterations)

def open_mask(mask, radius=1.8, iterations=1, voxel_spacing_arr=None):
    structure = calculate_structuring_element(radius, dim=mask.ndim, voxel_spacing_arr=voxel_spacing_arr)
    return binary_opening(mask, structure, iterations=iterations)

def close_mask(mask, radius=1.8, iterations=1, voxel_spacing_arr=None):
    structure = calculate_structuring_element(radius, dim=mask.ndim, voxel_spacing_arr=voxel_spacing_arr)
    return binary_closing(mask, structure, iterations=iterations)

def calculate_neighborhood_minimum_and_maximum(image, radius=1.8):
    # Define ball to consider
    int_radius = int(np.floor(radius)) # If radius is lower than an integer, no voxel of that distance is included

    # An array that saves which relative displacenemts will be considered
    shift_cube = np.array([[[(i,j,k) for k in range(2*int_radius+1)] for j in range(2*int_radius+1)] for i in range(2*int_radius+1)])-int_radius
    shift_arr = shift_cube.reshape((-1, 3))
    shift_arr = shift_arr[np.linalg.norm(shift_arr, axis=-1) <= radius] #Only include voxels within radius

    # pad image with nans so that no problem occurs with edges
    padded_image = np.pad(image, ((int_radius, int_radius), (int_radius, int_radius), (int_radius, int_radius)), constant_values=np.nan)

    nonnan_indices_nonpad = np.where(~np.isnan(image))
    nonnan_indices = (np.array(nonnan_indices_nonpad) + int_radius).reshape(3, 1, -1)

    all_displacements = nonnan_indices + shift_arr.transpose().reshape(3, -1, 1)
    min_values = np.nanmin(padded_image[all_displacements[0], all_displacements[1], all_displacements[2]], axis=0)
    max_values = np.nanmax(padded_image[all_displacements[0], all_displacements[1], all_displacements[2]], axis=0)

    min_image = np.nan*np.ones_like(image)
    min_image[nonnan_indices_nonpad] = min_values
    max_image = np.nan*np.ones_like(image)
    max_image[nonnan_indices_nonpad] = max_values

    return min_image, max_image

def calculate_gradient_from_min_and_max(data, min_data, max_data):
    #a = data-min_data
    #b = max_data-data
    return max_data-min_data

def calculate_gradient_from_min_and_max_p(data, min_data, max_data, p=0.5):
    a = data-min_data
    b = max_data-data
    return ((max_data)**p-(min_data)**p)**(1./p)

def calculate_gradient_from_min_and_max_maximum(data, min_data, max_data):
    a = data-min_data
    b = max_data-data
    return np.maximum(a, b)

def create_gradient_function_old(radius=1.8):
    # The normalizer needs a gradient function as input. This method provides that based on min and max
    def grad_func(array):
        min_arr, max_arr = calculate_neighborhood_minimum_and_maximum(array, radius)
        return calculate_gradient_from_min_and_max(array, min_arr, max_arr)
    return grad_func

def create_gradient_function_old_time(radius=1.8):
    # The normalizer needs a gradient function as input. This method provides that based on min and max
    def grad_func(array):
        start = time.time()
        min_arr, max_arr = calculate_neighborhood_minimum_and_maximum(array, radius)
        grad = calculate_gradient_from_min_and_max(array, min_arr, max_arr)
        print(f"Gradient for array with shape {array.shape} took {time.time()-start}s")
        return grad
    return grad_func

def create_gradient_function(radius=1.8, is_correct_near_miss=True):
    # The normalizer needs a gradient function as input. This method provides that based on min and max
    # If is_correct_near_miss, increase the radius slightly so it doesn't miss voxels due to floating point inaccuracies
    if is_correct_near_miss:
        radius *= 1.001
    def grad_func(array, voxel_spacing_arr=None):
        #start = time.time()
        structure_ball_mask = calculate_structuring_element(radius, dim=array.ndim, voxel_spacing_arr=voxel_spacing_arr)
        max_arr = maximum_filter(array, footprint=structure_ball_mask)
        min_arr = minimum_filter(array, footprint=structure_ball_mask)
        grad = calculate_gradient_from_min_and_max(array, min_arr, max_arr)
        #print(f"New gradient for array with shape {array.shape} took {time.time()-start}s")
        return grad
    return grad_func

def create_gradient_function_good_median(radius=1.8, is_correct_near_miss=True):
    # The normalizer needs a gradient function as input. This method provides that based on min and max
    # If is_correct_near_miss, increase the radius slightly so it doesn't miss voxels due to floating point inaccuracies
    if is_correct_near_miss:
        radius *= 1.001
    def grad_func(array, voxel_spacing_arr=None):
        #start = time.time()
        structure_ball_mask = calculate_structuring_element(radius, dim=array.ndim, voxel_spacing_arr=voxel_spacing_arr)
        median_filtered_array = median_filter(array, footprint=structure_ball_mask)
        max_arr = maximum_filter(median_filtered_array, footprint=structure_ball_mask)
        min_arr = minimum_filter(median_filtered_array, footprint=structure_ball_mask)
        grad = calculate_gradient_from_min_and_max(median_filtered_array, min_arr, max_arr)
        #print(f"New gradient for array with shape {array.shape} took {time.time()-start}s")
        return grad
    return grad_func

def create_gradient_function_p(radius=1.8, is_correct_near_miss=True):
    # The normalizer needs a gradient function as input. This method provides that based on min and max
    # If is_correct_near_miss, increase the radius slightly so it doesn't miss voxels due to floating point inaccuracies
    if is_correct_near_miss:
        radius *= 1.001
    def grad_func(array, voxel_spacing_arr=None):
        #start = time.time()
        structure_ball_mask = calculate_structuring_element(radius, dim=array.ndim, voxel_spacing_arr=voxel_spacing_arr)
        max_arr = maximum_filter(array, footprint=structure_ball_mask)
        min_arr = minimum_filter(array, footprint=structure_ball_mask)
        grad = calculate_gradient_from_min_and_max_p(array, min_arr, max_arr, 0.5)
        #print(f"New gradient for array with shape {array.shape} took {time.time()-start}s")
        return grad
    return grad_func

def create_gradient_function_sobel(radius=1.8):
    # The normalizer needs a gradient function as input. This method provides that based on min and max
    def grad_func(array, voxel_spacing_arr=None):
        #start = time.time()
        sobel_x = sobel(array, axis=0)
        sobel_y = sobel(array, axis=1)
        sobel_z = sobel(array, axis=2)
        grad = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)
        #print(f"New gradient for array with shape {array.shape} took {time.time()-start}s")
        return grad
    return grad_func

def split_sorted_array(array, n_sections):
    if n_sections == 1:
        return [array]
    section_size = int(np.ceil(array.shape[0] / n_sections)) # Mathematical section size

    # Old version (only good in 1d)
    # Look at highest index that is still equal to the section size element, take that as new section size
    # section_size = np.amax(np.argwhere(array == array[section_size-1])) + 1

    if len(array.shape) == 1:
        distance_list = array - array[section_size-1]
    elif len(array.shape) == 2:
        distance_list = np.linalg.norm(array - array[section_size-1], axis=1)
    else:
        raise ValueError(f"The dimension of the given array ({len(array.shape)}) is not supported")

    section_size = np.amax(np.argwhere(distance_list == 0)) + 1

    if section_size == array.size: # If the entire remainder of the array is in the section, stop making new sections
        return [array]
    return [array[:section_size]] + split_sorted_array(array[section_size:], n_sections-1)

def split_array_like_array_list(array, array_list):
    index_arr = np.append(0, np.cumsum([(a.shape[0]) for a in array_list]))
    return [array[index_arr[i]:index_arr[i+1]] for i in range(len(index_arr)-1)]

def split_sorted_array_old(array, n_sections):
    avg_section_size = array.size // n_sections
    number_larger_sections = array.size % n_sections
    section_size_arr = np.array([avg_section_size + int(i < number_larger_sections) for i in range(n_sections)])
    boundary_index_arr = np.array(np.append(0, np.cumsum(section_size_arr)))
    for index, boundary_index in enumerate(boundary_index_arr[1:-1]):
        # Find the largest index for a value that is equal to the previously established boundary value.
        # Choose that as the new boundary index
        new_boundary_index = np.amax(np.argwhere(array == array[boundary_index]))
        boundary_index_arr[index+1] = new_boundary_index
    return [array[boundary_index_arr[i]:boundary_index_arr[i+1]] for i in range(len(boundary_index_arr)-1)]


def calculate_split_intensity_mean(image_arr, factor=2./3.):
    mean_int = np.nanmean(image_arr)
    lower_int = np.nanmin(image_arr)

    split_int = lower_int + (mean_int-lower_int)*factor
    return split_int

def calculate_split_intensity_median(image_arr, factor=2./3., percentile=1.):
    median_int = np.nanmedian(image_arr)
    lower_int = np.nanpercentile(image_arr.ravel(), percentile)

    split_int = lower_int + (median_int-lower_int)*factor
    return split_int

def calculate_split_intensity_kde_withexperiments(image_arr, gradient_arr, mask, kde_grid, kde_bw, bottom_percentile=1., top_percentile=99.):
    flat_image_arr, flat_gradient_arr, _ = obtain_flat_arrays(image_arr, gradient_arr, mask)
    median_int = np.median(flat_image_arr)
    grid_median_index = np.searchsorted(kde_grid, median_int, 'right')

    lower_grad_index = np.searchsorted(flat_gradient_arr, np.percentile(flat_gradient_arr, bottom_percentile), 'right')
    higher_grad_index = np.searchsorted(flat_gradient_arr, np.percentile(flat_gradient_arr, top_percentile), 'left')

    lower_grad_image_arr = flat_image_arr[:lower_grad_index]
    higher_grad_image_arr = flat_image_arr[higher_grad_index:]

    lower_grad_pdf_nonnorm = calculate_KDE(lower_grad_image_arr, kde_grid, kde_bw)[1]*lower_grad_image_arr.size
    higher_grad_pdf_nonnorm = calculate_KDE(higher_grad_image_arr, kde_grid, kde_bw)[1]*higher_grad_image_arr.size

    #comparison_arr = (higher_grad_pdf_nonnorm-lower_grad_pdf_nonnorm)/(lower_grad_pdf_nonnorm+1)
    #normal_scale_size = 100000
    #correction_factor = (np.amax(flat_image_arr) - np.amin(flat_image_arr)) / normal_scale_size
    #lower_grad_pdf_nonnorm_scaled = lower_grad_pdf_nonnorm * correction_factor # Good idea, but doesn't work for some images
    bias_factor = 1000
    lower_grad_pdf_nonnorm_scaled = lower_grad_pdf_nonnorm / np.amax(lower_grad_pdf_nonnorm) * bias_factor
    comparison_arr = higher_grad_pdf_nonnorm/(lower_grad_pdf_nonnorm_scaled+1)
    split_int_index = np.argmax(comparison_arr[:grid_median_index])
    split_int = kde_grid[split_int_index]

    # For debugging
    all_pdf_nonnorm = calculate_KDE(flat_image_arr, kde_grid, kde_bw)[1]*flat_image_arr.size
    plt.figure()
    plt.title(f"Split intensity: {split_int}")
    plt.plot(kde_grid, all_pdf_nonnorm, label="all pdf")
    plt.plot(kde_grid, lower_grad_pdf_nonnorm, label="lower pdf")
    plt.plot(kde_grid, higher_grad_pdf_nonnorm, label="higher pdf")
    plt.plot(kde_grid, comparison_arr , label="comparison pdf")
    plt.vlines((median_int), np.amin(all_pdf_nonnorm), np.amax(all_pdf_nonnorm), colors='g', label="median")
    plt.vlines((split_int), np.amin(all_pdf_nonnorm), np.amax(all_pdf_nonnorm), colors='k', label="found split")
    plt.legend()
    plt.show()

    return split_int

def calculate_split_intensity_kde(image_arr, gradient_arr, mask, kde_grid, kde_bw, percentage_of_voxels=50., bias_factor=1000, is_search_below_median=True):
    flat_image_arr, flat_gradient_arr, _ = obtain_flat_arrays(image_arr, gradient_arr, mask)
    median_int = np.median(flat_image_arr)
    grid_median_index = np.searchsorted(kde_grid, median_int, 'right')

    lower_grad_index = np.searchsorted(flat_gradient_arr, np.percentile(flat_gradient_arr, percentage_of_voxels), 'right')
    higher_grad_index = np.searchsorted(flat_gradient_arr, np.percentile(flat_gradient_arr, 100.-percentage_of_voxels), 'left')

    lower_grad_image_arr = flat_image_arr[:lower_grad_index]
    higher_grad_image_arr = flat_image_arr[higher_grad_index:]

    lower_grad_pdf_nonnorm = calculate_KDE(lower_grad_image_arr, kde_grid, kde_bw)[1]*lower_grad_image_arr.size
    higher_grad_pdf_nonnorm = calculate_KDE(higher_grad_image_arr, kde_grid, kde_bw)[1]*higher_grad_image_arr.size

    lower_grad_pdf_nonnorm_scaled = lower_grad_pdf_nonnorm / np.amax(lower_grad_pdf_nonnorm) * bias_factor
    comparison_arr = higher_grad_pdf_nonnorm/(lower_grad_pdf_nonnorm_scaled+1)

    # Need to determine whether to search below or above median
    grid_lower_search_index = 0 if is_search_below_median else grid_median_index
    grid_higher_search_index = grid_median_index if is_search_below_median else kde_grid.size
    split_int_index = np.argmax(comparison_arr[grid_lower_search_index:grid_higher_search_index])
    split_int = kde_grid[split_int_index]

    return split_int

def calculate_split_intensity_kde_fail(image_arr, gradient_arr, mask, kde_grid, kde_bw, bottom_percentile=1., top_percentile=99.):
    flat_image_arr, flat_gradient_arr, _ = obtain_flat_arrays(image_arr, gradient_arr, mask)
    median_int = np.median(flat_image_arr)
    grid_median_index = np.searchsorted(kde_grid, median_int, 'right')

    higher_grad_index = np.searchsorted(flat_gradient_arr, np.percentile(flat_gradient_arr, top_percentile), 'left')

    higher_grad_image_arr = flat_image_arr[higher_grad_index:]

    higher_grad_pdf_nonorm = calculate_KDE(higher_grad_image_arr, kde_grid, kde_bw)[1]*higher_grad_image_arr.size

    all_pdf_nonnorm = calculate_KDE(flat_image_arr, kde_grid, kde_bw)[1]*flat_image_arr.size

    #comparison_arr = np.zeros_like(kde_grid) # Division approach. doesn't really work
    #comparison_arr[all_pdf_nonnorm > 0.] = higher_grad_pdf_nonorm[all_pdf_nonnorm > 0.]/all_pdf_nonnorm[all_pdf_nonnorm > 0.]

    comparison_arr = all_pdf_nonnorm - higher_grad_pdf_nonorm
    comparison_arr_belowmedian = comparison_arr[:grid_median_index]

    #split_int_index = np.argmax(comparison_arr[:grid_median_index]-comparison_arr[:grid_median_index]) #for division approach
    split_int_index = np.amax(np.nonzero(comparison_arr_belowmedian == np.amin(comparison_arr_belowmedian)))
    split_int = kde_grid[split_int_index]

    # For debugging
    plt.figure()
    plt.plot(kde_grid, all_pdf_nonnorm, label="all pdf")
    plt.plot(kde_grid, higher_grad_pdf_nonorm, label="higher pdf")
    plt.plot(kde_grid, comparison_arr, label="fraction of higher and all pdf (excluding zeros)")
    plt.vlines((median_int), np.amin(all_pdf_nonnorm), np.amax(all_pdf_nonnorm), colors='g', label="median")
    plt.vlines((split_int), np.amin(all_pdf_nonnorm), np.amax(all_pdf_nonnorm), colors='k', label="found split")
    plt.legend()
    plt.show()

    return split_int

def calculate_split_intensity_masks(image_arr, split_intensity=None, factor=2./3.):
    if split_intensity is None:
        split_intensity = calculate_split_intensity_median(image_arr, factor)

    mask_1 = image_arr <= split_intensity
    mask_2 = image_arr > split_intensity

    return mask_1, mask_2

def calculate_split_intensity_masks_kde(image_arr, gradient_arr, mask, kde_grid, kde_bw, percentage_of_voxels=50., bias_factor=1000):
    #split_intensity = calculate_split_intensity_kde(image_arr, gradient_arr, mask, kde_grid, kde_bw, bottom_percentile=bottom_percentile, top_percentile=top_percentile) # Old
    split_intensity = calculate_split_intensity_kde(image_arr, gradient_arr, mask, kde_grid, kde_bw, percentage_of_voxels=percentage_of_voxels, bias_factor=bias_factor)

    mask_1 = image_arr <= split_intensity
    mask_2 = image_arr > split_intensity

    return mask_1, mask_2

def calculate_voxel_spacing_from_affine(affine_mat):
    print("philipp function")
    print(np.linalg.norm(affine_mat[:-1,:-1], axis=0))
    print(np.linalg.norm(affine_mat[:-1, :-1], axis=0).shape)
    print(type(np.linalg.norm(affine_mat[:-1,:-1], axis=0)))
    return np.linalg.norm(affine_mat[:-1,:-1], axis=0)

def cutoff_percentiles(image_arr, mask=None, bottom_perc=1, top_perc=99):
    if mask is None:
        mask = ~np.isnan(image_arr)
    bottom_val = np.percentile(image_arr[mask], bottom_perc, method="lower")
    top_val = np.percentile(image_arr[mask], top_perc, method="higher")
    image_arr = np.maximum(image_arr, bottom_val)
    image_arr = np.minimum(image_arr, top_val)
    return image_arr

def hellinger_distance(p, q):
    # According to this: https://en.wikipedia.org/wiki/Hellinger_distance
    return np.sqrt(1 - np.vdot(np.sqrt(p), np.sqrt(q)))

def obtain_histogram_peaks_indices(hist, prom_fraction=0.1):
    # Determine all peak locations
    peaks = find_peaks(hist)[0]

    # Determine prominences and left and right bases for all peaks
    peak_proms, peak_l_bases, peak_r_bases = peak_prominences(hist, peaks)

    # Determine the indices of the peaks that are above the prominence threshold compared to the largest peak
    prominent_indices = peak_proms >= prom_fraction * np.amax(peak_proms)

    # Select only the peaks above the prominece thresholds
    peaks = peaks[prominent_indices]
    peak_proms = peak_proms[prominent_indices]
    peak_l_bases = peak_l_bases[prominent_indices]
    peak_r_bases = peak_r_bases[prominent_indices]
    peak_heights = hist[peaks]

    # Determine the split of the interval among the peaks. Start with the bases from the prominence algorithm
    peak_l_bound_arr = peak_l_bases.copy()
    peak_r_bound_arr = peak_r_bases.copy()

    for i, peak_index in enumerate(peaks):
        # Take as left bound the maximum of all lower index right bounds or the left bound itself.
        peak_l_bound_arr[i] = np.amax(np.append(peak_r_bases[peak_r_bases < peak_index], peak_l_bound_arr[i]))
        # Analogous for right bound
        peak_r_bound_arr[i] = np.amin(np.append(peak_l_bound_arr[peak_l_bound_arr > peak_index], peak_r_bound_arr[i]))

    return peaks, peak_heights, peak_l_bound_arr, peak_r_bound_arr

def obtain_histogram_peaks(grid, hist, prom_fraction=0.1):
    hist_peaks, hist_peak_heights, hist_peak_l_bound_arr, hist_peak_r_bound_arr = obtain_histogram_peaks_indices(hist, prom_fraction)
    return grid[hist_peaks], hist_peak_heights, grid[hist_peak_l_bound_arr], grid[hist_peak_r_bound_arr]

def obtain_gradient_thresholded_mask(gradient_arr, gradient_threshold, mask=None):
    if mask is None:
        mask = np.ones_like(gradient_arr).astype(bool)
    return (gradient_arr <= gradient_threshold) & mask

def calculate_last_intersection_point(arr_1, arr_2):
    diff = arr_2-arr_1
    return np.amax(np.argwhere(diff >= 0.))

def calculate_first_intersection_point(arr_1, arr_2):
    diff = arr_2-arr_1
    if np.any(diff < 0):
        return np.amin(np.argwhere(diff < 0.))
    else:
        print("Warning! there is no intersection between the two JSD curves")
        return 0

def calculate_optimal_interpolation_intensities(old_peak_intensities, new_peak_intensities):
    n = len(old_peak_intensities)
    D = np.zeros((n-1, n))
    D[np.arange(n-1), np.arange(n-1)] = -1.
    D[np.arange(n-1), np.arange(n-1)+1] = 1.
    K = np.zeros((n-1, n))
    K[np.arange(n-1), np.arange(n-1)+1] = 1.
    b = np.zeros(n)
    b[0] = 1.

    a = 2*np.dot(D, new_peak_intensities)/np.dot(D, old_peak_intensities)
    G = np.zeros((n, n))
    for i in range(n):
        G[np.arange(i, n), np.arange(0, n-i)] = (-1)**i
    Gdm = G - np.mean(G, axis=0).reshape((-1, n))

    x = np.dot(np.dot(Gdm.T, Gdm), b)

    g1 = - np.dot(x.T, np.dot(K.T, a)) / np.dot(x.T, b)

    g_vec = np.dot(G, g1*b + np.dot(K.T, a))

    new_split_intensities = new_peak_intensities[:-1] + 1./2. * np.dot(np.dot(np.diag(g_vec[:-1]), D), old_peak_intensities)
    return new_split_intensities

def main():
    old_peak_intensities = [0, 50, 100]#[0, 50, 100, 200, 500, 550, 680, 800]
    new_peak_intensities = [0, 80, 100]#[0, 80, 100, 180, 450, 570, 600, 800]
    old_peak_intensities_middle = 1./2.*(np.array(old_peak_intensities[:-1])+np.array(old_peak_intensities[1:]))
    result = calculate_optimal_interpolation_intensities(old_peak_intensities, new_peak_intensities)
    plt.figure()
    plt.plot([old_peak_intensities[0]] + list(old_peak_intensities_middle) + [old_peak_intensities[-1]], [new_peak_intensities[0]] + list(result) + [new_peak_intensities[-1]], label="space between peaks as landmarks")
    plt.plot(old_peak_intensities, new_peak_intensities, 'r', label="peaks as landmarks")
    plt.plot(old_peak_intensities, new_peak_intensities, 'rx')
    plt.legend()
    plt.xlabel("non-normalized intensity")
    plt.xlabel("normalized intensity")
    plt.show()

    # Basic test of the split sorted functionality
    array = np.arange(102)
    array[:20] = 0
    n_sections = 20
    #result = split_sorted_array(array, n_sections)
    #result_old = split_sorted_array_old(array, n_sections)


    # Test of the split sorted functionality in higher dimensions
    array = np.zeros((50, 50, 50))
    array[:,:25,:] = 1
    array_where = np.argwhere(array == 1)
    array_where_2 = np.argwhere(array == 0)

    result = split_sorted_array(array_where, n_sections)
    result_2 = split_array_like_array_list(array_where_2, result)

    result_1d = split_sorted_array(np.arange(array_where.shape[0]), n_sections)
    result_2_with1d = split_array_like_array_list(array_where_2, result_1d)

    print("End")


if __name__ == "__main__":
    main()