import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import maximum_filter, minimum_filter, binary_opening, binary_closing, binary_dilation, \
    binary_erosion
from scipy.interpolate import interp1d
from KDEpy import FFTKDE

CONTRASTS_PEAK_BELOW_MED = ["t1", "t1c", "flair"]
CONTRASTS_PEAK_ABOVE_MED = ["t2"]


def calculate_voxel_spacing_from_affine(affine_mat):
    """
    Calculates the voxel spacing, as used by the normalization, from the affine matrix, as is usually stored in nifti
    headers.
    :param affine_mat: numpy array (2 dimensions): float or integer (n+1)x(n+1) matrix that describes the transformation
    from image coordinates to real world coordinates, where n is the number of dimensions of the image.
    :return: numpy array (1 dimension): float with length n, where n is the number of dimensions of the image.s
    """
    return np.linalg.norm(affine_mat[:-1,:-1], axis=0)

def calculate_footprint(radius, dim=3, voxel_spacing_arr=None):
    """
    Calculates a footprint, in essence a binary spherical mask, for a given radius, dimension amount and
    voxel spacing.
    :param radius: integer or float: Radius of the footprint. Same units as given in voxel_spacing_arr
    :param dim: (optional) integer: number of dimensions used. Only relevant if no voxel_spacing_arr given
    :param voxel_spacing_arr: (optional) numpy array (1 dimension): Integer or float array with length n, where n is the
        number of dimensions of the image. For each dimension, this array contains the distance between voxels along
        that axis (assumed to be in mm). Default: assume all are 1
    :return: numpy array (n dimensions): boolean array that represents a binary mask where values with "True" are
    considered inside the footprint. The length of the array along each dimension is chosen dynamically.
    The resulting array can be used in filter functions of standard libraries
    """
    # If no voxel spacing array is given, use ones of the length of the dimension
    if voxel_spacing_arr is None:
        voxel_spacing_arr = np.ones(dim)
    # Calculate the number of voxels that are within the radius in each dimension
    int_radius_arr = np.floor(radius/voxel_spacing_arr).astype(np.int32)

    # Create coordinates from negative to positive the amount of voxels in each direction
    coordinate_list = [np.linspace(-int_r, int_r, 2*int_r+1)*d for d, int_r in zip(voxel_spacing_arr, int_radius_arr)]
    # Create a grid based on the coordinates, so that a new dimension is created
    grid_arr = np.array(np.meshgrid(*coordinate_list, indexing="ij"))
    # Calculate the norm of the grid along the last dimension in order to calculate the euclidean distance of every
    # point of the grid to the center
    dist_arr = np.linalg.norm(grid_arr, axis=0)
    # Create a boolean mask that is true for each voxel that is closer to the center than the radius.
    structuring_element = dist_arr <= radius
    return structuring_element

def create_local_intensity_change_function(radius=1.8, is_correct_near_miss=True):
    """
    Returns a function that calculates the local intensity change using a specified radius for each voxel in an input
    image. The function is designed this way to allow for custom choice of local intensity change function by the user
    (e.g. it is also possible to use a sobel filter instead of the output function defined here)
    :param radius: integer or float: Radius of the local neighborhood. Same units as given in voxel_spacing_arr
    :param is_correct_near_miss: boolean: if True, the radius is slightly increased over the given value so that
    floating-point uncertainties (e.g. in number sqrt(2)) do not cause points to be excluded unintentionally.
    :return: function: is used to calculate the local intensity change for a given image given in the form of an
    intensity array and a voxel spacing array
    """
    if is_correct_near_miss:
        radius *= 1.001
    def lic_func(intensity_arr, voxel_spacing_arr=None):
        footprint_mask_arr = calculate_footprint(radius, dim=intensity_arr.ndim, voxel_spacing_arr=voxel_spacing_arr)
        max_arr = maximum_filter(intensity_arr, footprint=footprint_mask_arr)
        min_arr = minimum_filter(intensity_arr, footprint=footprint_mask_arr)
        local_intensity_change_arr = max_arr - min_arr
        return local_intensity_change_arr
    return lic_func

def create_local_intensity_change_threshold_function(quantile=.25):
    """
    Returns a function that calculates a threshold value for a given set of local intensity change values. The threshold
    should divide homogeneous from non-homogeneous voxels. In this particular implementation, the threshold is simply
    chosen as a user-defined quantile among local intensity change values. The function is designed this way to allow
    for custom choice of local intensity change threshold function by the user (e.g. it is also possible to use an
    otsu threshold instead of the output function defined here)
    :param quantile: float: The quantile value which should be used to determine the threshold based on the local
    intensity change values.
    :return: function: is used to calculate the threshold for homogeneous voxels based on input local intensity change
    values.
    """
    def lic_threshold_func(local_intensity_change_arr, mask_arr):
        return np.nanquantile(local_intensity_change_arr[mask_arr], quantile)
    return lic_threshold_func

def create_piecewise_linear_transformation_function():
    """
    Returns a function that transforms the intensities in an image in a piecwise linear fashion (similar to Nyul et al)
    from a list of given intensity landmarks to a list of target landmarks. The function is designed this way to allow
    for custom choice of intensity transformation function by the user (e.g. it is also possible to use a polynomial
    transformation instead of the output function defined here)
    :return: function: is used to transform the image intensities to the normalized scale.
    """
    def pwl_transform_func(intensity_arr, peak_intensity_list, goal_intensity_list):
        f = interp1d(peak_intensity_list, goal_intensity_list, fill_value="extrapolate")
        norm_image_arr = f(intensity_arr)
        return norm_image_arr
    return pwl_transform_func

def calculate_KDE(array, grid=10_000, bw='ISJ', weights=None, is_normalized=False):
    """
    Returns the kernel density estimation of the data specified in the input array. The library KDEpy is used for the
    operation because it offers advanced bandwidth estimation techniques for KDE.
    :param array: numpy array: input data as integer or float
    :param grid: int or numpy array: If int, it is considered the number of grid points to be created. If array, it is
    considered to be the grid on which the KDE should be evaluated
    :param bw: string or float: If a float, it is taken as the bandwidth, if it's a string, it is taken as a bandwidth
    estimation strategy. Recommended: "ISJ" (Improved Sheather-Jones algorithm), which works better than alternatives in
     multimodal data.
    :param weights: (optional) numpy array (integer or float): needs to have same size as array. Specifies the weight of
    each data point of the array in the KDE.
    :param is_normalized: (optional) boolean: If True, simply return the values as given by the KDE, which integrate to
    1. If False, rescale the PDF so that it integrates to the number of voxels (if no weights given) or to the sum of
    weights (if given).
    :return: tuple: The first entry is an array describing the grid on which the KDE is defined and the second is an
    array of equal size that describes the probability density function at each point on the grid.
    """
    # Some very high intensity values may lead to numerical instability of this KDE method, therefore they are rescaled
    # as a workaround.
    threshold = 1e+4
    rescale_factor = 1.

    array_intensity_range = np.nanmax(array)-np.nanmin(array)
    is_rescale = array_intensity_range > threshold
    if is_rescale:
        rescale_factor = array_intensity_range/threshold
        array = array.copy() / rescale_factor
        # If grid or bandwidth are defined in terms of intensity units, have to be rescaled as well.
        if not isinstance(grid, int):
            grid = grid.copy()
            grid /= rescale_factor
        if not isinstance(bw, str):
            bw /= rescale_factor

    # Suggested for bw: automatic bw selection using Improved Sheather Jones (ISJ)
    # Always use gaussian kernel
    KDE = FFTKDE(kernel='gaussian', bw=bw) # Define KDE
    KDE.fit(array, weights=weights) # Fit data

    '''bw_ratio = KDE.bw / np.mean(np.diff(np.unique(array)))
    print(f"The ratio between bandwidth and average difference between unique intensities: {bw_ratio:.2f}")
    bw_ratio = KDE.bw / (np.percentile(array, 75) - np.percentile(array, 25)) #0.05 is good here.
    print(f"The ratio between bandwidth and intensity interquartile range: {bw_ratio:.3f}")'''

    # If grid is an integer, a new grid will be created by the evaluate function. Otherwise, the existing one is used
    if isinstance(grid, int):
        intensities, pdf_arr = KDE.evaluate(grid) # Evaluate KDE on grid
    else:
        pdf_arr = KDE.evaluate(grid) # Evaluate KDE on grid
        intensities = grid

    # The rescaling from the start has to be undone in case it was applied
    if is_rescale:
        intensities *= rescale_factor
        pdf_arr /= rescale_factor

    if not is_normalized:
        if not weights is None:
            pdf_arr *= np.nansum(weights)
        else:
            pdf_arr *= array.size

    return intensities, pdf_arr

def create_kde_calculation_function(grid=10_000, bw='ISJ', discretization_count=None, is_normalized=False):
    """
    Returns a function that calculates the kernel density estimation, a generalization of a histogram, for a set of
    input intensities. The function is designed this way to allow for custom choice of histogram calculation function
    by the user.
    :param grid: int or numpy array: If int, it is considered the number of grid points to be created. If array, it is
    considered to be the grid on which the KDE should be evaluated
    :param bw: string or float: If a float, it is taken as the bandwidth, if it's a string, it is taken as a bandwidth
    estimation strategy. Recommended: "ISJ", standing for Improved Sheather-Jones algorithm. It works better than
    alternatives in multimodal data.
    :param discretization_count: (optional) integer: if not None, the intensity values will be discretized into this
    number of bins before calculation of the kde. This may not be necessary on raw image data, but after bias field
    correction the number of unique values in the intensity array becomes very large, which can lead to problems with
    the bandwidth estimation for the KDE. It is recommended to use discretization of about 1024 in that case.
    :param is_normalized: (optional) boolean: If True, simply return the values as given by the KDE, which integrate to
    1. If False, rescale the PDF so that it integrates to the number of voxels (if no weights given) or to the sum of
    weights (if given).
    :return: function: calculates kernel density estimation based on the given settings. Takes as input an array with
    values for which to calculate the kde. Additionally optionally takes a weight array of the same shape as the
    intensity array, which defines the weight in the kde for each data point. Returns a tuple where the first element
    contains the intensity grid and the second the probability density function.
    """
    def kde_func(intensity_arr, weights_arr=None):
        # If chosen, discretize the values into the declared number of bins before calculating the KDE
        if not discretization_count is None:
            # Make sure not to change any values outside function
            intensity_arr = intensity_arr.copy()
            # Define grid of discrete values in intensity domain
            disc_grid = np.linspace(np.nanmin(intensity_arr), np.nanmax(intensity_arr), num=discretization_count,
                                    endpoint=True)
            # Digitize the intensity values to the discrete intensity values.
            intensity_arr = disc_grid[np.searchsorted(disc_grid, intensity_arr)]
        kde_grid, kde_pdf = calculate_KDE(intensity_arr, grid, bw, weights_arr, is_normalized)
        return kde_grid, kde_pdf
    return kde_func

def calculate_histogram_peak_indices(hist_arr, peak_prominence_fraction=0.1):
    """
    Finds the peaks from a given histogram array. If multiple peaks can be identified, only those that have a prominence
    of at least peak_prominence_fraction relative to the largest peak are considered. For each peak, the index of the
    peak as well as its height and the left and right bounds are returned.
    :param hist_arr: array (1 dimension): integer or float, contains the counts or frequency of the histogram in each
    bin.
    :param peak_prominence_fraction: float: value between 0 and 1, describes the minimum prominence a peak needs to
    have, as a fraction of that of the largest peak, in order to be considered a separate peak for the output.
    :return: tuple with 3 elements: First is an array that gives the indices of the identified substantial peaks. Second
    is an array of equal shape that gives the height of each of the substantial peaks. Third is a tuple with two
    elements that describe the two indices on the grid where the peak is the most relevant (i.e. the intensity scale
    is divided in such a way that each intensity belongs to exactly one peak. The boundaries of those regions for each
    peak are given with these two values.)
    """
    # Determine all peak locations
    peaks = find_peaks(hist_arr)[0]

    # Determine prominences and left and right bases for all peaks
    peak_proms, peak_l_bases, peak_r_bases = peak_prominences(hist_arr, peaks)

    # Determine the indices of the peaks that are above the prominence threshold compared to the largest peak
    prominent_indices = peak_proms >= peak_prominence_fraction * np.amax(peak_proms)

    # Select only the peaks above the prominece thresholds
    peaks = peaks[prominent_indices]
    peak_proms = peak_proms[prominent_indices]
    peak_l_bases = peak_l_bases[prominent_indices]
    peak_r_bases = peak_r_bases[prominent_indices]
    peak_heights = hist_arr[peaks]

    # Determine the split of the interval among the peaks. Start with the bases from the prominence algorithm
    peak_l_bound_arr = peak_l_bases.copy()
    peak_r_bound_arr = peak_r_bases.copy()

    for i, peak_index in enumerate(peaks):
        # Take as left bound the maximum of all lower index right bounds or the left bound itself.
        peak_l_bound_arr[i] = np.amax(np.append(peak_r_bases[peak_r_bases < peak_index], peak_l_bound_arr[i]))
        # Analogous for right bound
        peak_r_bound_arr[i] = np.amin(np.append(peak_l_bound_arr[peak_l_bound_arr > peak_index], peak_r_bound_arr[i]))

    #return grid_arr[peaks], peak_heights, grid_arr[peak_l_bound_arr], grid_arr[peak_r_bound_arr]

    return peaks, peak_heights, (peak_l_bound_arr, peak_r_bound_arr)

def choose_peak_from_list(grid_arr, hist_arr, peak_index_arr, peak_height_arr, peak_l_bound_arr, peak_r_bound_arr,
                          strategy="most"):
    """
    Requires a given histogram consisting of a grid array and a frequency / counts array as well as arrays defining the
    indices, heights and left and right index bounds for a selection of peaks. Furthermore, a strategy needs to be
    given which is then used to select a single peak from the given set of peaks.
    :param grid_arr: array (1 dimension): integer or float, contains the intensity value for each bin / point on the
    grid.
    :param hist_arr: array (1 dimension): integer or float, contains the counts or frequency of the histogram in each
    bin.
    :param peak_index_arr: array (1 dimension): integer, contains the indices on the grid of the peaks identified in
    a previous step. The length corresponds to the number of peaks.
    :param peak_height_arr: array (1 dimension): integer or float, contains the frequency / count height of the
    histogram peaks identified in a previous step. The length corresponds to the number of peaks.
    :param peak_l_bound_arr: array (1 dimension): integer, contains the index on the grid of the left boundary of each
    identified peak. The length corresponds to the number of peaks.
    :param peak_r_bound_arr: array (1 dimension): integer, contains the index on the grid of the right boundary of each
    identified peak. The length corresponds to the number of peaks.
    :param strategy: string: a keyword corresponding to a strategy for selecting the desired peak among the given list
    of substantial peaks. The options include.
    "most": the peak encompassing the largest number of voxels,
    "highest": the highest peak,
    "left": the peak with the lowest intensity,
    "right": the peak with the highest intensity.
    :return: a tuple with two elements. The first is the intensity of the chosen peak, the second is a tuple containing
    the left boundary of the peak and the right boundary of the peak. The boundary represents a symmetric interval
    around the peak where the histogram values are at least 75% of the peak.
    """
    # Have to choose one of the peaks among the given based on the selected strategy
    if strategy == "most":
        # Calculate probability density function area belonging to each peak. Select the peak with the largest area,
        # signifying the most contained voxels
        pdf_integral_list = [np.sum(hist_arr[peak_l_bound:peak_r_bound])
                             for peak_l_bound, peak_r_bound in zip(peak_l_bound_arr, peak_r_bound_arr)]
        chosen_index = np.argmax(pdf_integral_list)
    elif strategy == "highest":
        # Choose peak that is highest in terms of probability density function
        chosen_index = np.argmax(peak_height_arr)
    elif strategy == "left":
        # Choose peak that is most left on the histogram, i.e. the one with the lowest peak intensity
        chosen_index = np.argmin(peak_index_arr)
    elif strategy == "right":
        # Choose peak that is most right on the histogram, i.e. the one with the highest peak intensity
        chosen_index = np.argmax(peak_index_arr)
    else:
        print(f"Warning: Selected peak choice strategy \'{str(strategy)}\' is not implemented. Using default instead")
        return choose_peak_from_list(grid_arr, hist_arr, peak_index_arr, peak_height_arr, peak_l_bound_arr,
                                     peak_r_bound_arr)

    peak_index = peak_index_arr[chosen_index]
    peak_intensity = grid_arr[peak_index]

    # Calculate new left and right intensity bounds that corresponds to the voxels very closely around the peak.
    # Look at width of the peak at a given height fraction (counted from top) and then take the smaller of the two
    # widths as the width for the region.
    relative_peak_height_threshold = 0.25
    peak_width, _, peak_left_bound, peak_right_bound = peak_widths(
        hist_arr, [peak_index_arr[chosen_index]], relative_peak_height_threshold
    )

    # Convert the left and right bounds to intensity units based on the grid.
    # Formulation with interpolation is chosen to account for potential non-equidistant grid.
    index_intensity_conversion_func = interp1d(np.arange(grid_arr.size), grid_arr)
    peak_left_bound_intensity, peak_right_bound_intensity = index_intensity_conversion_func(
        np.array([peak_left_bound, peak_right_bound])
    )

    # Of the width to the left and to the right of the peak (returned by peak_widths), choose the smaller one
    lower_width = np.amin([np.absolute(peak_intensity-peak_left_bound_intensity),
                          np.absolute(peak_right_bound_intensity-peak_intensity)])

    # Return the peak intensity and the area of +/- lower width around it
    return peak_intensity, (peak_intensity - lower_width, peak_intensity + lower_width)

def create_peak_detection_function(peak_prominence_fraction=0.1, peak_selection_strategy="most"):
    """
    Defines a function that will determine a single, relevant peak on a histogram based on given settings, namely
    a peak prominence threshold for identifying relevant peaks and a strategy for selecting a single peak among the
    set of relevant ones. The function then returns the intensity of the selected peak as well as a left and right
    edge of the center of the peak. This function is designed this way to allow for custom choice of histogram
    peak detection by the user.
    :param peak_prominence_fraction: float: value between 0 and 1, describes the minimum prominence a peak needs to
    have, as a fraction of that of the largest peak, in order to be considered a substantial peak for the output.
    :param peak_selection_strategy: string: a keyword corresponding to a strategy for selecting the desired peak among
    the given list of substantial peaks. The options include.
    "most": the peak encompassing the largest number of voxels,
    "highest": the highest peak,
    "left": the peak with the lowest intensity,
    "right": the peak with the highest intensity.
    :return: function: Takes a histogram consisting of a grid array and a frequency / count array. Finds the primary
    peak inside the histogram based on the strategy specified and returns the peak intensity as well as an intensity
    interval around the center of the peak.
    """
    def peak_detection_function(grid_arr, hist_arr):
        peak_index_arr, peak_height_arr, (peak_l_bound_arr, peak_r_bound_arr) = calculate_histogram_peak_indices(
            hist_arr, peak_prominence_fraction
        )
        peak_intensity, peak_bounds = choose_peak_from_list(grid_arr, hist_arr, peak_index_arr, peak_height_arr,
                                                            peak_l_bound_arr, peak_r_bound_arr,
                                                            peak_selection_strategy)
        return peak_intensity, peak_bounds
    return peak_detection_function

def erode_mask(mask_arr, radius=1.8, iterations=1, voxel_spacing_arr=None):
    """
    Performs binary erosion of the given binary mask with a given radius and a given number of iterations. If a voxel
    spacing arr is given, it is assumed that it has the same units as the radius. If None is given, it is assumed that
    the voxel spacing in each direction is 1. Very large radii will cause memory issues, therefore it is recommended to
    keep the radius in the order of 2x-3x the smallest voxel size and use iterations to achieve the desired depth of
    erosion.
    :param mask_arr: numpy array (d dimensions): Boolean values where True means a given voxel is inside the mask and
    False means it is outside.
    :param radius: integer or float: Radius of the local neighborhood. Same units as given in voxel_spacing_arr.
    :param iterations: integer: The number of times the erosion should be performed.
    :param voxel_spacing_arr: (optional) numpy array (1 dimension): Integer or float array with length n, where n is the
    number of dimensions of the image. For each dimension, this array contains the distance between voxels along that
    axis (assumed to be in mm). Default: assume all are 1
    :return: numpy array (d dimensions): Boolean values, shape identical to that of the input mask array, but the voxels
    near the boundary have been excluded due to the erosion.
    """
    footprint_arr = calculate_footprint(radius, dim=mask_arr.ndim, voxel_spacing_arr=voxel_spacing_arr)
    return binary_erosion(mask_arr, footprint_arr, iterations=iterations)

def erode_brain(brain_mask, subarachnoid_removal_distance=20., voxel_spacing_arr=None):
    """
    Perform erosion on brain mask, automatically choosing the radius and iterations for the erosion based on the voxel
    spacing array. When it is not possible to erode exactly the given amount in all directions, the eroded distance is
    rounded up.
    :param brain_mask: numpy array (d dimensions): Boolean values where True means a given voxel is inside the brain and
    False means it is outside.
    :param subarachnoid_removal_distance: integer or float: Distance of tissue that should be eroded from the outside
    of the brain. Should have same unit as voxel_spacing_arr. Will be converted into an appropriate radius and number
    of iterations automatically before the erosion function is called.
    :param voxel_spacing_arr: (optional) numpy array (1 dimension): Integer or float array with length n, where n is the
    number of dimensions of the image. For each dimension, this array contains the distance between voxels along that
    axis (assumed to be in mm). Default: assume all are 1
    :return: numpy array (d dimensions): Boolean values, shape identical to that of the input mask array, but the voxels
    within the given distance of the boundary have been excluded due to the erosion.
    """
    eroded_brain_mask = brain_mask.copy()

    # Cannot erode None or negative distances.
    if not (subarachnoid_removal_distance is None or subarachnoid_removal_distance <= 0.):
        # Calculate maximum voxel spacing. Use this as radius so that we certainly erode all three dimensions
        max_voxel_spacing = np.amax(voxel_spacing_arr) if not voxel_spacing_arr is None else 1.
        # Calculate how many times we have to erode to get at least the given distance
        n_iterations = int(np.ceil(subarachnoid_removal_distance / max_voxel_spacing))
        # Erode the brain mask
        eroded_brain_mask = erode_mask(eroded_brain_mask, max_voxel_spacing,
                                       iterations=n_iterations, voxel_spacing_arr=voxel_spacing_arr)
    return eroded_brain_mask

def determine_brain_histogram_split_intensity(grid_arr, pdf_arr, lic_weighted_grid_arr, lic_weighted_pdf_arr,
                                              mr_contrast=None, outlier_cutoff_fraction=.01):
    """
    Determines the intensity at which the brain should be split for automatic delineation of the ventricles. Uses the
    conventional intensity histogram as well as an intensity histogram where the voxels are weighted by their local
    intensity change in order to determine a measure of the average local intensity change occuring at each intensity
    level of the grid. From the entire intensity interval, the regions corresponding to the top and bottom outlier
    voxels are cut off from both sides as specified. Furthermore, if an mr_contrast is specified, the location of the
    split between CSF and WM relative to the median intensity is known and thus the intensities corresponding to one
    half of the histogram are also cut off (which one depends on the sequence specified). For the remaining intensities,
    the intensity where the highest average local intensity change is observed is selected as the split intensity
    between CSF and GM/WM. The motivation for this is that in that intensity, the vast majority of voxels would not be
    homogeneous, but partial volume effect voxels and thus lie somewhere between tissues.
    :param grid_arr: array (1 dimension): integer or float, contains the intensity value for each bin / point on the
    grid.
    :param pdf_arr: array (1 dimension): integer or float, contains the counts or frequency of the histogram in each
    bin. Needs to have same shape as grid_arr.
    :param lic_weighted_grid_arr: array (1 dimension): integer or float, contains the intensity value for each bin /
    point on the grid for the local intensity change weighted histogram.
    :param lic_weighted_pdf_arr: array (1 dimension): integer or float, contains the counts or frequency of the
    local intensity change weighted histogram in each bin. Needs to have same shape as lic_weighted_grid_arr.
    :param mr_contrast: (optional) string: a description of the contrast of the sequence used, e.g. t1 or t2. For a list
    of implemented cases, have a look at CONTRASTS_PEAK_BELOW_MED and CONTRASTS_PEAK_ABOVE_MED in this file. If None is
    given, then no exclusion based on the median is done.
    :param outlier_cutoff_fraction: (optional) float: A fraction value which represent what fraction of all voxels is
    being cut off from both sides of the histogram for the purpose of determining the intensity interval on which the
    split is found. It is recommended to take a value between .01-0.02 at least, especially if some outlier voxels exist
    like artifacts or contrast-enhanced voxels, which may have very high local intensity change despite not being
    related to the split between CSF and GM/WM.
    :return: tuple with three entries: The first entry is the float or integer corresponding to the split intensity
    between the CSF intensities and GM/WM intensities of the histogram. The second entry contains the calculated
    comparison array (float, 1 dimensional, same shape as grid_arr) that describes the average local intensity change
    for each intensity on the grid_arr. The third entry contains that same array, but with values at excluded
    intensities being set to np.nan. The last two entries are primarily useful for plotting.
    """
    # Determine cutoff indices by considering the cumulative sum of the histogram.
    # Note that the current implementation does not give accurate results for non-uniform grid spacing.
    norm_cumsum_pdf_arr = np.cumsum(pdf_arr)/np.sum(pdf_arr)
    lower_bound_index, median_index, upper_bound_index = np.searchsorted(
        norm_cumsum_pdf_arr, [outlier_cutoff_fraction, .5, 1-outlier_cutoff_fraction]
    )

    # Use the modality information to restrict the investigated intensity interval to below or above the median.
    # This helps mitigate the determination of the wrong peak.
    if not mr_contrast is None:
        if mr_contrast.lower() in CONTRASTS_PEAK_BELOW_MED:
            upper_bound_index = median_index
        elif mr_contrast.lower() in CONTRASTS_PEAK_ABOVE_MED:
            lower_bound_index = median_index

    # If two grids are different, interpolate the local intensity change pdf to the regular grid here.
    f_grid_interpolate = interp1d(lic_weighted_grid_arr, lic_weighted_pdf_arr)
    lic_weighted_pdf_arr = f_grid_interpolate(grid_arr)

    # Create a comparison array that consists of ratio between lic_weighted_pdf_arr and pdf_arr.
    # The values in pdf_arr represent the density of points at each intensity. The values in lic_weighted_pdf_arr
    # represent the density of points at each intensity scaled by the local intensity change. The ratio of the two
    # cancels out the density and results in a measure for the average local intensity change at each intensity in the
    # grid.
    raw_comparison_arr = lic_weighted_pdf_arr / pdf_arr

    # Redefine an array that excludes all intensities we do not consider as candidates.
    comparison_arr = np.zeros_like(raw_comparison_arr)*np.nan
    comparison_arr[lower_bound_index:upper_bound_index] = raw_comparison_arr[lower_bound_index:upper_bound_index]

    # Determine the split intensity as the intensity where the comparison array (after exclusion of certain intensities)
    # takes the maximum value.
    split_index = np.nanargmax(comparison_arr)
    split_intensity = grid_arr[split_index]

    return split_intensity, raw_comparison_arr, comparison_arr

def create_ventricle_and_gmwm_masks(intensity_arr, brain_mask_arr, eroded_brain_mask_arr, split_intensity,
                                    mr_contrast=None):
    """
    Detemine binary mask for ventricles and gray/white matter regions of the brain for a given intensity array, brain
    mask, eroded brain mask and split intensity. The contrast needs to specified in order to specify if CSF or GM/WM
    should have higher intensity.
    :param intensity_arr: numpy array (n dimensions): Contains intensity values of an image as float or integer
    :param determined_peak_intensity_list: list of floats or integers: Contains peak intensities determined in masks
    of the image
    :param brain_mask_arr: numpy array (n dimensions): Binary mask in boolean with the same shape as intensity_arr,
    delineates the brain in the intensity array.
    :param eroded_brain_mask_arr: numpy array (n dimensions): Binary mask in boolean with the same shape as
    intensity_arr, delineates the brain in the intensity array, but with the outer regions eroded so that the main
    remaining CSF region consists of the ventricles.
    :param split_intensity: integer or float: The intensity value that splits the CSF voxels from the gray matter /
    white matter voxels.
    :param mr_contrast: (optional) string: a description of the contrast of the sequence used, e.g. t1 or t2. For a list
    of implemented cases, have a look at CONTRASTS_PEAK_BELOW_MED and CONTRASTS_PEAK_ABOVE_MED in this file. Is needed
    to determine if CSF should have lower or higher intensity than GM/WM voxels. If none is given, assume t1.
    :return: tuple with two entries: the first is a binary mask array of the same shape as the input brain mask array
    delineating the ventricles. The second is a binary mask array of the same shaep as the input brain mask array
    delineating the gray matter & white matter regions of the brain.
    """
    if not mr_contrast is None:
        # For ventricles, look at voxels inside the eroded brain mask that are in the CSF intensity range.
        # For gray/white matter voxels, look at voxels in the whole brain mask that are outside the CSF intensity
        # range.
        if mr_contrast.lower() in CONTRASTS_PEAK_BELOW_MED:
            csf_mask_arr = eroded_brain_mask_arr & (intensity_arr <= split_intensity)
            gmwm_mask_arr = brain_mask_arr & (intensity_arr > split_intensity)
        elif mr_contrast.lower() in CONTRASTS_PEAK_ABOVE_MED:
            csf_mask_arr = eroded_brain_mask_arr & (intensity_arr >= split_intensity)
            gmwm_mask_arr = brain_mask_arr & (intensity_arr < split_intensity)
    else:
        print(f"Warning, no mr_contrast specified for ventricle mask creation. Assuming t1.")
        return create_ventricle_and_gmwm_masks(intensity_arr, brain_mask_arr, eroded_brain_mask_arr, split_intensity,
                                               mr_contrast="t1")

    return csf_mask_arr, gmwm_mask_arr