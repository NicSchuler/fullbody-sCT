# From the official repository 
from npeaks_util import create_local_intensity_change_function, create_local_intensity_change_threshold_function, \
    create_kde_calculation_function, create_peak_detection_function, create_piecewise_linear_transformation_function, \
    erode_brain, determine_brain_histogram_split_intensity, create_ventricle_and_gmwm_masks

DEFAULT_LOCAL_INTENSITY_CHANGE_RADIUS = 1.8
DEFAULT_LOCAL_INTENSITY_CHANGE_THRESHOLD_QUANTILE = .25
DEFAULT_KDE_GRID_NUMBER = 10_000
DEFAULT_KDE_BANDWIDTH = "ISJ"
DEFAULT_KDE_DISCRETIZATION_COUNT = 1024
DEFAULT_PEAK_PROMINENCE_FRACTION = 0.1
DEFAULT_PEAK_SELECTION_STRATEGY = "most"

DEFAULT_BRAIN_MR_CONTRAST = "t1"
DEFAULT_BRAIN_EROSION_DISTANCE = 20.
DEFAULT_BRAIN_OUTLIER_CUTOFF_FRACTION = 0.01

class NPeaksNormalizer:

    def __init__(self, visualizer=None, brainsplit_visualizer=None):
        """Initialise the N-Peaks normalizer with optional visualizers.

        Args:
            visualizer: Optional visualizer object for the peak-finding step.
                When set, ``find_peaks_custom`` will call
                ``visualizer.plot(...)`` after processing each mask.
                See ``visualize.py`` for available implementations.
            brainsplit_visualizer: Optional visualizer for the brain-mask
                splitting step.  When set, ``split_brain_mask_custom`` will
                call ``brainsplit_visualizer.visualize_histogram_split(...)``
                after each split.
        """
        self.visualizer = visualizer
        self.brain_visualizer = brainsplit_visualizer

    def set_norm_visualizer(self, visualizer):
        """
        Allows the user to set a visualizer object which will plot the output of the normalization. For possible
        options, look at file "visualize.py"
        :param visualizer: object: visualizer which can plot the results of normalization.
        """
        self.visualizer = visualizer

    def set_brainsplit_visualizer(self, brainsplit_visualizer):
        """
        Allows the user to set a visualizer object which will plot the output of the brain split. For possible
        options, look at file "visualize.py"
        :param brainsplit_visualizer: object: visualizer which can plot the results of the automatic brain split.
        """
        self.brain_visualizer = brainsplit_visualizer

    def find_peaks_custom(self, intensity_arr, local_intensity_change_function,
                          mask_arr_list, local_intensity_change_threshold_function_list,
                          histogram_calculation_function,
                          peak_detection_function_list,
                          voxel_spacing_arr=None,
                          image_name=None):
        """
        Does all calculation steps of N-Peaks in order to identify the intensity peak of the homogeneous region in each
        mask of a given image. The functions to calculate local intensity change, local intensity change threshold,
        histogram and peak detection have to be specified by the user. If no specification is desired, use the function
        "find_peaks" instead, which uses default implementations of all these functions.
        If a plot path has been specified for the normalizer beforehand, a summary plot of the peak finding procedure
        will be saved there.
        :param intensity_arr: numpy array (n dimensions): Contains intensity values of an image as float or integer
        :param local_intensity_change_function: function: takes as inpuzt arguments the intensity array and the
        voxel spacing array. Should return an array of the same shape as the intensity array, but with each value
        representing the local intensity change.
        :param mask_arr_list: list of numpy arrays (n dimensions): Binary masks should be given in boolean and each have
        the same shape as intensity_arr. "True" means the specified voxel is inside the mask.
        :param local_intensity_change_threshold_function_list: list of functions (same length as mask_arr_list): takes
        as input the local intensity change array and the binary mask input by the user to define a threshold value for
        the local intensity change values. All voxels with values below that threshold are considered to be homogeneous.
        :param histogram_calculation_function: function: takes as input the intensity array and a binary mask. It should
        return a histogram of the intensities inside the mask in the form of a tuple containing an intensity grid
        as the first element and the corresponding frequencies as the second element.
        :param peak_detection_function_list: list of functions (same length as mask_arr_list): takes as input a
        histogram consisting of an intensity grid and the corresponding frequencies. Determines the desired tissue peak
        intensity and returns the intensity value as well as a binary mask that corresponds to the voxels at the center
        of the peak (for the plot).
        :param voxel_spacing_arr: (optional) numpy array (1 dimension): Integer or float array with length n, where n is
        the number of dimensions of intensity_arr. For each dimension, this array contains the distance between voxels
        along that axis (assumed to be in mm). Default: assume all are 1
        :param image_name: (optional) string: Is used for the purpose of plotting only. (For determining the file name
        and some text on the plot)
        :return: list: the homogeneous tissue peak intensity as a float for each mask defined in mask_arr_list
        """

        # Calculate local intensity change for whole image
        local_intensity_change_arr = local_intensity_change_function(intensity_arr, voxel_spacing_arr)

        # For each mask, find the threshold for local intensity change, which separates homogeneous from inhomogeneous
        # voxels.
        local_intensity_change_threshold_list = [
            local_intensity_change_threshold_function(local_intensity_change_arr, mask_arr)
            for mask_arr, local_intensity_change_threshold_function in
            zip(mask_arr_list, local_intensity_change_threshold_function_list)
        ]
        # For each mask, find the homogeneous region inside by applying a local intensity change threshold
        homogeneous_mask_arr_list = [
            mask_arr & (local_intensity_change_arr <= local_intensity_change_threshold)
            for mask_arr, local_intensity_change_threshold in zip(mask_arr_list, local_intensity_change_threshold_list)
        ]

        #mask_histogram_result_list = [histogram_calculation_function(intensity_arr[mask_arr])
        #                              for mask_arr in mask_arr_list]
        homogeneous_mask_histogram_result_list = [histogram_calculation_function(intensity_arr[homogeneous_mask_arr])
                                                  for homogeneous_mask_arr in homogeneous_mask_arr_list]

        # For each homogeneous mask, find histogram peak intensity and a boundary for the peak (for plotting)
        peak_intensity_list, peak_bounds_list = list(zip(*[
            peak_detection_function(histogram_grid_arr, histogram_freq_arr)
            for peak_detection_function, (histogram_grid_arr, histogram_freq_arr) in
            zip(peak_detection_function_list, homogeneous_mask_histogram_result_list)
        ]))

        # If the visualizer is defined, plot the result
        if not self.visualizer is None:
            for i, (mask_arr, local_intensity_change_threshold, peak_intensity, peak_bounds) in enumerate(zip(
                    mask_arr_list, local_intensity_change_threshold_list, peak_intensity_list, peak_bounds_list
            )):
                self.visualizer.plot(
                    intensity_arr, local_intensity_change_arr, mask_arr, local_intensity_change_threshold,
                    histogram_calculation_function, peak_intensity, peak_bounds,
                    lic_hist_string=None, hist_peak_string=None,
                    voxel_spacing_arr=voxel_spacing_arr, image_name=image_name, mask_name=str(i)
                )

        return peak_intensity_list

    def find_peaks(self, intensity_arr, mask_arr_list, voxel_spacing_arr=None,
                   local_intensity_change_radius=DEFAULT_LOCAL_INTENSITY_CHANGE_RADIUS,
                   local_intensity_change_threshold_quantile_list=None,
                   kde_grid_number=DEFAULT_KDE_GRID_NUMBER, kde_bandwidth=DEFAULT_KDE_BANDWIDTH,
                   kde_discretization_count=DEFAULT_KDE_DISCRETIZATION_COUNT,
                   peak_detection_prominence_fraction_list=None, peak_detection_peak_selection_strategy_list=None,
                   image_name=None
                   ):
        """
        Determines the intensity peak within each mask of a given image (in form of an intensity array and voxel spacing
        array). For each step of the procedure, the standard function implementation is used. The parameters need to be
        passed to this function or are assumed to be the default values declared in this file.
        If a plot folder has been defined, the results of the peak detection will automatically be plotted in that
        folder.
        The peak intensities returned by this function can be used to normalize the image to a standard scale.
        :param intensity_arr: numpy array (n dimensions): Contains intensity values of an image as float or integer
        :param mask_arr_list: list of numpy arrays (n dimensions): Binary masks should be given in boolean and each have
        the same shape as intensity_arr. "True" means the specified voxel is inside the mask.
        :param voxel_spacing_arr: (optional) numpy array (1 dimension): Integer or float array with length n, where n is
        the number of dimensions of intensity_arr. For each dimension, this array contains the distance between voxels
        along that axis (assumed to be in mm). Default: assume all are 1
        :param local_intensity_change_radius: integer or float: Radius of the local neighborhood for calculating local
        intensity change. Same units as given in voxel_spacing_arr
        :param local_intensity_change_threshold_quantile_list: list of floats: For each mask, give the
        quantile value which should be used to determine the threshold local intensity change value that separates the
        homogeneous from non-homogeneous regions of the mask.
        :param kde_grid_number: integer: The number of grid points to be created for the kernel density estimation of
        the intensity histogram.
        :param kde_bandwidth: string or float: If a float, it is taken as the bandwidth for the kernel density
        estimation, if it's a string, it is taken as a bandwidth estimation strategy as available in the library KDEpy.
        Recommended: "ISJ" (Improved Sheather-Jones algorithm), which works better than alternatives in multimodal data.
        :param kde_discretization_count: (optional) integer: if not None, the intensity values will be discretized into
        this number of bins before calculation of the kernel density estimation. This may not be necessary on raw image
        data, but after bias field correction the number of unique values in the intensity array becomes very large,
        which can lead to problems with the bandwidth estimation for the KDE. It is recommended to use discretization of
         about 1024 in that case.
        :param peak_detection_prominence_fraction_list: list of float: For each mask, determine parameter for histogram
        peak detection. Value between 0 and 1, describes the minimum prominence a peak needs to have, as a fraction of
        that of the largest peak, in order to be considered a separate peak.
        :param peak_detection_peak_selection_strategy_list: string: a keyword corresponding to a strategy for selecting
        the desired peak among the determined separate histogram peaks. The options include.
        "most": the peak encompassing the largest number of voxels,
        "highest": the highest peak,
        "left": the peak with the lowest intensity,
        "right": the peak with the highest intensity.
        :param image_name: (optional) string: Is used for the purpose of plotting only. (For determining the file name
        and some text on the plot)
        :return: list: the homogeneous tissue peak intensity as a float for each mask defined in mask_arr_list
        """
        # Define the default functions to be used for N-Peaks based on the input parameters
        if local_intensity_change_threshold_quantile_list is None:
            local_intensity_change_threshold_quantile_list = [
                DEFAULT_LOCAL_INTENSITY_CHANGE_THRESHOLD_QUANTILE for mask_arr in mask_arr_list
            ]
        if peak_detection_prominence_fraction_list is None:
            peak_detection_prominence_fraction_list = [
                DEFAULT_PEAK_PROMINENCE_FRACTION for mask_arr in mask_arr_list
            ]
        if peak_detection_peak_selection_strategy_list is None:
            peak_detection_peak_selection_strategy_list = [
                DEFAULT_PEAK_SELECTION_STRATEGY for mask_arr in mask_arr_list
            ]

        lic_function = create_local_intensity_change_function(local_intensity_change_radius)
        lic_threshold_function_list = [
            create_local_intensity_change_threshold_function(quantile)
            for quantile in local_intensity_change_threshold_quantile_list
        ]

        histogram_calculation_function = create_kde_calculation_function(kde_grid_number, kde_bandwidth,
                                                                         kde_discretization_count)

        peak_detection_function_list = [
            create_peak_detection_function(peak_detection_prominence_fraction, peak_detection_peak_selection_strategy)
            for peak_detection_prominence_fraction, peak_detection_peak_selection_strategy in
            zip(peak_detection_prominence_fraction_list, peak_detection_peak_selection_strategy_list)
        ]

        # Run the code using the default functions
        intensity_peak_list = self.find_peaks_custom(
            intensity_arr, mask_arr_list=mask_arr_list, voxel_spacing_arr=voxel_spacing_arr,
            local_intensity_change_function=lic_function,
            local_intensity_change_threshold_function_list=lic_threshold_function_list,
            histogram_calculation_function=histogram_calculation_function,
            peak_detection_function_list=peak_detection_function_list,
            image_name=image_name
        )

        return intensity_peak_list


    def transform_intensity_scale_custom(self, intensity_arr, determined_peak_intensity_list,
                                         target_peak_intensity_list, intensity_transformation_function):
        """
        Transform the intensities in the given array such that the intensities given in determined_peak_intensity_list
        will end up at intensities specified in target_peak_intensity_list
        :param intensity_arr: numpy array (n dimensions): Contains intensity values of an image as float or integer
        :param determined_peak_intensity_list: list of floats or integers: Contains peak intensities determined in masks
        of the image
        :param target_peak_intensity_list: list of floats or integers: Contains peak intensities in the output intensity
        scale. Must have same length as determined_peak_intensity_list
        :param intensity_transformation_function: function: takes as input the intensity array, determined peaks and
        target peaks and returns the transformed image. Can be custom-defined by the user.
        :return: numpy array (n dimensions): Contains intensity values of image after transformation as float or integer
        """
        normalized_intensity_arr = intensity_transformation_function(intensity_arr, determined_peak_intensity_list,
                                                                     target_peak_intensity_list)
        return normalized_intensity_arr

    def transform_intensity_scale(self, intensity_arr, determined_peak_intensity_list,
                                  target_peak_intensity_list):
        """
        Transform the intensities in the given array with a piecewise linear approach such that the intensities given in
        determined_peak_intensity_list will end up at intensities specified in target_peak_intensity_list
        :param intensity_arr: numpy array (n dimensions): Contains intensity values of an image as float or integer
        :param determined_peak_intensity_list: list of floats or integers: Contains peak intensities determined in masks
        of the image
        :param target_peak_intensity_list: list of floats or integers: Contains peak intensities in the output intensity
        scale. Must have same length as determined_peak_intensity_list
        :return: numpy array (n dimensions): Contains intensity values of image after transformation as float or integer
        """
        intensity_transformation_function = create_piecewise_linear_transformation_function()
        return self.transform_intensity_scale_custom(intensity_arr, determined_peak_intensity_list,
                                                     target_peak_intensity_list, intensity_transformation_function)

        # Piecewise linear transformation of intensities to common scale based on peak intensities

    def normalize(self, intensity_arr, mask_arr_list, target_peak_intensity_list, voxel_spacing_arr=None,
                  local_intensity_change_radius=DEFAULT_LOCAL_INTENSITY_CHANGE_RADIUS,
                  local_intensity_change_threshold_quantile_list=None,
                  kde_grid_number=DEFAULT_KDE_GRID_NUMBER, kde_bandwidth=DEFAULT_KDE_BANDWIDTH,
                  kde_discretization_count=DEFAULT_KDE_DISCRETIZATION_COUNT,
                  peak_detection_prominence_fraction_list=None, peak_detection_peak_selection_strategy_list=None,
                  image_name=None
                  ):
        """
        Normalizes the given image (in form of an intensity array and voxel spacing array) by determning the intensity
        peak of the homogeneous region in each mask and transforming the scale to the given intensity values. For each
        step of the procedure, the standard function implementation is used. The parameters need to be passed to this
        function or are assumed to be the default values declared in this file.
        If a plot folder has been defined, the results of the peak detection will automatically be plotted in that
        folder.
        :param intensity_arr: numpy array (n dimensions): Contains intensity values of an image as float or integer
        :param mask_arr_list: list of numpy arrays (n dimensions): Binary masks should be given in boolean and each have
        the same shape as intensity_arr. "True" means the specified voxel is inside the mask.
        :param target_peak_intensity_list: list of integers or floats: For each mask in mask_arr_list, give the target
        intensity that the homogeneous peak intensity in that mask should be transformed to on the normalized scale.
        :param voxel_spacing_arr: (optional) numpy array (1 dimension): Integer or float array with length n, where n is
        the number of dimensions of intensity_arr. For each dimension, this array contains the distance between voxels
        along that axis (assumed to be in mm). Default: assume all are 1
        :param local_intensity_change_radius: integer or float: Radius of the local neighborhood for calculating local
        intensity change. Same units as given in voxel_spacing_arr
        :param local_intensity_change_threshold_quantile_list: list of integers or floats: For each mask, give the
        quantile value which should be used to determine the threshold local intensity change value that separates the
        homogeneous from non-homogeneous regions of the mask.
        :param kde_grid_number: integer: The number of grid points to be created for the kernel density estimation of
        the intensity histogram.
        :param kde_bandwidth: string or float: If a float, it is taken as the bandwidth for the kernel density
        estimation, if it's a string, it is taken as a bandwidth estimation strategy as available in the library KDEpy.
        Recommended: "ISJ" (Improved Sheather-Jones algorithm), which works better than alternatives in multimodal data.
        :param kde_discretization_count: (optional) integer: if not None, the intensity values will be discretized into
        this number of bins before calculation of the kernel density estimation. This may not be necessary on raw image
        data, but after bias field correction the number of unique values in the intensity array becomes very large,
        which can lead to problems with the bandwidth estimation for the KDE. It is recommended to use discretization of
         about 1024 in that case.
        :param peak_detection_prominence_fraction_list: list of float: For each mask, determine parameter for histogram
        peak detection. Value between 0 and 1, describes the minimum prominence a peak needs to have, as a fraction of
        that of the largest peak, in order to be considered a separate peak.
        :param peak_detection_peak_selection_strategy_list: string: a keyword corresponding to a strategy for selecting
        the desired peak among the determined separate histogram peaks. The options include.
        "most": the peak encompassing the largest number of voxels,
        "highest": the highest peak,
        "left": the peak with the lowest intensity,
        "right": the peak with the highest intensity.
        :param image_name: (optional) string: Is used for the purpose of plotting only. (For determining the file name
        and some text on the plot)
        :return: list: the homogeneous tissue peak intensity as a float for each mask defined in mask_arr_list
        """
        # Find peaks
        peak_intensity_list = self.find_peaks(intensity_arr, mask_arr_list, voxel_spacing_arr,
                                              local_intensity_change_radius,
                                              local_intensity_change_threshold_quantile_list,
                                              kde_grid_number, kde_bandwidth,
                                              kde_discretization_count,
                                              peak_detection_prominence_fraction_list,
                                              peak_detection_peak_selection_strategy_list,
                                              image_name)

        # Transform scale of image
        norm_image = self.transform_intensity_scale(intensity_arr, peak_intensity_list, target_peak_intensity_list)

        return norm_image

    def split_brain_mask_custom(self, intensity_arr, brain_mask_arr,
                                local_intensity_change_function,
                                histogram_calculation_function,
                                brain_mr_contrast=DEFAULT_BRAIN_MR_CONTRAST,
                                brain_erosion_distance=DEFAULT_BRAIN_EROSION_DISTANCE,
                                brain_outlier_cutoff_fraction=DEFAULT_BRAIN_OUTLIER_CUTOFF_FRACTION,
                                voxel_spacing_arr=None,
                                image_name=None
                                ):
        """
        Perform an automatic segmentation of the ventricles of the brain based on histogram properties. The local
        intensity change is calculated at each voxel. The principle behind this technique is that the average local
        local intensity change among voxels in different regions of the histogram is calculated based on a calculation
        of a histogram that is weighted with the local intensity change at each voxel. Subsequently, the intensity with
        the highest local intensity change between WM and CSF is selected as the splitting intensity for the purpose
        of splitting the brain mask. Before the histograms are calculated, the brain mask is first eroded with a given
        distance in order to exclude most of the subarachnoid CSF regions and many GM reagions. The resulting splitting
        of the peaks is easier and the CSF region after the split would include mostly the ventricles.
        This function returns a list of two binary masks in the same shape as the input brain mask. The first includes
        the ventricles while the second includes the soft tissues (i.e. whole brain minus areas with CSF intensities)
        Further important details for finding the splitting intensity:
        - Since the relative location of the CSF and WM peak depend on the chosen MR sequence, a string describing the
        MR sequence needs to be passed as an argument.
        - Since outlier intensities may have excessively large local intensity change values (in particular contrast
        enhanced voxels), a certain fraction of voxels should be excluded as candidates for the splitting intensity
        from both ends of the histogram. That number can be specified.
        - Due to the erosion, the brain mask should not have any holes, because those would be strongly enlarged by the
        erosion. In case of holes (e.g. due to exclusion of ROIs), it is advised to either fill the holes first or to
        not perform the erosion.
        - The ventricle mask is obtained by applying the intensity threshold to the eroded brain, so that the outer
        voxels cannot interfere. However, the remaining brain mask (GM, WM) is determined by applying the intensity
        threshold to the entire input brain so as to not exclude the outer regions of the brain.
        - The splitting has shown to be relatively robust to different images and sequences, but may fail for very low
        resolutions and sometimes for FLAIR images where other dark brain regions interfere with the histogram detection
        of the CSF.
        :param intensity_arr: numpy array (n dimensions): Contains intensity values of an image as float or integer
        :param brain_mask_arr: numpy array (n dimensions): Binary mask in boolean with the same shape as intensity_arr,
        delineates the brain in the intensity array.
        :param local_intensity_change_function: function: takes as inpuzt arguments the intensity array and the
        voxel spacing array. Should return an array of the same shape as the intensity array, but with each value
        representing the local intensity change.
        :param histogram_calculation_function: function: takes as input the intensity array and a binary mask. It should
        return a histogram of the intensities inside the mask in the form of a tuple containing an intensity grid
        as the first element and the corresponding frequencies as the second element.
        :param brain_mr_contrast: string: Description of the contrast of the brain MRI. e.g. "t1", "t2", "t1c", "flair".
        :param brain_erosion_distance: integer or float: Distance of tissue that should be eroded from the outside
        of the brain. Should have same unit as voxel_spacing_arr. Will be converted into an appropriate radius and
        number of iterations automatically before the erosion function is called.
        :param brain_outlier_cutoff_fraction: float: Fraction of voxels that should be cut off from both ends of the
        intensity histogram for the purposes of selecting the split intensity between the brain regions.
        The purpose is to avoid the bias of the split location due to outlier intensities with very high local intensity
        change. It is recommended to use a value in the range of 0.01-0.02.
        :param voxel_spacing_arr: (optional) numpy array (1 dimension): Integer or float array with length n, where n is
        the number of dimensions of intensity_arr. For each dimension, this array contains the distance between voxels
        along that axis (assumed to be in mm). Default: assume all are 1.
        :param image_name: (optional) string: Is used for the purpose of plotting only. (For determining the file name
        and some text on the plot)
        :return: tuple with two entries: the first is a binary mask array of the same shape as the input brain mask
        array delineating the ventricles. The second is a binary mask array of the same shaep as the input brain mask
        array delineating the gray matter & white matter regions of the brain.
        """
        # Calculate local intensity change for whole image
        local_intensity_change_arr = local_intensity_change_function(intensity_arr, voxel_spacing_arr)

        # Erode the outside of the brain mask to exclude subarachnoid CSF tissue and part of the grey matter. The
        # remaining voxels should show a very clear distinction between CSF and WM, which will be used to split the
        # histogram.
        eroded_brain_mask_arr = erode_brain(brain_mask_arr, brain_erosion_distance, voxel_spacing_arr)

        grid_arr, pdf_arr = histogram_calculation_function(intensity_arr[eroded_brain_mask_arr])
        lic_weighted_grid_arr, lic_weighted_pdf_arr = histogram_calculation_function(
            intensity_arr[eroded_brain_mask_arr], local_intensity_change_arr[eroded_brain_mask_arr]
        )

        split_intensity, avg_lic_arr, truncated_avg_lic_arr = determine_brain_histogram_split_intensity(
            grid_arr, pdf_arr, lic_weighted_grid_arr, lic_weighted_pdf_arr, mr_contrast=brain_mr_contrast,
            outlier_cutoff_fraction=brain_outlier_cutoff_fraction
        )

        ventricle_mask_arr, gmwm_mask_arr = create_ventricle_and_gmwm_masks(
            intensity_arr, brain_mask_arr, eroded_brain_mask_arr, split_intensity, mr_contrast=brain_mr_contrast
        )

        # If the brain split visualizer is defined, plot the result
        if not self.brain_visualizer is None:
            self.brain_visualizer.visualize_histogram_split(
                grid_arr, pdf_arr, lic_weighted_pdf_arr, split_intensity, brain_erosion_distance,
                brain_outlier_cutoff_fraction, truncated_avg_lic_arr, avg_lic_arr, image_name=image_name
            )

        return ventricle_mask_arr, gmwm_mask_arr

    def split_brain_mask(self, intensity_arr, brain_mask_arr, voxel_spacing_arr=None,
                         local_intensity_change_radius=DEFAULT_LOCAL_INTENSITY_CHANGE_RADIUS,
                         kde_grid_number=DEFAULT_KDE_GRID_NUMBER, kde_bandwidth=DEFAULT_KDE_BANDWIDTH,
                         kde_discretization_count=DEFAULT_KDE_DISCRETIZATION_COUNT,
                         brain_mr_contrast=DEFAULT_BRAIN_MR_CONTRAST,
                         brain_erosion_distance=DEFAULT_BRAIN_EROSION_DISTANCE,
                         brain_outlier_cutoff_fraction=DEFAULT_BRAIN_OUTLIER_CUTOFF_FRACTION,
                         image_name=None
                         ):
        """
        Perform an automatic segmentation of the ventricles of the brain based on histogram properties. The local
        intensity change is calculated at each voxel with the default local intensity change function and the given
        radius. The kernel density estimation of the histogram is used for determining the split intensity between
        CSF and other brain tissues. For more details, consult the documentation of the function
        split_brain_mask_custom.
        :param intensity_arr: numpy array (n dimensions): Contains intensity values of an image as float or integer
        :param brain_mask_arr: numpy array (n dimensions): Binary mask in boolean with the same shape as intensity_arr,
        delineates the brain in the intensity array.
        :param voxel_spacing_arr: (optional) numpy array (1 dimension): Integer or float array with length n, where n is
        the number of dimensions of intensity_arr. For each dimension, this array contains the distance between voxels
        along that axis (assumed to be in mm). Default: assume all are 1.
        :param local_intensity_change_radius: integer or float: Radius of the local neighborhood for calculating local
        intensity change. Same units as given in voxel_spacing_arr
        :param kde_grid_number: integer: The number of grid points to be created for the kernel density estimation of
        the intensity histogram.
        :param kde_bandwidth: string or float: If a float, it is taken as the bandwidth for the kernel density
        estimation, if it's a string, it is taken as a bandwidth estimation strategy as available in the library KDEpy.
        Recommended: "ISJ" (Improved Sheather-Jones algorithm), which works better than alternatives in multimodal data.
        :param kde_discretization_count: (optional) integer: if not None, the intensity values will be discretized into
        this number of bins before calculation of the kernel density estimation. This may not be necessary on raw image
        data, but after bias field correction the number of unique values in the intensity array becomes very large,
        which can lead to problems with the bandwidth estimation for the KDE. It is recommended to use discretization of
        about 1024 in that case.
        :param brain_mr_contrast: string: Description of the contrast of the brain MRI. e.g. "t1", "t2", "t1c", "flair".
        :param brain_erosion_distance: integer or float: Distance of tissue that should be eroded from the outside
        of the brain. Should have same unit as voxel_spacing_arr. Will be converted into an appropriate radius and
        number of iterations automatically before the erosion function is called.
        :param brain_outlier_cutoff_fraction: float: Fraction of voxels that should be cut off from both ends of the
        intensity histogram for the purposes of selecting the split intensity between the brain regions.
        The purpose is to avoid the bias of the split location due to outlier intensities with very high local intensity
        change. It is recommended to use a value in the range of 0.01-0.02.
        :param image_name: (optional) string: Is used for the purpose of plotting only. (For determining the file name
        and some text on the plot)
        :return: tuple with two entries: the first is a binary mask array of the same shape as the input brain mask
        array delineating the ventricles. The second is a binary mask array of the same shaep as the input brain mask
        array delineating the gray matter & white matter regions of the brain.
        """
        lic_function = create_local_intensity_change_function(local_intensity_change_radius)

        histogram_calculation_function = create_kde_calculation_function(kde_grid_number, kde_bandwidth,
                                                                         kde_discretization_count)

        return self.split_brain_mask_custom(
            intensity_arr, brain_mask_arr, lic_function, histogram_calculation_function,
            brain_mr_contrast=brain_mr_contrast, brain_erosion_distance=brain_erosion_distance,
            brain_outlier_cutoff_fraction=brain_outlier_cutoff_fraction, voxel_spacing_arr=voxel_spacing_arr,
            image_name=image_name
        )