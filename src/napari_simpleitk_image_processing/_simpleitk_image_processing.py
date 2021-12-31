from enum import Enum
import numpy as np
import napari
from typing import Callable
from functools import wraps
from toolz import curry
import inspect
import numpy as np
import napari
from napari_tools_menu import register_function
from napari_time_slicer import time_slicer

@curry
def plugin_function(
        function: Callable,
        convert_input_to_float: bool = False
) -> Callable:
    # copied from https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/pyclesperanto_prototype/_tier0/_plugin_function.py
    @wraps(function)
    def worker_function(*args, **kwargs):
        import SimpleITK as sitk

        sig = inspect.signature(function)
        # create mapping from position and keyword arguments to parameters
        # will raise a TypeError if the provided arguments do not match the signature
        # https://docs.python.org/3/library/inspect.html#inspect.Signature.bind
        bound = sig.bind(*args, **kwargs)
        # set default values for missing arguments
        # https://docs.python.org/3/library/inspect.html#inspect.BoundArguments.apply_defaults
        bound.apply_defaults()

        

        # copy images to SimpleITK-types, and create output array if necessary
        for key, value in bound.arguments.items():
            np_value = None
            if isinstance(value, np.ndarray):
                np_value = value
            elif 'pyclesperanto_prototype._tier0._pycl.OCLArray' in str(type(value)) or \
                'dask.array.core.Array' in str(type(value)):
                # compatibility with pyclesperanto and dask
                np_value = np.asarray(value)

            if convert_input_to_float and np_value is not None:
                np_value = np_value.astype(float)

            if np_value is not None:
                if np_value.dtype == bool:
                    np_value = np_value * 1
                bound.arguments[key] = sitk.GetImageFromArray(np_value)

        # call the decorated function
        result = function(*bound.args, **bound.kwargs)

        if isinstance(result, sitk.SimpleITK.Image):
            return sitk.GetArrayFromImage(result)
        else:
            return result

    worker_function.__module__ = "napari_simpleitk_image_processing"

    return worker_function



@register_function(menu="Filtering / noise removal > Median (n-SimpleITK)")
@time_slicer
@plugin_function
def median_filter(image:napari.types.ImageData, radius_x: int = 1, radius_y: int = 1, radius_z: int = 0, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.Median(image, [radius_x, radius_y, radius_z])


@register_function(menu="Filtering / noise removal > Gaussian (n-SimpleITK)")
@time_slicer
@plugin_function
def gaussian_blur(image:napari.types.ImageData, variance_x: float = 1, variance_y: float = 1, variance_z: float = 0, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.DiscreteGaussian(image, variance=[variance_x, variance_y, variance_z])


@register_function(menu="Segmentation / binarization > Threshold (Otsu et al 1979, n-SimpleITK)")
@time_slicer
@plugin_function
def threshold_otsu(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    Binarize an image according to Otsu's method.

    Parameters
    ----------
    image: Image
    viewer: napari.Viewer

    See Also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/35_Segmentation_Shape_Analysis.html
    ..[1] https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1HuangThresholdImageFilter.html
    ..[2] https://www.insight-journal.org/browse/publication/811

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    import SimpleITK as sitk
    return sitk.OtsuThreshold(image,0,1)


@register_function(menu="Segmentation / binarization > Threshold (Kittler and Illingworth 1986, n-SimpleITK)")
@time_slicer
@plugin_function
def threshold_kittler_illingworth(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    Binarize an image according to the Kittler-Illingworth method.

    Parameters
    ----------
    image: Image
    viewer: napari.Viewer

    See Also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/35_Segmentation_Shape_Analysis.html
    ..[1] https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1HuangThresholdImageFilter.html
    ..[2] https://www.insight-journal.org/browse/publication/811

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    import SimpleITK as sitk
    return sitk.KittlerIllingworthThreshold(image,0,1)


@register_function(menu="Segmentation / binarization > Threshold (Li et al 1993, n-SimpleITK)")
@time_slicer
@plugin_function
def threshold_li(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    Binarize an image according to Li's method.

    Parameters
    ----------
    image: Image
    viewer: napari.Viewer

    See Also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/35_Segmentation_Shape_Analysis.html
    ..[1] https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1HuangThresholdImageFilter.html
    ..[2] https://www.insight-journal.org/browse/publication/811

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    import SimpleITK as sitk
    return sitk.LiThreshold(image,0,1)


@register_function(menu="Segmentation / binarization > Threshold (Moments, n-SimpleITK)")
@time_slicer
@plugin_function
def threshold_moments(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    Binarize an image according to the Moments method.

    Parameters
    ----------
    image: Image
    viewer: napari.Viewer

    See Also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/35_Segmentation_Shape_Analysis.html
    ..[1] https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1HuangThresholdImageFilter.html
    ..[2] https://www.insight-journal.org/browse/publication/811

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    import SimpleITK as sitk
    return sitk.MomentsThreshold(image,0,1)


@register_function(menu="Segmentation / binarization > Threshold (Renyi entropy, n-SimpleITK)")
@time_slicer
@plugin_function
def threshold_renyi_entropy(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    Binarize an image according to the Renyi-entropy method.

    Parameters
    ----------
    image: Image
    viewer: napari.Viewer

    See Also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/35_Segmentation_Shape_Analysis.html
    ..[1] https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1HuangThresholdImageFilter.html
    ..[2] https://www.insight-journal.org/browse/publication/811

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    import SimpleITK as sitk
    return sitk.RenyiEntropyThreshold(image,0,1)


@register_function(menu="Segmentation / binarization > Threshold (Shanbhag 1994, n-SimpleITK)")
@time_slicer
@plugin_function
def threshold_shanbhag(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    Binarize an image according to Shanbhag's method.

    Parameters
    ----------
    image: Image
    viewer: napari.Viewer

    See Also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/35_Segmentation_Shape_Analysis.html
    ..[1] https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1HuangThresholdImageFilter.html
    ..[2] https://www.insight-journal.org/browse/publication/811

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    import SimpleITK as sitk
    return sitk.ShanbhagThreshold(image,0,1)


@register_function(menu="Segmentation / binarization > Threshold (Yen et al 1995, n-SimpleITK)")
@time_slicer
@plugin_function
def threshold_yen(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    Binarize an image according to Yen's method.

    Parameters
    ----------
    image: Image
    viewer: napari.Viewer

    See Also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/35_Segmentation_Shape_Analysis.html
    ..[1] https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1HuangThresholdImageFilter.html
    ..[2] https://www.insight-journal.org/browse/publication/811

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    import SimpleITK as sitk
    return sitk.YenThreshold(image,0,1)


@register_function(menu="Segmentation / binarization > Threshold (IsoData, n-SimpleITK)")
@time_slicer
@plugin_function
def threshold_isodata(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    Binarize an image according to the IsoData method.

    Parameters
    ----------
    image: Image
    viewer: napari.Viewer

    See Also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/35_Segmentation_Shape_Analysis.html
    ..[1] https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1HuangThresholdImageFilter.html
    ..[2] https://www.insight-journal.org/browse/publication/811

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    import SimpleITK as sitk
    return sitk.IsoDataThreshold(image,0,1)


@register_function(menu="Segmentation / binarization > Threshold (Triangle, n-SimpleITK)")
@time_slicer
@plugin_function
def threshold_triangle(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    Binarize an image according to Otsu's method.

    Parameters
    ----------
    image: Image
    viewer: napari.Viewer

    See Also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/35_Segmentation_Shape_Analysis.html
    ..[1] https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1HuangThresholdImageFilter.html
    ..[2] https://www.insight-journal.org/browse/publication/811

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    import SimpleITK as sitk
    return sitk.OtsuThreshold(image,0,1)


@register_function(menu="Segmentation / binarization > Threshold (Huang and Wang 1995, n-SimpleITK)")
@time_slicer
@plugin_function
def threshold_huang(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    Binarize an image according to the triangle method.

    Parameters
    ----------
    image: Image
    viewer: napari.Viewer

    See Also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/35_Segmentation_Shape_Analysis.html
    ..[1] https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1HuangThresholdImageFilter.html
    ..[2] https://www.insight-journal.org/browse/publication/811

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    import SimpleITK as sitk
    return sitk.TriangleThreshold(image,0,1)


@register_function(menu="Segmentation / binarization > Threshold (Maximum entropy, n-SimpleITK)")
@time_slicer
@plugin_function
def threshold_maximum_entropy(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    Binarize an image according to maximum-entropy method.

    Parameters
    ----------
    image: Image
    viewer: napari.Viewer

    See Also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/35_Segmentation_Shape_Analysis.html

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    import SimpleITK as sitk
    return sitk.MaximumEntropyThreshold(image,0,1)


@register_function(menu="Segmentation post-processing > Binary fill holes (n-SimpleITK)")
@time_slicer
@plugin_function
def binary_fill_holes(binary_image:napari.types.LabelsData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    import SimpleITK as sitk
    return sitk.BinaryFillhole(binary_image)


@register_function(menu="Measurement > Signed Maurer Distance Map (n-SimpleITK)")
@time_slicer
@plugin_function
def signed_maurer_distance_map(binary_image:napari.types.LabelsData, viewer: napari.Viewer = None) -> napari.types.ImageData:
    """
    See also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/32_Watersheds_Segmentation.html
    """
    import SimpleITK as sitk
    return sitk.SignedMaurerDistanceMap(binary_image, insideIsPositive=True, squaredDistance=False, useImageSpacing=False)


@register_function(menu="Segmentation / labeling > Morphological watershed (n-SimpleITK)")
@time_slicer
@plugin_function
def morphological_watershed(distance_image:napari.types.ImageData, level:float = 1, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    See also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/32_Watersheds_Segmentation.html
    """
    import SimpleITK as sitk
    return sitk.MorphologicalWatershed( distance_image, markWatershedLine=False, level=level)


@register_function(menu="Segmentation / labeling > Connected component labeling (n-SimpleITK)")
@time_slicer
@plugin_function
def connected_component_labeling(binary_image:napari.types.LabelsData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    import SimpleITK as sitk
    return sitk.ConnectedComponent(binary_image)


@register_function(menu="Segmentation / labeling > Touching objects labeling (n-SimpleITK)")
@time_slicer
@plugin_function
def touching_objects_labeling(binary_image:napari.types.LabelsData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    Takes a binary image an splits touching objects into multiple similar to the Watershed segmentation in ImageJ.

    See also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/32_Watersheds_Segmentation.html#Multi-label-Morphology
    """
    import SimpleITK as sitk
    distance_map = sitk.SignedMaurerDistanceMap(binary_image, insideIsPositive=False, squaredDistance=False,
                                                useImageSpacing=False)

    ws = sitk.MorphologicalWatershed(distance_map, markWatershedLine=False, level=1)
    labels = sitk.Mask(ws, sitk.Cast(binary_image, ws.GetPixelID()))
    return labels


@register_function(menu="Segmentation / labeling > Watershed Otsu labeling (n-SimpleITK)")
@time_slicer
@plugin_function
def watershed_otsu_labeling(image:napari.types.ImageData, spot_sigma: float = 2, outline_sigma: float = 2, watershed_level:float = 10, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    The two sigma parameters and the level allow tuning the segmentation result. The first sigma controls how close detected cells
    can be (spot_sigma) and the second controls how precise segmented objects are outlined (outline_sigma). Under the
    hood, this filter applies two Gaussian blurs, spot detection, Otsu-thresholding and Voronoi-labeling. The
    thresholded binary image is flooded using the Voronoi approach starting from the found local maxima. Noise-removal
    sigma for spot detection and thresholding can be configured separately.

    The implementation here is similar to the Voronoi-Otsu-Labeling in clesperanto[1].

    See also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/32_Watersheds_Segmentation.html#Multi-label-Morphology
    ..[1] https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/voronoi_otsu_labeling.ipynb
    """
    import SimpleITK as sitk

    blurred_spots = sitk.GradientMagnitudeRecursiveGaussian(image, sigma=spot_sigma)

    blurred_outline = sitk.DiscreteGaussian(image, variance=[outline_sigma, outline_sigma, outline_sigma])
    binary_otsu = sitk.OtsuThreshold(blurred_outline, 0, 1)

    ws = sitk.MorphologicalWatershed(blurred_spots, markWatershedLine=False, level=watershed_level)
    labels = sitk.Mask(ws, sitk.Cast(binary_otsu, ws.GetPixelID()))

    return labels


@register_function(menu="Filtering / noise removal > Bilateral (n-SimpleITK)")
@time_slicer
@plugin_function
def bilateral_filter(image:napari.types.ImageData, radius: float = 1, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.Bilateral(image, radius)


@register_function(menu="Filtering / edge enhancement > Laplacian (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def laplacian_filter(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.Laplacian(image)


@register_function(menu="Filtering / edge enhancement > Laplacian of Gaussian (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def laplacian_of_gaussian_filter(image:napari.types.ImageData, sigma:float = 1, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.LaplacianRecursiveGaussian(image, sigma=sigma)


@register_function(menu="Filtering / noise removal > Binominal blur (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def binominal_blur_filter(image:napari.types.ImageData, repetitions:int = 1, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.BinomialBlur(image, repetitions)


@register_function(menu="Segmentation / binarization > Canny edge detection (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def canny_edge_detection(image:napari.types.ImageData, lower_threshold: float = 0, upper_threshold: float = 50, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    import SimpleITK as sitk
    return sitk.CannyEdgeDetection(image, lower_threshold, upper_threshold)


@register_function(menu="Filtering / edge enhancement > Gradient magnitude (n-SimpleITK)")
@time_slicer
@plugin_function
def gradient_magnitude(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.GradientMagnitude(image)


@register_function(menu="Filtering > H-Maxima (n-SimpleITK)")
@time_slicer
@plugin_function
def h_maxima(image:napari.types.ImageData, height: float = 10, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.HMaxima(image, height=height)


@register_function(menu="Filtering > H-Minima (n-SimpleITK)")
@time_slicer
@plugin_function
def h_minima(image:napari.types.ImageData, height: float = 10, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.HMinima(image, height=height)


@register_function(menu="Segmentation / binarization > Threshold Otsu, multiple thresholds (n-SimpleITK)")
@time_slicer
@plugin_function
def otsu_multiple_thresholds(image:napari.types.ImageData,
                             number_of_thresholds: int = 3,
                             label_offset: int = 0,
                             number_of_histogram_bins: int = 256,
                             viewer: napari.Viewer = None) -> napari.types.LabelsData:
    import SimpleITK as sitk
    return sitk.OtsuMultipleThresholds(image, numberOfThresholds=number_of_thresholds,
                                       labelOffset=label_offset,
                                       numberOfHistogramBins=number_of_histogram_bins)

@register_function(menu="Segmentation / binarization > Regional maxima (n-SimpleITK)")
@time_slicer
@plugin_function
def regional_maxima(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.RegionalMaxima(image)


@register_function(menu="Segmentation / binarization > Regional minima (n-SimpleITK)")
@time_slicer
@plugin_function
def regional_minima(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.RegionalMinima(image)


@register_function(menu="Filtering / deconvolution > Richardson-Lucy deconvolution (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def richardson_lucy_deconvolution(image:napari.types.ImageData, kernel:napari.types.ImageData, number_of_iterations: int = 10, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.RichardsonLucyDeconvolution(image, kernel, number_of_iterations)


@register_function(menu="Filtering / deconvolution > Wiener deconvolution (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def wiener_deconvolution(image:napari.types.ImageData, kernel:napari.types.ImageData, noise_variance: float = 0, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.WienerDeconvolution(image, kernel, noise_variance)


@register_function(menu="Filtering / deconvolution > Tikhonov deconvolution (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def tikhonov_deconvolution(image:napari.types.ImageData, kernel:napari.types.ImageData, regularization_constant: float = 0, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.TikhonovDeconvolution(image, kernel, regularization_constant)


@register_function(menu="Filtering > Rescale intensity (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def rescale_intensity(image:napari.types.ImageData, output_minimum: float = 0, output_maximum: float = 1, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.RescaleIntensity(image, outputMinimum=output_minimum, outputMaximum=output_maximum)


@register_function(menu="Filtering / edge enhancement > Sobel (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def sobel(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.SobelEdgeDetection(image)


@register_function(menu="Filtering / background removal > White top-hat (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def white_top_hat(image:napari.types.ImageData, radius_x: int = 10, radius_y: int = 10, radius_z: int = 0, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.WhiteTopHat(image, [radius_x, radius_y, radius_z])


@register_function(menu="Filtering / background removal > Black top-hat (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def black_top_hat(image:napari.types.ImageData, radius_x: int = 10, radius_y: int = 10, radius_z: int = 0, viewer: napari.Viewer = None) -> napari.types.ImageData:
    import SimpleITK as sitk
    return sitk.BlackTopHat(image, [radius_x, radius_y, radius_z])


@register_function(menu="Filtering > Adaptive histogram equalization (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def adaptive_histogram_equalization(
        image:napari.types.ImageData,
        alpha:float = 0.3,
        beta:float = 0.3,
        radius_x: int = 5,
        radius_y: int = 5,
        radius_z: int = 5,
        viewer: napari.Viewer = None) -> napari.types.ImageData:
    """
    Power Law Adaptive Histogram Equalization.

    Parameters
    ----------
    image
    alpha: float, optional
        controls how much the filter acts like the classical histogram equalization method (alpha=0) to how much the
        filter acts like an unsharp mask (alpha=1).
    beta: float, optional
        controls how much the filter acts like an unsharp mask (beta=0) to much the filter acts like pass through
        (beta=1, with alpha=1).
    radius_x: int, optional
        controls the size of the region over which local statistics are calculated. The size of the window is
        controlled by the radius the default radius is 5 in all directions.
    radius_y: int, optional
    radius_z: int, optional
    viewer: napari.Viewer, optional
        necessary for time-slicer

    Returns
    -------
    image

    See Also
    --------
    ..[0] https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1AdaptiveHistogramEqualizationImageFilter.html
    """

    import SimpleITK as sitk
    ahe = sitk.AdaptiveHistogramEqualizationImageFilter()
    ahe.SetAlpha(alpha)
    ahe.SetBeta(beta)
    ahe.SetRadius([radius_x, radius_y, radius_z])

    return ahe.Execute(image)



@register_function(menu="Filtering / noise removal > Curvature flow (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def curvature_flow_denoise(image:napari.types.ImageData,
        time_step:float = 0.05,
        number_of_iterations:int = 5,
        viewer: napari.Viewer = None) -> napari.types.ImageData:

    import SimpleITK as sitk
    cf = sitk.CurvatureFlowImageFilter()
    cf.SetNumberOfIterations(number_of_iterations)
    cf.SetTimeStep(time_step)
    return cf.Execute(image)


@register_function(menu="Segmentation post-processing > Relabel component (n-SimpleITK)")
@time_slicer
@plugin_function
def relabel_component(label_image:napari.types.LabelsData, minimumObjectSize:int=15, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    See Also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/35_Segmentation_Shape_Analysis.html
    """
    import SimpleITK as sitk
    return sitk.RelabelComponent(label_image, minimumObjectSize=minimumObjectSize)


@register_function(menu="Segmentation / binarization > Label contour (n-SimpleITK)")
@time_slicer
@plugin_function
def label_contour(label_image:napari.types.LabelsData, fully_connected: bool = True, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    Extract the outline of labels in a label image.

    See Also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/05_Results_Visualization.html
    """
    import SimpleITK as sitk
    return sitk.LabelContour(label_image, fullyConnected=fully_connected, backgroundValue=0)


@register_function(menu="Measurement > Measurements (n-SimpleITK)")
@time_slicer
@plugin_function
def label_statistics(
        intensity_image: napari.types.LayerData,
        label_image: napari.types.LabelsData,
        napari_viewer: napari.Viewer,
        size: bool = True, intensity: bool = True, perimeter: bool = False,
        shape: bool = False, position: bool = False, moments: bool = False):
    """
    See Also
    --------
    ..[0] https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1LabelShapeStatisticsImageFilter
    ..[1] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/35_Segmentation_Shape_Analysis.html
    """
    import SimpleITK as sitk

    intensity_stats = sitk.LabelStatisticsImageFilter()
    intensity_stats.Execute(intensity_image, label_image)

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.SetComputeFeretDiameter(True)
    shape_stats.SetComputeOrientedBoundingBox(False)
    shape_stats.SetComputePerimeter(True)
    shape_stats.Execute(label_image)

    results = {}

    for l in shape_stats.GetLabels():
        ##range(1, stats.GetNumberOfLabels() + 1):
        _append_to_column(results, "label", l)

        if intensity:
            _append_to_column(results, "maximum", intensity_stats.GetMaximum(l))
            _append_to_column(results, "mean", intensity_stats.GetMean(l))
            _append_to_column(results, "median", intensity_stats.GetMedian(l))
            _append_to_column(results, "minimum", intensity_stats.GetMinimum(l))
            _append_to_column(results, "sigma", intensity_stats.GetSigma(l))
            _append_to_column(results, "sum", intensity_stats.GetSum(l))
            _append_to_column(results, "variance", intensity_stats.GetVariance(l))

        if position:
            for i, value in enumerate(shape_stats.GetBoundingBox(l)):
                _append_to_column(results, "bbox_" + str(i), value)

            for i, value in enumerate(shape_stats.GetCentroid(l)):
                _append_to_column(results, "centroid_" + str(i), value)

        if shape:
            _append_to_column(results, "elongation", shape_stats.GetElongation(l))

            _append_to_column(results, "feret_diameter", shape_stats.GetFeretDiameter(l))
            _append_to_column(results, "flatness", shape_stats.GetFlatness(l))

            _append_to_column(results, "roundness", shape_stats.GetRoundness(l))

        if size:
            for i, value in enumerate(shape_stats.GetEquivalentEllipsoidDiameter(l)):
                _append_to_column(results, "equivalent_ellipsoid_diameter_" + str(i), value)

            _append_to_column(results, "equivalent_spherical_perimeter", shape_stats.GetEquivalentSphericalPerimeter(l))
            _append_to_column(results, "equivalent_spherical_radius", shape_stats.GetEquivalentSphericalRadius(l))

            _append_to_column(results, "number_of_pixels", shape_stats.GetNumberOfPixels(l))
            _append_to_column(results, "number_of_pixels_on_border", shape_stats.GetNumberOfPixelsOnBorder(l))

        if perimeter:
            _append_to_column(results, "perimeter", shape_stats.GetPerimeter(l))
            _append_to_column(results, "perimeter_on_border", shape_stats.GetPerimeterOnBorder(l))
            _append_to_column(results, "perimeter_on_border_ratio", shape_stats.GetPerimeterOnBorderRatio(l))

        if moments:
            for i, value in enumerate(shape_stats.GetPrincipalAxes(l)):
                _append_to_column(results, "principal_axes" + str(i), value)

            for i, value in enumerate(shape_stats.GetPrincipalMoments(l)):
                _append_to_column(results, "principal_moments" + str(i), value)

        # potential todo:
        # std::vector< double > 	GetOrientedBoundingBoxDirection (int64_t label) const
        # std::vector< double > 	GetOrientedBoundingBoxOrigin (int64_t label) const
        # std::vector< double > 	GetOrientedBoundingBoxSize (int64_t label) const
        # std::vector< double > 	GetOrientedBoundingBoxVertices (int64_t label) const
        # double 	GetPhysicalSize (int64_t label) const

    if napari_viewer is not None:
        from napari_workflows._workflow import _get_layer_from_data
        labels_layer = _get_layer_from_data(napari_viewer, label_image)
        # Store results in the properties dictionary:
        labels_layer.properties = results

        # turn table into a widget
        from napari_skimage_regionprops import add_table
        add_table(labels_layer, napari_viewer)
    else:
        return results


def _append_to_column(dictionary, column_name, value):
    if column_name not in dictionary.keys():
        dictionary[column_name] = []
    dictionary[column_name].append(value)


