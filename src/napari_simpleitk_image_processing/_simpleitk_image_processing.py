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
import SimpleITK as sitk

@curry
def plugin_function(
        function: Callable,
        convert_input_to_float: bool = False
) -> Callable:
    # copied from https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/pyclesperanto_prototype/_tier0/_plugin_function.py
    @wraps(function)
    def worker_function(*args, **kwargs):
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
    
    return sitk.Median(image, [radius_x, radius_y, radius_z])


@register_function(menu="Filtering / noise removal > Gaussian (n-SimpleITK)")
@time_slicer
@plugin_function
def gaussian_blur(image:napari.types.ImageData, variance_x: float = 1, variance_y: float = 1, variance_z: float = 0, viewer: napari.Viewer = None) -> napari.types.ImageData:
    
    return sitk.DiscreteGaussian(image, variance=[variance_x, variance_y, variance_z])


@register_function(menu="Segmentation / binarization > Threshold (Otsu et al 1979, n-SimpleITK)")
@time_slicer
@plugin_function
def threshold_otsu(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    
    return sitk.OtsuThreshold(image,0,1)

@register_function(menu="Segmentation post-processing > Binary fill holes (n-SimpleITK)")
@time_slicer
@plugin_function
def binary_fill_holes(binary_image:napari.types.LabelsData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    
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
    
    return sitk.MorphologicalWatershed( distance_image, markWatershedLine=False, level=level)


@register_function(menu="Segmentation / labeling > Connected component labeling (n-SimpleITK)")
@time_slicer
@plugin_function
def connected_component_labeling(binary_image:napari.types.LabelsData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    
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
    return sitk.Bilateral(image, radius)


@register_function(menu="Filtering / edge enhancement > Laplacian (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def laplacian_filter(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return sitk.Laplacian(image)


@register_function(menu="Filtering / edge enhancement > Laplacian of Gaussian (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def laplacian_of_gaussian_filter(image:napari.types.ImageData, sigma:float = 1, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return sitk.LaplacianRecursiveGaussian(image, sigma=sigma)


@register_function(menu="Filtering / noise removal > Binominal blur (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def binominal_blur_filter(image:napari.types.ImageData, repetitions:int = 1, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return sitk.BinomialBlur(image, repetitions)

@register_function(menu="Segmentation / binarization > Canny edge detection (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def canny_edge_detection(image:napari.types.ImageData, lower_threshold: float = 0, upper_threshold: float = 50, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    return sitk.CannyEdgeDetection(image, lower_threshold, upper_threshold)


@register_function(menu="Filtering / edge enhancement > Gradient magnitude (n-SimpleITK)")
@time_slicer
@plugin_function
def gradient_magnitude(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return sitk.GradientMagnitude(image)


@register_function(menu="Filtering > H-Maxima (n-SimpleITK)")
@time_slicer
@plugin_function
def h_maxima(image:napari.types.ImageData, height: float = 10, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return sitk.HMaxima(image, height=height)


@register_function(menu="Filtering > H-Minima (n-SimpleITK)")
@time_slicer
@plugin_function
def h_minima(image:napari.types.ImageData, height: float = 10, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return sitk.HMinima(image, height=height)


@register_function(menu="Segmentation / binarization > Threshold Otsu, multiple thresholds (n-SimpleITK)")
@time_slicer
@plugin_function
def otsu_multiple_thresholds(image:napari.types.ImageData,
                             number_of_thresholds: int = 3,
                             label_offset: int = 0,
                             number_of_histogram_bins: int = 256,
                             viewer: napari.Viewer = None) -> napari.types.LabelsData:
    return sitk.OtsuMultipleThresholds(image, numberOfThresholds=number_of_thresholds,
                                       labelOffset=label_offset,
                                       numberOfHistogramBins=number_of_histogram_bins)

@register_function(menu="Segmentation / binarization > Regional maxima (n-SimpleITK)")
@time_slicer
@plugin_function
def regional_maxima(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return sitk.RegionalMaxima(image)


@register_function(menu="Segmentation / binarization > Regional minima (n-SimpleITK)")
@time_slicer
@plugin_function
def regional_minima(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return sitk.RegionalMinima(image)


@register_function(menu="Filtering / deconvolution > Richardson-Lucy deconvolution (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def richardson_lucy_deconvolution(image:napari.types.ImageData, kernel:napari.types.ImageData, number_of_iterations: int = 10, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return sitk.RichardsonLucyDeconvolution(image, kernel, number_of_iterations)


@register_function(menu="Filtering / deconvolution > Wiener deconvolution (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def wiener_deconvolution(image:napari.types.ImageData, kernel:napari.types.ImageData, noise_variance: float = 0, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return sitk.WienerDeconvolution(image, kernel, noise_variance)


@register_function(menu="Filtering / deconvolution > Tikhonov deconvolution (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def tikhonov_deconvolution(image:napari.types.ImageData, kernel:napari.types.ImageData, regularization_constant: float = 0, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return sitk.TikhonovDeconvolution(image, kernel, regularization_constant)


@register_function(menu="Filtering > Rescale intensity (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def rescale_intensity(image:napari.types.ImageData, output_minimum: float = 0, output_maximum: float = 1, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return sitk.RescaleIntensity(image, outputMinimum=output_minimum, outputMaximum=output_maximum)


@register_function(menu="Filtering / edge enhancement > Sobel (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def sobel(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return sitk.SobelEdgeDetection(image)


@register_function(menu="Filtering / background removal > White top-hat (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def white_top_hat(image:napari.types.ImageData, radius_x: int = 10, radius_y: int = 10, radius_z: int = 0, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return sitk.WhiteTopHat(image, [radius_x, radius_y, radius_z])


@register_function(menu="Filtering / background removal > Black top-hat (n-SimpleITK)")
@time_slicer
@plugin_function(convert_input_to_float=True)
def black_top_hat(image:napari.types.ImageData, radius_x: int = 10, radius_y: int = 10, radius_z: int = 0, viewer: napari.Viewer = None) -> napari.types.ImageData:
    return sitk.BlackTopHat(image, [radius_x, radius_y, radius_z])

