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
            elif 'pyclesperanto_prototype._tier0._pycl.OCLArray' in str(type(value)):
                # compatibility with pyclesperanto
                np_value = np.asarray(value)

            if convert_input_to_float and np_value is not None:
                np_value = np_value.astype(float)

            if np_value is not None:
                bound.arguments[key] = sitk.GetImageFromArray(np_value)

        # call the decorated function
        result = function(*bound.args, **bound.kwargs)

        if isinstance(result, sitk.SimpleITK.Image):
            return sitk.GetArrayFromImage(result)
        else:
            return result

    return worker_function



@register_function(menu="Filtering > Median (n-SimpleITK)")
@time_slicer
@plugin_function
def median_filter(image:napari.types.ImageData, radius_x: int = 1, radius_y: int = 1, radius_z: int = 0, viewer: napari.Viewer = None) -> napari.types.ImageData:
    
    return sitk.Median(image, [radius_x, radius_y, radius_z])


@register_function(menu="Filtering > Gaussian (n-SimpleITK)")
@time_slicer
@plugin_function
def gaussian_blur(image:napari.types.ImageData, variance_x: float = 1, variance_y: float = 1, variance_z: float = 0, viewer: napari.Viewer = None) -> napari.types.ImageData:
    
    return sitk.DiscreteGaussian(image, variance=[variance_x, variance_y, variance_z])


@register_function(menu="Segmentation > Threshold (Otsu et al 1979, n-SimpleITK)")
@time_slicer
@plugin_function
def threshold_otsu(image:napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    
    return sitk.OtsuThreshold(image,0,1)

@register_function(menu="Segmentation > Binary fill holes (n-SimpleITK)")
@time_slicer
@plugin_function
def binary_fill_holes(binary_image:napari.types.LabelsData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    
    return sitk.BinaryFillhole(binary_image)


@register_function(menu="Segmentation > Signed Maurer Distance Map (n-SimpleITK)")
@time_slicer
@plugin_function
def signed_maurer_distance_map(binary_image:napari.types.LabelsData, viewer: napari.Viewer = None) -> napari.types.ImageData:
    """
    See also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/32_Watersheds_Segmentation.html
    """
    
    return sitk.SignedMaurerDistanceMap(binary_image, insideIsPositive=True, squaredDistance=False, useImageSpacing=False)


@register_function(menu="Segmentation > Morphological watershed (n-SimpleITK)")
@time_slicer
@plugin_function
def morphological_watershed(distance_image:napari.types.ImageData, level:float = 1, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    """
    See also
    --------
    ..[0] http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/32_Watersheds_Segmentation.html
    """
    
    return sitk.MorphologicalWatershed( distance_image, markWatershedLine=False, level=level)


@register_function(menu="Segmentation > Connected component labeling (n-SimpleITK)")
@time_slicer
@plugin_function
def connected_component_labeling(binary_image:napari.types.LabelsData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    
    return sitk.ConnectedComponent(binary_image)


@register_function(menu="Segmentation > Touching objects labeling (n-SimpleITK)")
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


@register_function(menu="Segmentation > Watershed Otsu labeling (n-SimpleITK)")
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
