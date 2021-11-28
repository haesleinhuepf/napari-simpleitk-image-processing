from enum import Enum
import numpy as np
from napari_plugin_engine import napari_hook_implementation
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
        function: Callable
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

        import SimpleITK as sitk

        # copy images to SimpleITK-types, and create output array if necessary
        for key, value in bound.arguments.items():
            if isinstance(value, np.ndarray):
                bound.arguments[key] = sitk.GetImageFromArray(value)
            elif 'pyclesperanto_prototype._tier0._pycl.OCLArray' in str(type(value)):
                # compatibility with pyclesperanto
                bound.arguments[key] = sitk.GetImageFromArray(np.asarray(value))

        # call the decorated function
        result = function(*bound.args, **bound.kwargs)

        if isinstance(result, sitk.SimpleITK.Image):
            return sitk.GetArrayFromImage(result)
        else:
            return result

    return worker_function

@napari_hook_implementation
def napari_experimental_provide_function():
    return [threshold_otsu]

@register_function(menu="Segmentation > Threshold (Otsu et al 1979, n-SimpleITK)")
@plugin_function
def threshold_otsu(image:napari.types.ImageData) -> napari.types.LabelsData:
    import SimpleITK as sitk
    return sitk.OtsuThreshold(image,0,1)

