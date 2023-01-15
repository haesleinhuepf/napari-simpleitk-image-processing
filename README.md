# napari-simpleitk-image-processing (n-SimpleITK)

[![License](https://img.shields.io/pypi/l/napari-simpleitk-image-processing.svg?color=green)](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-simpleitk-image-processing.svg?color=green)](https://pypi.org/project/napari-simpleitk-image-processing)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-simpleitk-image-processing.svg?color=green)](https://python.org)
[![tests](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/workflows/tests/badge.svg)](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/actions)
[![codecov](https://codecov.io/gh/haesleinhuepf/napari-simpleitk-image-processing/branch/main/graph/badge.svg)](https://codecov.io/gh/haesleinhuepf/napari-simpleitk-image-processing)
[![Development Status](https://img.shields.io/pypi/status/napari-simpleitk-image-processing.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-simpleitk-image-processing)](https://napari-hub.org/plugins/napari-simpleitk-image-processing)
[![DOI](https://zenodo.org/badge/432729955.svg)](https://zenodo.org/badge/latestdoi/432729955)

Process images using [SimpleITK](https://simpleitk.org/) in [napari]

## Usage

Filters, segmentation algorithms and measurements provided by this napari plugin can be found in the `Tools` menu. 
You can recognize them with their suffix `(n-SimpleITK)` in brackets.
Furthermore, it can be used from the [napari-assistant](https://www.napari-hub.org/plugins/napari-assistant) graphical user interface. 
Therefore, just click the menu `Tools > Utilities > Assistant (na)` or run `naparia` from the command line.

![img.png](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/raw/main/docs/screenshot_with_assistant.png)

All filters implemented in this napari plugin are also demonstrated in [this notebook](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/blob/main/docs/demo.ipynb).

### Gaussian blur

Applies a [Gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur)
to an image. This might be useful for denoising, e.g. before applying the Threshold-Otsu method.

![img.png](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/raw/main/docs/gaussian_blur.png)

### Median filter

Applies a [median filter](https://en.wikipedia.org/wiki/Median_filter) to an image. 
Compared to the Gaussian blur this method preserves edges in the image better. 
It also performs slower.

![img.png](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/raw/main/docs/median_filter.png)

### Bilateral filter

The [bilateral filter](https://en.wikipedia.org/wiki/Bilateral_filter) allows denoising an image
while preserving edges.

![img.png](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/raw/main/docs/bilateral.png)

### Threshold Otsu

Binarizes an image using [Otsu's method](https://ieeexplore.ieee.org/document/4310076).

![img.png](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/raw/main/docs/threshold_otsu.png)

### Connected Component Labeling

Takes a binary image and labels all objects with individual numbers to produce a label image.

![img.png](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/raw/main/docs/connected_component_labeling.png)

### Measurements

This function allows determining intensity and shape statistics from labeled images. I can be found in the `Tools > Measurement tables` menu.

![img.png](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/raw/main/docs/measurements.png)

### Signed Maurer distance map

A distance map (more precise: [Signed Maurer Distance Map](https://itk.org/ITKExamples/src/Filtering/DistanceMap/MaurerDistanceMapOfBinary/Documentation.html)) can be useful for visualizing distances within binary images between black/white borders. 
Positive values in this image correspond to a white (value=1) pixel's distance to the next black pixel.
Black pixel's (value=0) distance to the next white pixel are represented in this map with negative values.

![img.png](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/raw/main/docs/signed_maured_distance_map.png)

### Binary fill holes

Fills holes in a binary image.

![img.png](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/raw/main/docs/binary_fill_holes.png)

### Touching objects labeling

Starting from a binary image, touching objects can be splits into multiple regions, similar to the [Watershed segmentation in ImageJ](https://imagej.net/plugins/classic-watershed).

![img.png](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/raw/main/docs/Touching_object_labeling.png)

### Morphological Watershed

The [morhological watershed](http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/32_Watersheds_Segmentation.html)
allows to segment images showing membranes. Before segmentation, a filter such as the Gaussian blur or a median filter
should be used to eliminate noise. It also makes sense to increase the thickness of membranes using a maximum filter. 
See [this notebook](https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/segmentation_2d_membranes.ipynb) for details.

![img.png](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/raw/main/docs/morphological_watershed.png)

### Watershed-Otsu-Labeling

This algorithm uses [Otsu's thresholding method](https://ieeexplore.ieee.org/document/4310076) in combination with 
[Gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur) and the 
[Watershed-algorithm](https://en.wikipedia.org/wiki/Watershed_(image_processing)) 
approach to label bright objects such as nuclei in an intensity image. The alogrithm has two sigma parameters and a 
level parameter which allow you to fine-tune where objects should be cut (`spot_sigma`) and how smooth outlines 
should be (`outline_sigma`). The `watershed_level` parameter determines how deep an intensity valley between two maxima 
has to be to differentiate the two maxima. 
This implementation is similar to [Voronoi-Otsu-Labeling in clesperanto](https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/voronoi_otsu_labeling.ipynb).


![img.png](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/raw/main/docs/watershed_otsu_labeling.png)

### Richardson-Lucy Deconvolution

[Richardson-Lucy deconvolution](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution)
allows to restore image quality if the point-spread-function of the optical system used 
for acquisition is known or can be approximated.

![img.png](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/raw/main/docs/Richardson-Lucy-Deconvolution.png)


## Installation

You can install `napari-simpleitk-image-processing` via using `conda` and `pip`.
If you have never used `conda` before, please go through [this tutorial](https://biapol.github.io/blog/johannes_mueller/anaconda_getting_started/) first.

    conda install -c conda-forge napari
    pip install napari-simpleitk-image-processing

## See also

There are other napari plugins with similar functionality for processing images and extracting features:
* [morphometrics](https://www.napari-hub.org/plugins/morphometrics)
* [PartSeg](https://www.napari-hub.org/plugins/PartSeg)
* [napari-skimage-regionprops](https://www.napari-hub.org/plugins/napari-skimage-regionprops)
* [napari-cupy-image-processing](https://www.napari-hub.org/plugins/napari-cupy-image-processing)
* [napari-pyclesperanto-assistant](https://www.napari-hub.org/plugins/napari-pyclesperanto-assistant)
* [napari-allencell-segmenter](https://napari-hub.org/plugins/napari-allencell-segmenter)
* [RedLionfish](https://www.napari-hub.org/plugins/RedLionfish)
* [bbii-decon](https://www.napari-hub.org/plugins/bbii-decon)  
* [napari-segment-blobs-and-things-with-membranes](https://www.napari-hub.org/plugins/napari-segment-blobs-and-things-with-membranes)

Furthermore, there are plugins for postprocessing extracted measurements
* [napari-feature-classifier](https://www.napari-hub.org/plugins/napari-feature-classifier)
* [napari-clusters-plotter](https://www.napari-hub.org/plugins/napari-clusters-plotter)

## Contributing

Contributions are very welcome. There are many useful algorithms available in 
[SimpleITK](https://simpleitk.org/). If you want another one available here in this napari
plugin, don't hesitate to send a [pull-request](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/pulls).
This repository just holds wrappers for SimpleITK-functions, see [this file](https://github.com/haesleinhuepf/napari-simpleitk-image-processing/raw/main/src/napari_simpleitk_image_processing/_simpleitk_image_processing.py#L51) for how those wrappers
can be written.

## License

Distributed under the terms of the [BSD-3] license,
"napari-simpleitk-image-processing" is free and open source software

## Citation

For implementing this napari plugin, the 
[SimpleITK python notebooks](https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/) were very helpful. 
Thus, if you find the plugin useful, consider citing the SimpleITK notebooks:

Z. Yaniv, B. C. Lowekamp, H. J. Johnson, R. Beare, 
"SimpleITK Image-Analysis Notebooks: a Collaborative Environment for Education and Reproducible Research", \
J Digit Imaging., 31(3): 290-303, 2018, [https://doi.org/10.1007/s10278-017-0037-8](https://doi.org/10.1007/s10278-017-0037-8).

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/haesleinhuepf/napari-simpleitk-image-processing/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
