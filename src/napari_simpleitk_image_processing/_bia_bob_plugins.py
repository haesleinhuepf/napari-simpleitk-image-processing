def list_bia_bob_plugins():
    """List of function hints for bia_bob"""
    good_alternative_installed = False
    try:
        import napari_segment_blobs_and_things_with_membranes
        good_alternative_installed = True
    except:
        pass
    try:
        import pyclesperanto
        good_alternative_installed = True
    except:
        pass
    try:
        import pyclesperanto_prototype
        good_alternative_installed = True
    except:
        pass

    basic_hints = ""
    if not good_alternative_installed:
        basic_hints = """
        
    - Apply a median filter to an image to remove noise while preserving edges.
      nsitk.median_filter(image, radius=5)
    
    - Apply a Gaussian blur to smooth the image.
      nsitk.gaussian_blur(image, sigma=1.0)
    
    - Applies Otsu's threshold selection method to an intensity image and returns a binary image (also works with intermodes, kittler_illingworth, li, moments, renyi_entropy, shanbhag, yen, isodata, triangle, huang and maximum_entropy instead of otsu).
      nsitk.threshold_otsu(image)
    
    - Computes the gradient magnitude of an image.
      nsitk.gradient_magnitude(image)
    
    - Identifies the regional maxima of an image.
      nsitk.regional_maxima(image)
    
    - Rescales the intensity of an input image to a specified range.
      nsitk.rescale_intensity(image, output_min=0, output_max=255)
    
    - Applies the Sobel operator to an image to find edges.
      nsitk.sobel(image)
    
    - Applies a standard deviation filter to an image.
      nsitk.standard_deviation_filter(image)
    
    - Labels the connected components in a binary image.
      nsitk.connected_component_labeling(binary_image)
    
    - Applies the Laplacian of Gaussian filter to find edges in an image.
      nsitk.laplacian_of_gaussian_filter(image, sigma=1.0)
    
    - Identifies h-maxima of an image, suppressing maxima smaller than h.
      nsitk.h_maxima(image, h=10)
    
    - Removes background in an image using the Top-Hat filter.
      nsitk.white_top_hat(image, radius=5)
        """

    advanced_hints = """
    
    - Computes the signed Maurer distance map of the input image.
      nsitk.signed_maurer_distance_map(binary_image)
    
    - Detects edges in the image using Canny edge detection.
      nsitk.canny_edge_detection(image, variance=1.0)
    
    - Enhances the contrast of an image using adaptive histogram equalization.
      nsitk.adaptive_histogram_equalization(image)
    
    - Labels objects in a binary image and can split object that are touching..
      nsitk.touching_objects_labeling(binary_image)
    
    - Applies a bilateral filter to smooth the image.
      nsitk.bilateral_filter(image, domainSigma=2.0, rangeSigma=50.0)
    
    - Computes basic statistics for labeled object regions in an image.
      nsitk.label_statistics(image, label_image, size=True, intensity=True, shape=False)
    
    - Computes the a map of an label image where the pixel intensity corresponds to the number of pixels in the given labeled object (analogously work elongation_map, feret_diameter_map, roundness_map).
      nsitk.pixel_count_map(label_image)
        """

    return f"""
    ## napari-simpleitk-image-processing (nsitk)
    nsitk is a Python library that processes images using SimpleITK.
    When you use it, you always start by importing the library: `import napari_simpleitk_image_processing as nsitk`.
    When asked for how to use nsitk, you can adapt one of the following code snippets:
    
    {basic_hints}
    {advanced_hints}
    """