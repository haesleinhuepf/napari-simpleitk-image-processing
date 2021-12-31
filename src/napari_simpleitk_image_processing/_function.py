from napari_plugin_engine import napari_hook_implementation

@napari_hook_implementation
def napari_experimental_provide_function():
    from ._simpleitk_image_processing import median_filter, gaussian_blur, threshold_otsu, \
        signed_maurer_distance_map, morphological_watershed, connected_component_labeling, \
        touching_objects_labeling, watershed_otsu_labeling, binary_fill_holes, \
        bilateral_filter, laplacian_filter, laplacian_of_gaussian_filter, binominal_blur_filter, \
        canny_edge_detection, gradient_magnitude, h_maxima, \
        h_minima, otsu_multiple_thresholds, regional_maxima, regional_minima, \
        richardson_lucy_deconvolution, wiener_deconvolution, tikhonov_deconvolution, rescale_intensity, \
        sobel, black_top_hat, white_top_hat, relabel_component

    return [median_filter,
            gaussian_blur,
            threshold_otsu,
            signed_maurer_distance_map,
            morphological_watershed,
            connected_component_labeling,
            touching_objects_labeling,
            watershed_otsu_labeling,
            binary_fill_holes,
            bilateral_filter,
            laplacian_filter,
            laplacian_of_gaussian_filter,
            binominal_blur_filter,
            canny_edge_detection,
            gradient_magnitude,
            h_maxima,
            h_minima,
            otsu_multiple_thresholds,
            regional_maxima,
            regional_minima,
            richardson_lucy_deconvolution,
            wiener_deconvolution,
            tikhonov_deconvolution,
            rescale_intensity,
            sobel,
            black_top_hat,
            white_top_hat,
            relabel_component
            ]

