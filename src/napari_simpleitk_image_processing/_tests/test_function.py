# from napari_simpleitk_image_processing import threshold, image_arithmetic

# add your tests here...
import numpy as np


def test_something():
    from napari_simpleitk_image_processing import median_filter, gaussian_blur, threshold_otsu, \
        signed_maurer_distance_map, morphological_watershed, connected_component_labeling, \
        touching_objects_labeling, watershed_otsu_labeling, binary_fill_holes, \
        bilateral_filter, laplacian_filter, laplacian_of_gaussian_filter, binominal_blur_filter, \
        canny_edge_detection, gradient_magnitude, h_maxima, \
        h_minima, otsu_multiple_thresholds, regional_maxima, regional_minima, \
        richardson_lucy_deconvolution, wiener_deconvolution, tikhonov_deconvolution, rescale_intensity, \
        sobel, black_top_hat, white_top_hat

    image = np.asarray([[0, 1, 2, 3],
                        [2, 0, 1, 3],
                        [2, 0, 1, 3],
                        [2, 0, 1, 3]])

    for operation in  [median_filter,
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
            rescale_intensity,
            sobel,
            black_top_hat,
            white_top_hat]:
        print(operation)

        operation(image)


    for operation in  [richardson_lucy_deconvolution,
            wiener_deconvolution,
            tikhonov_deconvolution]:
        print(operation)

        operation(image, image)
