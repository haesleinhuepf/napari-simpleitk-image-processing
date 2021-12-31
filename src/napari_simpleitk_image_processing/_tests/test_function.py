# from napari_simpleitk_image_processing import threshold, image_arithmetic

# add your tests here...
import numpy as np


def test_something():
    from napari_simpleitk_image_processing import median_filter, gaussian_blur, threshold_otsu, threshold_intermodes, \
        threshold_kittler_illingworth, threshold_li, threshold_moments, threshold_renyi_entropy, \
        threshold_shanbhag, threshold_yen, threshold_isodata, threshold_triangle, threshold_huang, \
        threshold_maximum_entropy, \
        signed_maurer_distance_map, morphological_watershed, morphological_gradient, standard_deviation_filter, \
        simple_linear_iterative_clustering, scalar_image_k_means_clustering, \
        connected_component_labeling, \
        touching_objects_labeling, watershed_otsu_labeling, binary_fill_holes, invert_intensity, \
        bilateral_filter, laplacian_filter, laplacian_of_gaussian_filter, binominal_blur_filter, \
        canny_edge_detection, gradient_magnitude, h_maxima, \
        h_minima, otsu_multiple_thresholds, regional_maxima, regional_minima, \
        richardson_lucy_deconvolution, wiener_deconvolution, tikhonov_deconvolution, rescale_intensity, \
        sobel, black_top_hat, white_top_hat, adaptive_histogram_equalization, curvature_flow_denoise, \
        relabel_component, label_contour

    image = np.asarray([[0, 1, 2, 3],
                        [2, 0, 1, 3],
                        [2, 0, 1, 3],
                        [2, 0, 1, 3]])

    for operation in  [median_filter,
            gaussian_blur,
            threshold_otsu,
            threshold_intermodes,
            threshold_kittler_illingworth,
            threshold_li,
            threshold_moments,
            threshold_renyi_entropy,
            threshold_shanbhag,
            threshold_yen,
            threshold_isodata,
            threshold_triangle,
            threshold_huang,
            threshold_maximum_entropy,
            signed_maurer_distance_map,
            morphological_watershed,
            morphological_gradient,
            standard_deviation_filter,
            simple_linear_iterative_clustering,
            scalar_image_k_means_clustering,
            connected_component_labeling,
            touching_objects_labeling,
            watershed_otsu_labeling,
            binary_fill_holes,
            invert_intensity,
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
            white_top_hat,
            adaptive_histogram_equalization,
            curvature_flow_denoise,
            relabel_component,
            label_contour]:
        print(operation)

        operation(image)


    for operation in  [richardson_lucy_deconvolution,
            wiener_deconvolution,
            tikhonov_deconvolution]:
        print(operation)

        operation(image, image)

reference = {
        'label': [1, 2, 3, 4, 5],
        'maximum': [1., 2., 3., 4., 5.],
        'mean': [1., 2., 3., 4., 5.],
        'median': [1.00585938, 2.00195312, 2.99804688, 3.99414062, 4.99023438],
        'minimum': [1., 2., 3., 4., 5.],
        'sigma': [0., 0., 0., 0., 0.],
        'sum': [ 2.,  4.,  6., 16., 10.],
        'variance': [0., 0., 0., 0., 0.],
        'bbox_0': [1, 2, 3, 0, 0],
        'bbox_1': [0, 0, 0, 2, 3],
        'bbox_2': [1, 1, 1, 4, 2],
        'bbox_3': [2, 2, 2, 1, 1],
        'centroid_0': [1. , 2. , 3. , 1.5, 0.5],
        'centroid_1': [0.5, 0.5, 0.5, 2. , 3. ],
        'elongation': [0., 0., 0., 0., 0.],
        'feret_diameter': [1., 1., 1., 3., 1.],
        'flatness': [0., 0., 0., 0., 0.],
        'roundness': [1.09516279, 1.09516279, 1.09516279, 0.8470636 , 1.09516279],
        'equivalent_ellipsoid_diameter_0': [0., 0., 0., 0., 0.],
        'equivalent_ellipsoid_diameter_1': [0., 0., 0., 0., 0.],
        'equivalent_spherical_perimeter': [5.01325655, 5.01325655, 5.01325655, 7.0898154 , 5.01325655],
        'equivalent_spherical_radius': [0.79788456, 0.79788456, 0.79788456, 1.12837917, 0.79788456],
        'number_of_pixels': [2, 2, 2, 4, 2],
        'number_of_pixels_on_border': [1, 1, 2, 2, 2],
        'perimeter': [4.57763596, 4.57763596, 4.57763596, 8.36987376, 4.57763596],
        'perimeter_on_border': [1., 1., 3., 2., 3.],
        'perimeter_on_border_ratio': [0.21845337, 0.21845337, 0.65536011, 0.23895223, 0.65536011],
        'principal_axes0': [1., 1., 1., 0., 0.],
        'principal_axes1': [0., 0., 0., 1., 1.],
        'principal_axes2': [ 0.,  0.,  0., -1., -1.],
        'principal_axes3': [ 1.,  1.,  1., -0., -0.],
        'principal_moments0': [0., 0., 0., 0., 0.],
        'principal_moments1': [0.25, 0.25, 0.25, 1.25, 0.25]
    }

def test_statistics():
    from napari_simpleitk_image_processing import label_statistics

    image = np.asarray([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [4, 4, 4, 4],
        [5, 5, 0, 0]
    ])
    labels = image

    result = label_statistics(image, labels, None, size=True, intensity=True, perimeter=True, shape=True, position=True, moments=True )

    print(result)

    for k, v in result.items():
        assert np.allclose(result[k], reference[k], 0.001)

    for k, v in reference.items():
        assert np.allclose(result[k], reference[k], 0.001)

def test_statistics_with_viewer(make_napari_viewer):
    from napari_simpleitk_image_processing import label_statistics

    viewer = make_napari_viewer()

    image = np.asarray([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [4, 4, 4, 4],
        [5, 5, 0, 0]
    ])
    labels = image

    labels_layer = viewer.add_labels(labels, name="test1")

    label_statistics(image, labels, viewer, size=True, intensity=True, perimeter=True, shape=True, position=True, moments=True )

    result = labels_layer.properties
    print(result)

    for k, v in result.items():
        assert np.allclose(result[k], reference[k], 0.001)

    for k, v in reference.items():
        assert np.allclose(result[k], reference[k], 0.001)



def test_napari_api():
    from napari_simpleitk_image_processing import napari_experimental_provide_function
    napari_experimental_provide_function()
