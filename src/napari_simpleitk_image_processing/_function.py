from napari_plugin_engine import napari_hook_implementation

@napari_hook_implementation
def napari_experimental_provide_function():
    from ._simpleitk_image_processing import median_filter, gaussian_blur, threshold_otsu, \
        signed_maurer_distance_map, morphological_watershed, connected_component_labeling, \
        touching_objects_labeling, watershed_otsu_labeling, binary_fill_holes

    return [median_filter,
            gaussian_blur,
            threshold_otsu,
            signed_maurer_distance_map,
            morphological_watershed,
            connected_component_labeling,
            touching_objects_labeling,
            watershed_otsu_labeling,
            binary_fill_holes]

