Segmentation Model Configuration
================================

.. autoclass:: InnerEye.ML.config.SegmentationModelBase

   .. autoattribute:: activation_map_layers
   .. autoattribute:: architecture
   .. autoattribute:: loss_type
   .. autoattribute:: mixture_loss_components
   .. autoattribute:: loss_class_weight_power
   .. autoattribute:: focal_loss_gamma
   .. autoattribute:: dataset_expected_spacing_xyz
   .. autoattribute:: feature_channels
   .. autoattribute:: kernel_size
   .. autoattribute:: crop_size
   .. autoattribute:: image_channels
   .. autoattribute:: ground_truth_ids
   .. autoattribute:: mask_id
   .. autoattribute:: norm_method
   .. autoattribute:: window
   .. autoattribute:: level
   .. autoattribute:: output_range
   .. autoattribute:: debug_mode
   .. autoattribute:: tail
   .. autoattribute:: sharpen
   .. autoattribute:: trim_percentiles
   .. autoattribute:: padding_mode
   .. autoattribute:: inference_batch_size
   .. autoattribute:: test_crop_size
   .. autoattribute:: class_weights
   .. autoattribute:: ensemble_aggregation_type
   .. autoattribute:: posterior_smoothing_mm
   .. autoattribute:: store_dataset_sample
   .. autoattribute:: comparison_blob_storage_paths
   .. autoattribute:: slice_exclusion_rules
   .. autoattribute:: summed_probability_rules
   .. autoattribute:: disable_extra_postprocessing
   .. autoattribute:: ground_truth_ids_display_names
   .. autoattribute:: col_type_converters
   .. autoattribute:: is_plotting_enabled

.. automodule:: InnerEye.ML.config
   :members:
   :undoc-members:
   :exclude-members: SegmentationModelBase
