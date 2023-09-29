from importlib import import_module


def get_available_options_from_module(module):
    _module = import_module(module)
    return {k[:-1] if k.endswith('_') else k: getattr(_module, k) for k in getattr(_module, '__all__')}


class ConfigRPN:
    _rpn_key_ = 'rpn'
    _input_channels_key_ = 'input_channels'
    _roi_size_key_ = 'roi_size'
    _roi_type_key_ = 'roi_layer'
    _nms_threshold_key_ = 'nms_threshold'
    _nms_size_key_ = 'nms_size'
    _predictors_key_ = 'predictors'
    _mini_batch_size_key_ = 'mini_batch_size'
    _regression_weight_key_ = 'regression_weight'
    _classification_weight_key_ = 'classification_weight'
    _second_stage_detector_key_ = 'second_stage_detector'
    pass


class ConfigSecondStageDetector:
    _roi_size_key_ = 'roi_size'
    _roi_type_key_ = 'roi_layer'
    _nms_threshold_key_ = 'nms_threshold'
    _nms_size_key_ = 'nms_size'
    _mini_batch_size_key_ = 'mini_batch_size'
    _predictors_key_ = 'predictors'
    _localization_weight_key_ = 'regression_weight'
    _classification_weight_key_ = 'classification_weight'
    _orientation_weight_key_ = 'orientation_weight'
    pass


class ConfigFeatureExtractors:
    pass


class ConfigDataset:
    _experiment_ = 'experiment'

    _mode_ = 'mode'
    _base_dir_ = 'path'
    _image_dir_ = 'image_path'
    _label_dir_ = 'label_path'
    _planes_dir_ = 'plane_path'
    _annotation_prefix_ = 'annotation'
    _folder_mapping_ = 'folder_mapping'
    _point_cloud_dir_ = 'point_cloud_path'
    _calibration_dir_ = 'calibration_path'

    _classes_ = 'classes'
    _image_size_ = 'image_size'
    _difficulty_ = 'difficulty'
    _anchor_sizes_ = 'anchor_sizes'
    _num_clusters_ = 'num_anchors'
    _area_extents_ = 'area_extents'
    _anchor_stride_ = 'anchor_stride'
    _mini_batch_size_key_ = 'mini_batch_size'

    _transformation_config_ = 'transformation'
    _image_transform_config_ = 'transform'
    pass


class ConfigTrainer:
    _available_models = get_available_options_from_module('models')
    _datasets = get_available_options_from_module('data_utils.datasets')
    _transforms = get_available_options_from_module('data_utils.transforms')
    _weight_initialization_methods = get_available_options_from_module('to_be_renamed_later.weight_initializers')
    _lr_schedulers = get_available_options_from_module('to_be_renamed_later.learning_rate_schedulers')
    _optimizers = get_available_options_from_module('to_be_renamed_later.optimizers')

    _model_param_extraction_method_key_ = 'model_parameter_extraction_method'

    _available_modes = ['train', 'test', 'validation']
    _mode_key_ = 'mode'
    _dataset_type_key_ = 'name'
    _model_config_key_ = 'model_config'
    _dataset_config_key_ = 'dataset_config'
    _mode_config_params_key_ = 'operation_config'

    _optimizer_config_key_ = 'optimizer'
    _name_key_ = 'name'
    _epochs_key_ = 'epochs'
    _batch_size_key_ = 'batch_size'
    _gradient_clipping_ = 'gradient_clipping'
    _gradient_clipping_value_ = 'gradient_clipping_norm'
    _learning_rate_key_ = 'lr'
    _mixed_precision_key_ = 'mixed_precision'
    _learning_rate_type_key_ = 'type'
    _learning_rate_value_key_ = 'value'
    _learning_rate_scheduler_params_key_ = 'params'
    _override_default_weight_init_method_ = 'override_default_weight_init_method'
    _bias_initialization_cfg_ = 'bias'
    _weight_initialization_cfg_ = 'weight_initialization'
    _weight_initialization_method_ = 'method'
    _weight_initialization_method_params_ = 'params'

    _checkpoint_directory_key_ = 'checkpoints_dir'
    _checkpoint_name_key_ = 'checkpoint_name'
    _best_models_dir_key_ = 'best_models_dir'
    _load_key_ = 'load'

    _evaluation_interval_key_ = 'evaluation_interval'
    _gradient_accumulations_key_ = 'gradient_accumulations'
    _device_key_ = 'device'

    _optimizer_params_key_ = 'optimizer_params'


# ConfigAVOD = ConfigFEs + ConfigRPN + ConfigSSD
# ConfigExperiment = ConfigTrainer + ConfigAVOD
class ConfigAVOD:
    pass


# MAYBE CREATE  A DYNAMIC ATTRIBUTE LIST/DICT FOR THE TRAINER CLASS
if __name__ == '__main__':
    my_cfg = ConfigAVOD()
    print()
