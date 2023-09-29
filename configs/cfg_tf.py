def return_cfg(**kwargs):
    lr = kwargs.get('lr', 1e-4)
    epochs = kwargs.get('epochs', 30)
    step_size = kwargs.get('step_size', 10)
    evaluation_interval = kwargs.get('evaluation_interval', 2000)
    gradient_accumulations = kwargs.get('gradient_accumulations', 1)
    experiment_name = kwargs.get('experiment_name', 'experiment_0000')
    channels = kwargs.get('channels', 32)
    data_path = kwargs.get('data_path', '~/Kitti/object')
    load = kwargs.get('load', 'last')
    workers = kwargs.get('workers', 0)
    val_annotation = kwargs.get('val', 'val')
    train_annotation = kwargs.get('train', 'train')

    clip_gradient = kwargs.get('clip_gradient', True)
    gradient_clipping_norm = kwargs.get('gradient_clipping_norm', 1.0)

    optimizer = kwargs.get('optimizer', 'Adam')

    weight_decay = kwargs.get('weight_decay', 0.0005)
    weight_decay_rpn = kwargs.get('weight_decay_rpn', 0.005)
    weight_decay_ssd = kwargs.get('weight_decay_ssd', 0.005)

    apply_weight_init = kwargs.get('override_default_weight_init_method', False)
    weight_initialization_method = kwargs.get('weight_initialization_method', 'xavier_uniform')

    batch_normalization = kwargs.get('enable_batch_norm', True)
    activation_function_fe = kwargs.get('activation_function_fe', 'relu')
    activation_function_rpn = kwargs.get('activation_function_rpn', 'relu')
    activation_function_ssd = kwargs.get('activation_function_ssd', 'relu')

    rpn_detector_type = kwargs.get('rpn_type', 'linear')
    ssd_type = kwargs.get('ssd_type', 'linear')
    rpn_header_cls = kwargs.get('rpn_header_cls', [])
    rpn_header_reg = kwargs.get('rpn_header_reg', [])
    ssd_base = kwargs.get('ssd_base', [])
    path_drop = kwargs.get('path_drop', [1., 1.])
    roi_function = kwargs.get('roi_function', 'roi_align')
    dataset_transform = kwargs.get('dataset_transform',
                                   ['FlipTransform', 'PCAJitterTransform', 'ResizeTransform', 'MeanNormalize',
                                    'ToTfTensor'])

    parameter_extraction_method = kwargs.get('parameter_extraction_method', 'default')
    cls_header = [32, 32, 2]
    reg_header = [32, 32, 6]

    if channels == 32:
        cls_header = [256, 256, 2]
        reg_header = [256, 256, 6]
    if len(rpn_header_cls) > 0:
        cls_header = rpn_header_cls
    if len(rpn_header_reg) > 0:
        reg_header = rpn_header_reg
    if len(ssd_base) == 0:
        ssd_base = [2048, 2048, 2048]

    device = kwargs.get('device', 'cpu')
    device_str = 'cpu'
    if device == 'gpu':
        device_str = 'cuda:0'
    mixed_precision = kwargs.get('mixed_precision', False)

    default_config = {
        'mode': 'train',
        'operation_config': {
            'device': device_str,
            'workers': workers,
            'pin_mem': True,
            'shuffle': True,
            'epochs': epochs,
            'evaluation_interval': evaluation_interval,
            'gradient_accumulations': gradient_accumulations,
            'mixed_precision': mixed_precision,  # gpu support only
            'gradient_clipping': clip_gradient,
            'gradient_clipping_norm': gradient_clipping_norm,
            'batch_size': 1,
            'optimizer': optimizer,  # SGD, Adagrad, AdamW, Adam
            'optimizer_params': {
                'lr': {
                    'value': lr,
                    # steady, StepLR, ExponentialLR, ExponentialDecayLR, CosineAnnealingLR, CyclicLR
                    'type': 'ExponentialDecay',
                    'params': {
                        # 'gamma': 0.8,
                        # 'step_size': 1,  # 30000
                        # 'verbose': False,
                        # 'step_size': step_size,
                        'decay_steps': 30000,
                        'decay_rate': 0.8
                    },
                    # decay_factor
                    # 'decay_steps': 3000
                },  # learning rate
                # other optimizer parameters

                'model_parameter_extraction_method': parameter_extraction_method,
                'weight_decay_dict': {'model': weight_decay, 'feature_extractors': weight_decay,
                                      'rpn': weight_decay_rpn,
                                      'second_stage_detector': weight_decay_ssd},
                #     0.0005 for conv and 0.005 for fc

            },
            'override_default_weight_init_method': apply_weight_init,
            # 'weight_initialization': {'method': 'kaiming_uniform',
            #                           'params': {'nonlinearity': 'relu', 'mode': 'fan_out'}, 'bias': 'zeros_'},
            'weight_initialization': {'method': weight_initialization_method,
                                      'params': {}, 'bias': 'zeros'}

        },
        'global': {
            'area_extents': [-40, 40, -5, 3, 0, 70]
        },
        'dataset_config': {
            'name': 'KITTIDatasetTf',
            'iou_neg_range': [0.0, 0.3],
            'iou_pos_range': [0.5, 1.0],
            'experiment': experiment_name,
            'path': data_path,  # '~/Kitti/object', '/content/gdrive/MyDrive/MP/Kitti/object'
            # 'path': '/Kitti/object',
            'folder_mapping': {  # This dictionary  indicates which folder contains the data for each mode
                'train': 'training',
                'test': 'testing',
                'val': 'training'  # 'validating'
            },
            'mode': 'train',
            'annotation_train': train_annotation,
            'annotation_test': 'test',
            'annotation_val': val_annotation,
            'image_path': 'image_2',
            'point_cloud_path': 'velodyne',
            'label_path': 'label_2',
            'calibration_path': 'calib',
            'plane_path': 'planes',
            'image_size': [360, 1200],
            'difficulty': [0, 1, 2],
            # 'classes': ['Pedestrian', 'Cyclist'],
            'classes': ['Car'],
            'num_anchors': [2],
            # or clusters, indicates the number of anchors that will represent each class with each anchor
            # having  a different size that is the result of clustering. Thus the size of this list must be the same
            # as the above i,e, classes
            'anchor_stride': [0.5, 0.5],
            'anchor_sizes': [3.514, 1.581, 1.511, 4.236, 1.653, 1.547], # Car
            # 'anchor_sizes': [0.818, 0.628, 1.768, 1.771, 0.57, 1.723], #  Pedestrian & Cyclist
            'mini_batch_size': 512,

            # a list that is going to be reshaped to a (3,2) array

            'transformation': {
                'area_extents': [-40, 40, -5, 3, 0, 70],
                'voxel_size': 0.10000000149011612,
                'height_lo': -0.20000000298023224,
                'height_hi': 2.299999952316284,
                'num_slices': 5,
            },
            # 'transform': ['PCAJitterTransform', 'FlipTransform', 'ResizeTransform', 'MeanNormalize', 'ToTensor']
            'transform': dataset_transform,

            # 'PCAJitterTransform', 'FlipTransform',

        },
        'model_config': {
            'name': 'Avod',  # AVOD2
            'load': load,  # last or best model
            'checkpoint_num': 1,
            'checkpoint_name': experiment_name,  # 'test'
            'checkpoints_dir': 'checkpoints',
            'best_models_dir': 'best_models',
            'path_drop': path_drop,
            'feature_extractors': {
                'img_input_channels': 3,
                'bev_input_channels': 6,
                'out_channels': channels,
                'activation_function': activation_function_fe,
                'batch_normalization': batch_normalization,
                # 'use_bias':use_bias,
            },

            'rpn': {
                'batch_normalization': batch_normalization,
                'input_channels': channels,  # 32,
                'roi_size': 3,
                'roi_layer': roi_function,
                'predictors': {
                    'type': rpn_detector_type,
                    'args': 'common',
                    'headers': ['cls', 'reg'],
                    'input_channels': 1,
                    'headers_kernel_size': [3, 1, 1],
                    'headers_batch_norm': 3 * [False],
                    'headers_dropout': 2 * [True] + [False],
                    'headers_dropout_probability': 2 * [0.5],
                    'cls': cls_header,
                    'reg': reg_header,
                    'headers_activations': {'cls': 2 * [activation_function_rpn], 'reg': 2 * [activation_function_rpn]}
                },
                'iou_neg_range': [0.0, 0.3],
                'iou_pos_range': [0.5, 1.0],
                'nms_size_train': 1024,
                'nms_size_test': 300, # People: 1024, Car: 300
                'nms_threshold': 0.8,
                'mini_batch_size': 512,
                'regression_weight': 5.0,
                'classification_weight': 1.0

            },
            'second_stage_detector': {
                'num_classes': 2,
                'roi_size': 7,
                'roi_layer': roi_function,
                'predictors': {
                    'base': ssd_base,
                    'type': ssd_type,
                    'input_channels': channels,  # 32,
                    'base_activations': 3 * [activation_function_ssd],
                    'base_kernel_size': 3 * [3],
                    'base_batch_norm': 3 * [False],
                    'base_dropout': 3 * [True],
                    'base_dropout_probability': 3 * [0.5],
                    'headers': ['cls', 'reg', 'ang'],
                    'headers_kernel_size': [1],
                    'headers_batch_norm': [False],
                    'headers_dropout': [False],
                    'cls': [2],
                    'reg': [10],
                    'ang': [2],
                },

                'iou_neg_range': [0.0, 0.55],
                'iou_pos_range': [0.65, 1.0],
                'nms_size': 100,
                'nms_threshold': 0.01,
                'mini_batch_size': 1024,
                'regression_weight': 5.0,
                'orientation_weight': 1.0,
                'classification_weight': 1.0,

            }
        }
    }

    return default_config
