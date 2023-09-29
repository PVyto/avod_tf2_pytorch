default_config = {
    'mode': 'train',
    'operation_config': {
        'device': 'cuda:0',
        'workers': 4,
        'pin_mem': True,
        'shuffle': True,
        'epochs': 30,
        'evaluation_interval': 1,
        'gradient_accumulations': 64,
        'mixed_precision': False,  # gpu support only
        'gradient_clipping': False,
        'gradient_clipping_norm': 1.0,
        'batch_size': 1,
        'optimizer': 'Adam',  # SGD, Adagrad, AdamW, Adam
        'optimizer_params': {
            'lr': {
                # value refers to that of the initial learning rate
                'value': 0.0001,
                # type can take one of the following values:
                # steady, StepLR, ExponentialLR, ExponentialDecayLR, CosineAnnealingLR, CyclicLR
                'type': 'StepLR',
                'params': {
                    'gamma': 0.8,
                    # 'step_size': 1,  # 30000
                    'verbose': False,
                    'step_size': 10,
                },
                # decay_factor
                # 'decay_steps': 3000
            },  # learning rate
            # other optimizer parameters
            'weight_decay': 0.0005,

        },
        'override_default_weight_init_method': False,
        'weight_initialization': {'method': 'kaiming_uniform', 'params': {'nonlinearity': 'relu', 'mode': 'fan_out'}}
        #     {'nonlinearity': 'relu', 'mode': 'fan_in'}

    },
    'global': {
        'area_extents': [-40, 40, -5, 3, 0, 70]
    },
    'dataset_config': {
        'name': 'KITTIDataset',
        # if this list remains empty the anchor sizes will be determined by executing KMeans
        'experiment': 'experiment00010',

        'path': '~/Kitti/object',
        'folder_mapping': {  # This dictionary  indicates which folder contains the data for each mode
            'train': 'training',
            'test': 'testing',
            'val': 'training'
        },
        'mode': 'train',
        'annotation_train': 'train_10',
        'annotation_test': 'test',
        'annotation_val': 'val_10',
        'image_path': 'image_2',
        'point_cloud_path': 'velodyne',
        'label_path': 'label_2',
        'calibration_path': 'calib',
        'plane_path': 'planes',
        'image_size': [360, 1200],
        'difficulty': [0, 1, 2],
        'classes': ['Car'],
        'num_anchors': [2],
        # or clusters, indicates the number of anchors that will represent each class with each anchor
        # having  a different size that is the result of clustering. Thus the size of this list must be the same
        # as the above i,e, classes
        'anchor_stride': [0.5, 0.5],
        'anchor_sizes': [4.236, 1.653, 1.547, 3.514, 1.581, 1.511],
        'mini_batch_size': 512,

        # a list that is going to be reshaped to a (3,2) array

        'transformation': {
            'area_extents': [-40, 40, -5, 3, 0, 70],
            # 'name': 'BevSlicesCreator',
            # 'parameters': {  # parameters for the point cloud transformation i.e. the production of BEV maps
            'voxel_size': 0.10000000149011612,
            'height_lo': -0.20000000298023224,
            'height_hi': 2.299999952316284,
            'num_slices': 5,
            # }
        },
        'transform': ['PCAJitterTransform', 'ToTensor']

    },
    'model_config': {
        'name': 'AVOD',
        'load': 'last',  # last or best model
        'checkpoint_name': 'experiment00010',
        'checkpoints_dir': 'checkpoints',  
        'best_models_dir': 'best_models',  
        'feature_extractor': {
            'img_input_channels': 3,
            'bev_input_channels': 6,
            'out_channels': 32
        },

        'rpn': {
            'input_channels': 32,  # 32,
            'roi_size': 3,
            'fc': {
                'args': 'common',
                'headers': ['cls', 'reg'],
                # the base that follows is just for testing
                # 'base': [32],
                # 'base_activations': ['relu'],
                # 'base_kernel_size': [1],
                # 'base_batch_norm': [True],
                'type': 'conv',
                'input_channels': 1,
                'headers_kernel_size': [3, 1, 1],
                'headers_batch_norm': 3 * [False],
                'headers_dropout': 2 * [True] + [False],
                'headers_dropout_probability': 2 * [0.5],
                'cls': [256, 256, 2],
                'reg': [256, 256, 6],
                'headers_activations': {'cls': 2 * ['relu'], 'reg': 2 * ['relu']}
            },
            'iou_neg_range': [0.0, 0.3],
            'iou_pos_range': [0.5, 1.0],
            'nms_size': 1024,
            'nms_threshold': 0.8,
            'mini_batch_size': 512,
            'regression_weight': 5.0,
            'classification_weight': 1.0

        },
        'second_stage_detector': {
            'roi_size': 7,
            'fc': {
                'input_channels': 32,
                'args': 'common',
                'base': [2048, 2048, 2048],
                'base_activations': 3 * ['relu'],
                'base_norm': 3 * [False],
                'headers': ['cls', 'reg', 'ang'],
                'base_dropout': 3 * [True],
                'base_dropout_probability': 3 * [0.5],
                'type': 'linear',
                'cls': [2],
                'reg': [10],
                'ang': [2],
                # 'headers_activations': {'cls': ['softmax']}
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
