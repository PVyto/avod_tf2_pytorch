experiment_name = 'experiment_test'
channels = 8
default_config = {
    'mode': 'train',
    'operation_config': {
        'device': 'cpu',
        # dataloader parameters
        'workers': 1,
        'pin_mem': False,
        'shuffle': False,

        'epochs': 30,
        'evaluation_interval': 1,
        'gradient_accumulations': 1,
        'mixed_precision': False,  # gpu support only
        # whether to apply gradient clipping or not
        'gradient_clipping': False,
        'gradient_clipping_norm': 1.0,
        'batch_size': 1,
        # optimizer by name of class; must be one of the available under pytorch
        # https://pytorch.org/docs/stable/optim.html
        'optimizer': 'Adam',  # SGD, Adagrad, AdamW, Adam
        'optimizer_params': {
            'lr': {
                # value refers to that of the initial learning rate
                'value': 0.001,
                # learning rate scheduler
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
            # weight_decay adds an l2 penalty term to the loss function
            # https://pytorch.org/docs/stable/optim.html
            'weight_decay': 0.0005,

        },
        # whether to use a weight initialization method replacing the default one
        'override_default_weight_init_method': False,
        'weight_initialization': {'method': 'kaiming_uniform', 'params': {'nonlinearity': 'relu', 'mode': 'fan_out'}}
        #     {'nonlinearity': 'relu', 'mode': 'fan_in'}

    },
    'global': {
        'area_extents': [-40, 40, -5, 3, 0, 70]
    },
    'dataset_config': {
        # name of the dataset class
        'name': 'KITTIDataset',
        # if this list remains empty the anchor sizes will be determined by executing KMeans
        'experiment': experiment_name,
        # path to the data
        'path': '~/Kitti/object',
        # 'path': '~/Kitti/object',
        'folder_mapping': {  # This dictionary  indicates which folder contains the data for each mode
            'train': 'training',
            'test': 'testing',
            'val': 'validating'
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
        # size of the image that the network will be fed with
        'image_size': [360, 1200],
        'difficulty': [0, 1, 2],
        'classes': ['Car'],
        # parameters for the anchor genarator
        'num_anchors': [2],
        # or clusters, indicates the number of anchors that will represent each class with each anchor
        # having  a different size that is the result of clustering. Thus the size of this list must be the same
        # as the above i,e, classes
        'anchor_stride': [0.5, 0.5],
        'anchor_sizes': [3.514, 1.581, 1.511, 4.236, 1.653, 1.547],
        'mini_batch_size': 512,

        # a list that is going to be reshaped to a (3,2) array
        # TODO: change the name transformation with a more expressive one
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

        'transform': ['PCAJitterTransform', 'FlipTransform', 'ResizeTransform', 'MeanNormalize', 'ToTensor']

    },
    'model_config': {
        'name': 'AVOD',
        'load': 'last',  # last or best model
        'checkpoint_name': experiment_name,
        'checkpoints_dir': 'checkpoints',  
        'best_models_dir': 'best_models',  
        'feature_extractor': {
            'img_input_channels': 3,
            'bev_input_channels': 6,
            'out_channels': channels
        },

        'rpn': {
            # parameters for the first 'part' of the region proposal network
            'input_channels': channels,  # 32,
            'roi_size': 3,
            'predictors': {
                'args': 'common',
                'type': 'linear',
                'input_channels': 1,
                'headers': ['cls', 'reg'],
                'headers_batch_norm': 3 * [False],
                'headers_dropout': 2 * [True] + [False],
                'headers_dropout_probability': 2 * [0.5],

                'cls': [32, 32, 2],
                'reg': [32, 32, 6],
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
            'predictors': {
                'input_channels': channels,  # 32,
                'type': 'linear',
                'args': 'common',
                'base': [2048, 2048, 2048],
                'base_norm': 3 * [False],
                'base_dropout': 3 * [True],
                'base_dropout_probability': 3 * [0.5],
                'base_activations': 3 * ['relu'],

                'headers': ['cls', 'reg', 'ang'],
                'headers_dropout': [False],
                'headers_batch_norm': [False],

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

dataset_config = {
    'name': 'KITTIDataset',
    # if this list remains empty the anchor sizes will be determined by executing KMeans
    'experiment': 'experiment_test',
    'path': '~/Kitti/object',  # '~/Kitti/object',
    # 'path': '/Kitti/object',
    'folder_mapping': {  # This dictionary  indicates which folder contains the data for each mode
        'train': 'training',
        'test': 'testing',
        'val': 'training'
    },
    'mode': 'train',
    'annotation_train': 'train',
    'annotation_test': 'test',
    'annotation_val': 'val',
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
    'anchor_sizes': [3.514, 1.581, 1.511, 4.236, 1.653, 1.547],
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
    'transform': ['ResizeTransform', 'MeanNormalize', 'ToTfTensor']


}