import os
import logging
import datetime
import tensorflow as tf
from configs.cfg_people import return_cfg
from configs.cfg_conv import return_cfg
from configs.cfg_conv_full import return_cfg
# from configs.cfg import return_cfg
from models.main_models_tf import Avod
from trainer_helper_functions import fit_from_torch_dataloader, evaluate_from_torch_dataloader
from trainer_helper_functions import get_torch_train_dataloader, get_torch_val_dataloader
from tensorflow import summary

tf.get_logger().setLevel(logging.ERROR)
# create_dataloader = create_dataloader_v2

if __name__ == '__main__':
    summary_writer = None
    cfg = return_cfg(channels=32, path_drop=[.9, .9])
    dataset_config = cfg['dataset_config']
    dataset_config['path'] = '~/Kitti/object'

    experiment_name = 'experiment_1'

    dataset_config['annotation_train'] = 'train_all'
    dataset_config['annotation_val'] = 'val_all'

    dataset_config['experiment'] = experiment_name
    dataset_config['transform'] = ['FlipTransform', 'PCAJitterTransform',
                                   'ResizeTransform', 'MeanNormalize', 'ToTfTensor']
    # dataset_config['transform'] = ['ResizeTransform', 'MeanNormalize', 'ToTfTensor']

    model = Avod(config=cfg['model_config'], rpn_input_dims=[[360, 1200], [700, 800]])



    train_loader_torch, train_ds_torch = get_torch_train_dataloader(dataset_config)
    valid_loader_torch, valid_ds_torch = get_torch_val_dataloader(dataset_config)


    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=30000,
                                                                  decay_rate=0.8)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, experiment_name + "-ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, checkpoint_name=experiment_name + "-ckpt",
                                         max_to_keep=100)  # , checkpoint_interval=5, step_counter=tf.Variable(2))
    log_dir = "new_logs/"
    summary_writer = summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #

    fit_from_torch_dataloader(model=model, dataset_loader=train_loader_torch, optimizer=optimizer, clip_gradients=True,
                              starting_epoch=0, summary_writer=summary_writer,
                              epochs=30, checkpoint=checkpoint, checkpoint_prefix=checkpoint_prefix,
                              evaluation_interval=2000, evaluation_fn=evaluate_from_torch_dataloader,
                              evaluation_fn_params={'model': model, 'valid_loader': valid_loader_torch,
                                                    'valid_ds': valid_ds_torch,
                                                    'experiment_name': experiment_name})


