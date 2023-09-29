import os
import logging
import datetime
import tensorflow as tf
# from configs.cfg_people import return_cfg
# from configs.cfg_conv_full import return_cfg
from configs.cfg_conv import return_cfg
from models.main_models_tf import Avod
from trainer_helper_functions import load_checkpoint
from trainer_helper_functions import fit, evaluate, fit_from_torch_dataloader, fit_v2, evaluate_from_torch_dataloader
from trainer_helper_functions import get_tf_train_dataloader, get_tf_val_dataloader, get_torch_train_dataloader, \
    get_torch_val_dataloader, evaluate_v3

# create_dataloader = create_dataloader_v2

tf.get_logger().setLevel(logging.ERROR)

if __name__ == '__main__':
    # experiment_name = 'experiment_0002'
    # experiment_name = 'people_experiment_0003'
    # experiment_name = 'conv_experiment_0001'
    experiment_name = 'experiment_1'

    cfg = return_cfg(channels=32, path_drop=[.9, .9])
    dataset_config = cfg['dataset_config']
    dataset_config['annotation_train'] = 'train'
    dataset_config['annotation_val'] = 'val'
    dataset_config['experiment'] = experiment_name

    dataset_config['transform'] = ['FlipTransform', 'PCAJitterTransform',
                                   'ResizeTransform', 'MeanNormalize', 'ToTfTensor']

    model = Avod(config=cfg['model_config'], rpn_input_dims=[[360, 1200], [700, 800]])

    # train_loader, train_ds = get_tf_train_dataloader(dataset_config)
    # valid_loader, valid_ds = get_tf_val_dataloader(dataset_config)
    train_loader_torch, train_ds_torch = get_torch_train_dataloader(dataset_config)
    valid_loader_torch, valid_ds_torch = get_torch_val_dataloader(dataset_config)

    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=30000,
                                                                  decay_rate=0.8)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, experiment_name + "-ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, checkpoint_name=experiment_name + "-ckpt",
                                         max_to_keep=100)
    log_dir = "logs/"
    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # load_checkpoint(checkpoint, manager, load='specific', checkpoint_num=8)
    # print(optimizer.iterations)

    try:
        fit_from_torch_dataloader(model=model, dataset_loader=train_loader_torch, optimizer=optimizer,
                                  clip_gradients=True,
                                  starting_epoch=0, epochs=34, checkpoint=checkpoint,
                                  checkpoint_prefix=checkpoint_prefix,
                                  evaluation_interval=2000, evaluation_fn=evaluate_from_torch_dataloader,
                                  evaluation_fn_params={'model': model, 'valid_loader': valid_loader_torch,
                                                        'valid_ds': valid_ds_torch,
                                                        'experiment_name': experiment_name})
        os.system("sleep {};echo 'done sleeping'".format(660))
        os.system('sudo shutdown')
    except Exception as e:
        print(str(e))
        os.system("echo 'shutting down';sudo shutdown")


