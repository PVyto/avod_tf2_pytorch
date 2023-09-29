import os
import datetime
import tensorflow as tf

from tqdm import tqdm
from pathlib import Path
from .trainer_base import Trainer
from tensorflow import clip_by_norm
from torch.utils.data import DataLoader
from general_utils import get_available_options_from_module
from trainer.trainer_helper_functions import calculate_map_async, call_evaluation_script
from trainer.trainer_helper_functions import predictions_to_kitti_tf as predictions_to_kitti


class TrainerTf(Trainer):
    _backend = 'tf'
    _models = get_available_options_from_module('models.main_models_tf')
    _datasets = get_available_options_from_module('data_utils.datasets_tf')

    _optimizers = get_available_options_from_module('optimizers.optimizers_tf')
    _schedulers = get_available_options_from_module('learning_rate_schedulers.lr_schedulers_tf')

    # _model_parameters_methods = {'default': return_model_parameters, 'custom': custom_model_parameters}

    def _initialize_checkpoint_paths(self):

        super(TrainerTf, self)._initialize_checkpoint_paths()

        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir,
                                                  checkpoint_name=self.checkpoint_name,
                                                  max_to_keep=100)

    def _load_from_checkpoint(self):
        load = self.model_config.get(self._load_key_)
        checkpoint_num = self.model_config.get(self._checkpoint_num_, 0)

        import glob
        if load == 'last':
            print('Restoring  checkpoint with name: {}'.format(self.manager.latest_checkpoint))
            self._restore_checkpoint(self.manager.latest_checkpoint)
        elif load == 'specific':
            checkpoints = sorted([i[:-6] for i in glob.glob(self.manager._checkpoint_prefix + '*.index')])
            checkpoint_dict = {i.split('-')[-1]: i for i in checkpoints}
            print('Available checkpoints : ', checkpoint_dict)
            checkpoint_to_restore = checkpoint_dict.get(str(checkpoint_num))
            if checkpoint_to_restore:
                print('Restoring  checkpoint with name: {}'.format(checkpoint_to_restore))
                self._restore_checkpoint(checkpoint_to_restore)
            else:
                print('the checkpoint number that was provided does not exist')
        else:
            checkpoints = sorted([i[:-6] for i in glob.glob(self.manager._checkpoint_prefix + '*.index')])
            if len(checkpoints) < checkpoint_num - 1:
                checkpoint_num = len(checkpoints) - 1
            print('Available checkpoints : ', checkpoints)
            if not checkpoints:
                return
            print('Restoring  checkpoint with name: {}'.format(checkpoints[checkpoint_num]))
            self._restore_checkpoint(checkpoints[checkpoint_num])

    def _save_checkpoint(self):
        self.checkpoint.save(self.checkpoint_prefix)

    def _restore_checkpoint(self, path):
        self.checkpoint.restore(path)

    def _initialize_grad_scaler(self):
        pass


    def _load_config_file(self, config):
        pass

    # TODO: move to base class
    def _initialize_trainer(self):

        self._initialize_basic_parameters()

        self._extract_training_parameters()

        self.dataset = self._datasets[self.dataset_config.get(self._dataset_type_key_)]

        self._initialize_grad_scaler()

        self._prepare_optimizer_params()

        self._initialize_model()

        self._initialize_optimizer()

        self._initialize_checkpoint_paths()

        self._initialize_paths()

        self._load_from_checkpoint()

        if self.mode not in self._available_modes:
            raise Exception(
                'Mode parameter must take one of the following values: %s but %s' % (self._available_modes, self.mode))

        self.dataset = self._datasets[self.dataset_config.get(self._dataset_type_key_)]

        self._create_dataloaders()

        self._create_checkpoints_to_evaluate_file()
        # self._create_loggers_()
        self._create_result_folders_()

        self._save_config()

    def _create_dataloaders(self):
        self._train_dataset = self.dataset(mode='train', config=self.dataset_config)
        self._valid_dataset = self.dataset(mode='val', config=self.dataset_config)

        workers = 0
        shuffle = self.operation_parameters.get('shuffle')
        pin_mem = self.operation_parameters.get('pin_memory')

        self.train_loader = DataLoader(self._train_dataset, batch_size=self.batch_size,
                                       num_workers=workers,
                                       shuffle=shuffle,
                                       collate_fn=self._train_dataset.collate_fn, pin_memory=pin_mem)
        self.valid_loader = DataLoader(self._valid_dataset, batch_size=1,
                                       num_workers=workers,
                                       shuffle=False,
                                       collate_fn=self._valid_dataset.collate_fn, pin_memory=pin_mem)

    def _prepare_optimizer_params(self):
        self.optimizer_name = self.operation_parameters.get(self._optimizer_config_key_)
        self.optimizer_class = self._optimizers[self.optimizer_name]
        self.optimizer_params = self.operation_parameters.get(self._optimizer_params_key_)
        learning_rate_param = self.optimizer_params.get(self._learning_rate_key_)
        lr_type = learning_rate_param.get(self._learning_rate_type_key_)
        scheduler_params = learning_rate_param[self._learning_rate_scheduler_params_key_]
        self.optimizer_params_dict = {
            'initial_learning_rate': learning_rate_param.get(self._learning_rate_value_key_)}
        for param, v in self.optimizer_params.items():
            if param == 'lr' or 'weight_decay_dict':
                continue
            self.optimizer_params_dict[param] = v
        scheduler_params.update(self.optimizer_params_dict)
        self.scheduler_params = scheduler_params
        self.learning_rate_scheduler_class = self._schedulers[lr_type]

    def _initialize_optimizer(self):
        self.learning_rate_scheduler = self.learning_rate_scheduler_class(**self.scheduler_params)
        self.optimizer = self.optimizer_class(learning_rate=self.learning_rate_scheduler)

    def _initialize_model(self):
        self.model = self._models[self.model_config.get(self._name_key_)](self.model_config,
                                                                          rpn_input_dims=[[360, 1200], [700, 800]])

    def _calculate_gradients(self, tape, loss):
        return tape.gradient(loss, self.model.trainable_variables)

    def _clip_gradients(self, _gradients):
        if self.gradient_clipping:
            _gradients = [clip_by_norm(g, 1.) if g is not None else None for g in _gradients]
        return _gradients

    def _apply_gradients(self, _gradients):

        self.optimizer.apply_gradients(zip(_gradients, self.model.trainable_variables))

    def _train_step(self, input_sample, _gradient_list):
        self.model.train()
        with tf.GradientTape() as tape:
            *output, loss = self.model(*input_sample[:-1], enable_path_drop=True, training=True)
        _gradients = self._calculate_gradients(tape, loss)

        _gradients = self._clip_gradients(_gradients)

        self._apply_gradients(_gradients)  # use _gradient_list instead of _gradients
        return _gradient_list

    def _train(self):
        # self.display_experiment_parameters()
        # gradient_list is used when accumulating gradients
        _gradient_list = []
        self.current_step = 0
        N = len(self.train_loader)
        self.current_epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.epochs):
            for step, sample in enumerate(tqdm(self.train_loader, desc='Epoch {}/{}'.format(epoch + 1, self.epochs))):
                self.current_step = N * epoch + step + 1
                _gradient_list = self._train_step(sample, _gradient_list)
                if (N * epoch + step + 1) % self.eval_interval == 0 or (N * epoch + step + 1) == N * self.epochs:
                    self._evaluate(eval_step=N * epoch + step + 1)
                    self._save_checkpoint()

    def _evaluate(self, eval_step):
        self.model.eval()
        for step, sample in enumerate(tqdm(self.valid_loader, desc='Evaluating for step {}'.format(eval_step))):
            *outputs, loss = self.model(*sample[:-1], enable_path_drop=False, training=False)
            predictions_to_kitti(outputs, sample[-1], sample[4], sample[-2], current_step=eval_step,
                                 valid_ds=self._valid_dataset,
                                 experiment_name=self.checkpoint_name)

        root_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent
        evaluation_script_to_use = ''
        if set(self._valid_dataset.classes) & {'Pedestrian', 'Cyclist'}:
            evaluation_script_to_use = '05'

        calculate_map_async(root_dir=root_dir, results_parent_folder='results_tf',
                            experiment_name=self.checkpoint_name,
                            annotation_dir=self._valid_dataset.annotation_dir, step=eval_step,
                            evaluation_script_to_use=evaluation_script_to_use)


    def __call__(self, *args, **kwargs):
        if kwargs.get('mode') == 'train':
            print("{}: Starting the training process".format(datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")))
            self._train()
