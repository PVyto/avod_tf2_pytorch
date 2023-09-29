import os
from pathlib import Path
from configs.cfg import return_cfg
from abc import ABC, abstractmethod

_backends = {'tf', 'tensorflow', 'torch', 'pytorch'}


class Trainer(ABC):
    _backend = ''
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
    _checkpoint_num_ = 'checkpoint_num'

    _evaluation_interval_key_ = 'evaluation_interval'
    _gradient_accumulations_key_ = 'gradient_accumulations'
    _device_key_ = 'device'

    _optimizer_params_key_ = 'optimizer_params'

    _checkpoint_loss = 'loss'
    _checkpoint_epoch = 'epoch'
    _checkpoint_model = 'checkpoint'
    _checkpoint_optimizer = 'optimizer'
    _checkpoint_lr_scheduler = 'scheduler'
    _checkpoint_gradient_scaler = 'gradient_scaler'
    _checkpoint_losses = 'loss_dict'
    _checkpoint_maps = 'map_dict'
    _checkpoint_recalls = 'recall_dict'
    _checkpoint_precision = 'precision_dict'

    _checkpoint_train_metrics = 'train_dict'
    _checkpoint_val_metrics = 'val_dict'

    _mAP_easy = 'mAP_easy'
    _mAP_moderate = 'mAP_moderate'
    _mAP_hard = 'mAP_hard'
    _mAPs = 'mAPs'  # list with the mAP score of every epoch

    _metrics = ['map_easy', 'map_moderate', 'map_hard', 'loss']

    # @abstractmethod
    # def __init__(self, config=None, **kwargs):

    def __init__(self, config=None, **kwargs):
        if config is None:
            config = return_cfg()
        self._parse_config(config)
        self._initialize_trainer()
        self._prepare_evaluation_script()

    def _prepare_evaluation_script(self):
        command = 'make -C ' + self.evaluation_script_folder.as_posix()
        os.system(command)

    def _parse_config(self, cfg_dict):
        cfg = cfg_dict
        self.cfg = cfg
        self.globals = cfg.get('global')

        #  Model configuration follows
        self.model_config = cfg.get(self._model_config_key_)
        # Dataset configuration
        self.dataset_config = cfg.get(self._dataset_config_key_)
        # Train or test
        self.mode = cfg.get(self._mode_key_)
        # get the config for the above mode
        self.operation_parameters = cfg.get(self._mode_config_params_key_)

    def _initialize_basic_parameters(self):
        self.learning_rate = 0.001
        self.last_checkpoint = None
        self.has_lr_scheduler = False
        self.learning_rate_scheduler = None
        self.learning_rate_scheduler_params = {}
        self.learning_rate_scheduler_class = None
        self.resume = False
        self.start_epoch = 0
        self.current_epoch = 0
        self.best_loss = 1e+6
        self.current_loss = 1e+6
        self.mAP_easy = 0
        self.mAP_moderate = 0
        self.mAP_hard = 0
        self.best_mAP = 0
        self._best_mAP = 0

    def _extract_training_parameters(self):
        self.device = self.operation_parameters.get(self._device_key_)

        self.epochs = self.operation_parameters.get(self._epochs_key_)
        self.batch_size = self.operation_parameters.get(self._batch_size_key_)
        self.eval_interval = self.operation_parameters.get(self._evaluation_interval_key_)
        self.gradient_clipping = self.operation_parameters.get(self._gradient_clipping_, False)
        if self.gradient_clipping:
            self.gradient_clipping_norm = self.operation_parameters.get(self._gradient_clipping_value_)
        self.mixed_precision = self.operation_parameters.get(self._mixed_precision_key_, False)
        self.gradient_accumulations = self.operation_parameters.get(self._gradient_accumulations_key_)

    def _initialize_paths(self):
        self.log_dir = 'logs'
        experiment_id = self.checkpoint_name
        self.root_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent
        self.experiment_results_parent_folder = self.root_dir / ('results_' + self._backend) / experiment_id
        self.result_folder = self.root_dir / ('results_' + self._backend) / experiment_id / 'val'

        # self.result_folder = 'val'
        self.evaluation_script_folder = self.root_dir / 'kitti_native_eval'
        self.checkpoints_to_evaluate_path = self.root_dir / 'checkpoints_to_evaluate.json'

    def _initialize_checkpoint_paths(self):
        self.checkpoint_dir = self.model_config.get(self._checkpoint_directory_key_)
        self.checkpoint_name = self.model_config.get(self._checkpoint_name_key_)
        self.best_model_dir = self.model_config.get(self._best_models_dir_key_)
        self.checkpoint_path_prefix = os.path.join(self.checkpoint_dir, self.checkpoint_name)
        self.best_model_path_prefix = os.path.join(self.best_model_dir, self.checkpoint_name)
        self.best_model_path = self.best_model_path_prefix + '.pth.tar'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, self.checkpoint_name + "-ckpt")

    def _save_config(self):
        import json
        cfg_path = self.experiment_results_parent_folder / 'config.json'
        with open(cfg_path, "w") as f:
            f.write(json.dumps(self.cfg))

    def _create_result_folders_(self):
        # create checkpoint, best models and results directories in case they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.result_folder, exist_ok=True)

    def _create_checkpoints_to_evaluate_file(self):
        import json
        # if not os.path.exists(self.checkpoints_to_evaluate_path):
        default_eval_dct = {"current_eval_interval": self.eval_interval, "checkpoints_to_evaluate": [],
                            'disable_evaluation': False}
        with open(self.checkpoints_to_evaluate_path, 'w') as f:
            json.dump(default_eval_dct, f)

    @abstractmethod
    def _load_config_file(self, config):
        pass

    @abstractmethod
    def _initialize_trainer(self):
        pass

    @abstractmethod
    def _create_dataloaders(self):
        pass

    @abstractmethod
    def _prepare_optimizer_params(self):
        pass

    @abstractmethod
    def _initialize_optimizer(self):
        ...

    @abstractmethod
    def _initialize_model(self):
        ...

    # @abstractmethod
    # def _initialize_paths(self): ...

    @abstractmethod
    def _calculate_gradients(self, **kwargs):
        pass

    @abstractmethod
    def _clip_gradients(self, **kwargs):
        pass

    @abstractmethod
    def _apply_gradients(self, **kwargs):
        ...

    @abstractmethod
    def _train(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
