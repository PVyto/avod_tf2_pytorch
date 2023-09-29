import os
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from trainer_base import Trainer
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from general_utils import get_available_options_from_module
from trainer.trainer_helper_functions import calculate_map_async

from utils.model_utils import _weight_initializers, return_model_parameters, custom_model_parameters, weight_initializer


def cuda_opt(device=''):
    if 'cuda' in device or 'gpu' in device:
        torch.backends.cudnn.benchmark = True


class TrainerTorch(Trainer):
    _backend = 'torch'
    _models = get_available_options_from_module('models.main_models')
    _datasets = get_available_options_from_module('data_utils.datasets_torch')

    _optimizers = get_available_options_from_module('optimizers.optimizers_torch')
    _schedulers = get_available_options_from_module('learning_rate_schedulers.lr_schedulers_torch')

    # options for custom weight initialization
    _weight_initializers = _weight_initializers

    _model_param_extraction_method_key_ = 'model_parameter_extraction_method'
    # methods that return model's parameters, to initialize them using a strategy other than the default one using the
    # _weight_initializers dict above
    _model_parameters_methods = {'default': return_model_parameters, 'custom': custom_model_parameters}

    def to_device(self, sample):
        if isinstance(sample, dict):
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    sample[k] = v.to(self.device)
                if isinstance(v, np.ndarray):
                    sample[k] = torch.as_tensor(v, device=self.device)
        else:
            sample = list(sample)
            for i, s in enumerate(sample):
                if type(s) is tuple:
                    sample[i] = list(s)
                    for ssi, ss in enumerate(sample[i]):
                        if isinstance(ss, torch.Tensor):
                            sample[i][ssi] = ss.to(self.device)
                        if isinstance(ss, dict):
                            for k, v in ss.items():
                                if isinstance(v, torch.Tensor):
                                    sample[i][ssi][k] = v.to(self.device)
                                # if isinstance(v, np.ndarray) and self.device != 'cpu':
                                if isinstance(v, np.ndarray):
                                    sample[i][ssi][k] = torch.as_tensor(v, device=self.device)
                elif isinstance(s, torch.Tensor):
                    sample[i] = s.to(self.device)
                elif isinstance(s, np.ndarray):
                    sample[i] = torch.as_tensor(s, device=self.device)
        return sample

    def _load_from_checkpoint(self):
        ...

    def _save_checkpoint(self):
        ...

    def _load_config_file(self, config):
        pass

    def _initialize_trainer(self):
        pass

    def _create_dataloaders(self):
        self._train_dataset = self.dataset(mode='train', config=self.dataset_config)
        self._valid_dataset = self.dataset(mode='val', config=self.dataset_config)
        # self._test_dataset_ = self.dataset(mode='test', config=self.dataset_config)

        workers = self.operation_parameters.get('workers')
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
        self.optimizer_params = self.operation_parameters.get(self._optimizer_params_key_)
        learning_rate_param = self.optimizer_params.get(self._learning_rate_key_)
        self.learning_rate = learning_rate_param.get(self._learning_rate_value_key_)
        self.optimizer_params_dict = {
            'lr': learning_rate_param.get(self._learning_rate_value_key_)}
        lr_type = learning_rate_param.get(self._learning_rate_type_key_)
        if lr_type != 'steady':
            self.has_lr_scheduler = True
            scheduler_params = learning_rate_param[self._learning_rate_scheduler_params_key_]
            self.learning_rate_scheduler_class = self._schedulers[lr_type]
            self.learning_rate_scheduler_params = scheduler_params
        self.model_parameter_method_extraction_name = self.optimizer_params[self._model_param_extraction_method_key_]
        self.model_parameter_method_extraction = self._model_parameters_methods[
            self.model_parameter_method_extraction_name]
        self.weight_decay_dict = self.optimizer_params['weight_decay_dict']
        for param, v in self.optimizer_params.items():
            if param == 'lr' or 'weight_decay_dict':
                continue
            self.optimizer_params_dict[param] = v

    def _initialize_grad_scaler(self):
        # https://pytorch.org/docs/stable/amp.html#gradient-scaling
        self.scaler = GradScaler()

    def _initialize_optimizer(self):
        self.optimizer_name = self.operation_parameters.get(self._optimizer_config_key_).upper()
        self.optimizer_class = self._optimizers[self.optimizer_name]
        # self.model_parameter_method_extraction = self.operation_parameters

        parameter_dicts = [{'params': params, 'weight_decay': self.weight_decay_dict[k]} for k, params in
                           self.model_parameter_method_extraction(self.model).items()]
        self.optimizer = self.optimizer_class(parameter_dicts, **self.optimizer_params_dict)
        if self.has_lr_scheduler:
            self.learning_rate_scheduler = self.learning_rate_scheduler_class(optimizer=self.optimizer,
                                                                              **self.learning_rate_scheduler_params)

    def _initialize_model(self):
        self.model = self._models[self.model_config.get(self._name_key_)](self.model_config)
        self.model.to(self.device)

        override_weight_init = self.operation_parameters.get(self._override_default_weight_init_method_)
        if override_weight_init:
            self.weight_init_cfg = self.operation_parameters.get(self._weight_initialization_cfg_)

            self.weight_init_method_name = self.weight_init_cfg.get(self._weight_initialization_method_)
            self.weight_init_method_params = self.weight_init_cfg.get(self._weight_initialization_method_params_)

            self.weight_init_method = self._weight_initializers[self.weight_init_method_name]
            self.bias_init_ = self.weight_init_cfg.get(self._bias_initialization_cfg_)
            self.bias_init_method = self._weight_initializers.get(self.bias_init_, self.bias_init_)
            # assert self.bias_init_method in {'default', self.weight_init_method}
            # Initialize weights
            weight_initializer(model=self.model, method=self.weight_init_method, bias_method=self.bias_init_method,
                               **self.weight_init_method_params)

    def _calculate_gradients(self, **kwargs):
        loss = kwargs.get('loss')
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _clip_gradients(self, **kwargs):
        # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-unscaled-gradients
        if self.gradient_clipping:
            # unscale before clipping
            if self.mixed_precision:
                self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clipping_norm)

    def _apply_gradients(self, **kwargs):
        if self.mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def _train(self):
        cuda_opt(self.device)
        self.model.train()
        self.current_epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.epochs):
            # date1 = datetime.datetime.now()
            self.current_epoch = epoch
            # self._print_epoch_start(epoch)
            epoch_loss = torch.tensor([0.0], device=self.device)
            for batch_idx, sample in tqdm(enumerate(self.train_loader),
                                          desc=(self.lpad - 4) * ' ' + 'Epoch [%d/%d]' % (epoch + 1, self.epochs),
                                          total=len(self.train_loader), disable=not self.verbose):
                with autocast(enabled=self.mixed_precision):
                    sample = self.to_device(sample)
                    # torch.onnx.export(self.model, *sample, 'checkpoints/my_model.onnx')
                    *outputs, loss = self.model(*sample[:-1], enable_path_drop=True)
                    loss /= self.gradient_accumulations

                if (batch_idx + 1) % self.gradient_accumulations == 0 or (batch_idx + 1) == len(self.train_loader):
                    # use scaler if mixed_precision
                    self._calculate_gradients(loss=loss)
                    self._clip_gradients()
                    self._apply_gradients()

                epoch_loss += loss * self.gradient_accumulations
            if self.has_lr_scheduler:
                self.learning_rate_scheduler.step()
            if (epoch + 1) % self.eval_interval == 0:
                self._evaluate()
                self._save_checkpoint()

    def _evaluate(self):
        self.model.eval()
        valid_loss = torch.tensor([0.0], device=self.device)
        with torch.no_grad():
            for batch_idx, sample in tqdm(enumerate(self.valid_loader),
                                          desc=(self.lpad - 4) * ' ' + 'Evaluating',
                                          total=len(self.valid_loader), disable=not self.verbose):
                with autocast(enabled=self.mixed_precision):
                    sample = self.to_device(sample)
                    *outputs, loss = self.model(*sample[:-1], enable_path_drop=True)
                    valid_loss += loss
                    # sample[4] -> calib
                    if self.predictions_to_kitti(outputs, sample[-1], sample[4], sample[-2]) is None:
                        continue
        root_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent
        evaluation_script_to_use = ''
        if set(self._valid_dataset.classes) & {'Pedestrian', 'Cyclist'}:
            evaluation_script_to_use = '05'
        calculate_map_async(root_dir=root_dir, results_parent_folder='results_torch',
                            experiment_name=self.checkpoint_name,
                            annotation_dir=self._valid_dataset.annotation_dir, step=eval_step,
                            evaluation_script_to_use=evaluation_script_to_use)

    def __call__(self, *args, **kwargs):
        pass
