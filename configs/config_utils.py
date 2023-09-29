from configs.cfg import return_cfg
from general_utils import get_available_options_from_module


class Config:
    tensorflow_options = {'datasets': get_available_options_from_module('data_utils.datasets_tf'),
                          'optimizers': get_available_options_from_module('optimizers.optimizers_tf'),
                          'learning_rate_schedulers': get_available_options_from_module(
                              'learning_rate_schedulers.lr_schedulers_tf'),
                          'models': get_available_options_from_module('models.main_models_tf')}
    torch_options = {'datasets': get_available_options_from_module('data_utils.datasets_torch'),
                     'optimizers': get_available_options_from_module('optimizers.optimizers_torch'),
                     'learning_rate_schedulers': get_available_options_from_module(
                         'learning_rate_schedulers.lr_schedulers_torch'),
                     'models': get_available_options_from_module('models.main_models')}
    tensorflow_default_options = {'model':''}

    # backend_mapping = {'tf':}
    def return_cfg(self):
        pass
