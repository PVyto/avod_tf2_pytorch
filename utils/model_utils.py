import torch.nn as nn
from torch.nn.init import xavier_uniform_, xavier_normal_, kaiming_uniform_, uniform_, normal_, kaiming_normal_, \
    zeros_

_weight_initializers = {'normal': normal_, 'uniform': uniform_, 'xavier_normal': xavier_normal_,
                        'xavier_uniform': xavier_uniform_, 'kaiming_normal': kaiming_normal_,
                        'kaiming_uniform': kaiming_uniform_, 'zeros': zeros_}


def return_model_parameters(model):
    return {'model': model.parameters()}


def return_params_rpn(model, mods):
    for n, module_ in model.named_children():
        if n not in {'conv11_img', 'conv11_bev'}:  #
            try:
                # next checks if the parameters generator contains elements;if it does the module will be appended
                # to the list otherwise an exception will be raised
                next(module_.parameters())
                # append only if it has parameters
                mods.append(module_)
            except StopIteration:
                continue


def custom_model_parameters(model):
    rpn_mods = []
    return_params_rpn(model.rpn, rpn_mods)
    return {'feature_extractors': model.feature_extractors.parameters(),
            'rpn': rpn_mods[0].parameters(),
            'second_stage_detector': model.second_stage_detector.parameters()}


# recursive weight initialization for the defined layers
# method: refers to the   weight initialization function
def weight_initializer(model, method, bias_method=None, **kwargs):
    for module in model.children():
        # print(module)
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):  # nn.ConvTranspose2d, nn.Linear
            method(module.weight, **kwargs)
            if bias_method != 'default' and module.bias is not None:
                bias_method(module.bias, **kwargs)
        else:
            weight_initializer(module, method, bias_method, **kwargs)
