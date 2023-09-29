import os
import torch
import weakref
import numbers
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from typing import Union, Any
from scipy.linalg import block_diag
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter, FileWriter, RecordWriter, writer
from torchvision.utils import make_grid
from torch.nn.functional import interpolate
import tensorflow as tf

T = Union[torch.Tensor, np.ndarray]


class GeneralUtils(ABC):
    @staticmethod
    @abstractmethod
    def multiply(mat1, mat2): ...


class GeneralUtilsTensor(GeneralUtils):
    @staticmethod
    def multiply(mat1, mat2):
        N = mat1.shape[-1]
        rot = torch.cat([mat1[:, :, i] for i in range(N)], dim=1)
        c = torch.block_diag(*[mat2[:, :, i] for i in range(N)])
        return rot @ c


class GeneralUtilsArray(GeneralUtils):
    @staticmethod
    def multiply(mat1, mat2):
        """

        Parameters
        ----------
        mat1: rotation matrix with shape (3,3,N) -> (Y,N)
        mat2: corner matrix with shape (3,8,N) -> (3*N, 8*N): block diagonal

        Returns
        -------
        N matrices which are the result of mat1*mat2

        """
        N = mat1.shape[-1]
        # rotation matrix
        rot = np.block([mat1[:, :, i] for i in range(N)])
        c = block_diag(*[mat2[:, :, i] for i in range(N)])
        return rot @ c


def call_proper_method(method):
    def wrapper(*args, **kwargs):
        name = method.__name__
        functions = {np.ndarray: getattr(GeneralUtilsArray, name), torch.Tensor: getattr(GeneralUtilsTensor, name)}
        return functions[type(args[0])](*args, **kwargs)

    return wrapper


@call_proper_method
def multiply(mat1: T, mat2: T) -> T: ...


def create_rotation_matrix_2d(angle, rotation='clockwise'):
    matrix = np.array([[np.cos(angle), np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    if rotation == 'clockwise':
        matrix[0, 1] *= -1
    elif rotation == 'counterclockwise':
        matrix[1, 0] *= -1

    return matrix


def create_rotation_matrix_2d_tensor(angle, rotation='clockwise'):
    matrix = torch.stack([
        torch.stack([torch.cos(angle), torch.sin(angle)]),
        torch.stack([torch.sin(angle), torch.cos(angle)])
    ])
    if rotation == 'clockwise':
        matrix[0, 1] *= -1
    elif rotation == 'counterclockwise':
        matrix[1, 0] *= -1

    return matrix


def create_rotation_matrix_3d(angle, rotation='y'):
    cos = np.cos(angle)
    sin = np.sin(angle)
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)
    if rotation == 'x':
        matrix = np.array([[one, zero, zero],
                           [zero, cos, -sin],
                           [zero, sin, cos]])
    elif rotation == 'y':
        matrix = np.array([[cos, zero, sin],
                           [zero, one, zero],
                           [-sin, zero, cos]])
    elif rotation == 'z':
        matrix = np.array([[cos, -sin, zero],
                           [sin, cos, zero],
                           [zero, zero, one]])
    return matrix


def create_rotation_matrix_3d_tensor(angle, rotation):
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle, device=angle.device)
    zero = torch.zeros_like(angle, device=angle.device)
    if rotation == 'x':
        matrix = torch.stack([torch.stack([one, zero, zero]),
                              torch.stack([zero, cos, -sin]),
                              torch.stack([zero, sin, cos])])
    elif rotation == 'y':
        matrix = torch.stack([torch.stack([cos, zero, sin]),
                              torch.stack([zero, one, zero]),
                              torch.stack([-sin, zero, cos])])
    elif rotation == 'z':
        matrix = torch.stack([torch.stack([cos, -sin, zero]),
                              torch.stack([sin, cos, zero]),
                              torch.stack([zero, zero, one])])
    return matrix


def create_rotation_matrix(angle, dim=2, rotation='clockwise'):
    if dim not in {2, 3}:
        raise Exception("Invalid value for parameter dim")
    if rotation not in {'x', 'y', 'z', 'clockwise', 'counterclockwise'}:
        raise Exception("Invalid value for parameter rotation")
    available_functions_array = {2: create_rotation_matrix_2d, 3: create_rotation_matrix_3d}
    available_functions_tensor = {2: create_rotation_matrix_2d_tensor, 3: create_rotation_matrix_3d_tensor}
    if isinstance(angle, torch.Tensor):
        funct = available_functions_tensor[dim]
    elif isinstance(angle, np.ndarray):
        funct = available_functions_array[dim]
    return funct(angle=angle, rotation=rotation)


def check_if_iterable(obj: Any) -> bool:
    """
    checks if the given object is iterable
    """
    if hasattr(obj, '__iter__'):
        return True
    return False


def check_if_str(obj: Any) -> bool:
    if isinstance(obj, str):
        return True
    return False


def check_if_string_or_num(obj):
    return check_if_str(obj) or not check_if_iterable(obj)


def flatten(sequence, scalarp, result=None):
    if result is None:
        result = []
    for item in sequence:
        if scalarp(item):
            result.append(item)
        else:
            flatten(item, scalarp, result)
    return result


def separate_strings(*args):
    keys = []
    values = []
    if isinstance(args[0], dict):
        return list(args[0].keys()), list(args[0].values())
    flattened = flatten(args, scalarp=check_if_string_or_num)
    for arg in flattened:
        if isinstance(arg, str):
            keys.append(arg)
        else:
            values.append(arg)
    return keys, values


def make_dirs(fn):
    def wrapper(*args):
        self = args[0]
        os.makedirs(fn(self), exist_ok=True)
        return fn(self)

    return wrapper


def init_dir(parent_dir_attribute, dir_name):
    def init_path(fn):
        def wrapper(*args):
            self = args[0]
            attr_name = '_' + fn.__name__
            setattr(self, attr_name, os.path.join(getattr(self, parent_dir_attribute), dir_name))
            os.makedirs(getattr(self, attr_name), exist_ok=True)
            return fn(self)

        return wrapper

    return init_path


def handle_multiple_list_attributes(**decargs):
    _defaults = decargs.get('defaults', None)
    attrs = decargs.get('attrs')

    def dec(fn):
        def wrapper(self, *args):
            defaults_ = [len(attrs) * []]
            if _defaults:
                defaults_ = _defaults
            attr_name = fn.__name__
            if len(args) != 0:
                value, attr = args[0]
                new_attr = '_' + attr_name + '_' + attr
                if hasattr(self, new_attr):
                    attr_value = getattr(self, new_attr)
                    if not isinstance(value, list):
                        attr_value.append(value)
                    else:
                        attr_value += value
                    setattr(self, new_attr, attr_value)
                else:
                    if isinstance(value, list):
                        setattr(self, new_attr, value)
                    else:
                        setattr(self, new_attr, [value])
                return getattr(self, new_attr)
            else:
                dct = {}
                for i, attr in enumerate(attrs):
                    new_attr = '_' + attr_name + '_' + attr
                    if not hasattr(self, new_attr):
                        setattr(self, new_attr, defaults_[i])
                    dct[attr] = getattr(self, new_attr)
                return dct

        return wrapper

    return dec


class MeanMetricAccumulator:
    def __init__(self, metric_dict=None):
        self.count = 1
        self.dynamic_attrs = []
        self.computed_mean = False
        if metric_dict is not None:
            self.initialize_dynamic_attrs(metric_dict=metric_dict)

    def update(self, metric_dict):
        if len(self.dynamic_attrs) > 0:
            for k, v in metric_dict.items():
                value = getattr(self, k)
                setattr(self, k, value + v)
            self.count += 1
        else:
            self.initialize_dynamic_attrs(metric_dict=metric_dict)

    def initialize_dynamic_attrs(self, metric_dict):
        for k, v in metric_dict.items():
            self.dynamic_attrs.append(k)
            setattr(self, k, v)

    def calculate_mean_metrics(self):
        for attr in self.dynamic_attrs:
            value = getattr(self, attr)
            setattr(self, attr, value / self.count)
        self.computed_mean = True

    def print_metrics(self):

        for attr in self.dynamic_attrs:
            print("{}: {:.3f}".format(attr, getattr(self, attr)))

    def reset(self):
        for attr in self.dynamic_attrs:
            setattr(self, attr, 0)
        self.count = 1
        self.computed_mean = False

    def write_to_logger(self, logger, logger_method, epoch, prefix=''):
        for attr in self.dynamic_attrs:
            value = getattr(self, attr)
            getattr(logger, logger_method)(prefix + attr, value, epoch)

    def update_from_list(self, l):
        for metric_dct in l:
            self.update(metric_dct)


class MetricTrackerDecorator:
    _trackers = weakref.WeakValueDictionary()

    def __new__(cls, func_name, *args, **kwargs):
        tracker = cls._trackers.get(func_name)
        if not tracker:
            tracker = super().__new__(cls)
            cls._trackers[func_name] = tracker
        return tracker

    def __init__(self, func_name, metrics):
        if not hasattr(self, '_func_name'):
            self._func_name = func_name
            self._dynamic_attrs = []
            for attr in metrics:
                setattr(self, attr, [])
                self._dynamic_attrs.append(attr)

    def __call__(self, func):
        def wrapper(_self, *args):
            if len(args) == 0:
                return {attr: getattr(self, attr) for attr in self._dynamic_attrs}
            else:
                keys, values = separate_strings(*args)
                for i, attr in enumerate(keys):
                    attr_value = getattr(self, attr)
                    if isinstance(values[i], list):
                        attr_value += values[i]
                    else:
                        attr_value.append(values[i])
                    setattr(self, attr, attr_value)

        return wrapper


class MetricLogger:
    def __init__(self, log_dir, _metrics):
        self._metrics = _metrics
        self.writer = SummaryWriter(log_dir=log_dir)
        if _metrics:
            for _metric in _metrics:
                setattr(self, _metric, [])

    def __set__(self, instance, value):
        pass

    # def get
    def get_metric(self, attr_name):
        return getattr(self, attr_name)

    def update(self, attr_name, value, iteration):
        if attr_name not in self._metrics:
            raise Exception("{} is not in the defined metrics.Must be one of {}".format(attr_name, self._metrics))
        val = getattr(self, attr_name)
        self.writer.add_scalar(attr_name, value, iteration)
        val.append(value)
        setattr(self, attr_name, val)


class ExperimentManager:
    def __init__(self):
        pass

    def _parse_config_(self):
        pass

    def _build_experiment_(self):
        pass


# activation = {}


# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()

# return hook


def check_if_number(value):
    try:
        if isinstance(eval(value), numbers.Number):
            return True
        return False
    except NameError:
        return False


def register_method_to_model(model, attr_method, method, _name=''):
    delimiter = '.'
    for module_name, module in model.named_children():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            name = _name + delimiter + module_name
            if check_if_number(module_name):
                name = _name + delimiter + module._get_name()
            getattr(module, attr_method)(method(name))
        else:
            name = module_name
            if _name:
                name = _name + '.' + module_name
            register_method_to_model(module, attr_method, method, _name=name)


def register_method_to_model2(model, attr_method, method, _name=''):
    delimiter = '.'
    for module_name, module in model.named_children():
        if isinstance(module, (nn.ReLU)):
            name = _name + delimiter + module_name
            if check_if_number(module_name):
                name = _name + delimiter + module._get_name()
            getattr(module, attr_method)(method(name))
        else:
            name = module_name
            if _name:
                name = _name + '.' + module_name
            register_method_to_model2(module, attr_method, method, _name=name)


def prepare_image_for_grid(img):
    if img.shape[1] > 1:
        img = img.squeeze(0).unsqueeze(1)
    return img


def log_feature_maps(activation, summary_writer, epoch, keys_of_interest=None, sample_name=''):
    import matplotlib.pyplot as plt
    # plt.imshow
    if keys_of_interest is None:
        keys_of_interest = list(activation.keys())

    for k in keys_of_interest.keys():
        v = activation[k]
        suffix = ' feature maps'
        parent = 'Image'
        if 'bev' in k:
            parent = 'BEV'
        name = parent + suffix

        image = prepare_image_for_grid(v)
        image = normalize_output(image)
        # plt.imshow(image.squeeze(0).permute(1, 2, 0))
        # plt.show()
        image_grid = make_grid(image, nrow=2, pad_value=1)
        summary_writer.add_image(name + '/' + keys_of_interest[k] + '/' + sample_name, image_grid, epoch)


def log_activation_function_hist(activation, summary_writer, epoch, keys_of_interest=None, sample_name='', prefix=''):
    if keys_of_interest is None:
        keys_of_interest = list(activation.keys())

    for k in keys_of_interest:
        v = activation[k]
        suffix = ' histograms'
        parent = 'Image'
        if 'bev' in k:
            parent = 'BEV'
        name = prefix + parent + suffix

        # image = prepare_image_for_grid(v)
        # image = normalize_output(image)
        # plt.imshow(image.squeeze(0).permute(1, 2, 0))
        # plt.show()
        # image_grid = make_grid(image, nrow=2, pad_value=1)
        summary_writer.add_histogram(name + '/' + k, v, epoch)


def normalize_output(img, _type='whole', channel_dim=1):
    if channel_dim != 1:
        raise Exception("Channel dimension must be the 2nd one which implies a tensor in BCHW format")
    if _type not in {'per_channel', 'whole'}:
        raise Exception('Invalid _type must be one of the following {}'.format({'per_channel', 'whole'}))
    if _type == 'per_channel':
        for channel_idx in range(img.size(channel_dim)):
            min_value = img[:, channel_idx].min()
            img[:, channel_idx] = img[:, channel_idx] - min_value
            max_value = img[:, channel_idx].max()
            img[:, channel_idx] = img[:, channel_idx] / max_value * 255
    else:
        img = img - img.min()
        img = img / img.max()
    return img


def mean_unorm_image(image):
    from data_utils.transforms import UnormalizeTf
    # if image.dim() == 4:
    image = image
    mean_unorm = UnormalizeTf()
    mean_unormed_image = mean_unorm(image=image)['image']
    image = normalize_output(mean_unormed_image)
    return image.astype(np.float32)


def hist_model_attributes(model, summary_function, attributes=None, names_dict=None, _name='',
                          it=0,
                          delimiter='.',
                          **kwargs):
    if attributes is None:
        raise Exception('attributes argument must be provided')
    if names_dict is None:
        # names_dict = {'weight': ['_get_name', 'funct']}
        raise Exception('names_dict argument must be provided')

        # attributes = ['weight', 'bias']
    attributes_copy = attributes.copy()

    for module_name, module in model.named_children():
        # print(module_name)

        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            attr = attributes[0]
            name_ = names_dict.get(attr, attr)
            if names_dict.get(attr, '')[-1] == 'funct':
                name_ = getattr(module, name_[0])()
            mod_name = _name + module_name
            try:
                import numbers
                if isinstance(eval(module_name), numbers.Number):
                    # nn.ConvTranspose2d, nn.Linear
                    mod_name = _name + delimiter + name_ + module_name
                    # print(mod_name)
                    summary_function(mod_name, getattr(module, attr), **kwargs)
            except:
                summary_function(mod_name, module.weight, **kwargs)
            for attr in attributes[1:]:
                # if it's a list the first declares the name of the attribute to be used fro extracting the name else
                # the name is the same as attr
                name_ = names_dict.get(attr, attr)
                # check if name needs to be extracted from a method i.e. module.method() where or it is just provided
                if isinstance(name_, list):
                    if name_[-1] == 'funct':
                        name_ = getattr(module, name_[0])()
                # print(mod_name + delimiter + name_)
                summary_function(mod_name + delimiter + name_, getattr(module, attr), **kwargs)
        else:
            # case where module does not belong to the searching ones

            if _name:
                hist_model_attributes(module, summary_function, attributes=attributes_copy, names_dict=names_dict,
                                      _name=_name + delimiter + module_name, it=it + 1, **kwargs)
            else:
                hist_model_attributes(module, summary_function, attributes=attributes_copy, names_dict=names_dict,
                                      _name=module_name, it=it + 1, delimiter=delimiter, **kwargs)


def hist_model_parameters(model, summary_function, attributes=None, names_dict=None, _name='',
                          it=0,
                          delimiter='.',
                          **kwargs):
    if names_dict is None:
        names_dict = {'weight': ['_get_name', 'funct']}
    if attributes is None:
        attributes = ['weight', 'bias']
    hist_model_attributes(model, summary_function, attributes, names_dict, _name, it, delimiter, **kwargs)
 

def hist_model_gradients(model, summary_function, attributes=None, names_dict=None, _name='',
                         it=0,
                         delimiter='.',
                         **kwargs):
    if names_dict is None:
        names_dict = {'grad': ['_get_name', 'funct']}
    if attributes is None:
        attributes = ['grad']

    hist_model_attributes(model, summary_function, attributes, names_dict, _name, it, delimiter, **kwargs)
 

def plot_model_parameters(model, summary_function, _name='', epoch=0):
    for module_name, module in model.named_children():
        if '.' not in _name and _name:
            name = _name + '/' + module_name
        else:
            name = _name + '.' + module_name
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            kernel_grid = prepare_kernels(module.weight.detach())
            summary_function(_name, kernel_grid, epoch)
        else:
            if _name:
                plot_model_parameters(module, summary_function, name, epoch)
            else:
                plot_model_parameters(module, summary_function, module_name, epoch)


def normalize_labels(original_labels, extents):
    # labels = original_labels.clone()

    labels = original_labels
    labels[:, [0, 2]] /= extents[1]
    labels[:, [1, 3]] /= extents[0]
    # return torch.as_tensor(labels)
    return labels


def unormalize_labels(original_labels, extents):
    labels = original_labels  # .clone()
    labels[:, [0, 2]] *= extents[1]
    labels[:, [1, 3]] *= extents[0]
    return labels


def load_predictions(predictions_dir, sample_name, checkpoint_num, number_of_predictions=None, _type='2d'):
    import tensorflow as tf
    cols = ['minx', 'miny', 'maxx', 'maxy']
    if _type == '3d':
        cols = ['x', 'y', 'z', 'l', 'w', 'h', 'ry']
    folder = os.path.join(predictions_dir, 'data_' + str(checkpoint_num))
    file = os.path.join(folder, sample_name + '.txt')
    try:
        predictions = pd.read_csv(file, sep=" ", header=None,
                                  names=['class', 'truncation', 'occlusion', 'alpha', 'minx', 'miny', 'maxx',
                                         'maxy', 'h', 'w', 'l', 'x', 'y', 'z', 'ry', 'score'])
    except FileNotFoundError:
        return None
    if number_of_predictions:
        predictions = predictions.sort_values(by='score', ascending=False)[:number_of_predictions]

    return tf.convert_to_tensor(predictions.loc[:, cols].to_numpy())
    # return torch.as_tensor(predictions.loc[:, cols].to_numpy(), device=device)


def log_predictions2(summary_writer, metric_dict, sample_name, image, epoch,
                     dataset=None, batch_idx=None, original_image_shape=None, new_image_shape=None,
                     kitti_predictions_path=None):

    image = mean_unorm_image(image)
    import cv2
    tp_labels = metric_dict['tp labels']
    fn_labels = metric_dict['fn labels']
    tp_predictions = metric_dict['tp predictions']
    fp_predictions = metric_dict['fp predictions']

    pred_img = (image.copy() * 255).astype(np.uint8)


    gt_img = (image.copy() * 255).astype(np.uint8)

    grid = make_grid([torch.as_tensor(gt_img).permute(2, 0, 1), torch.as_tensor(pred_img).permute(2, 0, 1)],
                     pad_value=255, nrow=1)
    summary_writer.add_image('sample {} epoch {} | gt vs predicted'.format(sample_name, epoch),
                             img_tensor=grid,
                             global_step=epoch)
    # summary_writer.add_image_with_boxes('sample {} with gt labels'.format(sample_name),
    #                                     img_tensor=image,
    #                                     box_tensor=bbs,
    #                                     global_step=epoch, dataformats='HWC')
    #
    # summary_writer.add_image_with_boxes('sample {} with predicted labels'.format(sample_name),
    #                                     img_tensor=image,
    #                                     box_tensor=predicted_bbs,
    #                                     global_step=epoch, dataformats='HWC')


def load_predictions_and_labels(dataset, batch_idx, original_shape, new_shape, sample_name, predictions_path, epoch):
    bbs = load_labels(dataset, batch_idx, original_shape, new_shape)
    predicted_labels = load_predictions(predictions_path, sample_name, epoch)

    if predicted_labels is None:
        return bbs, None
    return bbs, tf.cast(predicted_labels, tf.float32)



def load_labels(dataset, batch_idx, original_shape, new_shape):
    import tensorflow as tf
    bbs = dataset._get_annotation(batch_idx, '2d')[0]
    bbs = normalize_labels(bbs, original_shape)
    bbs = unormalize_labels(bbs, new_shape)
    return tf.convert_to_tensor(bbs)


def log_predictions_and_metrics(summary_writer, metric_dict, sample_name, image, epoch, bbs):
    image = mean_unorm_image(image)
    import cv2

    if metric_dict.get('tp labels', None) is None:
        # case where there are no predictions above the defined conf score
        gt_img = (image.copy() * 255).astype(np.uint8)
        for bb in bbs.cpu().numpy().astype(np.uint16):
            x1, y1, x2, y2 = bb
            gt_img = cv2.rectangle(gt_img, (x1, y2), (x2, y1), (0, 0, 255), 2)
        summary_writer.add_image('gt vs predicted epoch {}/sample {}'.format(epoch, sample_name),
                                 img_tensor=torch.as_tensor(gt_img).permute(2, 0, 1),
                                 global_step=epoch)
        return

    tp_labels = metric_dict['tp labels'].cpu().numpy()
    fn_labels = metric_dict['fn labels'].cpu().numpy()
    tp_predictions = metric_dict['tp predictions'].cpu().numpy()
    fp_predictions = metric_dict['fp predictions'].cpu().numpy()

    pred_img = (image.copy() * 255).astype(np.uint8)
    for bb in tp_predictions.astype(np.uint16):
        x1, y1, x2, y2 = bb
        pred_img = cv2.rectangle(pred_img, (x1, y2), (x2, y1), (0, 255, 0), 2)
    for bb in fp_predictions.astype(np.uint16):
        x1, y1, x2, y2 = bb
        pred_img = cv2.rectangle(pred_img, (x1, y2), (x2, y1), (255, 0, 0), 2)

    gt_img = (image.copy() * 255).astype(np.uint8)
    for bb in tp_labels.astype(np.uint16):
        x1, y1, x2, y2 = bb
        gt_img = cv2.rectangle(gt_img, (x1, y2), (x2, y1), (0, 255, 0), 2)
    for bb in fn_labels.astype(np.uint16):
        x1, y1, x2, y2 = bb
        gt_img = cv2.rectangle(gt_img, (x1, y2), (x2, y1), (0, 0, 255), 2)

    grid = make_grid([torch.as_tensor(gt_img).permute(2, 0, 1), torch.as_tensor(pred_img).permute(2, 0, 1)],
                     pad_value=255, nrow=1)
    summary_writer.add_image('gt vs predicted epoch {}/sample {}'.format(epoch, sample_name),
                             img_tensor=grid,
                             global_step=epoch)

    summary_writer.add_scalars('per_image_metrics/sample {}'.format(sample_name),
                               {'precision': metric_dict['precision'],
                                'recall': metric_dict['recall'],
                                'f1': metric_dict['f1']},
                               epoch)


def plot_predictions_and_metrics(metric_dict, sample_name, image, epoch, bbs):
    image = mean_unorm_image(image)
    import cv2

    if metric_dict.get('tp labels', None) is None:
        # case where there are no predictions above the defined conf score
        gt_img = (image.copy() * 255).astype(np.uint8)
        for bb in bbs.numpy().astype(np.uint16):
            x1, y1, x2, y2 = bb
            gt_img = cv2.rectangle(gt_img, (x1, y2), (x2, y1), (0, 0, 255), 2)
        plt.imshow(gt_img)
        # plt.show()
        # summary_writer.add_image('gt vs predicted epoch {}/sample {}'.format(epoch, sample_name),
        #                          img_tensor=torch.as_tensor(gt_img).permute(2, 0, 1),
        #                          global_step=epoch)
        return

    tp_labels = metric_dict['tp labels'].numpy()
    fn_labels = metric_dict['fn labels'].numpy()
    tp_predictions = metric_dict['tp predictions'].numpy()
    fp_predictions = metric_dict['fp predictions'].numpy()

    pred_img = (image * 255).astype(np.uint8)
    for bb in tp_predictions.astype(np.uint16):
        x1, y1, x2, y2 = bb
        pred_img = cv2.rectangle(pred_img, (x1, y2), (x2, y1), (0, 255, 0), 2)
    for bb in fp_predictions.astype(np.uint16):
        x1, y1, x2, y2 = bb
        pred_img = cv2.rectangle(pred_img, (x1, y2), (x2, y1), (255, 0, 0), 2)

    gt_img = (image.copy() * 255).astype(np.uint8)
    for bb in tp_labels.astype(np.uint16):
        x1, y1, x2, y2 = bb
        gt_img = cv2.rectangle(gt_img, (x1, y2), (x2, y1), (0, 255, 0), 2)
    for bb in fn_labels.astype(np.uint16):
        x1, y1, x2, y2 = bb
        gt_img = cv2.rectangle(gt_img, (x1, y2), (x2, y1), (0, 0, 255), 2)

    grid = make_grid([torch.as_tensor(gt_img).permute(2, 0, 1), torch.as_tensor(pred_img).permute(2, 0, 1)],
                     pad_value=255, nrow=1)

    # print(metric_dict)
    fig, ax = plt.subplots(1, 2, figsize=(12, 12))

    ax[0].title.set_text('gt vs predicted epoch {}/sample {}'.format(epoch, sample_name))
    ax[0].imshow(grid.permute(1, 2, 0).numpy())

    table = ax[1].table(colLabels=['precision', 'recall', 'f1'], rowLabels=[sample_name],
                        cellText=[[np.round(x, 3) for x in [metric_dict['precision'].numpy().item(),
                                                            metric_dict['recall'].numpy().item(),
                                                            metric_dict['f1'].numpy().item()]]],
                        loc='center')
    table.scale(1, 1.5)
    table.set_fontsize(16)
    ax[1].axis('off')
    plt.show()

    print()
    # summary_writer.add_image('gt vs predicted epoch {}/sample {}'.format(epoch, sample_name),
    #                          img_tensor=grid,
    #                          global_step=epoch)
    #
    # summary_writer.add_scalars('per_image_metrics/sample {}'.format(sample_name),
    #                            {'precision': metric_dict['precision'],
    #                             'recall': metric_dict['recall'],
    #                             'f1': metric_dict['f1']},
    #                            epoch)


def prepare_kernels(kernels, resize=False, scale_f=3, mode='bilinear', plot_as_rgb=True):
    if resize:
        kernels = interpolate(kernels, scale_factor=scale_f, mode=mode)
    # print(kernels.shape)
    kernels = normalize_output(kernels)
    # img = kernels.permute(0, 2, 3, 1)
    if kernels.size(1) == 3 and plot_as_rgb:
        kernel_grid = make_grid(kernels, padding=1, nrow=4)
    else:
        kernels = make_grid(kernels, padding=1).unsqueeze(1)
        kernel_grid = make_grid(kernels, padding=0, nrow=int(kernels.size(0) / 4))

    # plot_tensor(kernel_grid)

    return kernel_grid
    # summary_writer.add_image('Kernels', kernel_grid, epoch)


def plot_tensor(tensor, **kwargs):
    plt.imshow(tensor.permute(1, 2, 0), **kwargs)
    plt.show()


def plot_weights(weights, resize=False, scale_f=3, mode='bilinear'):
    from torchvision.utils import make_grid
    if resize:
        weights = interpolate(weights, scale_factor=scale_f, mode=mode)
    weights = weights - weights.min()
    weights = weights / weights.max()
    weights = make_grid(weights).unsqueeze(1)
    final_grid = make_grid(weights)
    plot_tensor(final_grid)
    return final_grid


def log_inputs(summary_writer, inputs, input_name, epoch):
    image = inputs.get('image')
    bev_maps = inputs.get('bev_maps')
    image = mean_unorm_image(image)
    bev_maps = prepare_image_for_grid(bev_maps)

    bev_grid = make_grid(bev_maps * 255, nrow=3, pad_value=1)
    summary_writer.add_image('Inputs epoch {}/sample {}/image'.format(epoch, input_name), image, epoch,
                             dataformats='HWC')
    summary_writer.add_image('Inputs epoch {}/sample {}/bev maps'.format(epoch, input_name), bev_grid, epoch)


def log_images(summary_writer, sample, activation, epoch, sample_name):
    keys_of_interest = {'rpn.conv11_img.block.Conv2d': 'Img bottleneck',
                        'rpn.conv11_bev.block.Conv2d': 'BEV bottleneck(conv1x1)',
                        'feature_extractors.feature_extractor_img.block1.layers.Conv2d': 'first conv from img feature extractor ',
                        'feature_extractors.feature_extractor_bev.block1.layers.Conv2d': 'first conv from bev feature extractor ',
                        'feature_extractors.feature_extractor_bev.up_conv_concat3.layers.layer2.Conv2d': 'last conv from bev feature extractor ',
                        'feature_extractors.feature_extractor_img.up_conv_concat3.layers.layer2.Conv2d': 'last conv from img feature extractor '}
    if isinstance(sample, dict):
        images = sample['images']
        bev_maps = sample['bev_maps']
    elif isinstance(sample, list):
        images = sample[0][0]
        bev_maps = sample[1][0]
    log_inputs(summary_writer, {'image': images, 'bev_maps': bev_maps}, sample_name, epoch)
    log_feature_maps(activation, summary_writer, epoch, keys_of_interest)


def log_gradients():
    pass


def display_predictions(self, idx, _type='2d', _plot=True, predictions_dir='', checkpoint_num='',
                        _filter=False, colors=None, number_of_predictions=None):
    import pandas as pd
    # if not ax:
    #     fig, ax = plt.subplots(1, figsize=(15, 15))

    image = self._load_image(idx).astype(np.uint8)

    def load_predictions(number_of_predictions=None):
        cols = ['minx', 'miny', 'maxx', 'maxy']
        if _type == '3d':
            cols = ['x', 'y', 'z', 'l', 'w', 'h', 'ry']
        folder = os.path.join(predictions_dir, 'data_' + str(checkpoint_num))
        file = os.path.join(folder, self.sample_name + '.txt')
        try:
            predictions = pd.read_csv(file, sep=" ", header=None,
                                      names=['class', 'truncation', 'occlusion', 'alpha', 'minx', 'miny', 'maxx',
                                             'maxy', 'h', 'w', 'l', 'x',
                                             'y', 'z', 'ry', 'score'])
        except FileNotFoundError:
            return None, None, None
        if number_of_predictions:
            predictions = predictions.sort_values(by='score', ascending=False)[:number_of_predictions]

        return predictions['class'].to_numpy().tolist(), predictions.loc[:, cols].to_numpy(), \
               predictions['score'].to_numpy()

    categs, predictions, scores = load_predictions(number_of_predictions)
    if categs is None:
        return
    if _filter:
        if sum(scores > 0.1) > 0:
            predictions = predictions[scores > 0.1]
        else:
            max_ids = np.argmax(scores, axis=0)
            predictions = predictions[max_ids].reshape(-1, 4)
    # if _type == '3d':
    #     calib_dict = self._load_calibration_(idx)
    #     corners = extract_corners_from_label(predictions)
    #     projected_pts = project_lidar_to_img(corners, calib_dict=calib_dict)
    #     predictions = projected_pts.T
    return _display_image_with_2d_bb_(image=image, bbs=predictions, categs=categs,
                                      _plot=_plot, scores=scores, colors=colors)


def _display_image_with_2d_bb_(image, bbs, categs, _plot=True, scores=None, colors=None, fontsize=None):
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)  # figsize=(25,25)

    bbs_ = []
    i = 0
    if fontsize is None:
        fontsize = 'large'
    noc = len(set(categs))
    if colors is None:
        colors = np.random.rand(noc, 3)
    color_mapping = {colname: colors[i] for i, colname in enumerate(list(set(categs)))}
    if len(categs) == 1 and len(bbs) > len(categs):
        categs = len(bbs) * categs
    for i in range(bbs.shape[0]):
        # bl = bb[i * 2, :]
        bl = bbs[i, [0, 1]]
        h = bbs[i, 2] - bbs[i, 0]
        # w = bb[i * 2 + 1, 0] - bb[i * 2, 0]
        w = bbs[i, 3] - bbs[i, 1]
        # h = bb[i * 2 + 1, 1] - bb[i * 2, 1]
        # print('color:', color_mapping[categs[i]])
        # , facecolor = color_mapping[categs[i]]
        bbs_.append(patches.Rectangle(tuple(bl), h, w, edgecolor=color_mapping[categs[i]], facecolor='none'))
        if scores is not None:
            ax.text(bbs[i, 2], bbs[i, -1], str(round(scores[i], 4)),
                    horizontalalignment='right',
                    verticalalignment='bottom', color=color_mapping[categs[i]], size=fontsize)

    ax.imshow(image)
    edgecolor = None
    if noc == 1:
        edgecolor = color_mapping[categs[i]]
    ax.add_collection(PatchCollection(bbs_, linewidth=2, match_original=True))
    # ax.add_collection(PatchCollection(bbs_, linewidth=2, edgecolor=edgecolor, facecolor='none'))

    if _plot:
        plt.show()
    return ax


def calculate_image_metrics(gt_labels, predictions, threshold=0.4):
    from utils.box_utils_tf import calculate_iou

    ious = calculate_iou(gt_labels, tf.convert_to_tensor(predictions))
    max_val = tf.math.reduce_max(ious, axis=1)
    max_idx = tf.math.argmax(ious, axis=1)
    # max_val, max_idx = ious.max(1)
    tp_mask = (max_val > threshold)
    tp_mask_int = tf.cast(tp_mask, tf.int32)
    inv_tp_mask_int = tf.cast(~tp_mask, tf.int32)
    tp = tf.reduce_sum(tp_mask_int)

    tp_ids = tf.boolean_mask(max_idx, tp_mask)

    fp_ids = set(list(range(len(predictions)))) - set(tp_ids.numpy().tolist())
    fp_ids = tf.convert_to_tensor(list(fp_ids))
    tp_labels = tf.boolean_mask(gt_labels, tp_mask)

    fn_labels = tf.boolean_mask(gt_labels, ~tp_mask)
    tp_predictions = tf.gather(predictions, tp_ids)

    fp_predictions = tf.convert_to_tensor(np.array([]))
    if len(fp_ids) > 0:
        fp_predictions = tf.gather(predictions, fp_ids)

    fp = tf.cast(ious.shape[1] - tp, tf.float32)
    fn = tf.cast(ious.shape[0] - tp, tf.float32)
    tp = tf.cast(tp, tf.float32)
    recall = tp / (tp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    return {'recall': recall, 'precision': precision, 'tp predictions': tp_predictions, 'tp labels': tp_labels,
            'fp predictions': fp_predictions, 'fn labels': fn_labels, 'tp': tp, 'fp': fp, 'fn': fn}


def calculate_metrics_per_image(sample_name, gt_labels, predictions, threshold=0.4):
    if isinstance(predictions, np.ndarray):
        print(predictions.shape)
        predictions = tf.convert_to_tensor(predictions)

    metrics = calculate_image_metrics(gt_labels, predictions, threshold)
    precision, recall = metrics['precision'], metrics['recall']
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    # print('Sample {}: F1:{:.3f}, Precision:{:.3f}, Recall:{:.3f}'.format(sample_name, f1, precision, recall))
    # print()
    metrics['f1'] = f1

    return metrics
