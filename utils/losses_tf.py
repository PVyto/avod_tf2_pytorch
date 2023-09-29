import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras.losses import Loss, Huber, Reduction


def tf_to_torch(t):
    return torch.from_numpy(t.numpy())


class SmoothL1Loss(Loss):
    def __init__(self, reduction=Reduction.NONE):
        super(SmoothL1Loss, self).__init__(reduction=reduction)

    def call(self, predictions, targets):
        abs_diff = tf.abs(predictions - targets)
        abs_diff_lt_1 = tf.less(abs_diff, 1)
        smooth_l1norm = tf.reduce_sum(tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5), axis=1)
        return smooth_l1norm


class SoftmaxCELoss(Loss):

    def __init__(self, weight=1.0):
        super(SoftmaxCELoss, self).__init__()
        self.weight = weight

    @staticmethod
    def _forward(predictions, targets, weight):
        log_soft = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=predictions)
        return tf.math.reduce_sum(log_soft * weight) / tf.cast(len(targets), tf.float32)

    def call(self, predictions, targets):
        return self._forward(predictions, targets, tf.convert_to_tensor(self.weight))
        # log_soft = F.log_softmax(predictions, dim=1)
        # ls = tf.nn.log_softmax(predictions, axis=1)
        # k = (tf.math.reduce_sum(-tf.math.reduce_sum(targets * ls, -1)) * weight) / len(targets)
        # return (torch.sum(-torch.sum(targets * log_soft, -1)) * self.weight) / len(targets)


class WeightedSmoothL1Loss(Loss):
    def __init__(self, weight=1.0):
        super(WeightedSmoothL1Loss, self).__init__()
        self.weight = weight
        # self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        # self.smooth_l1 = Huber(reduction=Reduction.NONE)
        # self.smooth_l1_torch = nn.SmoothL1Loss(reduction='none')
        self.smooth_l1 = SmoothL1Loss()

    def forward_old(self, predictions, targets, mask=None):
        # noinspection PyArgumentList
        loss_p = torch.sum(self.smooth_l1(predictions, targets), axis=1) * self.weight
        if mask is not None:
            return torch.sum(loss_p[mask]) / sum(mask) if sum(mask) > 0 else 0.0
        return torch.sum(loss_p) / len(loss_p)

    # @torch.jit.script
    def __call__(self, predictions, targets, mask=None, objectness_gt=None):
        # loss_p = self.smooth_l1(predictions, targets) * targets.shape[-1] * self.weight
        loss_p = self.smooth_l1(predictions, targets) * self.weight
        # loss_p = tf.math.reduce_sum(self.smooth_l1(predictions, targets), axis=1) * self.weight
        if objectness_gt is not None:
            num_of_pos = tf.math.reduce_sum(objectness_gt[:, 1])
            zero_tensor = tf.constant(0.0)
            return tf.math.reduce_sum(loss_p * objectness_gt[:, 1]) / num_of_pos if num_of_pos > 0.9 else zero_tensor
        if mask is not None:
            mask_sum = tf.math.reduce_sum(tf.cast(mask, tf.float32))
            zero_tensor = tf.constant(0.0)
            return tf.math.reduce_sum(tf.boolean_mask(loss_p, mask)) / mask_sum if mask_sum > 0 else zero_tensor
        return tf.math.reduce_sum(loss_p) / len(loss_p)
