import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxCELoss(nn.Module):

    def __init__(self, weight=1.0):
        super(SoftmaxCELoss, self).__init__()
        self.weight = weight

    @staticmethod
    @torch.jit.script
    def _forward(predictions, targets, weight):
        log_soft = F.log_softmax(predictions, dim=1)
        return (torch.sum(-torch.sum(targets * log_soft, -1)) * weight) / len(targets)

    def forward(self, predictions, targets):
        return self._forward(predictions, targets, torch.as_tensor(self.weight, device=targets.device))



class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, weight=1.0):
        super(WeightedSmoothL1Loss, self).__init__()
        self.weight = weight
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward_old(self, predictions, targets, mask=None):
        # noinspection PyArgumentList
        loss_p = torch.sum(self.smooth_l1(predictions, targets), axis=1) * self.weight
        if mask is not None:
            return torch.sum(loss_p[mask]) / sum(mask) if sum(mask) > 0 else 0.0
        return torch.sum(loss_p) / len(loss_p)

    # @torch.jit.script
    def forward(self, predictions, targets, mask=None, objectness_gt=None):

        loss_p = torch.sum(self.smooth_l1(predictions, targets), dim=1) * self.weight
        if objectness_gt is not None:
            return torch.sum(loss_p * objectness_gt[:, 1]) / torch.sum(objectness_gt[:, 1]) if \
                torch.sum(objectness_gt[:, 1]) > 0.9 else torch.tensor([0.0], device=predictions.device)
        if mask is not None:
            return torch.sum(loss_p[mask]) / torch.sum(mask) if torch.sum(mask) > 0 \
                else torch.tensor([0.0], device=predictions.device)
        return torch.sum(loss_p) / len(loss_p)
