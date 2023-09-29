from torch.optim.lr_scheduler import _LRScheduler


# similar to tf.train.exponential_decay
class ExponentialDecayLR(_LRScheduler):
    def __init__(self, optimizer, step_size, staircase=False, gamma=0.1, last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.steps = step_size
        self.staircase = staircase
        self.name = 'Exponential decay learning rate scheduler'
        super(ExponentialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.steps != 0):
            # lr stays the same
            return [group['lr'] for group in self.optimizer.param_groups]
        # lr changes
        if self.staircase:
            # when self.staircase equals with True this class has the same behavior as StepLR
            return [base_lr * self.gamma ** (self.last_epoch // self.steps) for base_lr in self.base_lrs]
        return [base_lr * self.gamma ** (self.last_epoch / self.steps) for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch / self.steps) for base_lr in self.base_lrs]
