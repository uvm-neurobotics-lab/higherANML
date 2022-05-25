"""
Utilities for optimization.
"""
import torch
from torch.optim.lr_scheduler import _LRScheduler

from utils import get_arg_names


class DummyScheduler(_LRScheduler):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super(DummyScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # Never change learning rate, just use the existing one.
        return [group['lr'] for group in self.optimizer.param_groups]


def optimizer_from_config(config, params):
    """
    Create an optimizer for the given model, using the given config.

    Args:
        config (dict): The YAML config containing the desired optimization parameters.
        params (nn.Module): The set of parameters to optimize.

    Returns:
        torch.optim.Optimizer: The new optimizer.
    """
    optimizer_name = config.get("optimizer", "Adam")
    optimizer_args = config.get("optimizer_args", {})
    optimizer_args.setdefault("lr", config.get("lr"))
    cls = getattr(torch.optim, optimizer_name)
    return cls(params, **optimizer_args)


def scheduler_from_config(config, opt):
    """
    Create a learning rate scheduler for the given optimizer, using the given config.

    Args:
        config (dict): The YAML config containing the desired learning rate parameters.
        opt (torch.optim.Optimizer): The optimizer to put on schedule.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The new scheduler.
    """
    sched_name = config.get("lr_scheduler")
    if not sched_name:
        return DummyScheduler(opt)
    sched_args = config.get("lr_scheduler_args", {})
    cls = getattr(torch.optim.lr_scheduler, sched_name)
    if "verbose" in get_arg_names(cls):
        sched_args.setdefault("verbose", True)
    return cls(opt, **sched_args)
