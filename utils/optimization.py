"""
Utilities for optimization.
"""
import torch

from utils import get_arg_names


class DummyScheduler:
    def step(self, **kwargs):
        # Do nothing.
        pass


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
        return DummyScheduler()
    sched_args = config.get("lr_scheduler_args", {})
    cls = getattr(torch.optim.lr_scheduler, sched_name)
    if "input_shape" in get_arg_names(cls):
        sched_args.setdefault("verbose", True)
    return cls(opt, **sched_args)
