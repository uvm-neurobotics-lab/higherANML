"""
This file is borrowed with many thanks from [Few-Shot Meta-Baseline](https://github.com/yinboc/few-shot-meta-baseline),
by Yinbo Chen. It was copied on 2021-12-17. The license for this file can be found at ./few-shot-meta-baseline-LICENSE.

Some modifications have been made.
"""

from utils import get_arg_names

# Global registry of models.
models = {}


def register(name):
    """
    A decorator which can be used to register a model. It can decorate either a model class or a factory method. If a
    factory method, the method can return one of two things:
        - Just the model.
        - A tuple of (model, args), where `args` are the exact arguments that were passed to the model constructor.

    Args:
        name: The name to register the model under. Must be globally unique.
    """
    def decorator(callable_):
        models[name] = callable_
        return callable_
    return decorator


def make(name, input_shape=None, device=None, **kwargs):
    """
    Create a new instance of the specified model.

    Args:
        name (str): The name of the model in the global registry.
        input_shape (tuple): The shape of a single input that would be passed to this model. Some models will use this
            to auto-size their layers.
        device (str or device): The device to send the model to after instantiation.
        **kwargs: The constructor arguments for the specified model.

    Returns:
        tuple: A tuple of (model, args), where `args` are the exact arguments that were passed to the model constructor.
            These should be used to save the model using the `utils.storage` module.
    """
    if not name:
        return None, kwargs
    if "input_shape" in get_arg_names(models[name]):
        kwargs["input_shape"] = input_shape

    ret = models[name](**kwargs)
    if isinstance(ret, tuple):
        model, model_args = ret
    else:
        model = ret
        model_args = kwargs

    model.to(device)
    return model, model_args


def make_from_config(config, input_shape=None, device=None):
    """
    Create the model specified by the given config.

    Args:
        config (dict): The YAML config containing the desired model parameters.
        input_shape (tuple): The shape of a single input that would be passed to this model. Some models will use this
            to auto-size their layers.
        device (str or device): The device to send the model to after instantiation.

    Returns:
        tuple: A tuple of (model, args), where `args` are the exact arguments that were passed to the model constructor.
            These should be used to save the model using the `utils.storage` module.
    """
    model_name = config.get("model", None)
    model_args = config.get("model_args", {})
    return make(model_name, input_shape, device, **model_args)


def load(model_sv, name=None):
    """
    Load a model from a saved dict describing the model. We generally don't need this since we have the nifty
    `utils.storage` module.

    Requires the `model_sv` dict to have keys:
      - "<name>": The name of the model in the model registry.
      - "<name>_args": The args to use to construct the model.
      - "<name>_sd": The state_dict to load into the model.

    Args:
        model_sv (dict): A dict containing potentially multiple models, each prefaced by its name.
        name (str): The name of the model to instantiate.

    Returns:
        torch.nn.Module: The loaded model.
    """
    if name is None:
        name = 'model'
    model = make(model_sv[name], **model_sv[name + '_args'])
    model.load_state_dict(model_sv[name + '_sd'])
    return model

