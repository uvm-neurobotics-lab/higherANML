"""
This file is borrowed with many thanks from [Few-Shot Meta-Baseline](https://github.com/yinboc/few-shot-meta-baseline),
by Yinbo Chen. It was copied on 2021-12-17. The license for this file can be found at ./few-shot-meta-baseline-LICENSE.

Some modifications have been made.
"""

# Global registry of models.
models = {}


def register(name):
    """
    A decorator which can be used to register a model. It can decorate either a model class or a factory method.
    Args:
        name: The name to register the model under. Must be globally unique.
    """
    def decorator(callable_):
        models[name] = callable_
        return callable_
    return decorator


def make(name, device=None, **kwargs):
    """
    Create a new instance of the specified model.

    Args:
        name (str): The name of the model in the global registry.
        device (str or device): The device to send the model to after instantiation.
        **kwargs: The constructor arguments for the specified model.

    Returns:
        torch.nn.Module: The new model.
    """
    if name is None:
        return None
    model = models[name](**kwargs)
    model.to(device)
    return model


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

