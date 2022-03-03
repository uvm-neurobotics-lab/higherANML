"""
General utility functions.
"""

import itertools
from pathlib import Path
from typing import Dict, Iterable


def as_strings(l):
    """ Convert all items in a list into their string representation. """
    return [str(v) for v in l]


def unzip(l):
    """ Transpose a list of lists. """
    return list(zip(*l))


def flatten(list_of_lists):
    """ Flatten one level of nesting. """
    return itertools.chain.from_iterable(list_of_lists)


def divide_chunks(l, n):
    """ Yield successive n-sized chunks from l. """
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i: i + n]


def update_with_keys(src, dest, keys):
    """ Works like dict.update, but only copies the values from the given keys. """
    for k in keys:
        if k in src:
            dest[k] = src[k]


def load_yaml(yfile):
    """
    Read a YAML file. This is different from the default YAML loader because it can load YAML files that contain
    !include tags, which can recursively include other YAML files.

    Args:
        yfile (path-like): The path to the YAML file.

    Returns:
        dict: The loaded YAML object.
    """
    # Import locally so folks aren't mandated to depend on yaml unless they're using this function.
    import yaml
    from yamlinclude import YamlIncludeConstructor

    yfile = Path(yfile)
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=yfile.parent)

    with open(yfile, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def make_pretty(config):
    """
    Clean up the given YAML config object to make it nicer for printing and writing to file.

    Args:
        config (Any): The YAML config, or a sub-tree of the config.

    Returns:
        Any: The new config.
    """
    if isinstance(config, Dict):
        return {k: make_pretty(v) for k, v in config.items()}
    elif isinstance(config, Iterable) and not isinstance(config, str):
        # Also has the function of turning tuples into lists, so we get cleaner YAML output.
        return [make_pretty(v) for v in config]
    # Replace paths with fully-resolved path strings for improved readability when printing/writing config.
    elif isinstance(config, Path):
        return str(config.resolve())
    else:
        return config


def ensure_config_param(config, key, condition=None):
    """
    Function to check that a config parameter is present and satisfies a given condition.

    Args:
        config (dict): The config to check.
        key (str): The key that should be present in the config.
        condition (function): (Optional) A function of (obj) -> bool that returns whether the value for this key is
            valid.
    """
    if key not in config:
        raise RuntimeError(f'Required key "{key}" not found in config.')
    value = config[key]
    if condition and not condition(value):
        raise RuntimeError(f'Config parameter "{key}" has an invalid value: {value}')


def calculate_output_shape(module, input_shape):
    """
    Determines the output shape of the given module when fed with batches of the given input shape. This is useful when
    chaining modules together in a way that can adapt to different input shapes.

    Args:
        module: The feature extractor module.
        input_shape: The shape of inputs that will be fed to this extractor.

    Returns:
        tuple or torch.Size: The resulting output shape.
    """
    # Import torch locally so people can still use other util functions without having torch installed.
    import torch

    # Simulate a batch by adding an extra dim at the beginning.
    bsize = 3
    batch_shape = (bsize,) + tuple(input_shape)
    output_shape = module(torch.zeros(batch_shape)).shape
    # Then check that the batch dimension was preserved and trim it off before returning.
    if output_shape[0] != bsize:
        raise RuntimeError(f"Batch dimension was not preserved by the module. This is unexpected (output shape ="
                           f" {output_shape}).")
    return output_shape[1:]


def calculate_output_size_for_fc_layer(module, input_shape, max_size=int(1e4)):
    """
    Determines the output size of the given module when fed with batches of the given input shape. This is useful when
    you have an arbitrary feature extractor and you want to know the size of the resulting feature vector (how many
    neuronal activations will be at the end of the feature extractor).

    NOTE: This assumes the output is flattened at the end of the module, so that the result is a vector (or batch of
    vectors).

    Args:
        module: The feature extractor module.
        input_shape: The shape of inputs that will be fed to this extractor.
        max_size: (Optional) Throw an error if the result is larger than this size. This can be effectively disabled by
            passing inf or NaN.

    Returns:
        int: The number of dimensions in the resulting feature representation.
    """
    output_shape = calculate_output_shape(module, input_shape)
    if len(output_shape) != 1:
        raise RuntimeError(f"Module output should be a single feature vector, but got shape = {output_shape}.")
    feature_size = output_shape[-1]

    # Sanity check.
    if feature_size > max_size:
        raise RuntimeError(f"The output size of the given module is {feature_size} when given input of shape"
                           f" {input_shape}. Is this a mistake? You should add more pooling or longer strides to reduce"
                           " the features to a manageable size.")
    return feature_size


def memory_constrained_batches(dataset, indices, max_gb):
    """
    A function to batch up the desired samples from the given dataset into chunks that are as large as possible without
    exceeding the specified memory limit.

    This function assumes all data points are the same size. The first data point in the list will be used as a
    prototype to predict the memory usage of a batch.

    Args:
        dataset: The dataset to pull from.
        indices: The indices to sample from the dataset.
        max_gb: The maximum amount of memory to use for a single (inputs, labels) pair.

    Returns:
        list: A list of lists, where each list is a batch of indices. Indices will appear in the order given by
            `indices`. All batches will be the same size, except for the last batch which may be smaller.
    """
    if not indices:
        return []

    # Grab the first data point to estimate size.
    data, label = dataset[indices[0]]
    data_mem_bytes = data.element_size() * data.nelement()
    label_mem_bytes = label.element_size() * label.nelement()
    GB = 1024**4
    mem_per_sample_gb = (data_mem_bytes + label_mem_bytes) / GB
    num_allowable_samples = int(max_gb / mem_per_sample_gb)
    return list(divide_chunks(indices, num_allowable_samples))


def collate_images(samples, device=None):
    """
    Takes a list of (image, label) pairs and returns two tensors: [images], [labels]. Each pair could consist of a
    single image [H, W] or [C, H, W], or it could consist of a batch of images [B, C, H, W]. The resulting tensor will
    be a batch of images. We assume all elements in the list have the same dimensions.

    NOTE: NumPy arrays are not currently supported, but they could be. Labels are allowed to be tensors or numbers.

    Args:
        samples (list[tuple]): A list of pairs of (image, target) or (batch, targets) tensors.
        device (str or torch.device): (Optional) The device to send the tensors to, or None to use the default.

    Returns:
        torch.tensor: An image batch; dimensionality [B, C, H, W].
        torch.tensor: A label batch; dimensionality [B].
    """
    # Import torch locally so people can still use other util functions without having torch installed.
    import torch

    # Consistency checks.
    if not samples or not samples[0]:
        return torch.tensor([]), torch.tensor([])

    proto_im, proto_label = samples[0]
    assert 1 < proto_im.ndim < 5, f"Image dimensionality {proto_im.ndim} not expected."
    # Labels could be non-tensor objects, like floats, so guard against that here.
    if isinstance(proto_label, torch.Tensor):
        assert proto_label.ndim < 2, f"Label dimensionality {proto_label.ndim} not expected."
    if proto_im.ndim == 4:
        # If images are in batches, then check that labels are in a matching size.
        assert proto_label.ndim == 1, "Label batches are expected to be one-dimensional tensors."
        assert proto_im.shape[0] == proto_label.shape[0], (
            f"Image batch size ({proto_im.shape[0]}) and label batch size ({proto_label.shape[0]}) don't match.")

    if proto_im.ndim == 2:
        # Add channel dimension if not already present.
        images = [s[0].unsqueeze(0) for s in samples]
    else:
        images = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    if images[0].ndim == 3:
        # Single images, not batches, so stack them (adds a dimension).
        xs = torch.stack(images)
        ys = torch.tensor(labels)
    else:
        # Already in batches, so concat the batches (along the first dimension, by default).
        xs = torch.cat(images)
        ys = torch.cat(labels)

    # Transfer to device if desired.
    if device:
        xs = xs.to(device)
        ys = ys.to(device)

    return xs, ys


def compute_logits(feat, proto, metric="dot", temp=1.0):
    """
    Compute "logits" for zero-shot classification.

    The logits are given by a metric that quantifies similarity to a "prototype" of each class. The likelihood of being
    a member of a particular class is assumed to be proportional to this similarity. Thus, these logits can be passed
    into a softmax to produce a distribution over classes.

    TODO: Finish documenting.
    TODO: Change default to "cos"?

    Args:
        feat: A set of "query" feature vectors to classify.
        proto: A set of class prototype feature vectors. Prototype features should be the same size as the ones given by
            `feat`, and there should be one prototype per class.
        metric: The metric to use:
            "dot": Dot product similarity between query features and prototype features. (default)
            "cos": Cosine similarity between query features and prototype features.
            "sqr": Squared distance in feature space, between query features and prototype features.
        temp: TODO: don't know what this is.

    Returns:
        torch.Tensor: A set of logits for each given feature vector.
    """
    # Import torch locally so people can still use other util functions without having torch installed.
    import torch
    import torch.nn.functional as F

    if feat.dim() != proto.dim():
        raise RuntimeError(f"Feature dims ({feat.dim()}) not equal to prototype dims ({proto.dim()}).")

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1),
                               F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp
