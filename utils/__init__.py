"""
General utility functions.
"""

import itertools


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
