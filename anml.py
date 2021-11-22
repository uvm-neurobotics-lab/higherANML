from pathlib import Path

import higher
import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.nn.init import kaiming_normal_
from torch.optim import SGD, Adam

from model import ANML
from utils import divide_chunks
from utils.logging import Log


def lobotomize(layer, class_num):
    kaiming_normal_(layer.weight[class_num].unsqueeze(0))


def train(sampler, input_shape, rln_channels, nm_channels, inner_lr=1e-1, outer_lr=1e-3, its=30000, device="cuda"):
    assert inner_lr > 0
    assert outer_lr > 0
    assert its > 0

    # TODO: Auto-size this instead.
    # num_classes = max(sampler.num_train_classes(), sampler.num_test_classes())
    num_classes = 1000
    # For now, we identify the architecture of saved models using their filename.
    name = ""
    for num in (*input_shape, rln_channels, nm_channels, num_classes):
        name += f"{num}_"
    log = Log(name + "ANML")
    anml = ANML(input_shape, rln_channels, nm_channels, num_classes).to(device)

    # inner optimizer used during the learning phase
    inner_opt = SGD(
        list(anml.rln.parameters()) + list(anml.fc.parameters()), lr=inner_lr
    )
    # outer optimizer used during the remembering phase, the learning is propagate through the
    # inner loop optimizations computing second order gradients
    outer_opt = Adam(anml.parameters(), lr=outer_lr)

    for it in range(its):

        train_data, train_class, (valid_ims, valid_labels) = sampler.sample_train(device=device)

        # To facilitate the propagation of gradients through the model we prevent memorization of
        # training examples by randomizing the weights in the last fully connected layer corresponding
        # to the task that is about to be learned
        lobotomize(anml.fc, train_class)

        # higher turns a standard pytorch model into a functional version that can be used to
        # preserve the computation graph across multiple optimization steps
        with higher.innerloop_ctx(anml, inner_opt, copy_initial_weights=False) as (
                fnet,
                diffopt,
        ):
            # Inner loop of 1 random task (20 images), one by one
            for im, label in train_data:
                out = fnet(im)
                loss = cross_entropy(out, label)
                diffopt.step(loss)

            # Outer "loop" of 1 task (20 images) + 64 random chars, one batch of 84,1,28,28
            m_out = fnet(valid_ims)
            m_loss = cross_entropy(m_out, valid_labels)
            correct = (m_out.argmax(axis=1) == valid_labels).sum().item()
            m_acc = correct / len(valid_labels)
            m_loss.backward()

        outer_opt.step()
        outer_opt.zero_grad()

        log(it, m_loss, m_acc, anml)

    log.close(it, anml)


def test_test(model, test_data, test_examples=5):
    # Meta-test-test
    # given a meta-test-trained model, evaluate accuracy on the held out set
    # of classes used
    x, y = test_data
    with torch.no_grad():
        logits = model(x)
        # report performance per class
        ys = list(divide_chunks(y, test_examples))
        tasks = list(divide_chunks(logits, test_examples))
        t_accs = [
            torch.eq(task.argmax(dim=1), ys).sum().item() / test_examples
            for task, ys in zip(tasks, ys)
        ]
    return t_accs


def test_train(
        model_path,
        sampler,
        sampler_input_shape,
        num_classes=10,
        train_examples=15,
        device="cuda",
        lr=0.01,
):
    name = Path(model_path).name
    # Identify architecture by collecting all integers at the beginning of the filename.
    sizes = []
    for num in name.split("_")[:-1]:
        try:
            sizes.append(int(num))
        except ValueError:
            # Not an int, so consider this the end of the list.
            break
    sizes = tuple(sizes)

    if len(sizes) > 3:
        # We'll assume image shapes are 3 dimensions: (C, H, W). Remainder is assumed to be all other params.
        input_shape = sizes[:3]
        model = ANML(input_shape, *sizes[3:])
    elif len(sizes) == 3 and sizes[-1] == 2304:
        # Backward compatibility: Before we included `input_shape` and `num_classes`. At this time, `num_classes` was
        # always 1000 and we always used greyscale images.
        out_classes = 1000
        input_shape = (1, 28, 28)
        rln_chs, nm_chs = sizes[:2]
        model = ANML(input_shape, rln_chs, nm_chs, out_classes)
    else:
        # We currently don't need to support any other sizes, but could always change this.
        raise RuntimeError(f"Unsupported model shape: {sizes}")

    # Check if the images we are testing on match the dimensions of the images this model was built for.
    if tuple(input_shape) != tuple(sampler_input_shape):
        raise RuntimeError("The specified dataset image sizes do not match the size this model was trained for.\n"
                           f"Data size:  {sampler_input_shape}\n"
                           f"Model size: {input_shape}")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(device)

    torch.nn.init.kaiming_normal_(model.fc.weight)
    model.nm.requires_grad_(False)
    model.rln.requires_grad_(False)

    test_examples = 20 - train_examples
    train_tasks, test_data = sampler.sample_test(num_classes, train_examples, test_examples, device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for task in train_tasks:
        # meta-test-TRAIN
        for x, y in task:
            logits = model(x)
            opt.zero_grad()
            loss = cross_entropy(logits, y)
            loss.backward()
            opt.step()

    # meta-test-TEST
    t_accs = np.array(test_test(model, test_data, test_examples))

    return t_accs
