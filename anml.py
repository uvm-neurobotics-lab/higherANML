import torch
from torch.nn.functional import cross_entropy
from torch.nn.init import kaiming_normal_
from torch.optim import SGD, Adam
import higher
import numpy as np
from pathlib import Path

from model import ANML
from utils import Log
from datasets.OmniSampler import OmniSampler
from utils import divide_chunks


def lobotomize(layer, class_num):
    kaiming_normal_(layer.weight[class_num].unsqueeze(0))


def train(rln, nm, mask, inner_lr=1e-1, outer_lr=1e-3, its=30000, device="cuda"):

    log = Log(f"{rln}_{nm}_{mask}_ANML")
    omni_sampler = OmniSampler(root="../data/omni")

    anml = ANML(rln, nm, mask).to(device)

    # inner optimizer used during the learning phase
    inner_opt = SGD(
        list(anml.rln.parameters()) + list(anml.fc.parameters()), lr=inner_lr
    )
    # outer optimizer used during the remembering phase, the learning is propagate through the
    # inner loop optimizations computing second order gradients
    outer_opt = Adam(anml.parameters(), lr=outer_lr)

    for it in range(its):

        train_data, train_class, (valid_ims, valid_labels) = omni_sampler.sample_train(device=device)

        # To facilitate the propagation of gradients through the model we prevent memorization of
        # training examples by randomizi the weights in the last fully connected layer corresponding
        # to the task that is about to be learned
        lobotomize(anml.fc, train_class)

        # higher turns a standard pytorch model into a functional version that can be used to
        # preseve the computation graph across multiple optimization steps
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
    PATH,
    sampler,
    num_classes=10,
    train_examples=15,
    device="cuda",
    lr=0.01,
    quiet=False,
):

    name = Path(PATH).name
    sizes = [int(num) for num in name.split("_")[:3]]
    model = ANML(*sizes)
    model.load_state_dict(torch.load(PATH, map_location="cpu"))
    model = model.to(device)

    torch.nn.init.kaiming_normal_(model.fc.weight)
    model.nm.requires_grad_(False)
    model.rln.requires_grad_(False)

    test_examples = 20 - train_examples
    train_tasks, test_data, classes = sampler.sample_test(
        num_classes, train_examples, device
    )

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
