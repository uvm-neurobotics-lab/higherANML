"""
Utilities for saving/loading models to/from disk.
"""

import dis
import importlib
import inspect
import logging
import os
import sys
import tempfile
import time
from copy import deepcopy
from hashlib import blake2b
from pathlib import Path

import unittest

_logger = logging.getLogger(__name__)


def check_ext(file, expect=".net"):
    ext = os.path.splitext(file)[-1]
    if ext != expect:
        raise RuntimeError(f"Expected file extension to be {expect}, got {ext}.")


def load(file, device=None):
    """
    Load a saved PyTorch module.

    Args:
        file (str or Path): The file from which to load. Should have been saved using `save()`.
        device (str or torch.device): The device onto which to load the module. If None, this will be the same device
            where the module was originally saved from.

    Returns:
        torch.nn.Module: The re-constituted module.
    """

    # check_ext(file, expect="net")

    import torch

    info = torch.load(file, map_location=device)
    source = info["source"]
    # from https://stackoverflow.com/a/53080237
    # recover the module from the stored source code
    module_name = "coldmodels"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    cold_module = importlib.util.module_from_spec(spec)
    exec(source, cold_module.__dict__)
    # from https://stackoverflow.com/a/4821120
    # instantiate the original class from the cold module
    class_ = getattr(cold_module, info["class"])
    net = class_(**info["kwargs"])
    # previous version, not sure it avoids name conflicts
    # net = eval(instantiate, cold_module.__dict__)
    net.load_state_dict(info["state"])
    # stored retrieved info for future saving (need to preserve the original sources)
    del info["state"]
    net.serialize_info = info
    return net


def save(net, file, opt=None, check=True, **kwargs):
    """
    Save the given PyTorch module.

    Args:
        net (torch.nn.Module): The module to be saved.
        file (str or Path): Full path to save to.
        opt (torch.optim.Optimizer): The optimizer being used for training. Optional.
        check (bool): Whether to re-load the file immediately upon saving to check that it works properly.
        **kwargs: The exact arguments that were given to the model constructor. These will be saved so the model can
            be re-constructed in the same way upon loading.
    """
    # don't call these .pth or something you will never remember, use .net
    # because networks, right? and it's not like anyone else used it before right?
    check_ext(file, expect=".net")
    import torch
    import torchvision

    info = {}

    if net.__class__.__module__ == "coldmodels":
        # if the model has been retrieved already use the stored info
        info = net.serialize_info

    elif inspect.getmodule(net).__package__ == "torchvision.models":
        raise RuntimeError(f"This is not the save you are looking for. Use torch.save instead.")

    elif issubclass(net.__class__, torch.nn.Module):
        # if a module uses other local imports we want to include them
        imports = find_localimports(net)
        info["source"] = get_sources(imports)
    else:
        raise RuntimeError(f"Expected torch.nn.Module or coldmodels but got {net.__class__} instead.")

    # store additional info for forensic
    info["creator"] = get_creator()
    # get the class name and constructor arguments to re-instantiate it at load time
    info["class"] = net.__class__.__name__
    info["kwargs"] = kwargs

    if issubclass(opt.__class__, torch.optim.Optimizer):
        opt = opt_summary(opt)
    info["opt"] = opt

    info["state"] = deepcopy(net.state_dict())
    # info["state_hash"] = weights2hash(net)
    torch.save(info, file)
    # paranoia check
    if check:
        assert load(file) is not None
    _logger.info(f"Model saved to {file}")


def get_creator():

    try:
        __file__
        # print(f"{__file__=}")
    except NameError:
        # there is no __file__ in interactive sessions
        return ""

    # you wanted to use __file__ didn't you? nope, you gotta climb the calls stack!
    frame = inspect.getouterframes(inspect.currentframe())[-1]
    with open(frame.filename, "r") as f:
        creator = f.read()
    return creator


def find_localimports(obj):

    source_path = inspect.getsourcefile(obj.__class__)
    _logger.debug(f"Entry point for {obj.__class__}: {source_path}")
    found = [source_path]

    def _check_nested(source_path, level=0):
        # get the source code to look for imports
        with open(source_path, "r") as f:
            source = f.read()
        parent = Path(source_path).parent
        _logger.debug((" " * level) + f"Parent path: {parent}")
        # adapted from https://www.py4u.net/discuss/14448
        instructions = dis.get_instructions(source)
        imports = [__.argval for __ in instructions if "IMPORT_NAME" == __.opname]
        _logger.debug((" " * level) + f"Found {len(imports)} imports.")
        # e.g. imports = ['torch', 'torch.nn', 'utils']
        level += 2
        for imp in imports:
            s = importlib.util.find_spec(imp)
            _logger.debug((" " * level) + f"Spec for {imp} = {s}")
            name, orig = s.name, s.origin
            # if parent is in the parents then import is in the same dir tree
            if orig is not None and parent in Path(orig).parents:
                _logger.debug((" " * level) + f"Nested: {name} -> {orig} | parent {parent}")
                found.append(orig)
                _check_nested(orig, level + 2)

    _check_nested(source_path)
    # put the root import at the end
    return list(reversed(found))


def get_sources(paths):
    sources = []
    for path in paths:
        with open(path, "r") as f:
            source = f.read()
            # add a little note of where that chunk of code comes from
            sources.append(f"# from {path}\n{source}")
    return "\n\n".join(sources)


def summary(file, dsize=8):
    # print information about a stored model
    import torch

    if isinstance(file, str) or isinstance(file, Path):
        check_ext(file, expect=".net")
        info = torch.load(file)
    elif file.__class__.__module__ == "coldmodels":
        # if the model has been retrieved already use the stored info
        info = file.serialize_info
        info["state"] = file.state_dict()
    else:
        raise RuntimeError(f"Expected coldmodels or path-like, but got {type(file)} instead.")

    # "state": count params
    nparams = sum([v.numel() for v in info["state"].values()])

    # "source": hash for comparison purposes
    h = blake2b(digest_size=dsize)
    h.update(info["source"].encode("utf-8"))
    source = h.hexdigest()

    # "creator": hash if not empty
    creator = info["creator"]
    if creator != "":
        h = blake2b(digest_size=dsize)
        h.update(creator.encode("utf-8"))
        creator = h.hexdigest()

    # "class": print with kwargs
    # "kwargs": print with class
    cls = info["class"]
    kwargs = info["kwargs"]
    params = [f"{key}={val}" for key, val in kwargs.items()]
    instantiate = f"{cls}({', '.join(p for p in params)})"

    # state_hash = info["state_hash"]
    opt = info["opt"]

    pad = "=" * (14 + dsize * 2)
    return (
        f"{pad}\n"
        f"{instantiate}\n"
        f"{'source code':<12}= {source}\n"
        f"{'created by':<12}= {creator}\n"
        # f"{'state hash':<12}= {state_hash}\n"
        f"{'num. params':<12}= {nparams:,}\n"
        f"{'optimizer   ':<12}= {opt}\n"
        f"{pad}\n"
    )


def par2bytes(p):
    return p.detach().cpu().numpy().tobytes()


def weights2hash(model, dsize=8):
    # compute hash of a torch.nn.Module weights or a list of tensors
    import torch

    h = blake2b(digest_size=dsize)
    # state = {name:par2bytes(p) for name, p in net.named_parameters()}
    # names = sorted(state.keys()) # sort names for reproducibility
    # for name in names:
    #   b = state[name]
    #   h.update(b)
    if issubclass(model.__class__, torch.nn.Module):
        model = model.parameters()
    for p in model:
        h.update(par2bytes(p))
    return h.hexdigest()


def opt_summary(opt):
    param_groups = opt.param_groups
    opts_conf = []
    for param_group in opt.param_groups:
        args = ", ".join(
            [
                f"{key}={value}"
                for key, value in opt.param_groups[0].items()
                if type(value) != list
            ]
        )
        opts_conf.append(f"{opt.__class__.__name__}({args})")
    return "\n".join(opts_conf)


def module_from_file(name, path):
    # from https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


#%%


class TestStorageUtilities(unittest.TestCase):

    test_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self, channels=32, hid=800, out=100):\n"
        "        super().__init__()\n"
        "        self.rln = nn.Sequential(\n"
        "            nn.Conv2d(1, channels, 3),\n"
        "            nn.ReLU(inplace=True),\n"
        "            nn.MaxPool2d(2, 2),\n"
        "            nn.Conv2d(channels, channels, 3),\n"
        "            nn.ReLU(inplace=True),\n"
        "            nn.MaxPool2d(2, 2),\n"
        "            nn.Flatten(),\n"
        "        )\n"
        "        self.pln = nn.Linear(hid, 100)\n"
        "\n"
        "    def forward(self, x):\n"
        "        x = self.rln(x)\n"
        "        x = self.pln(x)\n"
        "        return x\n"
        "    pass\n"
        "dummy_inp = torch.rand(4,1,28,28)\n"
    )

    def test_load_custom_model(self):
        import torch

        with tempfile.NamedTemporaryFile(
            suffix=".py"
        ) as fpy, tempfile.NamedTemporaryFile(
            suffix=".net"
        ) as fpnet1, tempfile.NamedTemporaryFile(
            suffix=".net"
        ) as fpnet2:

            fpy.write(self.test_code.encode("utf-8"))
            fpy.flush()

            module = module_from_file("smallnet", fpy.name)

            net = module.Net()
            # print(net.__class__)
            out = net(module.dummy_inp)

            # test save/load of original model
            save(net, fpnet1.name)
            net1 = load(fpnet1.name)
            # print(net1.__class__)
            out1 = net1(module.dummy_inp)

            # test save/load of retrieved copy
            save(net, fpnet2.name)
            net2 = load(fpnet2.name)
            # print(net2.__class__)
            out2 = net2(module.dummy_inp)

            self.assertTrue(torch.all(out == out1))
            self.assertTrue(torch.all(out == out2))
            self.assertTrue(torch.all(out1 == out2))

    def test_save_model_arguments(self):
        import torch

        with tempfile.NamedTemporaryFile(
            suffix=".py"
        ) as fpy, tempfile.NamedTemporaryFile(suffix=".net") as fpnet1:

            fpy.write(self.test_code.encode("utf-8"))
            fpy.flush()

            module = module_from_file("smallnet", fpy.name)

            net = module.Net(channels=16, hid=400)
            opt = torch.optim.SGD(net.parameters(), lr=0.00001)
            pre = net(module.dummy_inp)
            save(net, fpnet1.name, opt=opt, channels=16, hid=400)
            print()
            print(summary(fpnet1.name))
            net1 = load(fpnet1.name)
            post = net(module.dummy_inp)

            self.assertTrue(torch.all(pre == post))

    def test_save_optimizer_info(self):
        import torch

        with tempfile.NamedTemporaryFile(
            suffix=".py"
        ) as fpy, tempfile.NamedTemporaryFile(suffix=".net") as fpnet1:

            fpy.write(self.test_code.encode("utf-8"))
            fpy.flush()

            module = module_from_file("smallnet", fpy.name)

            net = module.Net()
            opt = torch.optim.SGD(net.parameters(), lr=0.00001)
            save(net, fpnet1.name, opt=opt)

            print()
            print(summary(fpnet1.name))

    def test_save_torchvision(self):
        import torch
        from torchvision.models import resnet152

        with tempfile.NamedTemporaryFile(suffix=".net") as fpnet1, self.assertRaises(
            RuntimeError
        ):

            net = resnet152().cuda()
            opt = torch.optim.SGD(net.parameters(), lr=0.00001)
            save(net, fpnet1.name, opt=opt, check=True)

    def test_model_hashes(self):
        import torch
        from torchvision.models import resnet152

        with tempfile.NamedTemporaryFile(suffix=".pth") as fpth:
            net = resnet152().cuda()
            start = time.time()
            h1 = weights2hash(net)
            end = time.time()
            print(f"Hashing {net.__class__.__name__}: {end - start:.4f}s")
            torch.save(net, fpth.name)
            net2 = torch.load(fpth.name)
            h2 = weights2hash(net2)
            self.assertEqual(h1, h2, f"Expected hashes to match, got '{h1}' != '{h2}'")


if __name__ == "__main__":
    # test()
    unittest.main()

# def test():

#     import torch
#     from torchvision.models import resnet152

#     test_module = (
#         "import torch\n"
#         "import torch.nn as nn\n"
#         "class Net(nn.Module):\n"
#         "    def __init__(self, channels=32, hid=800, out=100):\n"
#         "        super().__init__()\n"
#         "        self.rln = nn.Sequential(\n"
#         "            nn.Conv2d(1, channels, 3),\n"
#         "            nn.ReLU(inplace=True),\n"
#         "            nn.MaxPool2d(2, 2),\n"
#         "            nn.Conv2d(channels, channels, 3),\n"
#         "            nn.ReLU(inplace=True),\n"
#         "            nn.MaxPool2d(2, 2),\n"
#         "            nn.Flatten(),\n"
#         "        )\n"
#         "        self.pln = nn.Linear(hid, 100)\n"
#         "\n"
#         "    def forward(self, x):\n"
#         "        x = self.rln(x)\n"
#         "        x = self.pln(x)\n"
#         "        return x\n"
#         "    pass\n"
#         "dummy_inp = torch.rand(4,1,28,28)\n"
#     )

#     with tempfile.NamedTemporaryFile(suffix=".py") as fpy, tempfile.NamedTemporaryFile(
#         suffix=".net"
#     ) as fpnet1, tempfile.NamedTemporaryFile(suffix=".net") as fpnet2:

#         fpy.write(test_module.encode("utf-8"))
#         fpy.flush()

#         path = fpy.name
#         name = "smallnet"
#         module = module_from_file(name, fpy.name)

#         net = module.Net()
#         print(net.__class__)
#         out = net(module.dummy_inp)

#         # test save/load of original model
#         save(net, fpnet1.name)
#         net1 = load(fpnet1.name)
#         print(net1.__class__)
#         out1 = net1(module.dummy_inp)

#         # test save/load of retrieved copy
#         save(net, fpnet2.name)
#         net2 = load(fpnet2.name)
#         print(net2.__class__)
#         out2 = net2(module.dummy_inp)

#         assert torch.all(out == out1)
#         assert torch.all(out == out2)
#         assert torch.all(out1 == out2)

#         print(summary(fpnet1.name))

#         print("Test saving optimizer info")
#         net = module.Net()
#         opt = torch.optim.SGD(net.parameters(), lr=0.00001)
#         save(net, fpnet1.name, opt=opt, check=True)
#         summary(fpnet1.name)

#         print("Test saving torchvision model")
#         try:
#             save(net, fpnet1.name, opt=opt, check=False)
#         except RuntimeError as e:
#             print(e)

#     with tempfile.NamedTemporaryFile(suffix=".pth") as fpth:
#         net = resnet152().cuda()
#         print(f"Hashing {net.__class__.__name__}")
#         start = time.time()
#         h1 = weights2hash(net)
#         end = time.time()
#         print(f"Elapsed {end - start:.4f}s")
#         torch.save(net, fpth.name)
#         net2 = torch.load(fpth.name)
#         h2 = weights2hash(net2)
#         assert h1 == h2, f"Expected hashes to match, got '{h1}' != '{h2}'"

# net = resnet152()
# for name, el in inspect.getmembers(torch.optim):
#     if el != torch.optim.Optimizer and inspect.isclass(el):
#         try:
#             opt = el(net.parameters(), lr=0.1)
#             confs = opt_summary(opt)
#             print(confs)
#             print()
#         except:
#             pass


# import inspect
# from torchvision.models import vgg11
# from models import SANML
# from omniglot import Omniglot

# net = SANML()
# print(net.__class__)
# print(net.__class__.__module__)
# net = vgg11()
# print(net.__class__)
# print(net.__class__.__module__)

# source_path = inspect.getsourcefile(net.__class__)

# s = Omniglot(root="../data/omni")
# find_localimports(s)


# from collections import OrderedDict
# from dataclasses import dataclass, field
# @dataclass()
# class Metadata:
#     cls: str
#     kwargs: dict
#     opt: str
#     state: OrderedDict
#     source: str = field(init=False)
#     creator: str = field(init=False)

#     def __post_init__(self):
#         self.state = deepcopy(self.state.state_dict())
#         self.opt = opt_summary(self.opt)
#         imports = find_localimports(net)
#         self.source = get_sources(imports)
#         self.creator = get_creator()

#     def __repr__(self):
#         dsize = 8
#         # "state": count params
#         nparams = sum([v.numel() for v in self.state.values()])

#         # "source": hash for comparison purposes
#         h = blake2b(digest_size=dsize)
#         h.update(self.source.encode("utf-8"))
#         source = h.hexdigest()

#         # "creator": hash if not empty
#         creator = self.creator
#         if creator != "":
#             h = blake2b(digest_size=dsize)
#             h.update(creator.encode("utf-8"))
#             creator = h.hexdigest()

#         # "class": print with kwargs
#         # "kwargs": print with class
#         kwargs = self.kwargs
#         params = [f"{key}={val}" for key, val in kwargs.items()]
#         instantiate = f"{self.cls}({', '.join(p for p in params)})"

#         pad = "=" * (14 + dsize * 2)
#         return (
#             f"{pad}\n"
#             f"{instantiate}\n"
#             f"{'source code':<12}= {source}\n"
#             f"{'created by':<12}= {creator}\n"
#             f"{'num. params':<12}= {nparams:,}\n"
#             f"{'optimizer   ':<12}= {self.opt}\n"
#             f"{pad}\n"
#         )


# import dataclasses
# import torch
# import torch.optim
# from models import SANML
# net = SANML()
# opt = torch.optim.SGD(net.parameters(), lr=0.00001)


# data = Metadata("a", {}, opt, net)
# print(data)

# safe = dataclasses.asdict(data)

# torch.save(safe,"tmp.pth")

# d = torch.load("tmp.pth")


# import torch

# torch.load("./DissectingANML_code/tmp.pth")

#%%
