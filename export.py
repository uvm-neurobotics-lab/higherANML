"""
A script to export models to TorchScript. The source format should be the one saved by the `storage` module.

Example command:
  python export.py trained_anmls/anml-1-28-28-29999.net
"""
import sys

import torch

import utils.argparsing as argutils
from models import load_model


def main(args=None):
    parser = argutils.create_parser(__doc__)
    parser.add_argument("model", type=argutils.existing_path, help="Path to the model to evaluate.")
    argutils.add_device_arg(parser)

    args = parser.parse_args(args)
    argutils.configure_logging(args)
    device = argutils.get_device(parser, args)

    # NOTE: This depends on the current naming scheme of the model files, which may change!
    input_shape = tuple(int(x) for x in args.model.stem.split("-")[-4:-1])
    model = load_model(args.model, input_shape, device)

    # Normally we'd just do the below, but this doesn't work for models loaded from cold storage. :-/
    # Instead, we'll need to use `torch.jit.trace()`.
    # scripted_module = torch.jit.script(model)

    # Simulate a batch by adding an extra dim at the beginning.
    bsize = 3
    batch_shape = (bsize,) + tuple(input_shape)
    script_module = torch.jit.trace(model, torch.zeros(batch_shape))

    dest_file = args.model.parent / (args.model.stem + ".pt")
    script_module.save(dest_file)
    print(f"Model:\n{script_module}\n")
    print(f"Model saved to: {dest_file}")


if __name__ == "__main__":
    sys.exit(main())
