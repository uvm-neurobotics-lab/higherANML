"""
ANML Training Batch Job
"""

import logging
import sys

import utils.argparsing as argutils
from train_anml import setup_and_train
from utils import load_yaml


def main(argv=None):
    parser = argutils.create_parser(__doc__)
    parser.add_argument("-c", "--config", metavar="PATH", type=argutils.existing_path, required=True,
                        help="Training config file.")
    argutils.add_verbose_arg(parser)

    args = parser.parse_args(argv)
    argutils.configure_logging(args, level=logging.INFO)
    config = load_yaml(args.config)
    setup_and_train(parser, config, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
