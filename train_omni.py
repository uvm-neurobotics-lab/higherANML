import argparse
from torch import manual_seed
from torch.cuda import is_available

from anml import train

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="ANML training")
    parser.add_argument(
        "--rln", type=int, default=256, help="number of channels to use in the RLN"
    )
    parser.add_argument(
        "--nm", type=int, default=112, help="number of channels to use in the NM"
    )
    parser.add_argument(
        "--mask",
        type=int,
        default=2304,
        help="size of the modulatory mask, needs to match extracted features size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30000,
        help="number of epochs to train (default: 30000)",
    )
    parser.add_argument(
        "--inner_lr",
        type=float,
        default=1e-1,
        help="inner learning rate (default: 1e-1)",
    )
    parser.add_argument(
        "--outer_lr",
        type=float,
        default=1e-3,
        help="outer learning rate (default: 1e-3)",
    )
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    args = parser.parse_args()
    device = "cpu" if not is_available() else args.device
    manual_seed(args.seed)

    train(
        args.rln,
        args.nm,
        args.mask,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        its=args.epochs,
        device=device,
    )
