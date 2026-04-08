import argparse

from sac_continuous import run_train as run_continuous
from sac_discrete import run_train as run_discrete


def parse_args():
    parser = argparse.ArgumentParser(description="SAC experiments")
    parser.add_argument(
        "--task",
        type=str,
        default="pendulum",
        choices=["pendulum", "cartpole"],
        help="pendulum: continuous SAC, cartpole: discrete SAC",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task == "pendulum":
        run_continuous(seed=args.seed)
    else:
        run_discrete(seed=args.seed)
