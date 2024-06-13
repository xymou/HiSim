import os
import logging
from argparse import ArgumentParser

from agentverse.logging import logger
from agentverse.simulation import Simulation

parser = ArgumentParser()
parser.add_argument("--debug", action="store_true")
parser.add_argument("--ckpt_dir", type=str, default=None)
args = parser.parse_args()

logger.set_level(logging.DEBUG if args.debug else logging.INFO)


def cli_main():
    agentverse = Simulation.from_ckpt(args.ckpt_dir)
    agentverse.run_from_ckpt()


if __name__ == "__main__":
    cli_main()
