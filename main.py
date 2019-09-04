import argparse
import logging
import sys
from solver import Solver

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train", choices=["train", "test"], required=True)
    parser.add_argument('--input_size', default=32)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--gpu', default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--train_iter', default=100)
    return parser


def main(config):
    logging.info(f"Program started with given config {vars(config)}")
    loader1, loader2 = ("loader1", "loader2")
    solver = Solver(config, loader1, loader2)
    # TODO trainer/model/loader
    return 1


if __name__ == "__main__":
    prepared_parser = create_parser()
    cfg = prepared_parser.parse_args()
    main(cfg)
