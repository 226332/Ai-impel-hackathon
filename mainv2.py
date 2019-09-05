import argparse
import logging
import sys
from solverv2 import Solver
from data_loader_better import DataLoaders

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train", choices=["train", "test"])
    parser.add_argument('--input_size', default=256, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=0.0005)
    parser.add_argument('--gpu', default=2, choices=[2, 3], type=int)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--dataset_name', default="forests_better", type=str)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--sample_output', default=False)
    return parser

def main(config):
    logging.info(f"Program started with given config {vars(config)}")
    dl = DataLoaders(config)
    loader1, loader2 = dl.a, dl.b
    solver = Solver(config, loader1, loader2)
    solver.train()
    return 1


if __name__ == "__main__":
    prepared_parser = create_parser()
    cfg = prepared_parser.parse_args()
    main(cfg)
