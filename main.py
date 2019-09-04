import argparse
import logging
import sys
from train import Solver
from test import inference_from_generator
from data_loader import DataLoaders

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s', datefmt='%H:%M:%S')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train", choices=["train", "test"], type=str)
    parser.add_argument('--input_size', default=256, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--gpu', default=2, choices=[2, 3], type=int)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--dataset_name', default="forests_better", type=str)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--sample_output', default=False)
    parser.add_argument('--decay', default=20, type=int)
    parser.add_argument('--lamb', default=10, type=float)
    parser.add_argument('--checkpoint', default=20, type=int)
    return parser

def main(config):
    logging.info(f"Program started with given config {vars(config)}")
    dl = DataLoaders(config)
    if config.mode == "train":
        loader1, loader2 = dl.a, dl.b
        solver = Solver(config, loader1, loader2)
        solver.train()
    else:
        inference_from_generator("./datasets/forests_final/normal_better/normal/10029939946_b1182350ff_o.jpg",
        "./models_params/3_forests_final_12:47:59_epoch_no_200/epoch_20/g12.pkl")


if __name__ == "__main__":
    prepared_parser = create_parser()
    cfg = prepared_parser.parse_args()
    main(cfg)
