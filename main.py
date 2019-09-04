import argparse
import logging


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train", choices=["train", "test"], required=True)
    parser.add_argument('--input_size', default=32)
    parser.add_argument('--batch-size', default=0)
    parser.add_argument('--gpu', default=0)
    return parser


def main(config):
    logging.info("Program started with given config %s" % vars(config))
    # TODO trainer/model/loader
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepared_parser = create_parser()
    cfg = prepared_parser.parse_args()
    main(cfg)
