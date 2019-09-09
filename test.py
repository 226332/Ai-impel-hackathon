import torch
from torch.autograd import Variable
from model import Generator256
import os
from PIL import Image
from torchvision import datasets, transforms, utils
import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', required=True,
                        type=str)
    parser.add_argument('--gen_path', required=True,
                        type=str)
    parser.add_argument('--out', default='created/generated.png',
                        type=str)
    return parser


def inference_from_generator(path_to_img, path_to_generator, out_path):
    # Data pre-processing
    loader = transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])

    # Test data
    img = Image.open(path_to_img)
    
    # Load model
    G = Generator256()
    G.cpu()
    G.load_state_dict(torch.load(path_to_generator, map_location=torch.device('cpu')))

    # Inference from generator
    real_A = loader(img).float()
    real_A = Variable(real_A.cpu())
    real_A.unsqueeze_(0)
    fake_B = G(real_A)
    save_image_from_tensor(fake_B, out_path)


def save_image_from_tensor(tensor, out_path):
    utils.save_image((tensor[:, :, :] * 0.5) + 0.5,
                         out_path)


def main(config):
    inference_from_generator(config.img_path, config.gen_path, config.out)
    return 1


if __name__ == "__main__":
    prepared_parser = create_parser()
    cfg = prepared_parser.parse_args()
    main(cfg)

