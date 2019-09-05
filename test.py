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
    parser.add_argument('--out', default='tested',
                        type=str)
    return parser


def main(config):
    inference_from_generator(config.img_path, config.gen_path, config.out)
    return 1


if __name__ == "__main__":
    prepared_parser = create_parser()
    cfg = prepared_parser.parse_args()
    main(cfg)


def inference_from_generator(path_to_img, path_to_generator, out_path):
    # Data pre-processing
    loader = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Scale((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])

    # Test data
    img = Image.open(path_to_img)
    # Load model
    G = Generator256()
    G.cpu()
    G.load_state_dict(torch.load(path_to_generator))

    # Test
    real_A = loader(img).float()
    real_A = Variable(real_A.cpu())
    real_A.unsqueeze_(0)
    fake_B = G(real_A)
    save_image_from_tensor(fake_B)


def save_image_from_tensor(tensor):
    parent_dir = out_path
    path = os.path.join(parent_dir, )
    if not os.path.exists(path):
        os.makedirs(path)

    def save_img_from_tensor(tensor, filename):
        utils.save_image((tensor[:, :, :] * 0.5) + 0.5,
                         os.path.join(path, filename))

    save_img_from_tensor(tensor, "kupa.png")
