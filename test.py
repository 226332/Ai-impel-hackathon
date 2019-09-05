import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder
from model import Generator256
import utils
import argparse
import os

# Directories for loading data and saving results
data_dir = '../Data/' + params.dataset + '/'
save_dir = params.dataset + '_test_results/'
model_dir = params.dataset + '_model/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Data pre-processing
transform = transforms.Compose([transforms.Scale(params.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Test data
test_data_A = DatasetFromFolder(data_dir, subfolder='testA', transform=transform)
test_data_loader_A = torch.utils.data.DataLoader(dataset=test_data_A,
                                                 batch_size=1,
                                                 shuffle=False)
test_data_B = DatasetFromFolder(data_dir, subfolder='testB', transform=transform)
test_data_loader_B = torch.utils.data.DataLoader(dataset=test_data_B,
                                                 batch_size=1,
                                                 shuffle=False)

# Load model
G_A = Generator256()
G_B = Generator256()
G_A.cuda()
G_B.cuda()
G_A.load_state_dict(torch.load(model_dir + 'generator_A_param.pkl'))
G_B.load_state_dict(torch.load(model_dir + 'generator_B_param.pkl'))

# Test
for i, real_A in enumerate(test_data_loader_A):

    # input image data
    real_A = Variable(real_A.cuda())

    # A -> B -> A
    fake_B = G_A(real_A)
    recon_A = G_B(fake_B)

    # Show result for test data
    utils.plot_test_result(real_A, fake_B, recon_A, i, save=True, save_dir=save_dir + 'AtoB/')

    print('%d images are generated.' % (i + 1))

    def save_output_tensors():
        data1 = iter(self.loader1).next()[0]
        data2 = iter(self.loader2).next()[0]

        parent_dir = 'output'
        path = os.path.join(
            parent_dir,
            f'{self.gpu}_{self.dataset_name}_{self.start_time}_epoch_no_{self.epoch}',
            f'epoch_{self.current_epoch}')
        if not os.path.exists(path):
            os.makedirs(path)

        def save_img_from_tensor(tensor, filename):
            utils.save_image((tensor[:, :, :] * 0.5) + 0.5,
                             os.path.join(path, filename))

        save_img_from_tensor(data1, "set_1_real.png")
        save_img_from_tensor(data2, "set_2_real.png")

        tensor1 = self.g21(self.__make_var(data2))
        tensor2 = self.g12(self.__make_var(data1))

        save_img_from_tensor(tensor1, "set_1_fake.png")
        save_img_from_tensor(tensor2, "set_2_fake.png")