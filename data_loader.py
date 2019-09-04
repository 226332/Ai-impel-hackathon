import logging
import os

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, utils


class DataLoaders:
    DATA_FOLDER = 'datasets'

    def __init__(self, opts):
        self.opts = opts

        self.transform_list = self.get_transforms_list()
        self.a, self.test_a, self.b, self.test_b = self.get_four_data_loaders()

        self.check_if_save_sample_output()

    def get_transforms_list(self):
        transforms_list = []
        return transforms_list

    def get_data_loaders(self, train_paths, test_paths):
        train_paths = sorted(train_paths)
        test_paths = sorted(test_paths)
        data_loaders = []

        for train, test in zip(train_paths, test_paths):
            train_dataset = self.get_extended_dataset(train)
            test_dataset = self.get_extended_dataset(test)

            data_loaders.append(DataLoader(dataset=train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=self.opts.num_workers))
            data_loaders.append(
                DataLoader(dataset=test_dataset,
                           batch_size=self.opts.batch_size,
                           shuffle=False, num_workers=self.opts.num_workers))
        return tuple(data_loaders)

    def get_four_data_loaders(self):

        path_to_datasets = os.path.join(f'./{self.DATA_FOLDER}',
                                        self.opts.dataset_name)

        test_path = []
        train_path = []
        for dataset_dir in os.listdir(path_to_datasets):
            if dataset_dir.startswith('Test_'):
                test_path.append(os.path.join(path_to_datasets, dataset_dir))
            else:
                train_path.append(os.path.join(path_to_datasets, dataset_dir))

        return self.get_data_loaders(train_path, test_path)

    def get_extended_dataset(self, vanilla_pics):
        dataset_list = []
        t = [
            transforms.Resize((self.opts.input_size, self.opts.input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        transform = transforms.Compose(t)
        dataset_list.append(datasets.ImageFolder(vanilla_pics, transform))

        t.insert(0, transforms.RandomHorizontalFlip(p=1))
        dataset_list.append(datasets.ImageFolder(vanilla_pics, transform))

        return ConcatDataset(dataset_list)

    def check_if_save_sample_output(self):
        sample_path = './samples/'
        output_file_name = 'sample_img.png'
        if self.opts.sample_output:
            tensor = iter(self.a).next()[0]
            utils.save_image(tensor[:, :, :],
                             sample_path + 'A' + output_file_name)
            tensor = iter(self.b).next()[0]
            utils.save_image(tensor[:, :, :],
                             sample_path + 'B' + output_file_name)
