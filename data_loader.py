import os
from typing import Tuple, List

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, utils
import itertools
from datetime import datetime


class DataLoaders:
    DATA_FOLDER = 'datasets'

    def __init__(self, opts):
        self.opts = opts
        self.a, self.test_a, self.b, self.test_b = self.get_four_data_loaders()
        self.check_if_save_sample_output()

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

    def get_extended_dataset(self, vanilla_pics):
        dataset_list = []
        basic_transforms = [
            transforms.Resize((self.opts.input_size, self.opts.input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        additional_transforms = [
            transforms.RandomHorizontalFlip(p=1),
            # transforms.ColorJitter(0, 0.1, 0.1),
            transforms.RandomResizedCrop(self.opts.input_size)
            # transforms.RandomCrop(450, padding_mode='edge')
        ]

        for transforms_sublist in self.list_combinations(additional_transforms):
            transforms_combined = list(transforms_sublist) + basic_transforms
            transform = transforms.Compose(transforms_combined)
            dataset_list.append(datasets.ImageFolder(vanilla_pics, transform))

        return ConcatDataset(dataset_list)

    @staticmethod
    def list_combinations(list1) -> List[Tuple]:
        out_list = []
        for i in range(len(list1) + 1):
            out_list.extend(
                list(itertools.combinations(list1, i))
            )
        return out_list

    def check_if_save_sample_output(self):
        sample_path = './samples/'
        output_file_name = 'data_loader_sample'
        output_file_format = '.png'
        if self.opts.sample_output:
            time = datetime.now().strftime("%H:%M:%S")

            def save_images_from_data_loader(data_loader, label):
                it = iter(data_loader)
                z = 0
                for tensor in it:
                    z += 1
                    utils.save_image(tensor[0][:, :, :],
                                     sample_path + output_file_name
                                     + f'_set_{label}_{z}_' + time
                                     + output_file_format)

            save_images_from_data_loader(self.a, 'A')
            save_images_from_data_loader(self.b, 'B')
