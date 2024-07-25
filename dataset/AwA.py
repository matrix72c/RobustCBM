"""
General utils for training, evaluation and data loading
"""

import os
from glob import glob
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import lightning as L


class AwADataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, data_path, resol=224, is_train=True):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        image_dir: default = 'images'. Will be append to the parent dir
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """

        self.path = data_path
        class_to_index = dict()
        with open(self.path + "Animals_with_Attributes2/classes.txt") as f:
            index = 0
            for line in f:
                class_name = line.split("\t")[1].strip()
                class_to_index[class_name] = index
                index += 1

        img_names = []
        img_label = []
        if is_train:
            with open(self.path + "Animals_with_Attributes2/trainclasses.txt") as f:
                for line in f:
                    class_name = line.strip()
                    FOLDER_DIR = os.path.join(
                        self.path + "Animals_with_Attributes2/JPEGImages", class_name
                    )
                    file_descriptor = os.path.join(FOLDER_DIR, "*.jpg")
                    files = glob(file_descriptor)

                    class_index = class_to_index[class_name]
                    for file_name in files:
                        img_names.append(file_name)
                        img_label.append(class_index)
                self.transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
        else:
            with open(self.path + "Animals_with_Attributes2/testclasses.txt") as f:
                for line in f:
                    class_name = line.strip()
                    FOLDER_DIR = os.path.join(
                        self.path + "Animals_with_Attributes2/JPEGImages", class_name
                    )
                    file_descriptor = os.path.join(FOLDER_DIR, "*.jpg")
                    files = glob(file_descriptor)

                    class_index = class_to_index[class_name]
                    for file_name in files:
                        img_names.append(file_name)
                        img_label.append(class_index)
                self.transform = transforms.Compose(
                    [
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

        self.img_names = img_names
        self.img_label = img_label
        self.label_to_num = class_to_index
        self.label_to_attr = np.array(
            np.genfromtxt(
                self.path + "Animals_with_Attributes2/predicate-matrix-binary.txt",
                dtype="float32",
            )
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = Image.open(self.img_names[idx]).convert("RGB")
        class_label = self.img_label[idx]
        if self.transform:
            img = self.transform(img)

        attr_label = self.label_to_attr[class_label, :]

        return img, class_label, torch.Tensor(attr_label)


class AwA(L.LightningDataModule):
    def __init__(self, data_path, resol, batch_size):
        super().__init__()
        self.data_path = data_path
        self.resol = resol
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train = AwADataset(self.data_path, self.resol, is_train=True)
        self.val = AwADataset(self.data_path, self.resol, is_train=False)
        self.test = AwADataset(self.data_path, self.resol, is_train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, shuffle=False, num_workers=8
        )


if __name__ == "__main__":
    data_path = "./data/"
    resol = 224
    batch_size = 128
    dm = AwA(data_path, resol, batch_size)
    dm.setup()
    for x, y, z in dm.train:
        print(x.shape, y, z.shape)
        break
    for x, y, z in dm.val:
        print(x.shape, y, z.shape)
        break
    for x, y, z in dm.test:
        print(x.shape, y, z.shape)
        break
