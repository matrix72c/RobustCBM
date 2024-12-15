"""
General utils for training, evaluation and data loading
"""

import pandas as pd
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import lightning as L


class AwADataset(Dataset):
    def __init__(self, data_path, stage, num_concepts):
        self.path = data_path
        self.num_concepts = num_concepts
        class_to_index = dict()
        with open(self.path + "Animals_with_Attributes2/classes.txt") as f:
            index = 1
            for line in f:
                class_name = line.split("\t")[1].strip()
                class_to_index[class_name] = index
                index += 1

        if stage == "fit":
            df = pd.read_csv(self.path + "Animals_with_Attributes2/train.csv")
            self.transform = transforms.Compose(
                [
                    transforms.ColorJitter(
                        brightness=32 / 255, saturation=(0.5, 1.5)
                    ),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
                ]
            )
        elif stage == "val" or stage == "test":
            df = pd.read_csv(self.path + "Animals_with_Attributes2/" + stage + ".csv")
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
                ]
            )
        self.img_names = df["img_name"].tolist()
        self.img_index = df["img_index"].tolist()
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
        class_label = self.img_index[idx] - 1
        if self.transform:
            img = self.transform(img)

        attr_label = torch.Tensor(self.label_to_attr[class_label, :])
        if self.num_concepts < 85:
            attr_label = attr_label[: self.num_concepts]
        else:
            attr_label = torch.cat(
                (attr_label, torch.zeros(self.num_concepts - 85)), dim=0
            )

        return img, class_label, attr_label


class AwA(L.LightningDataModule):
    def __init__(self, data_path, batch_size, num_concepts=85):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_concepts = num_concepts
        self.train = AwADataset(self.data_path, "fit", num_concepts)
        self.val = AwADataset(self.data_path, "val", num_concepts)
        self.test = AwADataset(self.data_path, "test", num_concepts)

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
