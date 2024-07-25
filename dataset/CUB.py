import torch
import pickle
import lightning as pl

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import cal_class_imbalance_weights
from torch.utils.data import DataLoader


class CUBDataSet(Dataset):
    def __init__(self, data_path, stage):
        self.data = []
        self.image_dir = "images"
        if data_path[-1] != "/":
            data_path += "/"
        data_path += "CUB_200_2011/"
        self.data_path = data_path
        if stage == "fit":
            self.data.extend(pickle.load(open(data_path + "train.pkl", "rb")))
            self.transform = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            if stage == "test":
                self.data.extend(pickle.load(open(data_path + "test.pkl", "rb")))
            else:
                self.data.extend(pickle.load(open(data_path + "val.pkl", "rb")))
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        self.imbalance_ratio = cal_class_imbalance_weights(self.data)

    def __getitem__(self, index):
        img_data = self.data[index]
        img_path = img_data["img_path"]
        idx = img_path.split("/").index("CUB_200_2011")
        if self.image_dir != "images":
            img_path = "/".join([self.image_dir] + img_path.split("/")[idx + 1 :])
            img_path = img_path.replace("images/", "")
        else:
            img_path = "/".join(img_path.split("/")[idx + 1 :])
        img = Image.open(self.data_path + img_path).convert("RGB")
        label = img_data["class_label"]
        img = self.transform(img)
        attr_label = img_data["attribute_label"]
        return img, label, torch.FloatTensor(attr_label)

    def __len__(self):
        return len(self.data)


class CUB(pl.LightningDataModule):
    def __init__(self, data_path, batch_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_data = CUBDataSet(self.data_path, "fit")
        self.val_data = CUBDataSet(self.data_path, "val")
        self.test_data = CUBDataSet(self.data_path, "test")
        self.imbalance_ratio = self.train_data.imbalance_ratio

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=24,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=24,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=24,
            pin_memory=True,
        )