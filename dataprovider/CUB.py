import os
import torch
import pickle

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CUB(Dataset):
    def __init__(self, data_path, resol, is_train=True):
        self.data = []
        self.is_train = is_train
        self.image_dir = "images"
        self.data_path = data_path
        self.data.extend(
            pickle.load(
                open(
                    data_path + "train.pkl" if is_train else data_path + "val.pkl", "rb"
                )
            )
        )
        if is_train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((resol, resol)),
                    # transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                    # transforms.RandomResizedCrop(resol),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # implicitly divides by 255
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
                    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
                ]
            )
            self.imbalance_ratio = cal_class_imbalance_weights(data_path + "train.pkl")
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((resol, resol)),
                    transforms.CenterCrop(resol),
                    transforms.ToTensor(),  # implicitly divides by 255
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
                    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
                ]
            )

    def __getitem__(self, index):
        img_data = self.data[index]
        img_path = img_data["img_path"]
        try:
            idx = img_path.split("/").index("CUB_200_2011")
            if self.image_dir != "images":
                img_path = "/".join([self.image_dir] + img_path.split("/")[idx + 1 :])
                img_path = img_path.replace("images/", "")
            else:
                img_path = "/".join(img_path.split("/")[idx + 1 :])
            img = Image.open(self.data_path + img_path).convert("RGB")
        except:
            img_path_split = img_path.split("/")
            split = "train" if self.is_train else "test"
            img_path = "/".join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(self.data_path + img_path).convert("RGB")
        label = img_data["class_label"]
        img = self.transform(img)
        attr_label = img_data["attribute_label"]
        return img, label, torch.FloatTensor(attr_label)

    def __len__(self):
        return len(self.data)


def cal_class_imbalance_weights(path):
    """
    Computes the class imbalance weights for a dataset
    """
    data = pickle.load(open(path, "rb"))
    imbalance_ratio = []
    n = len(data)
    n_attr = len(data[0]["attribute_label"])

    n_ones = [0] * n_attr
    total = [n] * n_attr
    for d in data:
        labels = d["attribute_label"]

        for i in range(n_attr):
            n_ones[i] += labels[i]
    for j in range(len(n_ones)):
        imbalance_ratio.append(total[j] / n_ones[j] - 1)
    return imbalance_ratio
