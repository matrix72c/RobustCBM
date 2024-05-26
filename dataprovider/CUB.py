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
        if data_path[-1] != "/":
            data_path += "/"
        data_path += "CUB_200_2011/"
        self.data_path = data_path
        if is_train:
            self.data.extend(pickle.load(open(data_path + "train.pkl", "rb")))
            self.transform = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                    transforms.RandomResizedCrop(resol),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.data.extend(pickle.load(open(data_path + "val.pkl", "rb")))
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(resol),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        self.imbalance_ratio = None
        self.mask = None

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

