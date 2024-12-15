import itertools
import torch
import pickle
import lightning as L

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class CUBDataSet(Dataset):
    def __init__(self, data_path, stage, num_concepts):
        self.data = []
        self.image_dir = "images"
        if data_path[-1] != "/":
            data_path += "/"
        data_path += "CUB_200_2011/"
        self.data_path = data_path
        self.num_concepts = num_concepts
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
        self.combos = list(itertools.combinations(range(112), 2))

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
        attr_label = torch.FloatTensor(img_data["attribute_label"])
        combo_attr = torch.zeros(len(self.combos))
        for i, (a, b) in enumerate(self.combos):
            combo_attr[i] = attr_label[a] * attr_label[b]
        if self.num_concepts < 112:
            attr_label = attr_label[: self.num_concepts]
        elif self.num_concepts > 112:
            attr_label = torch.cat(
                [attr_label, combo_attr[: self.num_concepts - 112]], dim=0
            )
        return img, label, attr_label

    def __len__(self):
        return len(self.data)


class CUB(L.LightningDataModule):
    def __init__(
        self,
        data_path,
        batch_size,
        num_concepts=112,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_concepts = num_concepts
        self.train_data = CUBDataSet(self.data_path, "fit", num_concepts)
        self.val_data = CUBDataSet(self.data_path, "val", num_concepts)
        self.test_data = CUBDataSet(self.data_path, "test", num_concepts)

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
