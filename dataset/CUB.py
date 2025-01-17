import itertools
import torch
import pickle
import lightning as L

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import cal_class_imbalance_weights


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
            if stage == "val":
                self.data.extend(pickle.load(open(data_path + "val.pkl", "rb")))
            elif stage == "test":
                self.data.extend(pickle.load(open(data_path + "test.pkl", "rb")))
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        concept_counts = torch.zeros(len(self.data[0]["attribute_label"]))
        for img_data in self.data:
            concept_counts += torch.FloatTensor(img_data["attribute_label"])
        most_common = concept_counts.argsort(descending=True)
        self.combos = list(itertools.combinations(most_common[:32], 2))

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
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_concepts = num_concepts
        self.real_concepts = 112
        self.num_classes = 200
        self.train_data = CUBDataSet(self.data_path, "fit", num_concepts)
        self.val_data = CUBDataSet(self.data_path, "val", num_concepts)
        self.test_data = CUBDataSet(self.data_path, "test", num_concepts)
        self.imbalance_weights = cal_class_imbalance_weights(self.train_data)

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
