import glob
import itertools
import os
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset

import lightning as L

from utils import cal_class_imbalance_weights


class AwADataset(Dataset):
    def __init__(self, data_path, stage, num_concepts, resol):
        self.path = data_path
        self.num_concepts = num_concepts
        class_to_index = dict()
        with open(self.path + "/Animals_with_Attributes2/classes.txt") as f:
            index = 1
            for line in f:
                class_name = line.split("\t")[1].strip()
                class_to_index[class_name] = index
                index += 1
        img_names = []
        img_index = []
        for c in class_to_index.keys():
            class_name = c
            FOLDER_DIR = os.path.join(
                f"{data_path}/Animals_with_Attributes2/JPEGImages", class_name
            )
            file_descriptor = os.path.join(FOLDER_DIR, "*.jpg")
            files = glob.glob(file_descriptor)

            class_index = class_to_index[class_name]
            for file_name in files:
                img_names.append(file_name)
                img_index.append(class_index)

        train_img_names, eval_img_names, train_img_index, eval_img_index = (
            train_test_split(img_names, img_index, test_size=0.3)
        )
        test_img_names, val_img_names, test_img_index, val_img_index = train_test_split(
            eval_img_names, eval_img_index, test_size=0.33
        )

        if stage == "fit":
            self.img_names = train_img_names
            self.img_index = train_img_index
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
            if stage == "val":
                self.img_names = val_img_names
                self.img_index = val_img_index
            elif stage == "test":
                self.img_names = test_img_names
                self.img_index = test_img_index
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(resol),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        self.label_to_num = class_to_index
        self.label_to_attr = np.array(
            np.genfromtxt(
                self.path + "/Animals_with_Attributes2/predicate-matrix-binary.txt",
                dtype="float32",
            )
        )
        concept_counts = np.sum(self.label_to_attr, axis=0)
        most_common = np.argsort(concept_counts)[::-1]
        self.combos = list(itertools.combinations(most_common[:32], 2))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = Image.open(self.img_names[idx]).convert("RGB")
        class_label = self.img_index[idx] - 1
        if self.transform:
            img = self.transform(img)

        attr_label = torch.Tensor(self.label_to_attr[class_label, :])
        combo_attr = torch.zeros(len(self.combos))
        for i, (a, b) in enumerate(self.combos):
            combo_attr[i] = attr_label[a] * attr_label[b]
        if self.num_concepts < 85:
            attr_label = attr_label[: self.num_concepts]
        else:
            attr_label = torch.cat(
                (attr_label, combo_attr[: self.num_concepts - 85]), dim=0
            )

        return img, class_label, attr_label


class AwA(L.LightningDataModule):
    def __init__(self, data_path, batch_size, num_concepts=85, resol=224, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_concepts = num_concepts
        self.real_concepts = 85
        self.num_classes = 50
        self.train = AwADataset(self.data_path, "fit", num_concepts, resol)
        self.val = AwADataset(self.data_path, "val", num_concepts, resol)
        self.test = AwADataset(self.data_path, "test", num_concepts, resol)
        sample = Subset(self.train, torch.arange(len(self.train) // 10))
        self.imbalance_weights = cal_class_imbalance_weights(sample)

        self.max_intervene_budget = 85
        self.concept_group_map = {}
        for idx in range(self.num_concepts):
            self.concept_group_map[idx] = [idx]
        self.group_concept_map = {}
        for name, idxs in self.concept_group_map.items():
            for idx in idxs:
                self.group_concept_map[idx] = name

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=24,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=24,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=24,
            pin_memory=True,
        )


if __name__ == "__main__":
    data_path = "./data/"
    resol = 224
    batch_size = 128
    dm = AwA(data_path, resol, batch_size)
    dm.setup("fit")
    for x, y, z in dm.train:
        print(x.shape, y, z.shape)
        break
    for x, y, z in dm.val:
        print(x.shape, y, z.shape)
        break
    for x, y, z in dm.test:
        print(x.shape, y, z.shape)
        break
