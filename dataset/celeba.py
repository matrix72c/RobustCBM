import torch
import lightning as L

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np


SELECTED_CONCEPTS = [
    2,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    33,
    39,
]

CONCEPT_SEMANTICS = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


class celebaDataset(Dataset):
    def __init__(self, celeba, num_concepts):
        self.celeba = celeba
        self.num_concepts = num_concepts

    def __getitem__(self, index):
        x, y = self.celeba[index]
        label, concept = y
        if self.num_concepts < 8:
            concept = concept[: self.num_concepts]
        else:
            concept = torch.cat(
                (concept, torch.zeros(self.num_concepts - concept.shape[0]))
            )
        return x, label, concept

    def __len__(self):
        return len(self.celeba)


class celeba(L.LightningDataModule):
    def __init__(self, data_path, batch_size, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        width = 1

        def _binarize(concepts, selected, width):
            result = []
            binary_repr = []
            concepts = concepts[selected]
            for i in range(0, concepts.shape[-1], width):
                binary_repr.append(str(int(np.sum(concepts[i : i + width]) > 0)))
            return int("".join(binary_repr), 2)

        celeba_train_data = torchvision.datasets.CelebA(
            root=data_path,
            split="all",
            download=True,
            target_transform=lambda x: x[0].long() - 1,
            target_type=["attr"],
        )

        concept_freq = (
            np.sum(celeba_train_data.attr.cpu().detach().numpy(), axis=0)
            / celeba_train_data.attr.shape[0]
        )
        sorted_concepts = list(
            map(
                lambda x: x[0],
                sorted(enumerate(np.abs(concept_freq - 0.5)), key=lambda x: x[1]),
            )
        )
        num_concepts = 6
        concept_idxs = sorted_concepts[:num_concepts]
        concept_idxs = sorted(concept_idxs)
        num_hidden = 2
        hidden_concepts = sorted(
            sorted_concepts[
                num_concepts : min((num_concepts + num_hidden), len(sorted_concepts))
            ]
        )
        celeba_train_data = torchvision.datasets.CelebA(
            root=data_path,
            split="all",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
            target_transform=lambda x: [
                torch.tensor(
                    _binarize(
                        x[1].cpu().detach().numpy(),
                        selected=(concept_idxs + hidden_concepts),
                        width=width,
                    ),
                    dtype=torch.long,
                ),
                x[1][concept_idxs].float(),
            ],
            target_type=["identity", "attr"],
        )
        label_remap = {}
        vals, counts = np.unique(
            list(
                map(
                    lambda x: _binarize(
                        x.cpu().detach().numpy(),
                        selected=(concept_idxs + hidden_concepts),
                        width=width,
                    ),
                    celeba_train_data.attr,
                )
            ),
            return_counts=True,
        )
        for i, label in enumerate(vals):
            label_remap[label] = i

        celeba_train_data = torchvision.datasets.CelebA(
            root=data_path,
            split="all",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
            target_transform=lambda x: [
                torch.tensor(
                    label_remap[
                        _binarize(
                            x[1].cpu().detach().numpy(),
                            selected=(concept_idxs + hidden_concepts),
                            width=width,
                        )
                    ],
                    dtype=torch.long,
                ),
                x[1][concept_idxs].float(),
            ],
            target_type=["identity", "attr"],
        )
        celeba_train_data = celebaDataset(celeba_train_data, num_concepts)

        # And subsample to reduce its massive size
        factor = 12
        train_idxs = np.random.choice(
            np.arange(0, len(celeba_train_data)),
            replace=False,
            size=len(celeba_train_data) // factor,
        )
        celeba_train_data = torch.utils.data.Subset(
            celeba_train_data,
            train_idxs,
        )
        total_samples = len(celeba_train_data)
        train_samples = int(0.7 * total_samples)
        test_samples = int(0.2 * total_samples)
        val_samples = total_samples - test_samples - train_samples
        self.train_data, self.test_data, self.val_data = torch.utils.data.random_split(
            celeba_train_data,
            [train_samples, test_samples, val_samples],
        )

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


if __name__ == "__main__":
    data_path = "./data/"
    batch_size = 128
    dm = celeba(data_path, batch_size)
    dm.setup("fit")
    loader = dm.test_dataloader()
    for batch in loader:
        img, concept, label = batch
        print(img.shape, concept.shape, label.shape)
        break
