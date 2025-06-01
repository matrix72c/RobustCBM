import itertools
import os
import numpy as np
import torch
import pickle
import lightning as L
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import cal_class_imbalance_weights

# Set of CUB attributes selected by original CBM paper
SELECTED_CONCEPTS = [
    1,
    4,
    6,
    7,
    10,
    14,
    15,
    20,
    21,
    23,
    25,
    29,
    30,
    35,
    36,
    38,
    40,
    44,
    45,
    50,
    51,
    53,
    54,
    56,
    57,
    59,
    63,
    64,
    69,
    70,
    72,
    75,
    80,
    84,
    90,
    91,
    93,
    99,
    101,
    106,
    110,
    111,
    116,
    117,
    119,
    125,
    126,
    131,
    132,
    134,
    145,
    149,
    151,
    152,
    153,
    157,
    158,
    163,
    164,
    168,
    172,
    178,
    179,
    181,
    183,
    187,
    188,
    193,
    194,
    196,
    198,
    202,
    203,
    208,
    209,
    211,
    212,
    213,
    218,
    220,
    221,
    225,
    235,
    236,
    238,
    239,
    240,
    242,
    243,
    244,
    249,
    253,
    254,
    259,
    260,
    262,
    268,
    274,
    277,
    283,
    289,
    292,
    293,
    294,
    298,
    299,
    304,
    305,
    308,
    309,
    310,
    311,
]

# Names of all CUB attributes
CONCEPT_SEMANTICS = [
    "has_bill_shape::curved_(up_or_down)",
    "has_bill_shape::dagger",
    "has_bill_shape::hooked",
    "has_bill_shape::needle",
    "has_bill_shape::hooked_seabird",
    "has_bill_shape::spatulate",
    "has_bill_shape::all-purpose",
    "has_bill_shape::cone",
    "has_bill_shape::specialized",
    "has_wing_color::blue",
    "has_wing_color::brown",
    "has_wing_color::iridescent",
    "has_wing_color::purple",
    "has_wing_color::rufous",
    "has_wing_color::grey",
    "has_wing_color::yellow",
    "has_wing_color::olive",
    "has_wing_color::green",
    "has_wing_color::pink",
    "has_wing_color::orange",
    "has_wing_color::black",
    "has_wing_color::white",
    "has_wing_color::red",
    "has_wing_color::buff",
    "has_upperparts_color::blue",
    "has_upperparts_color::brown",
    "has_upperparts_color::iridescent",
    "has_upperparts_color::purple",
    "has_upperparts_color::rufous",
    "has_upperparts_color::grey",
    "has_upperparts_color::yellow",
    "has_upperparts_color::olive",
    "has_upperparts_color::green",
    "has_upperparts_color::pink",
    "has_upperparts_color::orange",
    "has_upperparts_color::black",
    "has_upperparts_color::white",
    "has_upperparts_color::red",
    "has_upperparts_color::buff",
    "has_underparts_color::blue",
    "has_underparts_color::brown",
    "has_underparts_color::iridescent",
    "has_underparts_color::purple",
    "has_underparts_color::rufous",
    "has_underparts_color::grey",
    "has_underparts_color::yellow",
    "has_underparts_color::olive",
    "has_underparts_color::green",
    "has_underparts_color::pink",
    "has_underparts_color::orange",
    "has_underparts_color::black",
    "has_underparts_color::white",
    "has_underparts_color::red",
    "has_underparts_color::buff",
    "has_breast_pattern::solid",
    "has_breast_pattern::spotted",
    "has_breast_pattern::striped",
    "has_breast_pattern::multi-colored",
    "has_back_color::blue",
    "has_back_color::brown",
    "has_back_color::iridescent",
    "has_back_color::purple",
    "has_back_color::rufous",
    "has_back_color::grey",
    "has_back_color::yellow",
    "has_back_color::olive",
    "has_back_color::green",
    "has_back_color::pink",
    "has_back_color::orange",
    "has_back_color::black",
    "has_back_color::white",
    "has_back_color::red",
    "has_back_color::buff",
    "has_tail_shape::forked_tail",
    "has_tail_shape::rounded_tail",
    "has_tail_shape::notched_tail",
    "has_tail_shape::fan-shaped_tail",
    "has_tail_shape::pointed_tail",
    "has_tail_shape::squared_tail",
    "has_upper_tail_color::blue",
    "has_upper_tail_color::brown",
    "has_upper_tail_color::iridescent",
    "has_upper_tail_color::purple",
    "has_upper_tail_color::rufous",
    "has_upper_tail_color::grey",
    "has_upper_tail_color::yellow",
    "has_upper_tail_color::olive",
    "has_upper_tail_color::green",
    "has_upper_tail_color::pink",
    "has_upper_tail_color::orange",
    "has_upper_tail_color::black",
    "has_upper_tail_color::white",
    "has_upper_tail_color::red",
    "has_upper_tail_color::buff",
    "has_head_pattern::spotted",
    "has_head_pattern::malar",
    "has_head_pattern::crested",
    "has_head_pattern::masked",
    "has_head_pattern::unique_pattern",
    "has_head_pattern::eyebrow",
    "has_head_pattern::eyering",
    "has_head_pattern::plain",
    "has_head_pattern::eyeline",
    "has_head_pattern::striped",
    "has_head_pattern::capped",
    "has_breast_color::blue",
    "has_breast_color::brown",
    "has_breast_color::iridescent",
    "has_breast_color::purple",
    "has_breast_color::rufous",
    "has_breast_color::grey",
    "has_breast_color::yellow",
    "has_breast_color::olive",
    "has_breast_color::green",
    "has_breast_color::pink",
    "has_breast_color::orange",
    "has_breast_color::black",
    "has_breast_color::white",
    "has_breast_color::red",
    "has_breast_color::buff",
    "has_throat_color::blue",
    "has_throat_color::brown",
    "has_throat_color::iridescent",
    "has_throat_color::purple",
    "has_throat_color::rufous",
    "has_throat_color::grey",
    "has_throat_color::yellow",
    "has_throat_color::olive",
    "has_throat_color::green",
    "has_throat_color::pink",
    "has_throat_color::orange",
    "has_throat_color::black",
    "has_throat_color::white",
    "has_throat_color::red",
    "has_throat_color::buff",
    "has_eye_color::blue",
    "has_eye_color::brown",
    "has_eye_color::purple",
    "has_eye_color::rufous",
    "has_eye_color::grey",
    "has_eye_color::yellow",
    "has_eye_color::olive",
    "has_eye_color::green",
    "has_eye_color::pink",
    "has_eye_color::orange",
    "has_eye_color::black",
    "has_eye_color::white",
    "has_eye_color::red",
    "has_eye_color::buff",
    "has_bill_length::about_the_same_as_head",
    "has_bill_length::longer_than_head",
    "has_bill_length::shorter_than_head",
    "has_forehead_color::blue",
    "has_forehead_color::brown",
    "has_forehead_color::iridescent",
    "has_forehead_color::purple",
    "has_forehead_color::rufous",
    "has_forehead_color::grey",
    "has_forehead_color::yellow",
    "has_forehead_color::olive",
    "has_forehead_color::green",
    "has_forehead_color::pink",
    "has_forehead_color::orange",
    "has_forehead_color::black",
    "has_forehead_color::white",
    "has_forehead_color::red",
    "has_forehead_color::buff",
    "has_under_tail_color::blue",
    "has_under_tail_color::brown",
    "has_under_tail_color::iridescent",
    "has_under_tail_color::purple",
    "has_under_tail_color::rufous",
    "has_under_tail_color::grey",
    "has_under_tail_color::yellow",
    "has_under_tail_color::olive",
    "has_under_tail_color::green",
    "has_under_tail_color::pink",
    "has_under_tail_color::orange",
    "has_under_tail_color::black",
    "has_under_tail_color::white",
    "has_under_tail_color::red",
    "has_under_tail_color::buff",
    "has_nape_color::blue",
    "has_nape_color::brown",
    "has_nape_color::iridescent",
    "has_nape_color::purple",
    "has_nape_color::rufous",
    "has_nape_color::grey",
    "has_nape_color::yellow",
    "has_nape_color::olive",
    "has_nape_color::green",
    "has_nape_color::pink",
    "has_nape_color::orange",
    "has_nape_color::black",
    "has_nape_color::white",
    "has_nape_color::red",
    "has_nape_color::buff",
    "has_belly_color::blue",
    "has_belly_color::brown",
    "has_belly_color::iridescent",
    "has_belly_color::purple",
    "has_belly_color::rufous",
    "has_belly_color::grey",
    "has_belly_color::yellow",
    "has_belly_color::olive",
    "has_belly_color::green",
    "has_belly_color::pink",
    "has_belly_color::orange",
    "has_belly_color::black",
    "has_belly_color::white",
    "has_belly_color::red",
    "has_belly_color::buff",
    "has_wing_shape::rounded-wings",
    "has_wing_shape::pointed-wings",
    "has_wing_shape::broad-wings",
    "has_wing_shape::tapered-wings",
    "has_wing_shape::long-wings",
    "has_size::large_(16_-_32_in)",
    "has_size::small_(5_-_9_in)",
    "has_size::very_large_(32_-_72_in)",
    "has_size::medium_(9_-_16_in)",
    "has_size::very_small_(3_-_5_in)",
    "has_shape::upright-perching_water-like",
    "has_shape::chicken-like-marsh",
    "has_shape::long-legged-like",
    "has_shape::duck-like",
    "has_shape::owl-like",
    "has_shape::gull-like",
    "has_shape::hummingbird-like",
    "has_shape::pigeon-like",
    "has_shape::tree-clinging-like",
    "has_shape::hawk-like",
    "has_shape::sandpiper-like",
    "has_shape::upland-ground-like",
    "has_shape::swallow-like",
    "has_shape::perching-like",
    "has_back_pattern::solid",
    "has_back_pattern::spotted",
    "has_back_pattern::striped",
    "has_back_pattern::multi-colored",
    "has_tail_pattern::solid",
    "has_tail_pattern::spotted",
    "has_tail_pattern::striped",
    "has_tail_pattern::multi-colored",
    "has_belly_pattern::solid",
    "has_belly_pattern::spotted",
    "has_belly_pattern::striped",
    "has_belly_pattern::multi-colored",
    "has_primary_color::blue",
    "has_primary_color::brown",
    "has_primary_color::iridescent",
    "has_primary_color::purple",
    "has_primary_color::rufous",
    "has_primary_color::grey",
    "has_primary_color::yellow",
    "has_primary_color::olive",
    "has_primary_color::green",
    "has_primary_color::pink",
    "has_primary_color::orange",
    "has_primary_color::black",
    "has_primary_color::white",
    "has_primary_color::red",
    "has_primary_color::buff",
    "has_leg_color::blue",
    "has_leg_color::brown",
    "has_leg_color::iridescent",
    "has_leg_color::purple",
    "has_leg_color::rufous",
    "has_leg_color::grey",
    "has_leg_color::yellow",
    "has_leg_color::olive",
    "has_leg_color::green",
    "has_leg_color::pink",
    "has_leg_color::orange",
    "has_leg_color::black",
    "has_leg_color::white",
    "has_leg_color::red",
    "has_leg_color::buff",
    "has_bill_color::blue",
    "has_bill_color::brown",
    "has_bill_color::iridescent",
    "has_bill_color::purple",
    "has_bill_color::rufous",
    "has_bill_color::grey",
    "has_bill_color::yellow",
    "has_bill_color::olive",
    "has_bill_color::green",
    "has_bill_color::pink",
    "has_bill_color::orange",
    "has_bill_color::black",
    "has_bill_color::white",
    "has_bill_color::red",
    "has_bill_color::buff",
    "has_crown_color::blue",
    "has_crown_color::brown",
    "has_crown_color::iridescent",
    "has_crown_color::purple",
    "has_crown_color::rufous",
    "has_crown_color::grey",
    "has_crown_color::yellow",
    "has_crown_color::olive",
    "has_crown_color::green",
    "has_crown_color::pink",
    "has_crown_color::orange",
    "has_crown_color::black",
    "has_crown_color::white",
    "has_crown_color::red",
    "has_crown_color::buff",
    "has_wing_pattern::solid",
    "has_wing_pattern::spotted",
    "has_wing_pattern::striped",
    "has_wing_pattern::multi-colored",
]


class CUBDataSet(Dataset):
    def __init__(self, data_path, stage, resol):
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
                self.data.extend(pickle.load(open(data_path + "val.pkl", "rb")))
            elif stage == "test":
                self.data.extend(pickle.load(open(data_path + "test.pkl", "rb")))
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(resol),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

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
        return img, label, attr_label

    def __len__(self):
        return len(self.data)


class CUB(L.LightningDataModule):
    def __init__(
        self,
        data_path: str = "./data",
        resol: int = 224,
        batch_size: int = 128,
        num_workers: int = 12,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_concepts = 112
        self.num_classes = 200
        self.train_data = CUBDataSet(data_path, "fit", resol)
        self.val_data = CUBDataSet(data_path, "val", resol)
        self.test_data = CUBDataSet(data_path, "test", resol)
        self.imbalance_weights = cal_class_imbalance_weights(self.train_data)
        # Generate a mapping containing all concept groups in CUB generated
        # using a simple prefix tree
        CONCEPT_GROUP_MAP = defaultdict(list)
        for i, concept_name in enumerate(
            list(np.array(CONCEPT_SEMANTICS)[SELECTED_CONCEPTS])
        ):
            group = concept_name[: concept_name.find("::")]
            CONCEPT_GROUP_MAP[group].append(i)

        self.concept_group_map = CONCEPT_GROUP_MAP
        self.concept_names = list(np.array(CONCEPT_SEMANTICS)[SELECTED_CONCEPTS])
        self.max_intervene_budget = 29
        self.group_concept_map = {}
        for name, idxs in self.concept_group_map.items():
            for idx in idxs:
                self.group_concept_map[idx] = name

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


import numpy as np
from itertools import combinations
from collections import Counter


class CustomCUBDataSet(CUBDataSet):
    def __init__(self, data_path, stage, resol, custom_num):

        super().__init__(data_path, stage, resol)

        self.custom_num = custom_num

        all_attrs = []
        for item in self.data:
            all_attrs.append(item["attribute_label"])
        all_attrs = np.array(all_attrs)

        self.num_concepts = all_attrs.shape[1]

        if self.custom_num < self.num_concepts:
            attr_freq = np.sum(all_attrs, axis=0)

            top_indices = np.argsort(attr_freq)[-self.custom_num :][::-1]
            self.selected_indices = top_indices
            self.attr_type = "single"

        else:
            self.selected_indices = list(range(self.num_concepts))

            num_combinations = self.custom_num - self.num_concepts

            if num_combinations > 0:

                combination_freq = Counter()

                for i in range(self.num_concepts):
                    for j in range(i + 1, self.num_concepts):

                        count = np.sum((all_attrs[:, i] == 1) & (all_attrs[:, j] == 1))
                        combination_freq[(i, j)] = count

                if len(combination_freq) < num_combinations:
                    for combo in combinations(range(self.num_concepts), 3):
                        count = np.sum(np.all(all_attrs[:, combo] == 1, axis=1))
                        combination_freq[combo] = count

                top_combinations = sorted(
                    combination_freq.items(), key=lambda x: x[1], reverse=True
                )[:num_combinations]

                self.combinations = [combo[0] for combo in top_combinations]
                self.attr_type = "combined"

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

        original_attr_label = img_data["attribute_label"]

        if self.custom_num < self.num_concepts:

            attr_label = [original_attr_label[i] for i in self.selected_indices]
        else:

            attr_label = original_attr_label.copy()

            if hasattr(self, "combinations"):
                for combo in self.combinations:

                    combo_value = 1
                    for idx in combo:
                        combo_value = combo_value and original_attr_label[idx]
                    attr_label.append(combo_value)

        attr_label = torch.FloatTensor(attr_label)

        return img, label, attr_label


class CustomCUB(CUB):
    def __init__(
        self, custom_num, data_path: str = "./data", resol: int = 224, **kwargs
    ):
        super().__init__(**kwargs)
        self.train_data = CustomCUBDataSet(data_path, "fit", resol, custom_num)
        self.val_data = CustomCUBDataSet(data_path, "val", resol, custom_num)
        self.test_data = CustomCUBDataSet(data_path, "test", resol, custom_num)
        self.imbalance_weights = cal_class_imbalance_weights(self.train_data)
        self.num_concepts = custom_num


if __name__ == "__main__":
    dm = CustomCUB(200)
    print(len(dm.concept_group_map))
