"""
General utils for training, evaluation and data loading
"""
import os
from glob import glob
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class AwA(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, data_path, resol, is_train=True):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        image_dir: default = 'images'. Will be append to the parent dir
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        
        self.path = data_path
        class_to_index = dict()
        with open(self.path+'Animals_with_Attributes2/classes.txt') as f:
            index = 0
            for line in f:
                class_name = line.split('\t')[1].strip()
                class_to_index[class_name] = index
                index += 1
                
        img_names = []
        img_label = []
        with open(self.path+'Animals_with_Attributes2/testclasses.txt') as f:
            for line in f:
                class_name = line.strip()
                FOLDER_DIR = os.path.join(self.path+'Animals_with_Attributes2/JPEGImages', class_name)
                file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
                files = glob(file_descriptor)

                class_index = class_to_index[class_name]
                for file_name in files:
                    img_names.append(file_name)
                    img_label.append(class_index)
        
        with open(self.path+'Animals_with_Attributes2/trainclasses.txt') as f:
            for line in f:
                class_name = line.strip()
                FOLDER_DIR = os.path.join(self.path+'Animals_with_Attributes2/JPEGImages', class_name)
                file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
                files = glob(file_descriptor)

                class_index = class_to_index[class_name]
                for file_name in files:
                    img_names.append(file_name)
                    img_label.append(class_index)

        shape = (3, resol, resol)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean=mean,std=std)
        transform = transforms.Compose([
                            transforms.Resize(size=(resol, resol)),
                            transforms.ToTensor(),
                            normalize
                        ])
        
        self.img_names = img_names
        self.img_label = img_label
        self.transform = transform
        self.label_to_num = class_to_index
        self.label_to_attr = np.array(np.genfromtxt(self.path+'Animals_with_Attributes2/predicate-matrix-binary.txt', dtype='float32'))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):            
        img = Image.open(self.img_names[idx]).convert('RGB')
        class_label = self.img_label[idx]
        if self.transform:
            img = self.transform(img)
            
        attr_label = self.label_to_attr[class_label,:]
        
        return img, class_label, torch.Tensor(attr_label)

# def AwA_load_data(data_path, batch_size, shuffle, resol=224):
#     shape = (3, resol, resol)
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)
#     normalize = transforms.Normalize(mean=mean,std=std)
#     transform = transforms.Compose([
#                             transforms.Resize(size=(resol, resol)),
#                             transforms.ToTensor(),
#                             normalize
#                         ])

#     dataset = AwADataset(data_path, transform)

#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
#     return loader
