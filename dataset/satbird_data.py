from torch.utils.data.dataset import Dataset

import os
import torch
import fnmatch
import numpy as np
import pandas as pd
import pdb
import torchvision.transforms as transforms
from PIL import Image
import random
import torch.nn.functional as F
from loguru import logger

from models.satbird_utils import load_file


class SatBirdDataset(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, dataset_name:str="kenya", split='train', return_fp = False,resize_size=None,):
        
        self.num_channels = 3   # TODO: add multispectral option
        self.return_fp = return_fp

        self.split = split

        data_dir = "/datasets/ai/allenai/satlas_pretrain/satlaspretrain_finetune/datasets/satbird"
        if dataset_name.lower() == "kenya":
            dataset_name = "new_kenya2"
        self.data_base_dir = os.path.join(data_dir, dataset_name)
        logger.debug(f"self.data_base_dir: {self.data_base_dir}. split={split}")

        # Read the data file
        if split=="train":
            df = pd.read_csv(os.path.join(self.data_base_dir, "train_split.csv"))
        elif split=="test":
            df = pd.read_csv(os.path.join(self.data_base_dir, "test_split.csv"))
        elif split=="valid":
            df = pd.read_csv(os.path.join(self.data_base_dir, "valid_split.csv"))
        else:
            raise NotImplementedError

        self.df = df
        self.data_len = len(self.df)
        final_size = 512
        if resize_size is not None:
            final_size = resize_size
        self.transforms_list = [transforms.ToTensor()]
        self.transforms_list += [transforms.CenterCrop(size=(64,64))]   # following SatBird, crop at center
        self.transforms_list += [transforms.Resize(size=(final_size,final_size))]
        if "train" in self.split:
            self.transforms_list += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
            # TODO: Add other data augmentations: random noie, blur, contrast, brightness
        # if split == "train":
        #     self.data_len = 129

    def __getitem__(self, index):
        hotspot_id = self.df.iloc[index]['hotspot_id']
        img_path = os.path.join(self.data_base_dir, "images_visual", hotspot_id + '_visual.tif')
        species = load_file(os.path.join(self.data_base_dir, "targets", hotspot_id + '.json'))
        image = load_file(img_path) # size: (3, s, s)
        image = np.transpose(image, (1,2,0))    # (s,s,3)
        # logger.debug(f"image: {image.shape}")
        # sats = torch.from_numpy(img).float()
        label = np.array(species["probs"])    # (num_species,)

        num_complete_checklists = species["num_complete_checklists"]


        data_transforms = transforms.Compose(self.transforms_list)
        image = data_transforms(image).float()  # output of transforms: (3, 512, 512)
        # logger.debug(f"image: {image.shape}")

        # Min-max Normalization
        # Normalize data
        # """
        if (torch.max(image)-torch.min(image)):
            image = image - torch.min(image)
            image = image / torch.maximum(torch.max(image),torch.tensor(1))
        else:
            # logger.warning(f"all zero image. setting all pixels to zero. index: {index}. {self.split} {img_path}")
            image = torch.zeros_like(image)
            # all_data = torch.zeros_like(all_data)
        # """
        if self.return_fp:
            return (
                image,    # shape: (3, 512, 512)
                label,
                img_path,
            )

        else:
            return (
                image,    # shape: (3, 512, 512)
                label,
            )


    def __len__(self):
        return self.data_len

