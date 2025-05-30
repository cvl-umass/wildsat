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

import h5py
import json
import pickle
import ast

class GeoBenchSegmentation(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(
        self, 
        root="/datasets/ai/allenai/satlas_pretrain/satlaspretrain_finetune/datasets/geobench/segmentation_v1.0/",
        dataset_name:str="cashew", 
        split='train', 
        return_fp = False,
        resize_size=None,
    ):
        assert dataset_name in ["cashew", "crop"]
        assert split in ["train", "valid", "test"]

        if dataset_name == "cashew":
            self.dataset_name = "m-cashew-plant"
        elif dataset_name == "crop":
            self.dataset_name = "m-SA-crop-type"

        self.num_channels = 3   # TODO: add multispectral option
        self.return_fp = return_fp
        self.root = root
        

        self.split = split

        with open(os.path.join(root, self.dataset_name, "default_partition.json"), "r") as f:
            data_partition = json.load(f)
        # Read the data file
        all_fns = data_partition[split]
        self.fns = all_fns

        if dataset_name == "cashew":
            self.num_outputs = 7
            self.band_names = ["04 - Red_2019-12-10", "03 - Green_2019-12-10", "02 - Blue_2019-12-10"]   # bands to use in this order
        elif dataset_name == "crop":
            self.num_outputs = 10
            self.band_names = ["04 - Red", "03 - Green", "02 - Blue"]
        else:
            raise NotImplementedError

      
        self.data_len = len(self.fns)
        final_size = 512
        if resize_size is not None:
            final_size = resize_size
        self.transforms_list = [transforms.ToTensor()]
        self.transforms_list += [transforms.Resize(size=(final_size,final_size))]
        if "train" in self.split:
            self.transforms_list += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        # NOTE: if adding more transforms, make sure to apply same transforms to label!!
        # if split == "train":
        #     self.data_len = 45

    def __getitem__(self, index):
        fn = self.fns[index]
        fp = os.path.join(self.root, self.dataset_name, f"{fn}.hdf5")

        with h5py.File(fp, "r") as file_obj:
            bands = []
            if self.dataset_name == "m-cashew-plant":
                all_keys = sorted(list(file_obj.keys()))
                self.band_names = [all_keys[3], all_keys[2], all_keys[1]]   # R,G,B
            for band_name in self.band_names:
                h5_band = file_obj[band_name]
                band = np.array(h5_band)
                bands.append(band)
            image = np.stack(bands, axis=-1)   # (256,256,3)
            label = np.array(file_obj["label"])
            # logger.debug(f"unique label: {np.unique(label)}")
        data_transforms = transforms.Compose(self.transforms_list)

        data = np.concatenate((image,label[:,:,None]), axis=-1)
        trans_data = data_transforms(data)  # output of transforms: (5, 512, 512)
        image = trans_data[:self.num_channels, :, :].float()
        label = trans_data[-1, :, :].long()
        # logger.debug(f"unique label: {torch.unique(label)}")
        # image = data_transforms(image)  # output of transforms: (3, 512, 512)
        
        # Min-max Normalization
        # Normalize data
        # """
        if (torch.max(image)-torch.min(image)):
            image = image - torch.min(image)
            image = image / torch.maximum(torch.max(image),torch.tensor(1))
        else:
            logger.warning(f"all zero image. setting all labels to zero. index: {index}. {self.split} {fp}")
            image = torch.zeros_like(image).float()
            # all_data = torch.zeros_like(all_data)
        # """
        if self.return_fp:
            return (
                image,    # shape: (3, 512, 512)
                label,
                fp,
            )

        else:
            return (
                image,    # shape: (3, 512, 512)
                label,
            )


    def __len__(self):
        return self.data_len

