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

class GeoBenchClassification(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(
        self, 
        root="/datasets/ai/allenai/satlas_pretrain/satlaspretrain_finetune/datasets/geobench/classification_v1.0/",
        dataset_name:str="bigearthnet", 
        split='train', 
        return_fp = False,
        resize_size=None,
        use_graft=False,
        use_taxabind=False,
        use_remoteclip=False,
        sat_processor=None,
    ):
        assert dataset_name in ["bigearthnet", "so2sat"]
        assert split in ["train", "valid", "test"]

        if dataset_name == "bigearthnet":
            self.dataset_name = "m-bigearthnet"
        elif dataset_name == "so2sat":
            self.dataset_name = "m-so2sat"
            self.label_to_class = [
            "Compact High-Rise",
            "Compact middle-rise",
            "Compact low-rise",
            "Open high-rise",
            "Open middle-rise",
            "Open low-rise",
            "Lightweight low-rise",
            "Large low-rise",
            "Sparsely built",
            "Heavy industry",
            "Dense trees",
            "Scattered trees",
            "Bush, scrub",
            "Low plants",
            "Bare rock or paves",
            "Bare soil or sand",
            "water",
            ]

        self.num_channels = 3   # TODO: add multispectral option
        self.return_fp = return_fp
        self.root = root
        self.band_names = ["04 - Red", "03 - Green", "02 - Blue"]   # bands to use in this order

        self.split = split
        self.use_graft = use_graft
        self.use_taxabind = use_taxabind
        self.use_remoteclip = use_remoteclip

        with open(os.path.join(root, self.dataset_name, "default_partition.json"), "r") as f:
            data_partition = json.load(f)
        # Read the data file
        all_fns = data_partition[split]
        self.fns = all_fns

        with open(os.path.join(root, self.dataset_name, "label_stats.json")) as f:
            self.labels = json.load(f)
        if dataset_name == "bigearthnet":
            self.num_outputs = len(self.labels[all_fns[0]])
        elif dataset_name == "so2sat":
            self.num_outputs = 17
        else:
            raise NotImplementedError

      
        self.data_len = len(self.fns)
        final_size = 512
        if resize_size is not None:
            final_size = resize_size
        self.transforms_list = [transforms.ToTensor()]
        self.transforms_list += [transforms.Resize(size=(final_size,final_size))]
        if self.use_graft:
            logger.warning("Adding norm transform for GRAFT. Will disable min-max normalization")
            norm_transform = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
            self.transforms_list += [norm_transform]
        elif self.use_taxabind:
            self.transforms_list = [
                transforms.Resize((256, 256)),
            ]
            self.sat_processor = sat_processor
        elif self.use_remoteclip:
            logger.warning(f"Using RemoteCLIP image preprocess")
            self.transforms_list =[
                transforms.Resize((256, 256)),
            ]
            self.sat_processor = sat_processor
        if "train" in self.split:
            self.transforms_list += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        # if split == "train":
        #     self.data_len = 45

    def __getitem__(self, index):
        fn = self.fns[index]
        fp = os.path.join(self.root, self.dataset_name, f"{fn}.hdf5")
        if self.dataset_name == "m-bigearthnet":
            label = np.array(self.labels[fn])   # shape: (num_labels,)
        else:
            label = int(self.labels[fn])
        image = Image.open(fp)  # size: (512,512)

        with h5py.File(fp, "r") as file_obj:
            bands = []
            for band_name in self.band_names:
                h5_band = file_obj[band_name]
                band = np.array(h5_band)
                bands.append(band)
            image = np.stack(bands, axis=-1)   # (120,120,3)

        
        data_transforms = transforms.Compose(self.transforms_list)
        if (not self.use_taxabind) and (not self.use_remoteclip):
            image = data_transforms(image)  # output of transforms: (3, 512, 512)
        if self.use_taxabind or self.use_remoteclip:
            # logger.debug(f"image: {np.max(image)},{np.min(image)}, {image.shape}, {type(image)}")
            image = data_transforms(Image.fromarray((image*255).astype(np.uint8)))
            image = self.sat_processor(image)
        # Min-max Normalization
        # Normalize data
        # """
        if (not self.use_graft) and (not self.use_taxabind):
            if (torch.max(image)-torch.min(image)):
                image = image - torch.min(image)
                image = image / torch.maximum(torch.max(image),torch.tensor(1))
            else:
                logger.warning(f"all zero image. setting all labels to zero. index: {index}. {self.split} {fp}")
                image = torch.zeros_like(image)
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

