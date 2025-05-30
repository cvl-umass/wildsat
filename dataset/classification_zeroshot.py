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


class ClassificationDataset(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, dataset_name:str="aid", split='train50', return_fp = False,resize_size=None, use_graft=False,
        use_taxabind=False,
        use_remoteclip=False,
        sat_processor=None,):
        
        self.num_channels = 3   # TODO: add multispectral option
        self.return_fp = return_fp

        self.split = split
        self.use_graft = use_graft
        self.use_taxabind = use_taxabind
        self.use_remoteclip = use_remoteclip

        # Read the data file
        if split=="train50":
            df = pd.read_csv(f"data/{dataset_name}_train50.csv")    # train on just 50 examples (from satlas)
        elif split=="train":
            df = pd.read_csv(f"data/{dataset_name}_data.csv")
            df = df[df["split"]=="train"]   # train on full train set
        elif split=="test":
            df = pd.read_csv(f"data/{dataset_name}_data.csv")
            df = df[df["split"]=="test"]
        elif split=="all":
            df = pd.read_csv(f"data/{dataset_name}_data.csv")
            logger.warning(f"Using ALL data in dataset. only use for zeroshot eval. df.shape: {df.shape}")
        else:
            raise NotImplementedError

        with open(f"data/{dataset_name}_metadata.npy", "rb") as f:
            self.label_to_class = np.load(f, allow_pickle=True)[()]["classes"]  # maps from idx (0-29) to actual string
        self.fps = df["fp"].values
        self.labels = df["label"].values
        self.num_outputs = len(self.label_to_class)
      
        self.data_len = len(self.fps)
        final_size = 512
        if resize_size is not None:
            final_size = resize_size
        if self.use_graft:
            logger.warning("Using different transform for eurosat")
            if dataset_name == "eurosat":
                logger.warning(f"Using eurosat-specific normalization for GRAFT")
                norm_transform = transforms.Normalize(
                    (0.406, 0.456, 0.485),
                    (0.225, 0.224, 0.229)
                )
                self.transforms_list = [
                    transforms.Pad(padding=80, padding_mode='reflect'), 
                    transforms.ToTensor(), 
                    norm_transform
                ]
            else:
                logger.warning(f"Using default normalization for GRAFT")
                norm_transform = transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                )
                self.transforms_list = [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    norm_transform
                ]
        elif self.use_taxabind:
            logger.warning(f"Using taxabind-specific image preprocessing")
            self.transforms_list =[
                transforms.Resize((256, 256)),
            ]
            self.sat_processor = sat_processor
        elif self.use_remoteclip:
            logger.warning(f"Using RemoteCLIP image preprocess")
            self.transforms_list =[
                transforms.Resize((256, 256)),
            ]
            self.sat_processor = sat_processor
        else:
            self.transforms_list = []
            if dataset_name == "eurosat":
                logger.warning(f"Padding image with 80 reflected (adopted from GRAFT)")
                self.transforms_list += [transforms.Pad(padding=80, padding_mode='reflect')]
            self.transforms_list += [transforms.ToTensor()]
            # """
            if dataset_name not in ["ucm", "fmow_cls", "eurosat"]:
                # transforms.RandomCrop(size=(512,512)),
                self.transforms_list += [transforms.RandomCrop(size=(256,256))]
            elif dataset_name == "fmow_cls":
                self.transforms_list += [transforms.CenterCrop(size=(256,256))]
            self.transforms_list += [transforms.Resize(size=(final_size,final_size))]
            if "train" in self.split:
                self.transforms_list += [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                ]
        # """
        # if split == "train":
        #     self.data_len = 129

    def __getitem__(self, index):
        fp = self.fps[index]
        label = self.labels[index]
        # fp = os.path.join(self.root, tmp_fp)
        image = Image.open(fp)  # size: (512,512)

        data_transforms = transforms.Compose(self.transforms_list)
        image = data_transforms(image)  # output of transforms: (3, 512, 512)
        if self.use_taxabind or self.use_remoteclip:
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

