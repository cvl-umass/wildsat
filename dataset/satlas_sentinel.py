from torch.utils.data.dataset import Dataset

import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
# from pl_bolts.models.self_supervised.moco.transforms import GaussianBlur
from PIL import Image
import random
import torch.nn.functional as F
from loguru import logger

np.random.seed(123)
random.seed(123)
torch.manual_seed(123)

class SentinelwTextImg(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(
        self,
        root,
        split='train',
        metadata_path="data/dataloader_data.npy",    # 135140 groups; next res has 800k groups
        patch_size=512,
        limit_per_epoch=None,
        is_img_contrast=False,  # Set to true if satellite image is to be augmented
        use_text=False,
        return_fp = False, 
        resize_size=None,
        use_multispectral=False,
        # use_latlon=False,
    ):
        # TODO: for augmented image, use different image from the same h3 cell
        # (NOTE: need to make sure there are no other sat image from same h3 cell in batch)
        self.return_fp = return_fp
        self.num_channels = 3   # TODO: add multispectral option
        self.is_img_contrast = is_img_contrast
        self.use_text = use_text

        self.split = split
        self.root = os.path.expanduser(root)
        self.patch_size = patch_size
        # self.use_latlon = use_latlon
        self.use_multispectral = use_multispectral
        self.other_bands = ["b05","b06","b07","b08","b11","b12"]
        if self.use_multispectral:
            self.num_channels = len(self.other_bands)+3
            logger.warning(f"Using multispectral data. self.num_channels: {self.num_channels}")


        # Read the data file
        if split=="train":
            self.data_path = metadata_path
        else:
            raise NotImplementedError
        self.inat_data = None

        with open(metadata_path, "rb") as f:
            self.data = np.load(f, allow_pickle=True)[()]["dataloader_data"]
        self.text_embeddings = None
        self.transforms_list =[
            transforms.ToTensor(),
        ]
        final_size = 512
        if resize_size is not None:
            final_size = resize_size
        logger.debug(f"final_size: {final_size}")
        logger.debug(f"self.patch_size: {self.patch_size}")
        if self.patch_size != 512:
            self.transforms_list += [
                transforms.CenterCrop(size=(patch_size,patch_size)),
                transforms.Resize(size=(final_size,final_size))
            ]
        if final_size != 512:
            self.transforms_list += [transforms.Resize(size=(final_size,final_size))]
        self.transforms_list += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]

        if self.use_multispectral:
            self.augmentation_transforms = transforms.Compose([ # for augmenting positive sample
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.Resize(size=(final_size,final_size)),
            ])

        else:
            self.augmentation_transforms = transforms.Compose([ # for augmenting positive sample
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur((5,5), sigma=(0.1, 2.0))], p=0.5),  # NOTE: using kernel size (5,5)
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.Resize(size=(self.patch_size,self.patch_size)),
            ])
        if limit_per_epoch is None:
            self.data_len = len(self.data)
        else:
            self.data_len = limit_per_epoch
        # self.data_len = 65
        self.final_size = final_size

    def init_data_sources(self):
        logger.debug(f"Using separate image for augmentations. datapath: {self.data_path}")
        self.text_embeddings = torch.load("data/wiki_data_v4.pt")["data"]   # pre-computed embeddings from text; shape: (127484, 4096)

    def __getitem__(self, index):
        if self.text_embeddings is None:
            self.init_data_sources()
            logger.debug(f"Loaded text embeddings [{len(self.text_embeddings)}] and inat data [{len(self.inat_data) if self.inat_data is not None else 0}].")
        
        augmented_image = None
        inat_embed = None
        taxon_h3cell = self.data[index] # fp, lon, lat, text_emb_idx
        rand_idx = np.random.randint(low=0, high=len(taxon_h3cell)) # randomly choose satellite image - text pair
        tmp_fp, lon, lat, text_emb_idx, aug_fns = taxon_h3cell[rand_idx]
        aug_fn = np.random.choice(aug_fns)
        aug_fp = os.path.join(self.root, aug_fn)
        augmented_image = Image.open(aug_fp)  # size: (512,512)
        if self.use_multispectral:
            augmented_image = np.array(augmented_image) # (512,512,3)
            for b_name in self.other_bands:
                ob_fp = aug_fp.replace("tci",b_name)
                if os.path.exists(ob_fp):
                    new_band = np.array(Image.open(ob_fp))
                else:
                    new_band = np.zeros((augmented_image.shape[0],augmented_image.shape[1]), dtype=float)
                new_band = np.expand_dims(new_band, -1)
                # logger.debug(f"new_band: {type(new_band)}")
                augmented_image = np.concatenate((augmented_image, new_band), axis=-1)
        fp = os.path.join(self.root, tmp_fp)
        image = Image.open(fp)  # size: (512,512)
        
        text_embedding = self.text_embeddings[text_emb_idx]
        if self.use_multispectral:
            image = np.array(image) # (512,512,3)
            for b_name in self.other_bands:
                ob_fp = fp.replace("tci",b_name)
                if os.path.exists(ob_fp):
                    new_band = np.array(Image.open(ob_fp))
                else:
                    new_band = np.zeros((image.shape[0],image.shape[1]), dtype=float)
                new_band = np.expand_dims(new_band, -1)
                # logger.debug(f"new_band: {type(new_band)}")
                image = np.concatenate((image, new_band), axis=-1)

        # Random crop
        if self.split == "train":
            data_transforms = transforms.Compose(self.transforms_list)
        else:
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

        if self.is_img_contrast:
            if augmented_image is None:
                augmented_image = self.augmentation_transforms(image)
            else:
                augmented_image = self.augmentation_transforms(augmented_image)
            if not self.use_multispectral:
                augmented_image = data_transforms(augmented_image)
        image = data_transforms(image)  # output of transforms: (3, 512, 512)
        # logger.debug(f"image: {image.shape}")
        # Data Augmentation
        if self.split == "train":
            # Add Random channel mixing
            ccm = torch.eye(self.num_channels)[None,None,:,:]
            r = torch.rand(3,)*0.25 + torch.Tensor([0,1,0])
            filter = r[None, None, :, None]
            ccm = torch.nn.functional.conv2d(ccm, filter, stride=1, padding="same")
            ccm = torch.squeeze(ccm)
            # logger.debug(f"image: {type(image)}. ccm: {type(ccm)}")
            try:
                image = torch.tensordot(ccm, image, dims=([1],[0])) # not exactly the same perhaps
            except:
                pass    # NOTE: Error for multispectral images
            
            # Add Gaussian noise
            r = torch.rand(1,1)*0.04
            image = image + torch.normal(mean=0.0, std=r[0][0], size=(self.num_channels,image.shape[1],image.shape[2]))
        
        # Min-max Normalization
        # Normalize data
        if (torch.max(image)-torch.min(image)):
            image = image - torch.min(image)
            image = image / torch.maximum(torch.max(image),torch.tensor(1))
        else:
            logger.warning(f"all zero image. setting all labels to zero. index: {index}. {self.split} {fp}")
            image = torch.zeros_like(image)
            # all_data = torch.zeros_like(all_data)
        if self.is_img_contrast and (torch.max(image)-torch.min(image)):
            augmented_image = augmented_image - torch.min(augmented_image)
            augmented_image = augmented_image / torch.maximum(torch.max(augmented_image),torch.tensor(1))
        elif self.is_img_contrast:
            logger.warning(f"all zero augmented_image. setting all labels to zero. index: {index}. {self.split} {fp}")
            augmented_image = torch.zeros_like(augmented_image)
        
        return_fp = self.return_fp
        if return_fp:
            if not self.is_img_contrast:
                return (
                    image,    # shape: (3, 512, 512)
                    np.array([lon,lat]).astype(np.float32), # shape: (2, )
                    text_embedding, # shape: (4096, )
                    tmp_fp,
                )
            else:
                if inat_embed is None:
                    return (
                        image,    # shape: (3, 512, 512)
                        augmented_image,     # shape: (3, 512, 512)
                        np.array([lon,lat]).astype(np.float32), # shape: (2, )
                        text_embedding, # shape: (4096, )
                        tmp_fp,
                    )
                else:
                    return (
                        image,    # shape: (3, 512, 512)
                        augmented_image,     # shape: (3, 512, 512)
                        np.array([lon,lat]).astype(np.float32), # shape: (2, )
                        text_embedding, # shape: (4096, )
                        inat_embed,     # shape: (512, )
                        tmp_fp,
                    )
        if not self.is_img_contrast:
            return (
                image,    # shape: (3, 512, 512)
                np.array([lon,lat]).astype(np.float32), # shape: (2, )
                text_embedding, # shape: (4096, )
            )
        else:
            if inat_embed is None:
                return (
                    image,    # shape: (3, 512, 512)
                    augmented_image,     # shape: (3, 512, 512)
                    np.array([lon,lat]).astype(np.float32), # shape: (2, )
                    text_embedding, # shape: (4096, )
                )
            else:
                return (
                    image,    # shape: (3, 512, 512)
                    augmented_image,     # shape: (3, 512, 512)
                    np.array([lon,lat]).astype(np.float32), # shape: (2, )
                    text_embedding, # shape: (4096, )
                    inat_embed,     # shape: (512, )
                )



    def __len__(self):
        return self.data_len

