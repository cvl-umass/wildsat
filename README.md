# WildSAT: Learning Satellite Image Representations from Wildlife Observations
This is the official repository for the paper "WildSAT: Learning Satellite Image Representations from Wildlife Observations".
![overview](assets/overview.jpg)

## Overview
Species distributions encode valuable ecological and environmental information, yet their potential for guiding representation learning in remote sensing remains underexplored.
We introduce WildSAT, which pairs satellite images with millions of geo-tagged wildlife observations readily-available on citizen science platforms.
WildSAT employs a contrastive learning approach that jointly leverages satellite images, species occurrence maps, and textual habitat descriptions to train or fine-tune models.
This approach significantly improves performance on diverse satellite image recognition tasks, outperforming both ImageNet-pretrained models and satellite-specific baselines.
Additionally, by aligning visual and textual information, WildSAT enables zero-shot retrieval, allowing users to search geographic locations based on textual descriptions.
WildSAT surpasses recent cross-modal learning methods, including approaches that align satellite images with ground imagery or wildlife photos, demonstrating the advantages of our approach. Finally, we analyze the impact of key design choices and highlight the broad applicability of WildSAT to remote sensing and biodiversity monitoring.

## Setup the environment
1. Create a conda environment: `conda create -n wildsat python=3.9`
2. Activate the environment: `conda activate wildsat`
3. Install required packages `pip install -r requirements.txt`

## Quickstart
This shows how to extract features from satellite images and use them for retrieving relevant images.
1. Activate your environment and download the required package for GritLM: `pip install gritlm`
2. Download our sample model [here](https://drive.google.com/file/d/1IxBpf3nbEMzny4YJWS6stMBxel6gMiYE/view?usp=drive_link). This is an ImageNet pre-trained ViT-B/16 model that is further fine-tuned with WildSAT.
3. Download a small set of data [here](https://drive.google.com/file/d/18YL59DAPLj0WnLkX9fh_n0bDjMYutdLq/view?usp=drive_link)
4. Run the notebook `quickstart.ipynb`
  - Make sure to specify the location of the sample data downloaded in the previous step, and the location of the checkpoint in step 2 


## Dataset
1. Download the Sentinel satellite images from SatlasPretrain [here](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md)
2. Download the Wikipedia data from LE-SINR [here](https://drive.google.com/file/d/10n2rXajlwUxtrd8cNuTbFmsluYEgZP0c/view?usp=sharing). Place it in `data/wiki_data_v4.pt`
3. Download the bioclimatic variables [here](https://drive.google.com/file/d/15sCOevQVueDiXbtrymTg9eUwPwk1JNKp/view?usp=drive_link). Place it in `data/bioclim*.npy`. This is used by SINR to extract location features.
4. Download the mapping between satellite images, location, and text [here](https://drive.google.com/file/d/1jprjItXj3AflJc74dRhT15BCBW4wzLkE/view?usp=sharing). Place it in `data/dataloader_data.npy`

A sample code is provided for visualizing the dataset in `data_explore.ipynb`


## Training the model
1. Make sure all components of the dataset has been downloaded (see [Dataset](#dataset))
2. Download all [pre-trained model checkpoints](https://drive.google.com/file/d/1ha5UtNANBe4nlL1Kc4UMFU-VGpgfk-nu/view?usp=sharing). Extract it and place it in `wildsat/checkpoints/*`. This is needed for the different pre-trained models as the starting point. This is not needed if you want to start from a randomly initialized model or an ImagetNet pre-trained model.
3. Run training for a randomly initialized RN50 model: `python train.py --satellite_encoder "resnet50" --satellite_notpretrained`. For other model options see the table below.

| Architecture  | Pre-training  | Training command | Checkpoint when fully trained with WildSAT |
| ---           | ---           | ---              | ---        |
| ViT-B/16          | ImageNet1k        | `python train.py --satellite_encoder "vitb16"  --use_bnft --is_tunefc` | [link](https://drive.google.com/file/d/1IxBpf3nbEMzny4YJWS6stMBxel6gMiYE/view?usp=drive_link) |
| ViT-B/16          | CLIP        | `python train.py --satellite_encoder "vitb16" --satellite_encoder_ckpt "clip" --lora_layer_types 'attn.k_proj' 'attn.v_proj' 'attn.q_proj' 'attn.out_proj' 'visual_projection' --use_lora --use_dora` | [link](https://drive.google.com/file/d/1AjB9nWxv_-LTjZtxDFh4Tq50hwga65pK/view?usp=drive_link) |
| ViT-B/16          | Prithvi        | `python train.py --satellite_encoder "vitb16"  --satellite_encoder_ckpt "prithvi"` | [link](https://drive.google.com/file/d/12Mi1mr9Ktk7JEbyhyaaNhx6Nog--dPWs/view?usp=drive_link) |
| ViT-B/16          | SatCLIP        | `python train.py --satellite_encoder "vitb16"  --satellite_encoder_ckpt "checkpoints/satclip/satclip-vit16-l10.ckpt"` | [link](https://drive.google.com/file/d/1P5DVHJP_8idUoTOr-ycDTvCZ7hxUizUQ/view?usp=sharing) |
| ViT-B/16          | None (Random)        | `python train.py --satellite_encoder "vitb16" --satellite_notpretrained` | [link](https://drive.google.com/file/d/1jsabRt5QKvN8MAjsHKKZwlBqT16-6WPU/view?usp=drive_link) |
| Swin-T          | ImageNet1k        | `python train.py --satellite_encoder "swint"  --use_bnft --is_tunefc` | [link](https://drive.google.com/file/d/12rfb3ES9-RuIAk_RjzYrMQJG_5J1Iq5H/view?usp=drive_link) |
| Swin-T          | Satlas        | `python train.py --satellite_encoder "swint" --satellite_encoder_ckpt "satlas-backbone"` | [link](https://drive.google.com/file/d/1luX_a2nSp0Lg-TEWH29kViKFx5YES9-U/view?usp=drive_link) |
| Swin-T          | None (Random)        | `python train.py --satellite_encoder "swint" --satellite_notpretrained` | [link](https://drive.google.com/file/d/1nEKchTOxuXdJ6e-aLjnVZE4RmrJzZa-u/view?usp=drive_link) |
| RN50          | ImageNet1k      | `python train.py --satellite_encoder "resnet50" --use_bnft --is_tunefc ` | [link](https://drive.google.com/file/d/1LwCjzT1p0yFOtv1m0YD-rl7tP0eYXTb2/view?usp=drive_link) |
| RN50          | MoCov3        | `python train.py --satellite_encoder "resnet50" --satellite_encoder_ckpt "checkpoints/moco_v3/r-50-100ep.pth.tar" --use_bnft --is_tunefc` | [link](https://drive.google.com/file/d/1FvxUuRITM-156QgJDJ8OYCPkTN74OlC-/view?usp=drive_link) |
| RN50          | SatCLIP       | `python train.py --satellite_encoder "resnet50" --satellite_encoder_ckpt "checkpoints/satclip/satclip-resnet50-l10.ckpt"` | [link](https://drive.google.com/file/d/1Dca0pHQ53v20Jy22_ikIl1bPH1CKNVJr/view?usp=drive_link) |
| RN50          | Satlas        | `python train.py --satellite_encoder "resnet50" --satellite_encoder_ckpt "satlas-backbone"` | [link](https://drive.google.com/file/d/19fhXiSIrAYtXXuqklKG5YAdeS2jgTDjx/view?usp=drive_link) |
| RN50          | SeCo        | `python train.py --satellite_encoder "resnet50" --satellite_encoder_ckpt "checkpoints/seco/seco_resnet50_100k.ckpt"` | [link](https://drive.google.com/file/d/1GWrhoRLx4hnpfGOi8HYM2aKUyxVMvQEP/view?usp=drive_link) |
| RN50          | None (Random)        | `python train.py --satellite_encoder "resnet50" --satellite_notpretrained` | [link](https://drive.google.com/file/d/1qKphYzyPp3dM_l5CIno0-p4KPvypCOW_/view?usp=drive_link) |



## Citation
If you found this helpful, please cite our paper:
```
@article{daroya2024wildsat,
  title={WildSAT: Learning Satellite Image Representations from Wildlife Observations},
  author={Daroya, Rangel and Cole, Elijah and Mac Aodha, Oisin and Van Horn, Grant and Maji, Subhransu},
  journal={arXiv preprint arXiv:2412.14428},
  year={2024}
}
```