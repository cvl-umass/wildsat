# Adopted from https://github.com/pytorch/examples/blob/main/imagenet/main.py#L393
# NOTE: this is for using multiple GPUs for training

from enum import Enum
import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
import torch.distributed as dist
from loguru import logger as lgr
from datetime import datetime
from matplotlib import pyplot as plt
import os
import shutil
import argparse
import warnings
import errno

from models.get_model import get_model, SINREmbeddingModel

from dataset.satlas_sentinel import SentinelwTextImg

from models.clip import ClipLoss



DATE_STR = datetime.now().strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser(description='WildSAT')
parser.add_argument('--root', default='/datasets/ai/allenai/satlas_pretrain/sentinel2/', type=str, help='Path to dataset')
parser.add_argument('--metadata_path', default='data/dataloader_data.npy', type=str, help='Path to list of files for training')
parser.add_argument('--loss_type', default='mse', type=str, help="Type of loss for training. choices: [ bce]")
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--find_unused_param', default=0, type=int, help='Find unused param for distrib training')
parser.add_argument('--scheduler', default="none", type=str, help='Scheduler to use (if any) - choices: "none", "steplr", "cosineanneal"')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate to use')

parser.add_argument('--sinr_ckpt_path', default='checkpoints/baseline_sinr_pos+env_allspecies/model.pt', type=str, help="Path to pre-trained sinr")
parser.add_argument('--ckpt_path', default=None, type=str, help='specify location of checkpoint to load and use')
parser.add_argument('--out', default='./results/wildsat-trained', help='Directory to output the result and checkpoints')

parser.add_argument("--num_inp_channels", default=3, type=int, help="Number of channels in input (3 for RGB)")
parser.add_argument('--satellite_encoder', default='resnet50', type=str, help='encoder to get satellite features')
parser.add_argument('--satellite_head', default='threelinearnobias', type=str, help='satellite decoder')
parser.add_argument("--ckpt_interval", default=5, type=int, help="The epoch interval that ckpt is saved separately from latest")
parser.add_argument("--satellite_notpretrained", action='store_true', help="Use this to initial encoder weights to random (not imagenet pre-trained)")

# The following params are for multiple GPU training
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--freeze_backbone', action='store_true', help='Set to true to keep imagenet-pretrained backbone')
parser.add_argument("--patch_size", default=512, type=int, help="The size of the patches. max: 512. choices: [512, 256, 128, 64]")
parser.add_argument("--common_embed_dim", default=256, help="Dimension of common embedding space between SINR and satellite model")

parser.add_argument("--text_loss_weight", default=2, type=int, help="Weight of text loss")
parser.add_argument("--img_loss_weight", default=4, type=int, help="Weight of img loss")


parser.add_argument("--satellite_encoder_ckpt", default=None, type=str, help="Specify the filepath to weights of encoder (e.g., if moco pre-trained)")
parser.add_argument("--use_sinr", action='store_true', help="Set to true to use SINR features in contrastive training pipeline")
parser.add_argument("--use_satclip", action='store_true', help="Set to true to use SatCLIP features for lat/lon in contrastive training pipeline. use_sinr HAS to be FALSE")
parser.add_argument("--use_multispectral", action='store_true', help="Set to true to use all 9 bands from sentinel2")

parser.add_argument("--use_lora", action='store_true', help="Set to use lora in encoder")
parser.add_argument('--lora_layer_types', default=["conv"], nargs='+', help='Layers to use lora/dora on. use_lora MUST be true')
parser.add_argument("--use_dora", action='store_true', help="Set to use dora with lora. MUST set use_lora to true")
parser.add_argument("--use_bnft", action='store_true', help="Set to only finetune BN layers and final layer. MUST set use_lora to false and use_dora to false")
parser.add_argument("--is_tunefc", action='store_true', help="Set to true to also tune fc layer in BN FT. MUST set use_lora to false, use_dora to false, and use_bnft to true")



def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def plot_losses(train_losses, opt, suffix=None, is_multiple_losses=False, loss_labels=None):
    if is_multiple_losses:
        assert len(loss_labels) == len(train_losses)
        for idx, t_loss in enumerate(train_losses):
            plt.plot(list(range(1, len(t_loss)+1)), t_loss, marker="x", label=loss_labels[idx])
        plt.legend()
    else:
        plt.plot(list(range(1, len(train_losses)+1)), train_losses, marker="x")
    plt.xlabel("Epoch")
    plt.title("Train Loss")
    if suffix is None:
        plt.savefig(os.path.join(opt.out, f"{DATE_STR}_{opt.satellite_encoder}_{opt.satellite_head}.jpg"), bbox_inches="tight")
    else:
        plt.savefig(os.path.join(opt.out, f"{DATE_STR}_{suffix}_{opt.satellite_encoder}_{opt.satellite_head}.jpg"), bbox_inches="tight")
    plt.close()

def save_checkpoint(state, opt, epoch, filename='checkpoint.pth.tar'):
    if opt.satellite_notpretrained:
        filepath = os.path.join(opt.out, '{}_{}_{}_randominit_'.format(DATE_STR, opt.satellite_encoder, opt.satellite_head) + filename)
    else:
        filepath = os.path.join(opt.out, '{}_{}_{}_'.format(DATE_STR, opt.satellite_encoder, opt.satellite_head) + filename)
    torch.save(state, filepath)
    if (((epoch)%opt.ckpt_interval) == 0) and (epoch>0):
        shutil.copyfile(filepath, os.path.join(opt.out, '{}_{}_{}_'.format(DATE_STR, opt.satellite_encoder, opt.satellite_head) + f'e{epoch:03d}_checkpoint.pth'))

# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    opt = parser.parse_args()
    lgr.debug(f"opt: {opt}")


    if not os.path.isdir(opt.out):
        mkdir_p(opt.out)

    if opt.gpu is not None:
        lgr.warning('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    opt.distributed = opt.world_size > 1 or opt.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        opt.world_size = ngpus_per_node * opt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        # Simply call main_worker function
        main_worker(opt.gpu, ngpus_per_node, opt)
    
class LinearforCLIP(nn.Module): # TODO: fix this implementation
    def __init__(self, opt, embed_dim=256, normalize=False):
        super(LinearforCLIP, self).__init__()
        in_dim = 512
        self.head = nn.Linear(in_dim, embed_dim, bias=False)
        self.normalize = normalize

    def forward(self, text_emb):
        embed = self.head(text_emb)
        if self.normalize:
            embed = F.normalize(embed, dim=-1)
        return embed

class LinearforOrig(nn.Module): # TODO: fix this implementation
    def __init__(self, opt, in_dim=1000, embed_dim=256, normalize=False):
        super(LinearforOrig, self).__init__()
        self.head = nn.Linear(in_dim, embed_dim, bias=False)
        self.normalize = normalize

    def forward(self, text_emb):
        embed = self.head(text_emb)
        if self.normalize:
            embed = F.normalize(embed, dim=-1)

        return embed

class LinearforText(nn.Module): # TODO: fix this implementation
    def __init__(self, opt, embed_dim=256, normalize=False):
        super(LinearforText, self).__init__()
        in_dim = 4096
        self.head = nn.Linear(in_dim, embed_dim, bias=False)
        self.normalize = normalize

    def forward(self, text_emb):
        embed = self.head(text_emb)
        if self.normalize:
            embed = F.normalize(embed, dim=-1)

        return embed

class SatCLIPwithLinear(nn.Module):
    def __init__(self, sinr_train_params, opt, in_dim=256, embed_dim=256, normalize=False):
        super(SatCLIPwithLinear, self).__init__()
        from models.satclip.main import SatCLIPLightningModule
        sinr_train_params['hyper_parameters'].pop('eval_downstream')
        sinr_train_params['hyper_parameters'].pop('air_temp_data_path')
        sinr_train_params['hyper_parameters'].pop('election_data_path')
        lightning_model = SatCLIPLightningModule(**sinr_train_params['hyper_parameters'])
        self.sinr = lightning_model.model.location
        
        self.head = nn.Linear(in_dim, embed_dim, bias=False)
        self.normalize = normalize

    def forward(self, lonlat):
        sinr_feat = self.sinr(lonlat)  # torch.Size([batch size, 256])
        embed = self.head(sinr_feat)
        if self.normalize:
            embed = F.normalize(embed, dim=-1)
        return sinr_feat, embed

class SINRwithLinear(nn.Module):
    def __init__(self, sinr_train_params, opt, in_dim=256, embed_dim=256, normalize=False):
        super(SINRwithLinear, self).__init__()
        self.sinr = SINREmbeddingModel(sinr_train_params, return_logits=False)
        self.head = nn.Linear(in_dim, embed_dim, bias=False)
        self.normalize = normalize

    def forward(self, lonlat):
        sinr_feat = self.sinr(lonlat)
        embed = self.head(sinr_feat)
        if self.normalize:
            embed = F.normalize(embed, dim=-1)
        return sinr_feat, embed

def main_worker(gpu, ngpus_per_node, opt):
    if opt.use_bnft:
        assert opt.use_lora is False
        assert opt.use_dora is False
    if opt.use_lora or opt.use_dora:
        assert opt.use_bnft is False
    opt.gpu = gpu
    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))
    if opt.distributed:
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        if opt.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            opt.rank = opt.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
    sinr_train_params = torch.load(opt.sinr_ckpt_path, map_location='cpu')
    lgr.debug(f"Using SINR parameters from: {opt.sinr_ckpt_path}")
    satellite_model = get_model(
        encoder_name=opt.satellite_encoder, head=opt.satellite_head, 
        num_outputs=opt.common_embed_dim, num_inp_feats=opt.num_inp_channels,
        pretrained=(not opt.satellite_notpretrained), is_residual=False,
        normalize=False, 
        encoder_ckpt_path=opt.satellite_encoder_ckpt,
        use_lora=opt.use_lora,
        use_dora=opt.use_dora,
        lora_layer_types=opt.lora_layer_types,
    )


    if opt.use_sinr and opt.use_satclip:
        raise NotImplementedError
    if opt.use_satclip:
        lgr.debug(f"Using SatCLIP model for encoding lon/lat coordinates")
        sinr_model = SatCLIPwithLinear(sinr_train_params, opt, embed_dim=opt.common_embed_dim, normalize=False)
    else:
        lgr.debug(f"Using SINR model for encoding lon/lat coordinates")
        sinr_model = SINRwithLinear(sinr_train_params, opt, embed_dim=opt.common_embed_dim, normalize=False)
    text_emb_model = LinearforText(opt, embed_dim=opt.common_embed_dim, normalize=False)
    
    params = []

    # Freeze sinr backbone and not head
    if opt.use_sinr or opt.use_satclip:
        for param in sinr_model.sinr.parameters():
            param.requires_grad = False
        for param in sinr_model.head.parameters():
            param.requires_grad = True
        params += sinr_model.head.parameters()

    # train text embd model
    params += text_emb_model.parameters()
    
    if opt.freeze_backbone:
        lgr.debug(f"Freezing backbone with head={opt.satellite_head}")
        satellite_model.backbone.eval()
        for param in satellite_model.backbone.parameters():
            param.requires_grad = False
        for param in satellite_model.decoder.parameters():
            param.requires_grad = True
        params += satellite_model.decoder.parameters()
    elif opt.use_bnft:
        lgr.debug(f"Freezing all layers except batchnorm and fc")
        for idx, (name, param) in enumerate(satellite_model.named_parameters()):
            if ('bn' in name) or ('norm' in name) or ('ln' in name):  # 'norm' for satlas/swin models; 'bn' for pytorch resnet models; 'ln' for vit models
                param.requires_grad = True
            elif opt.is_tunefc and (("fc" in name) or ("head" in name) or ("proj") in name):    # proj for remoteclip
                param.requires_grad = True
            else:
                param.requires_grad = False
        params += satellite_model.parameters()
    else:
        params += satellite_model.parameters()


    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif opt.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                satellite_model.cuda(opt.gpu)
                sinr_model.cuda(opt.gpu)
                text_emb_model.cuda(opt.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                opt.batch_size = int(opt.batch_size / ngpus_per_node)
                opt.workers = int((opt.workers + ngpus_per_node - 1) / ngpus_per_node)
                satellite_model = torch.nn.parallel.DistributedDataParallel(satellite_model, device_ids=[opt.gpu], find_unused_parameters=(opt.find_unused_param==1))
                sinr_model = torch.nn.parallel.DistributedDataParallel(sinr_model, device_ids=[opt.gpu], find_unused_parameters=(opt.find_unused_param==1))
                text_emb_model = torch.nn.parallel.DistributedDataParallel(text_emb_model, device_ids=[opt.gpu], find_unused_parameters=(opt.find_unused_param==1))
            else:
                satellite_model.cuda()
                sinr_model.cuda()
                text_emb_model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                satellite_model = torch.nn.parallel.DistributedDataParallel(satellite_model, find_unused_parameters=(opt.find_unused_param==1))
                sinr_model = torch.nn.parallel.DistributedDataParallel(sinr_model, find_unused_parameters=(opt.find_unused_param==1))
                text_emb_model = torch.nn.parallel.DistributedDataParallel(text_emb_model, find_unused_parameters=(opt.find_unused_param==1))
    elif opt.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        satellite_model = satellite_model.cuda(opt.gpu)
        sinr_model = sinr_model.cuda(opt.gpu)
        text_emb_model = text_emb_model.cuda(opt.gpu)

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        satellite_model = satellite_model.to(device)
        sinr_model = sinr_model.to(device)
        text_emb_model = text_emb_model.to(device)

    else:
        satellite_model = torch.nn.DataParallel(satellite_model).cuda()
        sinr_model = torch.nn.DataParallel(sinr_model).cuda()
        text_emb_model = torch.nn.DataParallel(text_emb_model).cuda()


    if torch.cuda.is_available():
        if opt.gpu:
            device = torch.device('cuda:{}'.format(opt.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    optimizer = optim.Adam(params, lr=opt.lr)
    scheduler = None
    if opt.scheduler == "steplr":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif opt.scheduler == "cosineanneal":
        pass
    lgr.debug(f"Using scheduler: {scheduler}")

    if opt.multiprocessing_distributed:
        if opt.use_sinr:
            new_sinr_ckpt = {f"module.sinr.model.{k}":v for k,v in sinr_train_params["state_dict"].items()}
            sinr_train_params['state_dict'] = new_sinr_ckpt
        else:
            new_sinr_ckpt = {f"module.sinr.{k.split('model.location.')[-1]}":v for k,v in sinr_train_params["state_dict"].items()}
            sinr_train_params['state_dict'] = new_sinr_ckpt
    else:
        if opt.use_sinr:
            new_sinr_ckpt = {f"sinr.model.{k}":v for k,v in sinr_train_params["state_dict"].items()}
            sinr_train_params['state_dict'] = new_sinr_ckpt
        else:
            new_sinr_ckpt = {f"sinr.{k.split('model.location.')[-1]}":v for k,v in sinr_train_params["state_dict"].items()}
            sinr_train_params['state_dict'] = new_sinr_ckpt
    if opt.use_sinr or opt.use_satclip:
        tmp = sinr_model.load_state_dict(sinr_train_params['state_dict'], strict=False)
        lgr.debug(f"sinr_model load state: missing_keys={tmp.missing_keys}")
    if (opt.satellite_encoder_ckpt is not None) and ("satclip" in opt.satellite_encoder_ckpt.lower()):
        lgr.debug(f"Loading checkpoint from {opt.satellite_encoder_ckpt}")
        satclip_ckpt = torch.load(opt.satellite_encoder_ckpt, map_location='cpu')
        new_satclip_ckpt = {f"module.backbone.{k.split('model.visual.')[-1]}":v for k,v in satclip_ckpt["state_dict"].items()}
        satclip_ckpt['state_dict'] = new_satclip_ckpt
        tmp = satellite_model.load_state_dict(satclip_ckpt['state_dict'], strict=False)
        lgr.debug(f"satclip model load state: missing_keys={tmp.missing_keys}")
        lgr.debug(f"Chaning satclip weights to float")
        satellite_model = satellite_model.float()
        # lgr.debug(f"satclip model load state: tmp={tmp}")

    start_epoch = 0
    if opt.ckpt_path is not None:
        global DATE_STR
        DATE_STR = opt.ckpt_path.split("/")[-1].split("_")[0]
        lgr.debug(f"Loading checkpoint: {opt.ckpt_path}. Changing DATE_STR to {DATE_STR}")
        if opt.gpu is None:
            checkpoint = torch.load(opt.ckpt_path)
        elif torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(opt.gpu)
            checkpoint = torch.load(opt.ckpt_path, map_location=loc)
        start_epoch = checkpoint['epoch']
        tmp = satellite_model.load_state_dict(checkpoint['satellite_state_dict'])
        lgr.debug(f"satellite_model load state: {tmp}")
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('sinr_state_dict'):
            tmp = sinr_model.load_state_dict(checkpoint['sinr_state_dict'])
            lgr.debug(f"sinr_model load state: {tmp}")
        if checkpoint.get('text_emb_model'):
            tmp = text_emb_model.load_state_dict(checkpoint['text_emb_model'])
            lgr.debug(f"text_emb_model load state: {tmp}")
        if (scheduler is not None) and (checkpoint.get("scheduler")):
            scheduler = checkpoint["scheduler"]
            lgr.debug(f"Loaded scheduler from checkpoint")
        lgr.debug(f"Successfully loaded checkpoint. Starting from epoch: {start_epoch}")
    else:
        lgr.debug(f"No checkpoint loaded.")


    lgr.debug('Parameter Space: ABS: {:.1f}\n'.format(count_parameters(satellite_model)))

    # define dataset path
    resize_size = None
    if (opt.satellite_encoder == "vitb16") or (opt.satellite_encoder == "vitl16"):
        resize_size = 224
        lgr.warning(f"NOTE: changing size of input image to 224 to fit pretrained models of vitb16")
    elif (opt.satellite_encoder_ckpt is not None) and ("remoteclip" in opt.satellite_encoder_ckpt.lower()):
        resize_size = 224
        lgr.warning(f"NOTE: changing size of input image to 224 for RemoteCLIP model")

    train_dataset = SentinelwTextImg(
        root=opt.root, split="train", metadata_path=opt.metadata_path, 
        is_img_contrast=True, resize_size=resize_size, use_text=True, 
        use_multispectral=opt.use_multispectral,
    )
    lgr.debug(f"Found {len(train_dataset)} train_dataset")

    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    lgr.debug(f"Found {len(train_loader)} train_loader")

    # define parameters
    total_epoch = opt.epochs
    train_batch = len(train_loader)
    lgr.debug(f"train_batch={train_batch}")
    loss_fn = ClipLoss(cache_labels=True)
    if opt.use_satclip:
        lgr.debug(f"Chaning sinr_model weights to float")
        sinr_model = sinr_model.float()
    if opt.use_sinr or opt.use_satclip:
        if opt.multiprocessing_distributed:
            sinr_model.module.sinr.eval()
            sinr_model.module.head.train()
        else:
            sinr_model.sinr.eval()
            sinr_model.head.train()
    text_emb_model.train()

    init_logit_scale = np.log(10)
    logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    train_losses = []
    text_losses, sinr_losses, sat_losses = [], [], []
    text_accs, sinr_accs, sat_accs = [], [], []
    for epoch in range(start_epoch, total_epoch):
        # iteration for all batches
        if opt.freeze_backbone:
            if opt.multiprocessing_distributed:
                satellite_model.module.backbone.eval()
                satellite_model.module.decoder.train()
            else:
                satellite_model.backbone.eval()
                satellite_model.decoder.train()
        else:
            satellite_model.train()
        train_dataset = iter(train_loader)
        
        train_loss0 = AverageMeter('trainLoss0', ':.4e')
        sinr_loss0 = AverageMeter('sinrLoss0', ':.4e')
        text_loss0 = AverageMeter('textLoss0', ':.4e')
        sat_loss0 = AverageMeter('satLoss0', ':.4e')
        sinr_acc0 = AverageMeter('sinrAcc0', ':.4e')
        text_acc0 = AverageMeter('textAcc0', ':.4e')
        sat_acc0 = AverageMeter('satAcc0', ':.4e')
        for k in range(train_batch):
            optimizer.zero_grad()
            imgs, aug_imgs, lonlats, text_emb = next(train_dataset)
            aug_imgs = aug_imgs.to(device, non_blocking=True)
            imgs = imgs.to(device, non_blocking=True)
            if (opt.satellite_encoder_ckpt is not None) and ("satclip" in opt.satellite_encoder_ckpt.lower()):
                # for SatCLIP, make other channels zero (satclip channels: B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, and B12)
                # BGR = B2, B3, B4
                tmp = torch.zeros((imgs.shape[0], 13, imgs.shape[2], imgs.shape[3])).to(device, non_blocking=True)
                tmp[:, 1, :, :] = imgs[:, 2, :, :]  # blue
                tmp[:, 2, :, :] = imgs[:, 1, :, :]  # green
                tmp[:, 3, :, :] = imgs[:, 0, :, :]  # red
                if opt.use_multispectral:
                    tmp[:, 4, :, :] = imgs[:, 3, :, :]  # band 5
                    tmp[:, 5, :, :] = imgs[:, 4, :, :]  # band 6
                    tmp[:, 6, :, :] = imgs[:, 5, :, :]  # band 7
                    tmp[:, 7, :, :] = imgs[:, 6, :, :]  # band 8
                    tmp[:, 11, :, :] = imgs[:, 7, :, :]  # band 11
                    tmp[:, 12, :, :] = imgs[:, 8, :, :]  # band 12
                imgs = tmp
                # Image augmentations
                tmp2 = torch.zeros((aug_imgs.shape[0], 13, aug_imgs.shape[2], aug_imgs.shape[3])).to(device, non_blocking=True)
                tmp2[:, 1, :, :] = aug_imgs[:, 2, :, :]  # blue
                tmp2[:, 2, :, :] = aug_imgs[:, 1, :, :]  # green
                tmp2[:, 3, :, :] = aug_imgs[:, 0, :, :]  # red
                if opt.use_multispectral:
                    tmp2[:, 4, :, :] = aug_imgs[:, 3, :, :]  # band 5
                    tmp2[:, 5, :, :] = aug_imgs[:, 4, :, :]  # band 6
                    tmp2[:, 6, :, :] = aug_imgs[:, 5, :, :]  # band 7
                    tmp2[:, 7, :, :] = aug_imgs[:, 6, :, :]  # band 8
                    tmp2[:, 11, :, :] = aug_imgs[:, 7, :, :]  # band 11
                    tmp2[:, 12, :, :] = aug_imgs[:, 8, :, :]  # band 12
                aug_imgs = tmp2
            elif (opt.satellite_encoder_ckpt is not None) and ("prithvi" in opt.satellite_encoder_ckpt.lower()):
                # for Prithvi, make other channels zero (Prithvi channels: Blue, Green, Red, Narrow NIR, SWIR1, SWIR2); NOTE: Narrow NIR is B8A, SWIR1=B11, SWIR2=B12 in sentinel2
                # BGR = B2, B3, B4
                tmp = torch.zeros((imgs.shape[0], 6, imgs.shape[2], imgs.shape[3])).to(device, non_blocking=True)
                tmp[:, 0, :, :] = imgs[:, 2, :, :]  # blue
                tmp[:, 1, :, :] = imgs[:, 1, :, :]  # green
                tmp[:, 2, :, :] = imgs[:, 0, :, :]  # red
                if opt.use_multispectral:
                    tmp[:, 4, :, :] = imgs[:, 7, :, :]  # SWIR1
                    tmp[:, 5, :, :] = imgs[:, 8, :, :]  # SWIR2
                imgs = tmp.unsqueeze(2) # add dimension for time
                # Image augmentations
                tmp2 = torch.zeros((aug_imgs.shape[0], 6, aug_imgs.shape[2], aug_imgs.shape[3])).to(device, non_blocking=True)
                tmp2[:, 0, :, :] = aug_imgs[:, 2, :, :]  # blue
                tmp2[:, 1, :, :] = aug_imgs[:, 1, :, :]  # green
                tmp2[:, 2, :, :] = aug_imgs[:, 0, :, :]  # red
                if opt.use_multispectral:
                    tmp2[:, 4, :, :] = aug_imgs[:, 7, :, :]  # SWIR1
                    tmp2[:, 5, :, :] = aug_imgs[:, 8, :, :]  # SWIR2
                aug_imgs = tmp2.unsqueeze(2) # add dimension for time
            lonlats = lonlats.to(device, non_blocking=True)
            text_emb = text_emb.to(device, non_blocking=True)

            all_imgs = torch.cat([imgs, aug_imgs], dim=0)
            all_feats, all_preds = satellite_model(all_imgs)
            assert opt.satellite_head == "threelinearnobias"    # means three outputs: text, sinr, sat
            tmp_text_pred, tmp_sinr_pred, tmp_img_pred = all_preds
            sat_model_text_pred, sat_model_sinr_pred, sat_model_img_pred = tmp_text_pred[:len(imgs)], tmp_sinr_pred[:len(imgs)], tmp_img_pred[:len(imgs)]
            sat_model_img_pred_aug = tmp_img_pred[len(imgs):]
            
            text_preds = text_emb_model(text_emb)

            
            loss = 0
            # Text features
            text_loss, sat_txt_logits, txt_logits = loss_fn(sat_model_text_pred, text_preds, logit_scale, return_logits=True)
            text_loss = text_loss*opt.text_loss_weight
            txt_num_corr = torch.sum(torch.argmax(sat_txt_logits, dim=1) == loss_fn.labels[sat_model_text_pred.device])
            loss += text_loss
            
            # Location features (from SINR/SatCLIP)
            sinr_feats, sinr_preds = sinr_model(lonlats)
            sinr_loss, sat_sinr_logits, sinr_logits = loss_fn(sat_model_sinr_pred, sinr_preds, logit_scale, return_logits=True)
            sinr_num_corr = torch.sum(torch.argmax(sat_sinr_logits, dim=1) == loss_fn.labels[sat_sinr_logits.device])
            loss += sinr_loss
            
            # Image augmentation loss
            sat_img_loss, sat_img_logits, img_logits = loss_fn(sat_model_img_pred, sat_model_img_pred_aug, logit_scale, return_logits=True)
            sat_img_loss = sat_img_loss*opt.img_loss_weight
            img_num_corr = torch.sum(torch.argmax(sat_img_logits, dim=1) == loss_fn.labels[sat_img_logits.device])
            loss += sat_img_loss

            loss.backward()
            optimizer.step()
            
            train_loss0.update(loss.item(), imgs.shape[0])
            # Text
            text_loss0.update(text_loss.item(), imgs.shape[0])
            text_acc0.update(txt_num_corr.item()/imgs.shape[0], imgs.shape[0])
            # Location
            sinr_loss0.update(sinr_loss.item(), imgs.shape[0])
            sinr_acc0.update(sinr_num_corr.item()/imgs.shape[0], imgs.shape[0])
            # Image augmentation
            sat_loss0.update(sat_img_loss.item(), imgs.shape[0])
            sat_acc0.update(img_num_corr.item()/imgs.shape[0], imgs.shape[0])



        if scheduler is not None:   # updated every epoch
            scheduler.step()
            
        if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed
                and opt.rank % ngpus_per_node == 0):
            lgr.debug(f"[{epoch:03d}/{total_epoch:03d}] Loss: {train_loss0.avg}")
            train_losses.append(train_loss0.avg)
            all_losses_plot = [train_losses]
            all_losses_names = ["train"]
            all_accs_plot = []
            all_accs_names = []
            # Text
            text_losses.append(text_loss0.avg)
            all_losses_plot.append(text_losses)
            all_losses_names.append("text")
            text_accs.append(text_acc0.avg)
            all_accs_plot.append(text_accs)
            all_accs_names.append("text")
            #Location
            sinr_losses.append(sinr_loss0.avg)
            all_losses_plot.append(sinr_losses)
            all_losses_names.append("sinr")
            sinr_accs.append(sinr_acc0.avg)
            all_accs_plot.append(sinr_accs)
            all_accs_names.append("sinr")
            # image Augmentation
            sat_losses.append(sat_loss0.avg)
            all_losses_plot.append(sat_losses)
            all_losses_names.append("sat")
            sat_accs.append(sat_acc0.avg)
            all_accs_plot.append(sat_accs)
            all_accs_names.append("sat")

            # print(train_losses, text_losses, sinr_losses)
            # plot_losses(train_losses, opt)
            plot_losses(all_losses_plot, opt, is_multiple_losses=True, loss_labels=all_losses_names)
            if len(all_accs_plot) > 0:
                plot_losses(all_accs_plot, opt, suffix="accuracy", is_multiple_losses=True, loss_labels=all_accs_names)
            save_checkpoint({
                'epoch': epoch + 1,
                'satellite_state_dict': satellite_model.state_dict(),
                'sinr_state_dict': sinr_model.state_dict(),
                'text_emb_model': text_emb_model.state_dict() ,
                'optimizer' : optimizer.state_dict(),
                'opt': opt,
                'loss': train_loss0.avg,
                'scheduler': scheduler if scheduler is not None else None,
                'all_losses_plot': all_losses_plot,
                'all_losses_names': all_losses_names,
                'all_accs_plot': all_accs_plot,
                'all_accs_names': all_accs_names,
            }, opt, epoch, f'_latest_checkpoint.pth')
            

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)
        
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

