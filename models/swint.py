import torch
from torch import nn
import torchvision


class SwinTModel(torch.nn.Module):
    def __init__(self, num_inp_feats=6, pretrained=True):
        super(SwinTModel, self).__init__()
        
        if num_inp_feats != 3:
            self.first = nn.Sequential(
                nn.Conv2d(num_inp_feats, 3, kernel_size=1), # change channels from 6 to 3
                nn.Conv2d(3, 3, kernel_size=66, stride=2, padding=0), # from 512x512 to 224x224
            )
        else:
            self.first = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=66, stride=2, padding=0), # from 512x512 to 224x224
            )
        self.backbone = torchvision.models.swin_t(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone.head  = nn.Identity()
        self.backbone_channels = 768

    def forward(self, x):
        x = self.first(x)
        x = self.backbone(x)
        return x

class SwinTHead(torch.nn.Module):
    def __init__(self, backbone_channels, out_channels):
        super(SwinTHead, self).__init__()

        self.head1 = nn.Linear(backbone_channels, 1024)
        upscaling_block = lambda in_channels, out_channels: nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(kernel_size=3, in_channels=in_channels, out_channels=out_channels, padding=1), 
            nn.ReLU()
        )
        self.head = nn.Sequential(
            upscaling_block(1,8),
            *[
            upscaling_block(8,8) for i in range(3)
            ],
            nn.Conv2d(kernel_size=1, in_channels=8, out_channels=out_channels),
        )

    def forward(self, x):
        out = self.head1(x)
        out = out.contiguous().reshape(-1, 1, 32, 32)
        out = self.head(out)
        return out