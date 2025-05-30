import torch
from torch import nn
from enum import Enum
import models
import datasets
import utils
import os


class EmbeddingModel(nn.Module):
    def __init__(self, checkpoint_path, return_logits=True):
        super(EmbeddingModel, self).__init__()
        self.return_logits = return_logits

        # load model
        train_params = torch.load(checkpoint_path, map_location='cpu')
        model = models.get_model(train_params['params'], inference_only=True)
        model.load_state_dict(train_params['state_dict'], strict=False)
        model.eval()

        self.train_params = train_params
        self.model = model

        if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
            raster = datasets.load_env()
        else:
            raster = None
        enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster,
                                 input_dim=train_params['params']['input_dim'])
        self.enc = enc

    def forward(self, lonlat):
        normalization = torch.tensor([[1/180, 1/90]], device=lonlat.device)
        locs_enc = self.enc.encode(lonlat*normalization, normalize=False)
        if self.train_params['params']['model'] == 'HyperNet':
            out = self.model.pos_enc(locs_enc)
        else:
            out = self.model(locs_enc, return_feats=True)

        if not self.return_logits:
            return out

        if self.train_params['params']['model'] == 'HyperNet':
            logits = out @ (self.model.species_params.T)
        else:
            logits = self.model.class_emb(out)
        return logits


if __name__ == '__main__':
    model = EmbeddingModel('experiments/exp_sinr_baseline/model.pt', False).cuda()
    out = model(torch.from_numpy(utils.coord_grid((100,200))).cuda())