"""
Extracts features from a trained network for each geo location, performs 
dimensionality reduction, and generates an output image.

Different seed values will result in different mappings of locations to colors. 
"""

import torch
import numpy as np
import datasets
import matplotlib.pyplot as plt
import os
from sklearn import decomposition
from skimage import exposure
import json
import utils
import models
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image

def viz_ica(model, input_dim=4, show=True):
    op_dir = 'visualizations/'
    num_ds_dims = 3
    device = next(model.parameters()).device
    with open('../paths.json', 'r') as f:
        paths = json.load(f)
    enc = utils.CoordEncoder('sin_cos', input_dim=input_dim)
    # load ocean mask
    mask = np.load(os.path.join(paths['masks'], 'ocean_mask.npy'))
    mask_inds = np.where(mask.reshape(-1) == 1)[0]

    locs = utils.coord_grid(mask.shape)
    locs = locs[mask_inds, :]
    locs = torch.from_numpy(locs)
    locs_enc = enc.encode(locs).to(device)
    with torch.no_grad():
        feats = model(locs_enc, return_feats=True).cpu().numpy()

    # standardize the features
    f_mu = feats.mean(0)
    f_std = feats.std(0)
    feats = feats - f_mu
    feats = feats / f_std
    assert not np.any(np.isnan(feats))
    assert not np.any(np.isinf(feats))

    # downsample features - choose middle time step
    print('Performing dimensionality reduction.')
    dsf = decomposition.FastICA(n_components=num_ds_dims, random_state=2001, whiten='unit-variance', max_iter=1000)
    dsf.fit(feats)

    feats_ds = dsf.transform(feats)

    # equalize - doing this means there is no need to do the mean normalization
    for cc in range(num_ds_dims):
        feats_ds[:, cc] = exposure.equalize_hist(feats_ds[:, cc])

    # convert into image
    op_im = np.ones((mask.shape[0] * mask.shape[1], num_ds_dims))
    op_im[mask_inds] = feats_ds
    op_im = op_im.reshape((mask.shape[0], mask.shape[1], num_ds_dims))

    # save output
    op_path = os.path.join(op_dir, 'ica.png')
    print('Saving image to: ' + op_path)
    plt.imsave(op_path, (op_im * 255).astype(np.uint8))
    if show:
        plt.imshow((op_im * 255).astype(np.uint8))
        plt.show()


if __name__ == '__main__':
    # params - specify model of interest here
    seed = 2001
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    num_ds_dims = 3
    # model_path = './experiments/exp_sinr_baseline_env/model.pt'
    model_path = "/work/pi_smaji_umass_edu/jmhamilton/sinr/experiments/exp_sinr_baseline_env/model.pt"

    op_dir = 'visualizations/'
    if not os.path.isdir(op_dir):
        os.makedirs(op_dir)

    eval_params = {}
    if 'device' not in eval_params:
        eval_params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_params['model_path'] = model_path

    # load model
    train_params = torch.load(eval_params['model_path'], map_location='cpu')
    model2 = models.get_model(train_params['params'], inference_only=True)
    model2.load_state_dict(train_params['state_dict'], strict=False)
    #TODO
    model2 = model2.to(eval_params['device'])
    model2.eval()
    if train_params['params']['model'] == 'HyperNet':
        model = lambda x, return_feats=True: model2.pos_enc(x)
    else:
        model = model2
    if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
        raster = datasets.load_env()
    else:
        raster = None
    enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster, input_dim=train_params['params']['input_dim'])

    # load ocean mask
    mask = np.load(os.path.join(paths['masks'], 'ocean_mask.npy'))
    mask_inds = np.where(mask.reshape(-1) == 1)[0]

    locs = utils.coord_grid(mask.shape)
    locs = locs[mask_inds, :]
    locs = torch.from_numpy(locs)
    locs_enc = enc.encode(locs).to(eval_params['device'])
    with torch.no_grad():
        feats = model(locs_enc, return_feats=True).cpu().numpy()

    # standardize the features
    f_mu = feats.mean(0)
    f_std = feats.std(0)
    f_mask = f_std > 0
    feats = feats - f_mu
    feats = feats / (f_std + 1e-8)
    assert not np.any(np.isnan(feats))
    assert not np.any(np.isinf(feats))
    # downsample features - choose middle time step
    print('Performing dimensionality reduction.')
    dsf = decomposition.FastICA(n_components=num_ds_dims, random_state=seed, whiten='unit-variance', max_iter=1000)
    #dsf.fit(feats)

    all_data = np.load("sample_imgs_with_lonlats.npy", allow_pickle=True)
    for i in range(len(all_data)):
    # for i in range(16,17):
        data = all_data[i]
        # data = np.load("/work/pi_smaji_umass_edu/rdaroya/satellite-representations/big_sample_with_lonlat.npy", allow_pickle=True).item()
        # print(f"data: {data.shape}")
        # data['lonlat'] = data['lonlat'][::20,::20,:]
        locs2 = data['lonlat']
        #print(data['image_fp'])
        s = locs2.shape
        locs2 = torch.from_numpy(locs2).reshape(-1, 2).float()
        locs_enc2 = enc.encode(locs2).to(eval_params['device'])
        with torch.no_grad():
            feats2 = model(locs_enc2, return_feats=True).cpu().numpy()
        raw_feats = feats2.copy()
        mask = ~np.isnan(feats2.sum(1))
        # standardize the features
        f_mu2 = feats2[mask].mean(0)
        f_std2 = feats2[mask].std(0)
        feats2 = feats2 - f_mu2
        feats2 = feats2 / (f_std2 + 1e-8)

        dsf.fit(feats2[mask])
        feats_ds = dsf.transform(feats2)
        torch.save({'lonlat': data['lonlat'], 'features_normalized': feats2, 'features_raw': raw_feats, 'pca': feats_ds},'features.pt')
        # equalize - doing this means there is no need to do the mean normalization
        '''for cc in range(num_ds_dims):
            feats_ds[:, cc] = exposure.equalize_hist(feats_ds[:, cc])'''
        feats_ds = (feats_ds + 1)/(2)

        # convert into image
        '''op_im = np.ones((mask.shape[0]*mask.shape[1], num_ds_dims))
        op_im[mask_inds] = feats_ds
        op_im = op_im.reshape((mask.shape[0], mask.shape[1], num_ds_dims))'''
        feats_ds = feats_ds[-s[0]*s[1]:]
        op_im = feats_ds.reshape(s[0], s[1], num_ds_dims)

        im = read_image(data['image_fp'])/255.0
        # im = torch.from_numpy(data['image_rgb'][::20,::20,:]).permute(2,0,1)

        # save output
        op_file_name = os.path.basename(model_path[:-3]) + f'_ica({i}).png'
        op_path = os.path.join(op_dir, op_file_name)
        print('Saving image to: ' + op_path)
        save_image(make_grid([im, torch.from_numpy(op_im).permute(2, 0, 1)]), op_path)

        op_file_name2 = os.path.basename(model_path[:-3]) + f'_ica({i})_combined.png'
        op_path2 = os.path.join(op_dir, op_file_name2)
        im = np.transpose(im, (1,2,0))
        op_im = (op_im - op_im.min())/(op_im.max()-op_im.min())
        new_im = im*0.3 + (op_im*0.7)
        print(f"im: {im.max()} {im.min()}")
        print(f"op_im: {op_im.max()} {op_im.min()}")
        new_im = new_im.detach().cpu().numpy()
        plt.imsave(op_path2, (new_im*255).astype(np.uint8))
