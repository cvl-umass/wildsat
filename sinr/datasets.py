import os
import numpy as np
import json
import pandas as pd
from calendar import monthrange
import torch
import utils
import random
# from h3.unstable import vect
# import h3.api.numpy_int as h3

class LocationDataset(torch.utils.data.Dataset):
    def __init__(self, locs, labels, classes, class_to_taxa, input_enc, device, dates=None, input_dim=4, time_dim=0, noise_time=False):

        # handle input encoding:
        self.input_enc = input_enc
        if self.input_enc in ['env', 'sin_cos_env']:
            raster = load_env()
        else:
            raster = None
        self.enc = utils.CoordEncoder(input_enc, raster, input_dim=input_dim)

        # define some properties:
        self.locs = locs
        self.loc_feats = self.enc.encode(self.locs)
        self.labels = labels
        self.classes = classes
        self.class_to_taxa = class_to_taxa
        if dates is not None:
            self.dates = dates
            self.enc_time = utils.TimeEncoder()

        # useful numbers:
        self.num_classes = len(np.unique(labels))
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.noise_time = noise_time

        if self.enc.raster is not None:
            self.enc.raster = self.enc.raster.to(device)

    def __len__(self):
        return self.loc_feats.shape[0]

    def __getitem__(self, index):
        loc_feat  = self.loc_feats[index, :]
        loc       = self.locs[index, :]
        class_id  = self.labels[index]

        if self.time_dim > 0:
            date = self.dates[index]
            # add noise
            if self.noise_time:
                noise_level = random.random()
                noise = (2*random.random() - 1) * (0.5*noise_level)
                loc_feat = torch.cat([loc_feat, self.enc_time.encode_fast([date.item()+noise,noise_level])])
            else:
                raise NotImplementedError()
                loc_feat = torch.cat([loc_feat, torch.tensor(self.enc_time.encode([2*date.item()-1], normalize=False))])
            return loc_feat, torch.cat([loc, date[None]]), class_id
        else:
            return loc_feat, loc, class_id

def load_env():
    # with open('paths.json', 'r') as f:
    #     paths = json.load(f)
    # raster = load_context_feats(os.path.join(paths['env'],'bioclim_elevation_scaled.npy'))
    raster = load_context_feats('data/bioclim_elevation_scaled.npy')
    return raster

def load_context_feats(data_path):
    context_feats = np.load(data_path).astype(np.float32)
    context_feats = torch.from_numpy(context_feats)
    return context_feats

_file_cache = {}
def load_inat_data(ip_file, taxa_of_interest=None):
    if os.path.exists('.datacache.pt'):
        print('\nLoading cached data')
        if '.datacache.pt' not in _file_cache:
            # If not in the cache, read the file and store its content in the cache
            _file_cache['.datacache.pt'] = torch.load('.datacache.pt')
        locs, taxa, users, dates, years, obs_ids = _file_cache['.datacache.pt']
    else:
        print('\nLoading  ' + ip_file)
        data = pd.read_csv(ip_file)

        # remove outliers
        num_obs = data.shape[0]
        data = data[((data['latitude'] <= 90) & (data['latitude'] >= -90) & (data['longitude'] <= 180) & (data['longitude'] >= -180) )]
        if (num_obs - data.shape[0]) > 0:
            print(num_obs - data.shape[0], 'items filtered due to invalid locations')

        if 'accuracy' in data.columns:
            data.drop(['accuracy'], axis=1, inplace=True)

        if 'positional_accuracy' in data.columns:
            data.drop(['positional_accuracy'], axis=1, inplace=True)

        if 'geoprivacy' in data.columns:
            data.drop(['geoprivacy'], axis=1, inplace=True)

        if 'observed_on' in data.columns:
            data.rename(columns = {'observed_on':'date'}, inplace=True)

        num_obs_orig = data.shape[0]
        data = data.dropna()
        size_diff = num_obs_orig - data.shape[0]
        if size_diff > 0:
            print(size_diff, 'observation(s) with a NaN entry out of' , num_obs_orig, 'removed')

        # keep only taxa of interest:
        if taxa_of_interest is not None:
            num_obs_orig = data.shape[0]
            data = data[data['taxon_id'].isin(taxa_of_interest)]
            print(num_obs_orig - data.shape[0], 'observation(s) out of' , num_obs_orig, 'from different taxa removed')

        print('Number of unique classes {}'.format(np.unique(data['taxon_id'].values).shape[0]))

        locs = np.vstack((data['longitude'].values, data['latitude'].values)).T.astype(np.float32)
        taxa = data['taxon_id'].values.astype(np.int64)

        if 'user_id' in data.columns:
            users = data['user_id'].values.astype(np.int64)
            _, users = np.unique(users, return_inverse=True)
        elif 'observer_id' in data.columns:
            users = data['observer_id'].values.astype(np.int64)
            _, users = np.unique(users, return_inverse=True)
        else:
            users = np.ones(taxa.shape[0], dtype=np.int64)*-1

        # Note - assumes that dates are in format YYYY-MM-DD
        temp = np.array(data['date'], dtype='S10')
        temp = temp.view('S1').reshape((temp.size, -1))
        years = temp[:,:4].view('S4').astype(int)[:,0]
        months = temp[:,5:7].view('S2').astype(int)[:,0]
        days = temp[:,8:10].view('S2').astype(int)[:,0]
        days_per_month = np.cumsum([0] + [monthrange(2018, mm)[1] for mm in range(1, 12)])
        dates  = days_per_month[months-1] + days-1
        dates  = np.round((dates) / 364.0, 4).astype(np.float32)
        if 'id' in data.columns:
            obs_ids = data['id'].values
        elif 'observation_uuid' in data.columns:
            obs_ids = data['observation_uuid'].values
        torch.save((locs, taxa, users, dates, years, obs_ids), '.datacache.pt')

    return locs, taxa, users, dates, years, obs_ids

def choose_aux_species(current_species, num_aux_species, aux_species_seed, taxa_file):
    if num_aux_species == 0:
        return []
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    data_dir = paths['train']
    taxa_file = os.path.join(data_dir, taxa_file)
    with open(taxa_file, 'r') as f:
        inat_large_metadata = json.load(f)
    aux_species_candidates = [x['taxon_id'] for x in inat_large_metadata]
    aux_species_candidates = np.setdiff1d(aux_species_candidates, current_species)
    print(f'choosing {num_aux_species} species to add from {len(aux_species_candidates)} candidates')
    rng = np.random.default_rng(aux_species_seed)
    idx_rand_aux_species = rng.permutation(len(aux_species_candidates))
    aux_species = list(aux_species_candidates[idx_rand_aux_species[:num_aux_species]])
    return aux_species

def get_taxa_of_interest(species_set='all', num_aux_species=0, aux_species_seed=123, taxa_file=None, taxa_file_snt=None):
    if species_set == 'all':
        return None
    if species_set == 'snt_birds':
        assert taxa_file_snt is not None
        with open(taxa_file_snt, 'r') as f: #
            taxa_subsets = json.load(f)
        taxa_of_interest = list(taxa_subsets['snt_birds'])
    else:
        raise NotImplementedError
    # optionally add some other species back in:
    aux_species = choose_aux_species(taxa_of_interest, num_aux_species, aux_species_seed, taxa_file)
    taxa_of_interest.extend(aux_species)
    return taxa_of_interest

def get_idx_subsample_observations(labels, hard_cap=-1, hard_cap_seed=123, subset=None, subset_cap=-1):
    if hard_cap == -1:
        if subset_cap != -1:
            raise NotImplementedError('subset_cap set but not hard_cap')
        return np.arange(len(labels))
    print(f'subsampling (up to) {hard_cap} per class for the training set')
    ids, counts = np.unique(labels, return_counts=True)
    count_ind = np.cumsum(counts)
    count_ind[1:] = count_ind[:-1]
    count_ind[0] = 0
    ss_rng = np.random.default_rng(hard_cap_seed)
    idx_rand = ss_rng.permutation(len(labels))

    ordered_inds = np.argsort(labels[idx_rand], kind='stable')
    caps = hard_cap + np.zeros_like(counts)
    if subset is not None and subset_cap != -1:
        caps[subset] = subset_cap
    idx_ss = idx_rand[np.concatenate([ordered_inds[i:i+min(limit, cap)] for i, limit, cap in zip(count_ind, counts, caps)])]
    print(f'final training set size: {len(idx_ss)}')
    return idx_ss

def uniform_sample_h3(cells, low, high):
    '''uniformly sample points in a batch of h3 cells'''
    out = np.empty((len(cells), 2))
    invalid_mask = np.arange(len(cells))
    cell_ids_buffer = np.empty(len(cells), dtype='uint64')
    while len(invalid_mask) > 0:
        #print(len(invalid_mask))
        pts = np.random.random((len(invalid_mask), 2))
        pts = high + pts*(low - high)

        vect._vect.geo_to_h3_vect(pts[:,0], pts[:,1], 5, cell_ids_buffer)

        valid_mask = (cell_ids_buffer[:len(cells)] == cells)
        out[invalid_mask[valid_mask]] = pts[valid_mask]
        neg_mask = ~valid_mask
        invalid_mask = invalid_mask[neg_mask]
        low = low[neg_mask]
        high = high[neg_mask]
        cells = cells[neg_mask]

    return out


class LocationIUCNDataset(torch.utils.data.Dataset):
    def __init__(self, locs, labels, classes, class_to_taxa, input_enc, device, dates=None, input_dim=4, time_dim=0, noise_time=False):
        # handle input encoding:
        self.input_enc = input_enc
        if self.input_enc in ['env', 'sin_cos_env']:
            raster = load_env()
        else:
            raster = None
        self.enc = utils.CoordEncoder(input_enc, raster, input_dim=input_dim)

        if os.path.exists('iucndataset_nocap.pt'):
            mask = torch.load('iucndataset_nocap.pt')
        else:
            from tqdm import tqdm
            # load iucn data
            with open('paths.json', 'r') as f:
                paths = json.load(f)
            with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
                data = json.load(f)
            obs_locs = np.array(data['locs'], dtype=np.float32)
            taxa = {int(tt):tk for tt, tk in data['taxa_presence'].items()}
            cells = vect.geo_to_h3(obs_locs[:, 1], obs_locs[:, 0], 5)
            mask = np.zeros(len(locs), dtype=bool)
            for i, data_taxa in tqdm(enumerate(class_to_taxa)):
                if data_taxa in taxa:
                    data_cells = vect.geo_to_h3(locs[labels==i, 1], locs[labels==i, 0], 5)
                    data = np.array(cells[taxa[data_taxa]])
                    data_inds = data.argsort()
                    search = np.searchsorted(data[data_inds], data_cells)
                    search = search.clip(0, len(data)-1)
                    mask[labels==i] = data[data_inds][search] == data_cells
                else:
                    mask[labels==i] = False
            torch.save(mask, 'iucndataset_nocap.pt')
        print('Reduced Size: ', mask.sum())

        # remove locations that are not in the iucn dataset
        locs = locs[mask]
        labels = labels[mask]
        if dates is not None:
            dates = dates[mask]

        labels_uniq, labels = np.unique(labels, return_inverse=True)
        classes = {class_to_taxa[i]: classes[class_to_taxa[i]] for i in labels_uniq}
        class_to_taxa = [class_to_taxa[i] for i in labels_uniq]
        # define some properties:
        self.locs = locs
        self.loc_feats = self.enc.encode(self.locs)
        self.labels = torch.from_numpy(labels)
        self.classes = classes
        self.class_to_taxa = class_to_taxa
        if dates is not None:
            self.dates = dates
            self.enc_time = utils.TimeEncoder()

        # useful numbers:
        self.num_classes = len(classes)
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.noise_time = noise_time
        if self.time_dim > 0:
            self.__getitem__ = self._get_item_time

        if self.enc.raster is not None:
            self.enc.raster = self.enc.raster.to(device)


    def viz_map(self, taxa_id, high_res=False):
        from matplotlib import pyplot as plt
        # load params
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
            data = json.load(f)
        obs_locs = np.array(data['locs'], dtype=np.float32)
        taxa = {int(tt): tk for tt, tk in data['taxa_presence'].items()}
        taxa_cells = vect.geo_to_h3(obs_locs[:, 1], obs_locs[:, 0], 5)
        # load taxa of interest
        if taxa_id in self.class_to_taxa:
            class_of_interest = self.class_to_taxa.index(taxa_id)
        else:
            print(f'Error: Taxa specified that is not in the model: {taxa_id}')
            return False
        print(f'Loading taxa: {taxa_id}')

        # load ocean mask
        if high_res:
            mask = np.load(os.path.join(paths['masks'], 'ocean_mask_hr.npy'))
        else:
            mask = np.load(os.path.join(paths['masks'], 'ocean_mask.npy'))
        mask_shape = mask.shape
        mask_inds = np.where(mask.reshape(-1) == 1)[0]

        # generate input features
        locs = utils.coord_grid(mask_shape)
        locs_cells = vect.geo_to_h3(locs[:,1], locs[:,0], 5)

        # mask iucn
        iucn_cells = np.sort(taxa_cells[taxa[taxa_id]])
        mask = iucn_cells[np.searchsorted(iucn_cells, locs_cells).clip(max=len(iucn_cells)-1)] == locs_cells
        mask_inds = np.where(mask == 1)[0]
        mask = mask.reshape(mask_shape)

        cell_inds, cell_counts = np.unique(vect.geo_to_h3(90*self.locs[self.labels==class_of_interest, 1], 180*self.locs[self.labels==class_of_interest, 0], 5),return_counts=True)
        search_inds = np.searchsorted(cell_inds, locs_cells).clip(max=len(cell_inds)-1)
        preds = np.zeros(len(locs))
        cell_mask = cell_inds[search_inds] == locs_cells
        preds[cell_mask] = cell_counts[search_inds][cell_mask]
        preds = preds/preds.sum()

        # Convert preds to log scale
        preds = np.log(preds)
        center = np.median(preds[np.isfinite(preds)])
        preds = 0.5*preds/(preds.max()-center)
        preds = (preds + 1 - preds.max()).clip(min=0)

        # mask data
        op_im = np.ones((mask.shape[0] * mask.shape[1])) * np.nan  # set to NaN
        op_im[mask_inds] = preds[mask_inds]

        # reshape and create masked array for visualization
        op_im = op_im.reshape((mask.shape[0], mask.shape[1]))
        op_im = np.ma.masked_invalid(op_im)

        # set color for masked values
        cmap = plt.cm.plasma
        cmap.set_bad(color='none')
        vmax = np.max(op_im)

        # save image
        save_loc = os.path.join('./images/', str(taxa_id) + '_map.png')
        print(f'Saving image to {save_loc}')
        plt.imsave(fname=save_loc, arr=op_im, vmin=0, vmax=vmax, cmap=cmap)

    def __len__(self):
        return self.loc_feats.shape[0]

    def __getitem__(self, index):
        return index


    def collate_fn(self, batch):
        if isinstance(batch[0], int):
            loc_feat = self.loc_feats[batch, :]
            loc = self.locs[batch, :]
            class_id = self.labels[batch]
            return loc_feat, loc, class_id
        else:
            return torch.utils.data.default_collate(batch)


    def _get_item_time(self, index):
        loc_feat  = self.loc_feats[index, :]
        loc       = self.locs[index, :]
        class_id  = self.labels[index]

        date = self.dates[index]
        # add noise
        if self.noise_time:
            noise_level = random.random()
            # noise = (2*random.random() - 1) * (0.5*(365 ** (noise_level - 1)))
            noise = (2 * random.random() - 1) * (0.5 * noise_level)
            loc_feat = torch.cat([loc_feat, self.enc_time.encode_fast([date.item() + noise, noise_level])])
        else:
            raise NotImplementedError()
            loc_feat = torch.cat([loc_feat, torch.tensor(self.enc_time.encode([2 * date.item() - 1], normalize=False))])
        return loc_feat, torch.cat([loc, date[None]]), class_id


class UniformH3Dataset(torch.utils.data.Dataset):
    def __init__(self, locs, labels, classes, class_to_taxa, input_enc, device, dates=None, input_dim=4, time_dim=0, noise_time=False, snt=False):
        if dates is not None or time_dim > 0 or noise_time:
            raise NotImplementedError()
        # handle input encoding:
        self.input_enc = input_enc
        if self.input_enc in ['env', 'sin_cos_env']:
            raster = load_env()
        else:
            raster = None
        self.enc = utils.CoordEncoder(input_enc, raster, input_dim=input_dim)
        self._enc = utils.CoordEncoder(input_enc, raster, input_dim=input_dim)

        # load h3 data:
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        if snt:
            D = np.load(os.path.join(paths['snt'], 'snt_res_5.npy'), allow_pickle=True)
            D = D.item()
            loc_indices_per_species = D['loc_indices_per_species']
            taxa = D['taxa']
            loc_indices_per_species = {t:ls for t,ls in zip(taxa, loc_indices_per_species)}
            obs_locs = D['obs_locs']
        else:
            with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
                data = json.load(f)
            obs_locs = np.array(data['locs'], dtype=np.float32)
            loc_indices_per_species = data['taxa_presence']
        self.taxa = {int(tt):tk for tt, tk in loc_indices_per_species.items()}
        self.cells = vect.geo_to_h3(obs_locs[:,1], obs_locs[:,0], 5)
        self.ind_to_cell = vect.geo_to_h3(obs_locs[:, 1], obs_locs[:, 0], 5)
        self.low_b = np.stack([np.array(h3.h3_to_geo_boundary(c)).min(axis=0) for c in self.cells])
        self.high_b = np.stack([np.array(h3.h3_to_geo_boundary(c)).max(axis=0) for c in self.cells])

        if os.path.exists('iucndataset_nocap.pt') and not snt:
            mask = torch.load('iucndataset_nocap.pt')
        elif os.path.exists('sntdataset_nocap.pt') and snt:
            mask = torch.load('sntdataset_nocap.pt')
        else:
            from tqdm import tqdm
            mask = np.zeros(len(locs), dtype=bool)
            for i, data_taxa in tqdm(enumerate(class_to_taxa)):
                if data_taxa in self.taxa:
                    data_cells = vect.geo_to_h3(locs[labels==i, 1], locs[labels==i, 0], 5)
                    data = np.array(self.cells[self.taxa[data_taxa]])
                    data_inds = data.argsort()
                    search = np.searchsorted(data[data_inds], data_cells)
                    search = search.clip(0, len(data)-1)
                    mask[labels==i] = data[data_inds][search] == data_cells
                else:
                    mask[labels==i] = False
            torch.save(mask, 'sntdataset_nocap.pt' if snt else 'iucndataset_nocap.pt')
        print('Reduced Size: ', mask.sum())
        # remove locations that are not in the iucn dataset
        locs = locs[mask]
        labels = labels[mask]

        labels_uniq, labels = np.unique(labels, return_inverse=True)
        classes = {class_to_taxa[i]: classes[class_to_taxa[i]] for i in labels_uniq}
        class_to_taxa = [class_to_taxa[i] for i in labels_uniq]

        # calculate species statistics
        _, counts = np.unique(labels, return_counts=True)
        self.num_obs = counts.sum()
        self.counts = counts / self.num_obs

        # define some properties:
        self.classes = classes
        self.class_to_taxa = class_to_taxa

        # useful numbers:
        self.num_classes = len(self.class_to_taxa)
        self.input_dim = input_dim

        if self.enc.raster is not None:
            self.enc.raster = self.enc.raster.to(device)

    def __len__(self):
        return self.num_obs

    def __getitem__(self, index, species=None):
        if species is None:
            class_id = np.random.choice(np.arange(self.num_classes), p=self.counts)
            species = self.class_to_taxa[class_id]
        else:
            class_id = -1
        ind = random.choice(self.taxa[species])
        cell = self.cells[ind]
        high, low = self.high_b[ind], self.low_b[ind]
        return cell, high, low, class_id

    def collate_fn(self, batch):
        cell, high, low, class_id = zip(*batch)
        cell = np.array(cell)
        high = np.stack(high)
        low = np.stack(low)
        class_id = torch.tensor(class_id, dtype=torch.long)
        pts = torch.from_numpy(uniform_sample_h3(cell, high, low)).flip(1)
        return self._enc.encode(pts, normalize=True).float(), pts, class_id


class MultiUniformH3Dataset(torch.utils.data.Dataset):
    def __init__(self, locs, labels, classes, class_to_taxa, input_enc, device, dates=None, input_dim=4, time_dim=0, noise_time=False):
        if dates is not None or time_dim > 0 or noise_time:
            raise NotImplementedError()
        # handle input encoding:
        self.input_enc = input_enc
        if self.input_enc in ['env', 'sin_cos_env']:
            raster = load_env()
        else:
            raster = None
        self.enc = utils.CoordEncoder(input_enc, raster, input_dim=input_dim)
        self._enc = utils.CoordEncoder(input_enc, raster, input_dim=input_dim)

        # load h3 data:
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
            data = json.load(f)
        obs_locs = np.array(data['locs'], dtype=np.float32)
        self.taxa = {int(tt): tk for tt, tk in data['taxa_presence'].items()}
        self.cells = vect.geo_to_h3(obs_locs[:, 1], obs_locs[:, 0], 5)
        self.ind_to_cell = vect.geo_to_h3(obs_locs[:, 1], obs_locs[:, 0], 5)
        self.low_b = np.stack([np.array(h3.h3_to_geo_boundary(c)).min(axis=0) for c in self.cells])
        self.high_b = np.stack([np.array(h3.h3_to_geo_boundary(c)).max(axis=0) for c in self.cells])

        if os.path.exists('iucndataset_nocap.pt'):
            mask = torch.load('iucndataset_nocap.pt')
        else:
            from tqdm import tqdm
            mask = np.zeros(len(locs), dtype=bool)
            for i, data_taxa in tqdm(enumerate(class_to_taxa)):
                if data_taxa in self.taxa:
                    data_cells = vect.geo_to_h3(locs[labels == i, 1], locs[labels == i, 0], 5)
                    data = np.array(self.cells[self.taxa[data_taxa]])
                    data_inds = data.argsort()
                    search = np.searchsorted(data[data_inds], data_cells)
                    search = search.clip(0, len(data) - 1)
                    mask[labels == i] = data[data_inds][search] == data_cells
                else:
                    mask[labels == i] = False
            torch.save(mask, 'iucndataset_nocap.pt')
        print('Reduced Size: ', mask.sum())
        # remove locations that are not in the iucn dataset
        locs = locs[mask]
        labels = labels[mask]

        labels_uniq, labels = np.unique(labels, return_inverse=True)
        classes = {class_to_taxa[i]: classes[class_to_taxa[i]] for i in labels_uniq}
        class_to_taxa = [class_to_taxa[i] for i in labels_uniq]

        # calculate species statistics
        _, counts = np.unique(labels, return_counts=True)
        self.num_obs = counts.sum()
        self.counts = counts / self.num_obs

        # define some properties:
        self.classes = classes
        self.class_to_taxa = class_to_taxa

        if os.path.exists('taxa_inverse.pt'):
            self.taxa_inverse = torch.load('taxa_inverse.pt')
        else:
            from collections import defaultdict
            ctt_i = {c: i for i, c in enumerate(class_to_taxa)}
            self.taxa_inverse = defaultdict(list)
            for k, vs in self.taxa.items():
                for v in vs:
                    self.taxa_inverse[v].append(ctt_i[k])
            torch.save(self.taxa_inverse, 'taxa_inverse.pt')

        # useful numbers:
        self.num_classes = len(self.class_to_taxa)
        self.input_dim = input_dim

        if self.enc.raster is not None:
            self.enc.raster = self.enc.raster.to(device)

    def __len__(self):
        return self.num_obs

    def __getitem__(self, index):
        cell_ind = np.random.randint(len(self.cells))
        cell = self.cells[cell_ind]
        label = np.zeros(len(self.classes), dtype=np.float32)
        label[self.taxa_inverse[cell_ind]] = 1
        high, low = self.high_b[cell_ind], self.low_b[cell_ind]
        return cell, high, low, label

    def collate_fn(self, batch):
        cell, high, low, label = zip(*batch)
        cell = np.array(cell)
        high = np.stack(high)
        low = np.stack(low)
        labels = torch.from_numpy(np.stack(label))
        pts = torch.from_numpy(uniform_sample_h3(cell, high, low)).flip(1)
        return self._enc.encode(pts, normalize=True).float(), pts.float(), labels


class MultiLocationIUCNDataset(torch.utils.data.Dataset):
    def __init__(self, locs, labels, classes, class_to_taxa, input_enc, device, dates=None, input_dim=4, time_dim=0, noise_time=False):

        # handle input encoding:
        self.input_enc = input_enc
        if self.input_enc in ['env', 'sin_cos_env']:
            raster = load_env()
        else:
            raster = None
        self.enc = utils.CoordEncoder(input_enc, raster, input_dim=input_dim)

        # load h3 data:
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
            data = json.load(f)
        obs_locs = np.array(data['locs'], dtype=np.float32)
        self.taxa = {int(tt): tk for tt, tk in data['taxa_presence'].items()}
        self.cells = vect.geo_to_h3(obs_locs[:, 1], obs_locs[:, 0], 5)
        self.cells_inverse = {c: i for i, c in enumerate(self.cells)}

        if os.path.exists('iucndataset_nocap.pt'):
            mask = torch.load('iucndataset_nocap.pt')
        else:
            from tqdm import tqdm
            mask = np.zeros(len(locs), dtype=bool)
            for i, data_taxa in tqdm(enumerate(class_to_taxa)):
                if data_taxa in self.taxa:
                    data_cells = vect.geo_to_h3(locs[labels==i, 1], locs[labels==i, 0], 5)
                    data = np.array(self.cells[self.taxa[data_taxa]])
                    data_inds = data.argsort()
                    search = np.searchsorted(data[data_inds], data_cells)
                    search = search.clip(0, len(data)-1)
                    mask[labels==i] = data[data_inds][search] == data_cells
                else:
                    mask[labels==i] = False
            torch.save(mask, 'iucndataset_nocap.pt')
        print('Reduced Size: ', mask.sum())

        # remove locations that are not in the iucn dataset
        locs = locs[mask]
        labels = labels[mask]
        if dates is not None:
            dates = dates[mask]

        labels_uniq, labels = np.unique(labels, return_inverse=True)
        classes = {class_to_taxa[i]: classes[class_to_taxa[i]] for i in labels_uniq}
        class_to_taxa = [class_to_taxa[i] for i in labels_uniq]
        # define some properties:
        self.locs = locs
        self.loc_cells = vect.geo_to_h3(locs[:, 1], locs[:, 0], 5)
        self.loc_feats = self.enc.encode(self.locs)
        self.labels = labels
        self.classes = classes
        self.class_to_taxa = class_to_taxa
        if dates is not None:
            self.dates = dates
            self.enc_time = utils.TimeEncoder()

        if os.path.exists('taxa_inverse.pt'):
            self.taxa_inverse = torch.load('taxa_inverse.pt')
        else:
            from collections import defaultdict
            ctt_i = {c: i for i, c in enumerate(class_to_taxa)}
            self.taxa_inverse = defaultdict(list)
            for k, vs in self.taxa.items():
                for v in vs:
                    self.taxa_inverse[v].append(ctt_i[k])
            torch.save(self.taxa_inverse, 'taxa_inverse.pt')

        # useful numbers:
        self.num_classes = len(classes)
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.noise_time = noise_time

        if self.enc.raster is not None:
            self.enc.raster = self.enc.raster.to(device)

    def __len__(self):
        return self.loc_feats.shape[0]

    def __getitem__(self, index):
        loc_feat  = self.loc_feats[index, :]
        loc       = self.locs[index, :]
        cell     = self.loc_cells[index]
        label = np.zeros(len(self.classes), dtype=np.float32)
        label[self.taxa_inverse[self.cells_inverse[cell]]] = 1

        if self.time_dim > 0:
            date = self.dates[index]
            # add noise
            if self.noise_time:
                noise_level = random.random()
                #noise = (2*random.random() - 1) * (0.5*(365 ** (noise_level - 1)))
                noise = (2*random.random() - 1) * (0.5*noise_level)
                loc_feat = torch.cat([loc_feat, self.enc_time.encode_fast([date.item()+noise,noise_level])])
            else:
                raise NotImplementedError()
                loc_feat = torch.cat([loc_feat, torch.tensor(self.enc_time.encode([2*date.item()-1], normalize=False))])
            return loc_feat, torch.cat([loc, date[None]]), label
        else:
            return loc_feat, loc, label


dataset_classes = {'inat': LocationDataset, 'iucn_inat': LocationIUCNDataset, 'iucn_uniform': UniformH3Dataset, 'multi_iucn_uniform': MultiUniformH3Dataset, 'multi_iucn_inat': MultiLocationIUCNDataset}


def get_train_data(params):
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    data_dir = paths['train']
    obs_file  = os.path.join(data_dir, params['obs_file'])
    taxa_file = os.path.join(data_dir, params['taxa_file'])
    taxa_file_snt = os.path.join(data_dir, 'taxa_subsets.json')

    taxa_of_interest = get_taxa_of_interest(params['species_set'], params['num_aux_species'], params['aux_species_seed'], params['taxa_file'], taxa_file_snt)

    locs, labels, _, dates, _, _ = load_inat_data(obs_file, taxa_of_interest)
    if params['zero_shot']:
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
            data = json.load(f)
        D = np.load(os.path.join(paths['snt'], 'snt_res_5.npy'), allow_pickle=True)
        D = D.item()
        taxa_snt = D['taxa'].tolist()
        taxa = [int(tt) for tt in data['taxa_presence'].keys()]
        taxa = list(set(taxa + taxa_snt))
        mask = labels != taxa[0]
        for i in range(1, len(taxa)):
            mask &= (labels != taxa[i])
        locs = locs[mask]
        dates = dates[mask]
        labels = labels[mask]
    unique_taxa, class_ids = np.unique(labels, return_inverse=True)
    class_to_taxa = unique_taxa.tolist()

    # load class names
    class_info_file = json.load(open(taxa_file, 'r'))
    class_names_file = [cc['latin_name'] for cc in class_info_file]
    taxa_ids_file = [cc['taxon_id'] for cc in class_info_file]
    classes = dict(zip(taxa_ids_file, class_names_file))

    subset = None
    if params['subset_cap_name'] is not None:
        if params['subset_cap_name'] == 'iucn':
            with open('paths.json', 'r') as f:
                paths = json.load(f)
            with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
                data = json.load(f)
            taxa = [int(tt) for tt in data['taxa_presence'].keys()]
            # get classes to eval
            subset = np.zeros((len(taxa),), dtype=int)
            for tt_id, tt in enumerate(taxa):
                class_of_interest = np.where(np.array(class_to_taxa) == tt)[0]
                if len(class_of_interest) != 0:
                    subset[tt_id] = class_of_interest
        else:
            raise NotImplementedError(f'Uknown subset name: {params["subset_cap_name"]}')

    idx_ss = get_idx_subsample_observations(labels, params['hard_cap_num_per_class'], params['hard_cap_seed'], subset, params['subset_cap_num_per_class'])

    locs = torch.from_numpy(np.array(locs)[idx_ss]) # convert to Tensor

    labels = torch.from_numpy(np.array(class_ids)[idx_ss])

    dates = 364/365*torch.from_numpy(np.array(dates)[idx_ss]) if params['input_time'] else None

    ds = dataset_classes[params['dataset']](locs, labels, classes, class_to_taxa, input_enc=params['input_enc'], device=params['device'], dates=dates, input_dim=params['input_dim'], time_dim=params['input_time_dim'], noise_time=params['noise_time'])

    return ds


def test_dataset():
    import setup
    from tqdm import tqdm
    train_params = {}
    train_params['species_set'] = 'all'
    train_params['hard_cap_num_per_class'] = -1
    train_params['num_aux_species'] = 0
    train_params['input_enc'] = 'sin_cos_env'
    train_params['input_dim'] = 8
    train_params['input_time'] = False
    train_params['input_time_dim'] = 0
    train_params['num_epochs'] = 50
    train_params['noise_time'] = False
    train_params['loss'] = 'an_full'
    train_params['dataset'] = 'iucn_inat'
    params = setup.get_default_params_train(train_params)
    train_dataset = get_train_data(params)
    train_dataset.viz_map(10070)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=getattr(train_dataset, 'collate_fn', None))
    for _ in tqdm(train_loader):
        pass


if __name__ == '__main__':
    test_dataset()