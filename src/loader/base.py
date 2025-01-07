import os
import sys
import pickle
import logging

import uuid
import numpy as np
import multiprocessing
from tqdm import tqdm
from sklearn import preprocessing
from scipy.interpolate import interp1d
from iblutil.numerical import bincount2D

import torch

from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader

from from utils.registry import target_registry

logging.basicConfig(level=logging.INFO)

def to_tensor(x, device):
    return torch.tensor(x).to(device)

def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result

def bin_spike_count(
    times, 
    units, 
    start, 
    end, 
    length=None,
    binsize=0.01, 
    n_workers=1
):
    num_chunk = len(start)
    if length is None:
        length = int(min(end - start))
    num_bin = int(np.ceil(length / binsize))

    unit_index = np.unique(units)
    unit_count = len(unit_index)

    @globalize
    def count_spike_per_chunk(chunk):
        chunk_id, t_beg, t_end = chunk
        mask = (times >= t_beg) & (times < t_end)
        times_curr = times[mask]
        clust_curr = units[mask]

        if len(times_curr) == 0:
            spike_per_chunk = np.zeros((unit_count, num_bin))
            tbin_ids = np.arange(unit_count)
        else:
            spike_per_chunk, tbin_ids, unit_ids = bincount2D(
                times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end]
            )
            _, tbin_ids, _ = np.intersect1d(unit_index, unit_ids, return_indices=True)

        return spike_per_chunk[:,:num_bin], tbin_ids, chunk_id

    spike_count = np.zeros((num_chunk, unit_count, num_bin))

    chunks = list(zip(np.arange(num_chunk), start, end))
    
    if n_workers == 1:
        for chunk in chunks:
            res = count_spike_per_chunk(chunk)
            spike_count[res[-1], res[1], :] += res[0]
    else:
        with multiprocessing.Pool(processes=n_workers) as pool:
            with tqdm(total=num_chunk) as pbar:
                for res in pool.imap_unordered(count_spike_per_chunk, chunks):
                    pbar.update()
                    spike_count[res[-1], res[1], :] += res[0]
            pbar.close()

    return spike_count

def bin_target():
    pass


class BaseDataset(Dataset):
    def __init__(
        self, 
        session_id,
        target, 
        data_dir="./processed", 
        split="train", 
        region=None,
        device="cpu", 
    ):
        
        with open(f"{data_dir}/{session_id}.pkl", "rb") as f:
            session_dict = pickle.load(f)

        spike_count = bin_spike_count(
            session_dict["data"]["spikes"], 
            session_dict["data"]["units"]["unit_index"], 
            start=session_dict["splits"]["drifting_gratings"]["train"].T[0],
            end=session_dict["splits"]["drifting_gratings"]["train"].T[1],
            binsize=0.02,
        )

        target = bin_target(
            session_dict["data"][target]["timestamps"], 
            session_dict["data"][target][target], 
            # start=, 
            # end=, 
            binsize=0.02,
        )

        self.train_spike = get_binned_spikes(dataset['train'])
        self.train_behavior = np.array(dataset['train'][beh_name])
        self.neuron_regions = np.array(dataset['train']['cluster_regions'])[0]
        
        if split == 'val':
            try:
                # if 'val' exists, load pre-partitioned validation set
                self.spike_data = get_binned_spikes(dataset[split])
            except:
                # if not, partition training data into 'train' and 'val'
                tmp_dataset = dataset['train'].train_test_split(test_size=0.1, seed=seed)
                self.train_spike = get_binned_spikes(tmp_dataset['train'])
                self.spike_data = get_binned_spikes(tmp_dataset['test'])
        else:
            self.spike_data = get_binned_spikes(dataset[split])
            
        if region and region != 'all':
            neuron_idxs = np.argwhere(self.neuron_regions == region).flatten()
            self.spike_data = self.spike_data[:,:,neuron_idxs]
            self.regions = np.array([region] * len(self.spike_data))
        else:
            self.regions = np.array(['all'] * len(self.spike_data))

        self.sessions = np.array([eid] * len(self.spike_data))
        self.n_t_steps, self.n_units = self.spike_data.shape[1], self.spike_data.shape[2]

        self.behavior = np.array(dataset[split][beh_name])

        if target == 'clf':
            enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
            self.behavior = enc.fit_transform(self.behavior.reshape(-1, 1)).toarray()
        elif target == 'reg' or self.behavior.shape[1] == self.n_t_steps:
            self.scaler = preprocessing.StandardScaler().fit(self.train_behavior)
            self.behavior = self.scaler.transform(self.behavior) 

        if np.isnan(self.behavior).sum() != 0:
            self.behavior[np.isnan(self.behavior)] = np.nanmean(self.behavior)
            print(f'{beh_name} in session {eid} contains NaNs; interpolate with trial-average.')

        self.spike_data = to_tensor(self.spike_data, device).double()
        self.behavior = to_tensor(self.behavior, device).double()
  
    def __len__(self):
        return len(self.spike_data)

    def __getitem__(self, trial_idx):
        return (
            self.spike_data[trial_idx], self.behavior[trial_idx], self.regions[trial_idx], self.sessions[trial_idx]
        )

    
class SingleSessionDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config['dirs']['data_dir']
        self.eid = config['eid']
        self.beh_name = config['target']
        self.target = config['model']['target']
        self.region = config['region']
        self.device = config['training']['device']
        self.load_local = config['training']['load_local']
        self.batch_size = config['training']['batch_size']
        self.n_workers = config['data']['num_workers']

    def setup(self, stage=None):
        """Call this function to load and preprocess data."""
        self.train = BaseDataset(
            self.data_dir, self.eid, self.beh_name, self.target, 
            self.device, 'train', self.region, self.load_local
        )
        self.val = BaseDataset(
            self.data_dir, self.eid, self.beh_name, self.target, 
            self.device, 'val', self.region, self.load_local
        )
        self.test = BaseDataset(
            self.data_dir, self.eid, self.beh_name, self.target, 
            self.device, 'test', self.region, self.load_local
        )
        self.config.update({
            'n_units': self.train.n_units, 'n_t_steps': self.train.n_t_steps,
            'eid': self.eid, 'region': self.region
        })

    def train_dataloader(self):
        data_loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        return data_loader

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, drop_last=True)


class MultiSessionDataModule(LightningDataModule):
    def __init__(self, eids, configs):
        """Load and preprocess multi-session datasets.
            
        Args:
            eids: a list of session IDs.
            configs: a list of data configs for each session.
        """
        super().__init__()
        self.eids = eids
        self.configs = configs
        self.batch_size = configs[0]['training']['batch_size']

    def setup(self, stage=None):
        """Call this function to load and preprocess data."""
        self.train, self.val, self.test = [], [], []
        for config in self.configs:
            dm = SingleSessionDataModule(config)
            dm.setup()
            self.train.append(
                DataLoader(dm.train, batch_size = self.batch_size, shuffle=True)
            )
            self.val.append(
                DataLoader(dm.val, batch_size = self.batch_size, shuffle=False, drop_last=True)
            )
            self.test.append(
                DataLoader(dm.test, batch_size = self.batch_size, shuffle=False, drop_last=True)
            )

    def train_dataloader(self):
        data_loader = CombinedLoader(self.train, mode = "max_size_cycle")
        return data_loader

    def val_dataloader(self):
        data_loader = CombinedLoader(self.val)
        return data_loader

    def test_dataloader(self):
        data_loader = CombinedLoader(self.test)
        return data_loader


class MultiRegionDataModule(LightningDataModule):
    def __init__(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
        
    