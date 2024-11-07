import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from one.api import ONE
from datasets import DatasetDict
from utils.ibl_data_utils import (
    prepare_data, 
    select_brain_regions, 
    list_brain_regions, 
    bin_spiking_data,
    bin_behaviors,
    align_spike_behavior
)
from utils.dataset_utils import create_dataset, upload_dataset

from reproducible_ephys_functions import filter_recordings
from fig_PCA.fig_PCA_load_data import load_dataframe

ap = argparse.ArgumentParser()
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--n_workers", type=int, default=1)
args = ap.parse_args()

SEED = 42

np.random.seed(SEED)

one = ONE(
    base_url='https://openalyx.internationalbrainlab.org', 
    password='international', silent=True,
    cache_dir = args.base_path
)

freeze_file = 'data/bwm_release.csv'
bwm_df = pd.read_csv(freeze_file, index_col=0)

concat_df = load_dataframe()
concat_df = filter_recordings(concat_df, min_regions=0)

# Trial setup
params = {
    'interval_len': 2, 'binsize': 0.05, 'single_region': False,
    'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)
}

beh_names = ['choice', 'reward', 'block', 'wheel-speed']

include_eids = np.unique(concat_df.eid)
bad_eids = [
    "38d95489-2e82-412a-8c1a-c5377b5f1555",
    "51e53aff-1d5d-4182-a684-aba783d50ae5"
]

print(f"Preprocess a total of {len(include_eids)} EIDs.")

num_neurons = []
for eid_idx, eid in enumerate(include_eids):

    if eid in bad_eids:
        continue

    print('==========================')
    print(f'Preprocess session {eid}:')

    # Load and preprocess data
    neural_dict, behave_dict, meta_data, trials_data = prepare_data(one, eid, bwm_df, params, n_workers=args.n_workers)
    regions, beryl_reg = list_brain_regions(neural_dict, **params)
    region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)
    binned_spikes, clusters_used_in_bins = bin_spiking_data(
        region_cluster_ids, neural_dict, trials_df=trials_data['trials_df'], n_workers=args.n_workers, **params
    )
    binned_behaviors, behavior_masks = bin_behaviors(
        one, eid, beh_names[3:], trials_df=trials_data['trials_df'], 
        allow_nans=True, n_workers=args.n_workers, **params
    )
    
    # Ensure neural and behavior data match for each trial
    aligned_binned_spikes, aligned_binned_behaviors = align_spike_behavior(
        binned_spikes, binned_behaviors, beh_names, trials_data['trials_mask']
    )
    print("spike data shape: ", aligned_binned_spikes.shape)
    num_neurons.append(aligned_binned_spikes.shape[-1])

    # Partition dataset (train: 0.7 val: 0.1 test: 0.2)
    max_num_trials = len(aligned_binned_spikes)
    trial_idxs = np.random.choice(np.arange(max_num_trials), max_num_trials, replace=False)
    train_idxs = trial_idxs[:int(0.7*max_num_trials)]
    val_idxs = trial_idxs[int(0.7*max_num_trials):int(0.8*max_num_trials)]
    test_idxs = trial_idxs[int(0.8*max_num_trials):]

    if len(train_idxs) == 0 or len(val_idxs) == 0 or len(test_idxs) == 0:
        print(f"Skip {eid} due to empty set.")
        continue

    train_beh, val_beh, test_beh = {}, {}, {}
    for beh in aligned_binned_behaviors.keys():
        train_beh.update({beh: aligned_binned_behaviors[beh][train_idxs]})
        val_beh.update({beh: aligned_binned_behaviors[beh][val_idxs]})
        test_beh.update({beh: aligned_binned_behaviors[beh][test_idxs]})
    
    train_dataset = create_dataset(
        aligned_binned_spikes[train_idxs], bwm_df, eid, params, 
        binned_behaviors=train_beh, meta_data=meta_data
    )
    val_dataset = create_dataset(
        aligned_binned_spikes[val_idxs], bwm_df, eid, params, 
        binned_behaviors=val_beh, meta_data=meta_data
    )
    test_dataset = create_dataset(
        aligned_binned_spikes[test_idxs], bwm_df, eid, params, 
        binned_behaviors=test_beh, meta_data=meta_data
    )

    # Create dataset
    partitioned_dataset = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset}
    )
    print(partitioned_dataset)

    # Cache dataset
    save_path = Path(args.base_path)/'cached_re_data'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    partitioned_dataset.save_to_disk(f'{save_path}/{eid}')

    print(f'Cached session {eid}.')
    print(f'Progress: {eid_idx+1} / {len(include_eids)} sessions cached.')

print(f"Min: {min(num_neurons)} max: {max(num_neurons)} mean: {sum(num_neurons)/len(num_neurons)} # of neurons.")
