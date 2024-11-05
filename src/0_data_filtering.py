"""
Adapted from:
https://github.com/int-brain-lab/paper-reproducible-ephys/blob/develop/fig_PCA/fig_PCA_prepare_data.py
"""
import argparse
import pandas as pd
import numpy as np
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
from one.api import ONE
from reproducible_ephys_functions import combine_regions, BRAIN_REGIONS, get_insertions, save_data_path, save_dataset_info
from reproducible_ephys_processing import compute_psth, compute_new_label
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import compute_comparison_statistics
from fig_PCA.fig_PCA_load_data import load_dataframe, load_data
from iblutil.numerical import ismember

ba = AllenAtlas()

# Defaults parameters for psth computation
default_params = {'bin_size': 0.02,
                  'align_event': 'stim', # was 'move'
                  'event_epoch': [-0.5, .1], # was -0.5, 1
                  'base_event': 'stim', # was 'move'
                  'base_epoch': [-0.5, 0],
                  'norm': 'z_score',
                  'smoothing': None,
                  'split': 'rt',
                  'slide_kwargs': {'n_win': 5, 'causal': 1}}


def prepare_data(insertions, one, recompute=False, **kwargs):

    bin_size = kwargs.get('bin_size', default_params['bin_size'])
    align_event = kwargs.get('align_event', default_params['align_event'])
    event_epoch = kwargs.get('event_epoch', default_params['event_epoch'])
    base_event = kwargs.get('base_event', default_params['base_event'])
    base_epoch = kwargs.get('base_epoch', default_params['base_epoch'])
    norm = kwargs.get('norm', default_params['norm'])
    smoothing = kwargs.get('smoothing', default_params['smoothing'])
    split = kwargs.get('split', default_params['split'])
    slide_kwargs = kwargs.get('slide_kwargs', default_params['slide_kwargs'])

    params = {'bin_size': bin_size,
              'align_event': align_event,
              'event_epoch': event_epoch,
              'base_event': base_event,
              'base_epoch': base_epoch,
              'norm': norm,
              'smoothing': smoothing,
              'split': split,
              'slide_kwargs': slide_kwargs}

    if not recompute:
        data_exists = load_data(event=align_event, split='rt', norm=norm, smoothing=smoothing, exists_only=True)
        if data_exists:
            df = load_dataframe()
            pids = np.array([p['probe_insertion'] for p in insertions])
            isin, _ = ismember(pids, df['pid'].unique())
            if np.all(isin):
                print('Already computed data for set of insertions. Will load in data. To recompute set recompute=True')
                data = load_data()
                return df, data

    all_df = []
    for iIns, ins in enumerate(insertions):

        print(f'processing {iIns + 1}/{len(insertions)}')
        eid = ins['session']['id']
        probe = ins['probe_name']
        pid = ins['probe_insertion']

        data = {}

        # Load in spikesorting
        sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)

        clusters['rep_site_acronym'] = combine_regions(clusters['acronym'])
        # Find clusters that are in the repeated site brain regions and that have been labelled as good
        cluster_idx = np.sort(np.where(np.bitwise_and(np.isin(clusters['rep_site_acronym'], BRAIN_REGIONS),
                                                      clusters['label'] == 1))[0])
        data['cluster_ids'] = clusters['cluster_id'][cluster_idx]
        data['cluster_idx'] = cluster_idx

        # Find spikes that are from the clusterIDs
        spike_idx = np.isin(spikes['clusters'], data['cluster_idx'])
        if np.sum(spike_idx) == 0:
            continue

        # Load in trials data
        trials = one.load_object(eid, 'trials', collection='alf')
        # For this computation we use correct, non zero contrast trials
        trial_idx = np.bitwise_and(trials['feedbackType'] == 1,
                                   np.bitwise_or(trials['contrastLeft'] > 0, trials['contrastRight'] > 0))
        # Find nan trials
        nan_trials = np.bitwise_or(np.isnan(trials['stimOn_times']), np.isnan(trials['firstMovement_times']))

        # Find trials that are too long
        stim_diff = trials['feedback_times'] - trials['stimOn_times']
        rm_trials = stim_diff > 10
        # Remove these trials from trials object
        rm_trials = np.bitwise_or(rm_trials, nan_trials)

        eventMove = trials['firstMovement_times'][np.bitwise_and(trial_idx, ~rm_trials)]
        eventStim = trials['stimOn_times'][np.bitwise_and(trial_idx, ~rm_trials)]

        if split == 'choice':
            trial_l_idx = np.where(trials['choice'][np.bitwise_and(trial_idx, ~rm_trials)] == 1)[0]
            trial_r_idx = np.where(trials['choice'][np.bitwise_and(trial_idx, ~rm_trials)] == -1)[0]
        elif split == 'block':
            trial_l_idx = np.where(trials['probabilityLeft'][np.bitwise_and(trial_idx, ~rm_trials)] == 0.2)[0]
            trial_r_idx = np.where(trials['probabilityLeft'][np.bitwise_and(trial_idx, ~rm_trials)] == 0.8)[0]
        elif split == 'rt':
            rt = eventMove - eventStim
            trial_l_idx = np.where(rt < 0.16)[0]
            trial_r_idx = np.where(rt > 0.16)[0]

        if align_event == 'move':
            eventTimes = eventMove
        elif align_event == 'stim':
            eventTimes = eventStim

        if base_event == 'move':
            eventBase = eventMove
        elif base_event == 'stim':
            eventBase = eventStim

        # Compute firing rate
        fr_l, _, t = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_idx'],
                                  eventTimes[trial_l_idx], align_epoch=event_epoch, bin_size=bin_size,
                                  baseline_events=eventBase[trial_l_idx], base_epoch=base_epoch,
                                  smoothing=smoothing, norm=norm)
        fr_r, _, t = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_idx'],
                                  eventTimes[trial_r_idx], align_epoch=event_epoch, bin_size=bin_size,
                                  baseline_events=eventBase[trial_r_idx], base_epoch=base_epoch,
                                  smoothing=smoothing, norm=norm)
        frs = np.c_[fr_l, fr_r]

        # Find responsive neurons
        # Baseline firing rate
        intervals = np.c_[eventStim - 0.2, eventStim]
        counts, cluster_ids = get_spike_counts_in_bins(spikes.times[spike_idx], spikes.clusters[spike_idx], intervals)
        fr_base = counts / (intervals[:, 1] - intervals[:, 0])

        # Post-move firing rate
        intervals = np.c_[eventMove - 0.05, eventMove + 0.2]
        counts, cluster_ids = get_spike_counts_in_bins(spikes.times[spike_idx], spikes.clusters[spike_idx], intervals)
        fr_post_move = counts / (intervals[:, 1] - intervals[:, 0])

        data['responsive'], data['p_responsive'], _ = \
            compute_comparison_statistics(fr_base, fr_post_move, test='signrank')

        # Add other cluster information
        data['region'] = clusters['rep_site_acronym'][cluster_idx]
        data['x'] = clusters['x'][cluster_idx]
        data['y'] = clusters['y'][cluster_idx]
        data['z'] = clusters['z'][cluster_idx]

        df = pd.DataFrame.from_dict(data)
        df['eid'] = eid
        df['pid'] = pid
        df['subject'] = ins['session']['subject']
        df['probe'] = ins['probe_name']
        df['date'] = ins['session']['start_time'][:10]
        df['lab'] = ins['session']['lab']

        all_df.append(df)
        if iIns == 0:
            all_frs = frs
        else:
            all_frs = np.r_[all_frs, frs]

    concat_df = pd.concat(all_df, ignore_index=True)
    data = {'all_frs': all_frs, 'time': t, 'params': params}

    save_path = save_data_path(figure='fig_PCA')
    print(f'Saving data to {save_path}')
    concat_df.to_csv(save_path.joinpath('fig_PCA_dataframe.csv'))
    smoothing = smoothing or 'none'
    norm = norm or 'none'
    np.savez(save_path.joinpath(f'fig_PCA_data_event_{align_event}_split_{split}_smoothing_{smoothing}_norm_{norm}.npz'), **data)

    return concat_df, data


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
    args = ap.parse_args()

    one = ONE(
        base_url='https://openalyx.internationalbrainlab.org', 
        password='international', silent=True,
        cache_dir = args.base_path
    )
    one.record_loaded = True
    insertions = get_insertions(level=0, one=one)
    prepare_data(insertions, one=one, **default_params)
    save_dataset_info(one, figure='fig_PCA')

