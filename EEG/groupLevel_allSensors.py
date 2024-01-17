#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 09:34:36 2023

@author: ambric
"""

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import listdir
from mpl_toolkits.axes_grid1 import make_axes_locatable

#need this to handle event IDs
def eventMapper(markerString):
    #start of recording
    if markerString == 'New Segment/':
        eventID = 9999
    else:
        eventID = int(markerString.replace('Stimulus/S', ''))
        
    return eventID

#############
##LOAD DATA##
#############

#set file paths

pathEEG = "../EEG_data_clean/"
pathBehavioral = "../behavioral_data_clean/"

#get list of subjects
subjects = [i.split("_")[0] for i in listdir(pathEEG) if i.split(".")[-1 == ".fif"]]

#putting the TF powers in here
epochs_power_0 = []
epochs_power_1 = []
epochs_power_2 = []

#loop over subjects
for subject in subjects:

    fileEEG = pathEEG + subject + "_eeg.fif"
    fileBehavioral = pathBehavioral + subject + ".csv"
    
    #import EEG file
    rawData = mne.io.read_raw_fif(fileEEG, preload = True)
    
    #import the behavioral file
    behavioralData = pd.read_csv(fileBehavioral)
    
    
    #######################################
    ##STEP 6: APPLY DESIRED EVENT MARKERS##
    #######################################
    
    #markers for response routine
    respMarkers = (200, 201, 222)
    
    ##get events object. structure = [[starting sample, duration, maker], . . .]
    events, event_dict = mne.events_from_annotations(rawData, event_id = eventMapper)
    
    #literal mappingdifferent marker for each block
    markerMapping = {"T-M" : 300, "T-A" : 301, 
                     "B-M" : 310, "B-A" : 311, 
                     "K-M" : 320, "K-A" : 321
                     }
    
    #list of block for every trial (all 468)
    blockList = list(behavioralData.block)
    blockList.reverse() #to work with pop() method
    
    #list of all responses
    respList = list(behavioralData.resp)
    respList.reverse()
    
    #list of all correct responses
    corRespList = list(behavioralData.cor_resp)
    corRespList.reverse()
    
    #list of all trial types
    trialTypeList = list(behavioralData.trial_type)
    trialTypeList.reverse()
    
    
    #loop over every decision event (decision only)
    changeCounter = 0
    for i in range(1, len(events)): #event 0 = marker for recording start
        
        #decision frame = one before response frame
        if events[i][2] in respMarkers:
            block = blockList.pop()
            resp = respList.pop()
            corResp = corRespList.pop()
            trialType = trialTypeList.pop()
            
            #correct attribution only (perception)
            if resp == "left" and corResp == 1:
                #decision frame
                events[i-10][2] = markerMapping[block]
            
            changeCounter += 1
    
    #################    
    ##STEP 7: EPOCH##
    #################
    
    #simple mapping
    event_id = {'truth' : 300, 'belief' : 311, 'knowledge' : 322}
    
    epochs = mne.Epochs(rawData, events, event_id = markerMapping, 
                        tmin = -1, tmax = 7.8)
    
    reject_criteria = dict(eeg=200e-6)  #200 µV
    epochs.drop_bad(reject=reject_criteria)
    
    #########################
    ##STEP 8: TFR and Stats##
    #########################
    
    #power
    freqs = np.logspace(*np.log10([4, 35]), num=10)
    n_cycles = freqs / 2.  # different number of cycle per frequency
    
    #conditions you'll be contrasting
    contrast = ['B-A', 'T-A', 'K-A']
    
    #do the TFRs
    this_tfr_0 = mne.time_frequency.tfr_morlet(epochs[contrast[0]], freqs, n_cycles=n_cycles,
                     decim=3, average=True, return_itc=False)
    
    
    this_tfr_1 = mne.time_frequency.tfr_morlet(epochs[contrast[1]], freqs, n_cycles=n_cycles,
                     decim=3, average=True, return_itc=False)
    
    #baseline corrections
    baselineRange = (-1, 0)
    
    this_tfr_0.apply_baseline(mode='ratio', baseline = baselineRange)
    this_tfr_1.apply_baseline(mode='ratio', baseline = baselineRange)
    
    #add to main list
    epochs_power_0.append(this_tfr_0.data)
    epochs_power_1.append(this_tfr_1.data)
    
#get in right shape (f bands, samples, chanels)
epochs_power_0 = np.array([np.transpose(x, (1,2,0)) for x in epochs_power_0])
epochs_power_1 = np.array([np.transpose(x, (1,2,0)) for x in epochs_power_1])

#final stats object
X = [epochs_power_0, epochs_power_1]

#print data shape
print("subjects, f bands, samples, channels:", epochs_power_0.shape)

#Get channel adjacency
adjacency, ch_names = mne.channels.find_ch_adjacency(epochs.info, ch_type = "eeg")

#then this thing
tfr_adjacency = mne.stats.combine_adjacency(len(freqs), len(this_tfr_0.times), adjacency)

#permutation test
threshold_tfce = dict(start=0, step=0.2)
threshold_F = 10

cluster_stats = mne.stats.spatio_temporal_cluster_test(
    X, n_permutations=1000, threshold=threshold_tfce, tail=1, n_jobs=None,
    buffer_size=None, adjacency=tfr_adjacency)

#alpha value
p_accept = .1

F_obs, clusters, p_values, _ = cluster_stats

good_cluster_inds = np.where(p_values < p_accept)[0]

print("min p-value:", min(p_values))
print("significant p-values:", [p for p in p_values if p <= p_accept])


#vizualize
#taken from: https://mne.tools/stable/auto_tutorials/stats-sensor-space/75_cluster_ftest_spatiotemporal.html
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    freq_inds, time_inds, space_inds = clusters[clu_idx]
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)
    freq_inds = np.unique(freq_inds)

    # get topography for F stat
    f_map = F_obs[freq_inds].mean(axis=0)
    f_map = f_map[time_inds].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = epochs.times[time_inds]

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # plot average test statistic and mark significant sensors
    f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs.info, tmin=0)
    f_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo, cmap='Reds',
                          vlim=(np.min, np.max), show=False, colorbar=False,
                          mask_params=dict(markersize=10))
    image = ax_topo.images[0]

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

    # remove the title that would otherwise say "0.000 s"
    ax_topo.set_title("")

    # add new axis for spectrogram
    ax_spec = divider.append_axes('right', size='300%', pad=1.2)
    title = 'Cluster #{0}, {1} spectrogram'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += " (max over channels)"
    F_obs_plot = F_obs[..., ch_inds].max(axis=-1)
    F_obs_plot_sig = np.zeros(F_obs_plot.shape) * np.nan
    F_obs_plot_sig[tuple(np.meshgrid(freq_inds, time_inds))] = \
        F_obs_plot[tuple(np.meshgrid(freq_inds, time_inds))]

    for f_image, cmap in zip([F_obs_plot, F_obs_plot_sig], ['gray', 'autumn']):
        c = ax_spec.imshow(f_image, cmap=cmap, aspect='auto', origin='lower',
                           extent=[epochs.times[0], epochs.times[-1],
                                   freqs[0], freqs[-1]])
    ax_spec.set_xlabel('Time (s)')
    ax_spec.set_ylabel('Frequency (Hz)')
    ax_spec.set_title(title)

    # add another colorbar
    ax_colorbar2 = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(c, cax=ax_colorbar2)
    ax_colorbar2.set_ylabel('F-stat')

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    plt.show()
