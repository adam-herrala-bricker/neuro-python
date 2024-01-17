#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 18:46:57 2023

@author: ambric

This takes raw EEG recording for a single subject, pre-processes it, and saves
it as a .fif file. Epoching is done in the next script.

This works best for pre-processing/QC checking as data is recorded. A more 
automated script would be better if trying to pre-process the entire data set 
at once.
"""
import mne

##############################
##STEP 0: INITIAL PARAMETERS##
##############################

#lets the script know whether we have EOG electodes
EOG = False

#EOG channel names
EOG_channels = ("LEYE_beside", "LEYE_below")

###########################
##STEP 1: IMPORT THE DATA##
###########################

##set file paths
subject = "S27"

#True if session split into multiple recordings
multipleRecordings = False
subjectAdditional = ["S40b"]

pathEEG = "../EEG_data_raw/"

fileEEG = pathEEG + subject + ".vhdr"

##import the EEG file
rawData = mne.io.read_raw_brainvision(fileEEG, preload = True, eog = EOG_channels)

#handle sessions with multiple recordings
if multipleRecordings:
    for s in subjectAdditional:
        pathAdditional = pathEEG + s + ".vhdr"
        rawDataAdditional = mne.io.read_raw_brainvision(pathAdditional, preload = True, eog = EOG_channels)
        rawData.append(rawDataAdditional)


#someone typed the name of this channel wrong in the lab computer
mne.channels.rename_channels(rawData.info, {"FPz" : "Fpz"})

#set montage (where channels are located)
rawData.set_montage("easycap-M1")

#can plot channel locations for QC
#rawData.plot_sensors(show_names = True)

#drop EOG channels if not used for this recording
if not EOG:
    rawData.drop_channels(EOG_channels)


###############################
##STEP 2: REPAIR BAD CHANNELS##
###############################

rawData.plot(block=True)

rawData.interpolate_bads()

#can verify that it worked
#rawData.plot(block=True)

########################################
##STEP 3: RE-REFERENCE TO CHANNEL MEAN##
########################################

rawData.set_eeg_reference(ref_channels='average')

#####################
##STEP 4: FILTERING##
#####################
"""
A note on filtering: The ICA documentation says it needs at least 1 Hz HP to
work properly, but there's evidence that that's too aggressive for ERPs.

The solution: 
    
    1. Filter at (1,40) for the ICA and (.1, 40) for the main data.
    2. Apply the ICA to the main data.

(A reviewer pointed this out on my previous paper.)
"""

dataICA = rawData.copy()

rawData.filter(.1,40)
dataICA.filter(1, 40)

#check that everything looks good
rawData.plot(block=True)

###############
##STEP 5: ICA##
###############

ica = mne.preprocessing.ICA(n_components=15, max_iter='auto')
ica.fit(dataICA)

#manual ICA if no EOG channels
if not EOG:
    ica.plot_components()

    ica.plot_sources(rawData, block = True)

ica.apply(rawData)

#verify that it worked
rawData.plot(block=True)

###########################
##SAVE PRE-PROCESSED DATA##
###########################

outPath = "../EEG_data_clean/"

rawData.save(outPath + subject + "_eeg.fif", overwrite = True)


