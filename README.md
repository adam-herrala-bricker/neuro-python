# Neuro Python

## About

This repository contains a few scripts written for analyzing behavioral and EEG data using pandas (behavioral) and MNE (EEG). It also contains demo scripts for the psychophysics course I taught to students in the human neuroscience master's program at the University of Turku.

## Psychophysics Demos

- Single-subject and group-level analysis of reaction time (RT) data between two conditions.
- Single-subject and group-level analysis of the d' and criterion metrics from signal detection theory (SDT).
- Demo data used for the scripts.
- Exported data that can be used with the [SDT Kamu app](https://github.com/adam-herrala-bricker/SDT-buddy).
- PDFs of the instructions for context.

>[!NOTE]
>Keep in mind that these demo scripts were written as examples for students, so they're over-commented, less compact than they could be, and littered with `print()` statements.


## Behavioral Data

- Preprocessing script to convert the multiple raw .csv files collected for each subject into a single, "clean" file that only contains the data necessary for analysis.
- Analysis script:
  - Subject rejection based on a priori accuracy requirements.
  - Statistical analysis of RTs and responses using linear mixed-effects models.
  - Visualization of RTs and responses.
- Sample raw and preprocessed data files.
- An auxiliary ordering file used for preprocessing.
- Examples of figures output for the full data set.

>[!NOTE]
>Because the full data set is quite large, only a few subjects have been included here. This is just for illustrative purposes.

## EEG Data

- Preprocessing script for BrainVision data
  - Bad channel interpolation.
  - Re-reference to mean of all channels.
  - Filtering.
  - Manual ICA to remove eye and muscle artifacts.
- Group-level time-frequency analysis script for .fif data:
  - Re-structure event markers and epoch data.
  - Compute mean time-frequency representations (TFRs) for each subject/condition.
  - Perform spatiotemporal cluster test on group-level TFRs (using threshold-free cluster enhancement).
  - Visualize significant spatiotemporal clusters.

>[!NOTE]
>Because raw EEG recordings are so large, I haven't included any sample data here.
