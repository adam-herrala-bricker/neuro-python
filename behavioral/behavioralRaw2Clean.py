#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 10:20:51 2023

@author: ambric

This script takes the raw behavioral .csv files and outputs a single file with
every trial, in order, and only the information we need.

Automatically cleans raw files that haven't been cleaned yet.
"""
import pandas as pd

from os import listdir
from sys import exit

#list of block order for each participant
orderingPath = "./auxiliary_files/block_order.csv"
blockOrder = pd.read_csv(orderingPath)

#List of subjects in behavioral_data_raw
rawPath = "./behavioral_data_raw/"

rawSubjects = list(set([i.split("_")[0] for i in listdir(rawPath)]))

#List of subjects already in behavioral_data_clean

cleanPath = "./behavioral_data_clean/"

cleanSubjects = list(set([i.split(".")[0] for i in listdir(cleanPath)]))

#Only working with subjects we haven't already processed
subjects = [i for i in rawSubjects if i not in cleanSubjects and i[0] == "S"]

subjects.sort()

print("New subjects found:", subjects)

if subjects == []:
    print('no new subject data found')
    exit()


#list of subject files we need, in order of block presentation
targetFiles = [i + "_" + j + ".csv" for i in blockOrder.columns for j in blockOrder[i] if i in subjects]
print("Target files in order:", targetFiles)

#list of raw subject files in directory
subjectFiles = [i for i in listdir(rawPath) if i.split("_")[0] in subjects and i.split(".")[-1] == "csv"]

subjectFiles.sort()

#check if files needed = files found
if set(targetFiles) != set(subjectFiles):
    print("ERROR. Target files don't match files found.")
    print("Subject files found:", subjectFiles)
    exit()

#data frame to hold all the important data
data = pd.DataFrame(columns = ['S', 'block', 'trial_count', 
                               'trial_type', 'trial_code', 'trial_num',
                               'cor_resp', 'resp', 'RT'])

#will need this to pick out the correct response column
corRespMap = {'T-M' : 'T_hit', 'T-A' : 'T_hit', 
              'B-M' : 'B_hit', 'B-A' : 'B_hit',
              'K-M' : 'K_hit', 'K-A' : 'K_hit'}

#fill the data frame
for file in targetFiles:
    #import raw data
    newData = pd.read_csv(rawPath + file)

    #new column values
    newS = pd.concat([data['S'], newData['subject']], ignore_index = True)
    newBlock = pd.concat([data['block'], newData['session']], ignore_index = True)
    newCount = pd.concat([data['trial_count'], newData['trials.thisN'] + 1 ], ignore_index = True) #start from trial 1
    newType = pd.concat([data['trial_type'], newData['cond']], ignore_index = True)
    newCode = pd.concat([data['trial_code'], newData['trial_code']], ignore_index = True)
    newNum = pd.concat([data['trial_num'], newData['trial_num']], ignore_index = True)
    
    #figure out column for correct responses
    currentBlock = newData['session'][0]
    respCode = corRespMap[currentBlock]
    
    newCorResp = pd.concat([data['cor_resp'], newData[respCode]], ignore_index = True)
    newResp = pd.concat([data['resp'], newData['keyResp.keys']], ignore_index = True)
    newRT = pd.concat([data['RT'], newData['keyResp.rt']], ignore_index = True)
    
    
    data = pd.DataFrame(data = {'S' : newS, 'block' : newBlock, 
                                'trial_count' : newCount, 'trial_type' : newType,
                                'trial_code' : newCode, 'trial_num' : newNum,
                                'cor_resp' : newCorResp, 'resp' : newResp,
                                'RT' : newRT})
    
    
#drop rows that != trials
data = data.dropna(subset = ['trial_count'])


#export out each subject as new .csv file
for subject in subjects:
    subjectData = data[data.S == subject]
    subjectData = subjectData.reset_index()
    
    outPath = cleanPath + subject + ".csv"
    #correct number of trials
    if len(subjectData) == 468:
        subjectData.to_csv(outPath)
        print("New file saved:", outPath)
    else:
        print("subject", subject, "number of trials incorrect")
        print("number of trials found:", len(subjectData))
