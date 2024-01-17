#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:52:15 2023

@author: ambric

Takes clean behavioral files and calculates HRs and FARs

Later will add stats

NOTE: should also add a thing to look at HR for per and mem K seperately
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from os import listdir
from statistics import mean
from scipy.stats import sem

#makes a "ball plot"
def ballPlot(values, axis, color = 'red', y_cord = 0):
    axis.set_xlim([-.1,1.1])
    axis.set_xticks([0, .2, .4, .6, .8, 1])
    
    axis.set_yticks([])
    axis.set_ylim([-4.3,4.4])
    
    sign = 1 # for flipping sign
    values.sort()
    for i in range(0, len(values)):
        axis.scatter(values[i], sign*y_cord, color = color, edgecolor = 'black')
        
        #next value = repeat
        if i < len(values) - 1:
            if values[i + 1] == values[i]:
                y_cord += .1
                sign = sign * -1
            else:
                y_cord = 0


#Running the visualization bit?
viz = False


#list of clean data files
dataPath = "./behavioral_data_clean/"

figOutPath = "./behavioral_figs/"

dataFiles = [i for i in listdir(dataPath) if i.split(".")[-1] == "csv"]

dataFiles.sort()
print("Data files found:", dataFiles)

#list of subjects
subjects = [i.split(".")[0] for i in dataFiles]

#summary data frame for accuracy
summary = pd.DataFrame(index = subjects, 
                       columns = ['Overall_HR', 'Overall_FAR',
                                  'T_HR', 'B_HR', 'K_HR',
                                  'T_FAR', 'B_FAR', 'K_FAR'])

#summary data frame, fine-grained accuracy
summaryFine = pd.DataFrame(index = subjects,
                           columns = ['K-M_HR', 'K-A_HR', 'K-M_FAR', 'K-A_FAR',
                                      'K_PER_HR', 'K_MEM_HR',
                                      'B-M_HR', 'B-A_HR', 'B-M_FAR', 'B-A_FAR',
                                      'B_PER_HR', 'B_MEM_HR',
                                      'T-M_HR', 'T-A_HR', 'T-M_FAR', 'T-A_FAR',
                                      'T_PER_HR', 'T_MEM_HR'])


summary.index.rename('subject', inplace = True)

#for the response LMs
responseData = pd.DataFrame(columns = ['S', 'judgement', 'block', 'basis', 'HR', 'FAR'])
respInd = 0


#big loop over every subject
for i in range(0,len(subjects)):
    subjectFile = dataFiles[i]
    subject = subjects[i]
    
    subjectData = pd.read_csv(dataPath + subjectFile)
    
    #boolean conditions for indexing
    respYes = subjectData.resp == 'left'
    resNo = subjectData.resp == 'right'
    
    corRespYes = subjectData.cor_resp == 1
    corRespNo = subjectData.cor_resp == 0
    
    basisPer = subjectData.trial_type == 'per_K'
    basisMem = subjectData.trial_type == 'mem_K'
    
    trueBeliefOnly = (subjectData.trial_num < 41) | (subjectData.trial_num > 49)
    
    kBlock = (subjectData.block == 'K-M')|(subjectData.block == 'K-A')
    bBlock = (subjectData.block == 'B-M')|(subjectData.block == 'B-A')
    tBlock = (subjectData.block == 'T-M')|(subjectData.block == 'T-A')
    
    kBlock_M = subjectData.block == 'K-M'
    bBlock_M = subjectData.block == 'B-M'
    tBlock_M = subjectData.block == 'T-M'
    
    kBlock_A = subjectData.block == 'K-A'
    bBlock_A = subjectData.block == 'B-A'
    tBlock_A = subjectData.block == 'T-A'
    
    #hit and FA counts
    overallHits = len(subjectData[respYes & corRespYes])
    overallFAs = len(subjectData[respYes & corRespNo])
    
    kHits = len(subjectData[respYes & corRespYes & kBlock])
    kFARs = len(subjectData[respYes & corRespNo & kBlock])
    
    kHits_M = len(subjectData[respYes & corRespYes & kBlock_M])
    kFARs_M = len(subjectData[respYes & corRespNo & kBlock_M])
    
    kHits_A = len(subjectData[respYes & corRespYes & kBlock_A])
    kFARs_A = len(subjectData[respYes & corRespNo & kBlock_A])
    
    kHits_PER =len(subjectData[respYes & corRespYes & kBlock & basisPer])
    kHits_MEM = len(subjectData[respYes & corRespYes & kBlock & basisMem])
    
    bHits = len(subjectData[respYes & corRespYes & bBlock])
    bFARs = len(subjectData[respYes & corRespNo & bBlock])
    
    bHits_M = len(subjectData[respYes & corRespYes & bBlock_M])
    bFARs_M = len(subjectData[respYes & corRespNo & bBlock_M])
    
    bHits_A = len(subjectData[respYes & corRespYes & bBlock_A])
    bFARs_A = len(subjectData[respYes & corRespNo & bBlock_A])
    
    bHits_PER =len(subjectData[respYes & corRespYes & bBlock & basisPer & trueBeliefOnly]) #compare apples and apples w K and T
    bHits_MEM = len(subjectData[respYes & corRespYes & bBlock & basisMem & trueBeliefOnly])
    
    tHits = len(subjectData[respYes & corRespYes & tBlock])
    tFARs = len(subjectData[respYes & corRespNo & tBlock])
    
    tHits_M = len(subjectData[respYes & corRespYes & tBlock_M])
    tFARs_M = len(subjectData[respYes & corRespNo & tBlock_M])
    
    tHits_A = len(subjectData[respYes & corRespYes & tBlock_A])
    tFARs_A = len(subjectData[respYes & corRespNo & tBlock_A])
    
    tHits_PER =len(subjectData[respYes & corRespYes & tBlock & basisPer])
    tHits_MEM = len(subjectData[respYes & corRespYes & tBlock & basisMem])
    
    #target and nontarget counts
    overallTargets = len(subjectData[corRespYes])
    overallNontargets = len(subjectData[corRespNo])
    
    kTargets = len(subjectData[corRespYes & kBlock])
    kNontargets = len(subjectData[corRespNo & kBlock])
    
    kTargets_M = len(subjectData[corRespYes & kBlock_M])
    kNontargets_M = len(subjectData[corRespNo & kBlock_M])
    
    kTargets_A = len(subjectData[corRespYes & kBlock_A])
    kNontargets_A = len(subjectData[corRespNo & kBlock_A])
    
    kTargets_PER = len(subjectData[corRespYes & kBlock & basisPer])
    kTargets_MEM = len(subjectData[corRespYes & kBlock & basisMem])
    
    bTargets = len(subjectData[corRespYes & bBlock])
    bNontargets = len(subjectData[corRespNo & bBlock])
    
    bTargets_M = len(subjectData[corRespYes & bBlock_M])
    bNontargets_M = len(subjectData[corRespNo & bBlock_M])
    
    bTargets_A = len(subjectData[corRespYes & bBlock_A])
    bNontargets_A = len(subjectData[corRespNo & bBlock_A])
    
    bTargets_PER = len(subjectData[corRespYes & bBlock & basisPer & trueBeliefOnly])
    bTargets_MEM = len(subjectData[corRespYes & bBlock & basisMem & trueBeliefOnly])
    
    tTargets = len(subjectData[corRespYes & tBlock])
    tNontargets = len(subjectData[corRespNo & tBlock])
    
    tTargets_M = len(subjectData[corRespYes & tBlock_M])
    tNontargets_M = len(subjectData[corRespNo & tBlock_M])
    
    tTargets_A = len(subjectData[corRespYes & tBlock_A])
    tNontargets_A = len(subjectData[corRespNo & tBlock_A])
    
    tTargets_PER = len(subjectData[corRespYes & tBlock & basisPer])
    tTargets_MEM = len(subjectData[corRespYes & tBlock & basisMem])
    
    #add HR and FAR to summary dataframe
    summary.loc[subject, 'Overall_HR'] = overallHits/overallTargets
    summary.loc[subject, 'Overall_FAR'] = overallFAs/overallNontargets
    
    summary.loc[subject, 'K_HR'] = kHits/kTargets
    summary.loc[subject, 'K_FAR'] = kFARs/kNontargets
    
    summary.loc[subject, 'B_HR'] = bHits/bTargets
    summary.loc[subject, 'B_FAR'] = bFARs/bNontargets
    
    summary.loc[subject, 'T_HR'] = tHits/tTargets
    summary.loc[subject, 'T_FAR'] = tFARs/tNontargets
    
    #add additional HR and FAR to summaryFine dataframe
    summaryFine.loc[subject,'K-M_HR'] = kHits_M/kTargets_M
    summaryFine.loc[subject, 'K-A_HR'] = kHits_A/kTargets_A
    summaryFine.loc[subject, 'K-M_FAR'] = kFARs_M/kNontargets_M
    summaryFine.loc[subject, 'K-A_FAR'] = kFARs_A/kNontargets_A
   
    summaryFine.loc[subject, 'K_PER_HR'] = kHits_PER/kTargets_PER
    summaryFine.loc[subject, 'K_MEM_HR'] = kHits_MEM/kTargets_MEM
    
    summaryFine.loc[subject,'B-M_HR'] = bHits_M/bTargets_M
    summaryFine.loc[subject, 'B-A_HR'] = bHits_A/bTargets_A
    summaryFine.loc[subject, 'B-M_FAR'] = bFARs_M/bNontargets_M
    summaryFine.loc[subject, 'B-A_FAR'] = bFARs_A/bNontargets_A
    
    summaryFine.loc[subject, 'B_PER_HR'] = bHits_PER/bTargets_PER
    summaryFine.loc[subject, 'B_MEM_HR'] = bHits_MEM/bTargets_MEM
    
    summaryFine.loc[subject,'T-M_HR'] = tHits_M/tTargets_M
    summaryFine.loc[subject, 'T-A_HR'] = tHits_A/tTargets_A
    summaryFine.loc[subject, 'T-M_FAR'] = tFARs_M/tNontargets_M
    summaryFine.loc[subject, 'T-A_FAR'] = tFARs_A/tNontargets_A
    
    summaryFine.loc[subject, 'T_PER_HR'] = tHits_PER/tTargets_PER
    summaryFine.loc[subject, 'T_MEM_HR'] = tHits_MEM/tTargets_MEM
    
    #for the DF used in the LM
    for j in ['K', 'B', 'T']:
        for b in ['M', 'A']:
            for basis in ['per', 'mem']:
                responseData.loc[respInd, 'S'] = subject
                responseData.loc[respInd, 'judgement'] = j
                responseData.loc[respInd, 'block'] = b
                responseData.loc[respInd, 'basis'] = basis
                
                responseData.loc[respInd, 'HR'] = len(subjectData[(subjectData.block == j + '-' + b) & (subjectData.trial_type == basis + '_K') & respYes & corRespYes]) / len(subjectData[(subjectData.block == j + '-' + b) & (subjectData.trial_type == basis + '_K') & corRespYes]) 
                responseData.loc[respInd, 'FAR'] = len(subjectData[(subjectData.block == j + '-' + b) & respYes & corRespNo]) / len(subjectData[(subjectData.block == j + '-' + b) & corRespNo]) 
                
                
                respInd += 1
 


#####################
##SUBJECT EXCLUSION##
#####################

#set criteria
minHR = .75
maxFAR = .25

#counters for different bases for exclusion
highMiss = 0
highFA = 0
highBoth = 0

#list of excluded participants
excluded_A = []

for subject in subjects:
    #check HR
    if summaryFine.loc[subject, 'T-A_HR'] < minHR or summaryFine.loc[subject, 'B-A_HR'] < minHR:
        excluded_A.append(subject)
        
    elif summaryFine.loc[subject, 'T-A_FAR'] > maxFAR or summaryFine.loc[subject, 'B-A_FAR'] > maxFAR:
        excluded_A.append(subject)
        
    
    #see what they're in for
    if (summaryFine.loc[subject, 'T-A_HR'] < minHR or summaryFine.loc[subject, 'B-A_HR'] < minHR) and not (summaryFine.loc[subject, 'T-A_FAR'] > maxFAR or summaryFine.loc[subject, 'B-A_FAR'] > maxFAR):
        highMiss += 1
        
    elif (summaryFine.loc[subject, 'T-A_FAR'] > maxFAR or summaryFine.loc[subject, 'B-A_FAR'] > maxFAR) and not (summaryFine.loc[subject, 'T-A_HR'] < minHR or summaryFine.loc[subject, 'B-A_HR'] < minHR):
        highFA += 1
    
    elif (summaryFine.loc[subject, 'T-A_HR'] < minHR or summaryFine.loc[subject, 'B-A_HR'] < minHR) and (summaryFine.loc[subject, 'T-A_FAR'] > maxFAR or summaryFine.loc[subject, 'B-A_FAR'] > maxFAR):
        highBoth += 1

print('number of subjects excluded (A):', len(excluded_A))
print('excluded subjects (A):', excluded_A)

######################
##DATA FRAME FOR RTs##
######################

#data frame for every trial w RT data
allRTs = pd.DataFrame(columns = ['S', 'block', 'judgement', 'basis', 'RT'])

#data frame for every trial for logit/response analysis
allResps = pd.DataFrame(columns = ['S', 'block', 'judgement', 'basis', 'resp'])

#RT means for every subject over every condition (used for sem for error bars)
subjectRTs = pd.DataFrame(columns = ['T-M', 'T-A', 'B-M', 'B-A', 'K-M', 'K-A'])

indexRT = 0 #for keeping track of the index in the allRTs data frame
indexResp = 0 #same for allResps

for i in range(0,len(subjects)):
    subjectFile = dataFiles[i]
    subject = subjects[i]
    
    #first check to make sure this S in included
    if subject not in excluded_A:
    
        subjectData = pd.read_csv(dataPath + subjectFile)
        
        #only trials where correct resp = yes
        subjectData_1 = subjectData[subjectData.cor_resp == 1]
        subjectData_1 = subjectData_1.reset_index()
        
        #all responses --> logit
        for j in range(0,len(subjectData_1)):
            allResps.loc[indexResp, 'S'] = subjectData_1.loc[j, 'S']
            allResps.loc[indexResp, 'block'] = subjectData_1.loc[j, 'block'][-1] # just the perspective
            allResps.loc[indexResp, 'judgement'] = subjectData_1.loc[j, 'block'][0] # just the judgement
            allResps.loc[indexResp, 'basis'] = subjectData_1.loc[j, 'trial_type'][0:3]
            
            if subjectData_1.loc[j, 'resp'] == "left":
                allResps.loc[indexResp, 'resp'] = 1
                
            elif subjectData_1.loc[j, 'resp'] == "right":
                allResps.loc[indexResp, 'resp'] = 0
                
            else:
                allResps.loc[indexResp, 'resp'] = None
            
            indexResp += 1
    
        #just correct attributions
        subjectData = subjectData[(subjectData.resp == "left") & (subjectData.cor_resp == 1)]
        subjectData = subjectData.reset_index()
        
        #subject means to get the sem
        for condition in subjectRTs:
            subjectRTs.loc[i, condition] = mean(subjectData[subjectData.block == condition].RT)
        
       
        #individual entries for the LM
        for j in range(0, len(subjectData)):
            
            allRTs.loc[indexRT, 'S'] = subjectData.loc[j, 'S']
            allRTs.loc[indexRT, 'block'] = subjectData.loc[j, 'block'][-1] # just the perspective
            allRTs.loc[indexRT, 'judgement'] = subjectData.loc[j, 'block'][0] # just the judgement
            allRTs.loc[indexRT, 'basis'] = subjectData.loc[j, 'trial_type'][0:3]
            allRTs.loc[indexRT, 'RT'] = subjectData.loc[j, 'RT']
            
            indexRT += 1


#make RT column numeric
allRTs["RT"] = pd.to_numeric(allRTs["RT"])



#########
##STATS##
#########


#responses (all)
print("K mem > .75:", len(summaryFine[summaryFine['K_MEM_HR'] > .75]))
print("K mem < .25:", len(summaryFine[summaryFine['K_MEM_HR'] < .25]))
print("K mem = 0:", len(summaryFine[summaryFine['K_MEM_HR'] == 0]))

#responses (post-exclusion)
summaryFine_inc = summaryFine[~summaryFine.index.isin(excluded_A)]
print("K mem > .75 (post-excluded):", len(summaryFine_inc[summaryFine_inc['K_MEM_HR'] > .75]))
print("K mem < .25 (post-excluded):", len(summaryFine_inc[summaryFine_inc['K_MEM_HR'] < .25]))
print("K mem = 0 (post-excluded):", len(summaryFine_inc[summaryFine_inc['K_MEM_HR'] == 0]))

#make HR and FAR columns numeric
responseData["HR"] = pd.to_numeric(responseData["HR"])
responseData["FAR"] = pd.to_numeric(responseData["FAR"])

#only included participants
responseData = responseData[~responseData.S.isin(excluded_A)]


##LM for HRs
model_HR = smf.mixedlm("HR ~ judgement * C(block, Treatment(reference = 'M')) * C(basis, Treatment(reference = 'per'))", 
                       data = responseData, groups = responseData["S"],
                       missing = "drop")

mdf = model_HR.fit()
print("HR MODEL:")
print(mdf.summary())






##LM for FARs
#there is no 'basis' for a trial with where the correct response is no K/B,
#so the FARs ended up getting dulicated when making the DF above
#this corrects that duplication so each appears only once for the LM
dataFAR = responseData[responseData.basis == 'per']

model_FAR = smf.mixedlm("FAR ~ C(judgement, Treatment(reference = 'B')) * C(block, Treatment(reference = 'M'))",
                        data = dataFAR, groups = dataFAR["S"],
                        missing = "drop")

mdf = model_FAR.fit()
print("FAR MODEL:")
print(mdf.summary())



"""
##logit for responses

#need responses in numeric (0 and 1)
allResps["resp"] = pd.to_numeric(allResps["resp"])

#then also this?
allResps.index = allResps["S"]

#the model
logit_hits = smf.logit("resp ~ judgement * block * basis",
                       data = allResps, subset = allResps["S"],
                       missing = "drop")

mdf = logit_hits.fit()
print("Hits model:")
print(mdf.summary())
"""







#summary of RT

summaryRTs = pd.DataFrame(index = ['per', 'mem'], columns = ['T-M', 'T-A', 'B-M', 'B-A', 'K-M', 'K-A'])

judgeT = allRTs.judgement == "T"
judgeB = allRTs.judgement == "B"
judgeK = allRTs.judgement == "K"

blockM = allRTs.block == "M"
blockA = allRTs.block == "A"

basisPer = allRTs.basis == "per"
basisMem = allRTs.basis == "mem"


meanT_M_per = mean(allRTs[judgeT & blockM & basisPer].RT)
meanT_M_mem = mean(allRTs[judgeT & blockM & basisMem].RT)

meanT_A_per = mean(allRTs[judgeT & blockA & basisPer].RT)
meanT_A_mem = mean(allRTs[judgeT & blockA & basisMem].RT)

meanB_M_per = mean(allRTs[judgeB & blockM & basisPer].RT)
meanB_M_mem = mean(allRTs[judgeB & blockM & basisMem].RT)

meanB_A_per = mean(allRTs[judgeB & blockA & basisPer].RT)
meanB_A_mem = mean(allRTs[judgeB & blockA & basisMem].RT)

meanK_M_per = mean(allRTs[judgeK & blockM & basisPer].RT)
meanK_M_mem = mean(allRTs[judgeK & blockM & basisMem].RT)

meanK_A_per = mean(allRTs[judgeK & blockA & basisPer].RT)
meanK_A_mem = mean(allRTs[judgeK & blockA & basisMem].RT)

meanT_M = mean(allRTs[judgeT & blockM].RT)
meanT_A = mean(allRTs[judgeT & blockA].RT)

meanB_M = mean(allRTs[judgeB & blockM].RT)
meanB_A = mean(allRTs[judgeB & blockA].RT)

meanK_M = mean(allRTs[judgeK & blockM].RT)
meanK_A = mean(allRTs[judgeK & blockA].RT)



summaryRTs.loc['per','T-M'] = meanT_M_per
summaryRTs.loc['per','T-A'] = meanT_A_per

summaryRTs.loc['per','B-M'] = meanB_M_per
summaryRTs.loc['per','B-A'] = meanB_A_per

summaryRTs.loc['per','K-M'] = meanK_M_per
summaryRTs.loc['per','K-A'] = meanK_A_per

summaryRTs.loc['mem','T-M'] = meanT_M_mem
summaryRTs.loc['mem','T-A'] = meanT_A_mem

summaryRTs.loc['mem','B-M'] = meanB_M_mem
summaryRTs.loc['mem','B-A'] = meanB_A_mem

summaryRTs.loc['mem','K-M'] = meanK_M_mem
summaryRTs.loc['mem','K-A'] = meanK_A_mem

"""
#mixed model
model_RT = smf.mixedlm("RT ~ C(judgement, Treatment(reference = 'K')) * C(block, Treatment(reference = 'M')) + basis", 
                       data = allRTs, groups = allRTs["S"],
                       missing = "drop")

mdf = model_RT.fit()
print("REACTION TIMES MODEL:")
print(mdf.summary())
"""


###############
##PLOT THINGS##
###############



if viz:
    
    ##RT bars
    positions = [.2,.3,.5,.6, .8, .9]
    means = [meanT_M, meanT_A, meanK_M, meanK_A, meanB_M, meanB_A]

    fig, ax = plt.subplots(layout = 'tight')

    bars = ax.bar(positions, means, width=.09, color = "slategrey")

    ax.set_title('Mean Reaction Time by Condition', fontsize = 16)

    ax.set_xticks(positions)
    ax.set_xticklabels(["T-M", "T-A", "K-M", "K-A", "B-M", "B-A"])

    ax.set_ylabel('RT (s)')
    ax.set_ylim(top=.75, bottom=.3)

    #error bars

    SEMs = [2*sem(i) for i in [subjectRTs[condition] for condition in subjectRTs]]
    ax.errorbar(x = positions, y = means, yerr = SEMs, capsize = 3, ecolor = "black", fmt = "none")


    #plt.show()

    #plt.savefig(figOutPath + 'RTs', dpi = 800)
    
    
    ##density of K_per vs. K_mem
    fig, (ax1, ax2) = plt.subplots(2,1, figsize = (6,6), layout = 'tight')
    
    ballPlot(list(summaryFine["K_PER_HR"]), ax1, color = 'deeppink')
    ballPlot(list(summaryFine["K_MEM_HR"]), ax2, color = 'lightsteelblue')
    
    ax1.set_title('Knowledge from Perception', fontsize = 16)
    ax2.set_title('Knowledge from Memory', fontsize = 16)
    
    ax2.set_xlabel('attribution rate')
    
    plt.show()
    
    plt.savefig(figOutPath + 'K_per-K_mem.png', dpi = 800)
    
    ##density of B_per vs. B_mem
    fig, (ax1, ax2) = plt.subplots(2,1, figsize = (6,6), layout = 'tight')
    
    ballPlot(list(summaryFine["B_PER_HR"]), ax1, color = 'deeppink')
    ballPlot(list(summaryFine["B_MEM_HR"]), ax2, color = 'lightsteelblue')
    
    ax1.set_title('(True) Belief from Perception', fontsize = 16)
    ax2.set_title('(True) Belief from Memory', fontsize = 16)
    
    ax2.set_xlabel('attribution rate')
    
    plt.show()
    
    plt.savefig(figOutPath + 'B_per-B_mem.png', dpi = 800)
    
    ##density of B_per vs. B_mem
    fig, (ax1, ax2) = plt.subplots(2,1, figsize = (6,6), layout = 'tight')
    
    ballPlot(list(summaryFine["T_PER_HR"]), ax1, color = 'deeppink')
    ballPlot(list(summaryFine["T_MEM_HR"]), ax2, color = 'lightsteelblue')
    
    ax1.set_title('Location from Perception', fontsize = 16)
    ax2.set_title('Location from Memory', fontsize = 16)
    
    ax2.set_xlabel('hit rate')
    
    plt.show()
    
    plt.savefig(figOutPath + 'T_per-T_mem.png', dpi = 800)
    
        
