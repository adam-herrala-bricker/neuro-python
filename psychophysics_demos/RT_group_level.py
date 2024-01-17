import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, sem
from os import listdir

#prevent from truncating dataframes when printing to console
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#######################
##1. SPECIFY THE DATA##
#######################

#path to data
dataPath = './sample_data/RT/'

#all files in the given directory
fileList = listdir(dataPath)

#only .csv files
dataFiles = [file for file in fileList if file.split('.')[-1] == 'csv']

dataFiles.sort() #sort by subject code in file name

######################################
##2. DATAFRAMES FOR GROUP-LEVEL DATA##
######################################

#dataframe for subject means for each condition
#(used for t-test, bar graph, and box plot)
groupDataMeans = pd.DataFrame(columns = ['subj', 'cond', 'RT'])

#dataframe for EVERY TRIAL for every subject
#(used for histogram and scatter plot; could also use for a linear mixed model)
groupDataAll = pd.DataFrame(columns= ['subj', 'cond', 'resp', 'RT'])


#############################
##3. LOOP OVER EACH SUBJECT##
#############################

#for keeping track of the next row in groupData
rowIndex = 0

for file in dataFiles:
    ##3A. read data into pandas df
    rawData = pd.read_csv(dataPath + file)

    ##3B. clean up the dataframe

    #select only the columns we need
    rawData = pd.DataFrame(rawData, columns = ['participant', 'condition', 'key_resp.keys', 'key_resp.rt'])

    #rename those columns to more user-friendly labels
    newLabels = {
        'participant': 'subj', 
        'condition': 'cond', 
        'key_resp.keys': 'resp',
        'key_resp.rt': 'RT'
        }

    rawData = rawData.rename(columns = newLabels)

    #variable for the name of this subject (will use later)
    thisSubject = rawData.loc[1, 'subj']

    #remove first and last row (since those are not trials)
    rawData = rawData.loc[1:44, :]

    #quality control: verify that we have the expected number of trials
    expectedTrialCount = 44
    trialCount = len(rawData)

    if trialCount != expectedTrialCount:
        print('unexpected number of trials for subject', thisSubject)
    
    ##3C. filter the data

    #condition variable for not a catch trial
    notCatch = rawData.cond != 'catch'

    #condition variable for response registered
    response = rawData.resp == 'space'

    #remove catch trials
    filtData = rawData[notCatch & response]

    ##3D. add data from every trial to groupDataAll
    groupDataAll = pd.concat([groupDataAll, filtData])

    ##3E. calculate mean RTs

    #condition variable for each experimental condition
    condHigh = filtData.cond == 'high'
    condLow = filtData.cond == 'low'

    #mean RT for each condition
    highMeanRT = filtData[condHigh].RT.mean()
    lowMeanRT = filtData[condLow].RT.mean()

    ##3F. add to groupData df: [subject, condition, mean RT]

    #high condition
    groupDataMeans.loc[rowIndex] = [thisSubject, 'high', highMeanRT]
    rowIndex += 1

    #low condition
    groupDataMeans.loc[rowIndex] = [thisSubject, 'low', lowMeanRT]
    rowIndex += 1

##3G. check that everything looks right
print(groupDataMeans)
print(groupDataAll)

#########################
##4. MEAN RTs AND STATS##
#########################

#condition variables for each experimental condition
condHighGroup = groupDataMeans.cond == 'high'
condLowGroup = groupDataMeans.cond == 'low'

#subject mean RTs for each condition
highRTsGroup = groupDataMeans[condHighGroup].RT
lowRTsGroup = groupDataMeans[condLowGroup].RT

#plus both conditions
allRTsGroup = groupDataMeans.RT

#print group means
print('group mean RT (high opacity): ', highRTsGroup.mean())
print('group mean RT (low opacity): ', lowRTsGroup.mean())
print('group mean RT (both conditions): ', allRTsGroup.mean())

#t-test
statsResults = ttest_ind(highRTsGroup, lowRTsGroup, equal_var= False)
print(statsResults)


################
##5. Visualize##
################

#histogram
"""
plt.hist(groupDataAll.RT, bins = 10)
plt.ylabel("Frequency")
plt.xlabel("RT (s)")
plt.title("Demo Histogram")
plt.show()
"""


#scatter plot
"""
plt.scatter(groupDataAll.index, groupDataAll.RT)
plt.ylim(bottom=0, top = .6)
plt.ylabel("RT (s)")
plt.xlabel("Trial Number")
plt.title("Reaction Time Sequence")
plt.show()
"""


#bar graph
"""
fig, ax = plt.subplots()

bars = ax.bar([.5,1.5], [highRTsGroup.mean(), lowRTsGroup.mean()], width=.4)

ax.set_ylabel('RT (s)')
ax.set_ylim(top=.5)
ax.set_title('Mean Reaction Time by Stimulus Opacity')
ax.set_xticks([.5,1.5])
ax.set_xticklabels(["High Opacity", "Low Opacity"])

#standard error of the mean for error bars
highSEM = sem(highRTsGroup)
lowSEM = sem(lowRTsGroup)
ax.errorbar(x = [.5,1.5], 
            y = [highRTsGroup.mean(), lowRTsGroup.mean()], 
            yerr = [2*highSEM, 2*lowSEM], 
            capsize = 3, 
            ecolor = "black", 
            fmt = "none")

plt.show()
"""


#box plot
"""
fig, ax = plt.subplots()

box = ax.boxplot([highRTsGroup, lowRTsGroup])

ax.set_ylabel('RT (s)')
ax.set_title('Reaction Time by Opacity')
ax.set_xticklabels(['High Opacity', 'Low Opacity'])

plt.show()
"""
