import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from scipy.stats import norm, ttest_ind

#prevent from truncating dataframes when printing to console
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

####################
##0. SDT FUNCTIONS##
####################

#note that the typing isn't required, just a helpful reminder for what type the functions take arguments/return

#d' function
def dPrime(HR: float, FAR: float) -> float:
    result = norm.ppf(HR) - norm.ppf(FAR)
    return result

#criterion function
def criterion(HR: float, FAR: float ) -> float:
    result = -.5*(norm.ppf(HR) + norm.ppf(FAR))
    return result

##################
##1. LOCATE DATA##
##################

#path to file folder + file name
dataPath = './sample_data/SDT/'
fileList = listdir(dataPath)

dataFiles = [file for file in fileList if file.split('.')[-1] == 'csv']

dataFiles.sort() #sort by subject code in file name

######################################
##2. DATAFRAMES FOR GROUP-LEVEL DATA##
######################################

#SDT metrics for each subject/condition (used for stats + viz)
groupDataSDT = pd.DataFrame(columns = ['subj', 'cond', 'HR', 'FAR', 'dPrime', 'criterion'])

#response data for every trial for every subject (used for checking results via SDT Kamu app)
groupDataAll = pd.DataFrame(columns = ['subj', 'cond', 'side', 'resp'])

#############################
##3. LOOP OVER EACH SUBJECT##
#############################

rowIndex = 0
conditions = ['all', 'long', 'short']

for file in dataFiles:
    ##3A. read data into pandas df
    rawData = pd.read_csv(dataPath + file)

    ##3B. clean up data frame

    #select only the columns we need
    rawData = pd.DataFrame(rawData, columns = ['participant', 
                                            'durationCondition',
                                            'position', 
                                            'key_resp_main.keys'])

    #rename those columns to more user-friendly labels
    newLabels = {
        'participant': 'subj', 
        'durationCondition': 'cond',
        'position': 'side', 
        'key_resp_main.keys': 'resp'
        }

    rawData = rawData.rename(columns = newLabels)

    #remove rows that != trials
    isTrials = rawData.cond.notnull() #condition != NaN
    trialData = rawData[isTrials]

    #index -> trial number
    trialData = trialData.reset_index(drop=True) #reset index
    trialData.index += 1 #start from 1, not 0

    #subject name to use later (assumes every file is only for a SINGLE SUBJECT)
    thisSubject = trialData.loc[1, 'subj']

    #add trialData to groupDataAll
    groupDataAll = pd.concat([groupDataAll, trialData])

    ##3C. calculate SDT metrics
    #note: here the left grating will be defined as the "target"

    #condition variables
    isTarget = trialData.side == 'left'
    notTarget = trialData.side != 'left'

    respTarget = trialData.resp == 'left'

    condLong = trialData.cond == 'long'
    condShort = trialData.cond == 'short'

    #number of trials with target presented + not presented (all, long, and short conditions)
    targetTrials = {}
    notTargetTrials = {}

    targetTrials['all'] = len(trialData[isTarget])
    notTargetTrials['all'] = len(trialData[notTarget])

    targetTrials['long'] = len(trialData[isTarget & condLong])
    notTargetTrials['long'] = len(trialData[notTarget & condLong])

    targetTrials['short'] = len(trialData[isTarget & condShort])
    notTargetTrials['short'] = len(trialData[notTarget & condShort])


    #QC that found number of trials == the expected amount
    expectedTargets = {'all': 40, 'long': 20, 'short': 20}
    expectedNotTargets = {'all': 40, 'long': 20, 'short': 20}

    for cond in conditions:
        if targetTrials[cond] != expectedTargets[cond] or notTargetTrials[cond] != expectedNotTargets[cond]:
            print(thisSubject, '- Possible error with trial counts')
            print('Expected targets:', expectedTargets, 'Found targets:', targetTrials)
            print('Expected non-targets:', expectedNotTargets, 'Found non-targets:', notTargetTrials)
        

    #hit and FA count
    hitTrials = trialData[isTarget & respTarget]
    faTrials = trialData[notTarget & respTarget]

    hitCount = {}
    faCount = {}

    hitCount['all'] = len(hitTrials)
    faCount['all'] = len(faTrials)

    hitCount['long'] = len(hitTrials.loc[condLong]) #using .loc[] to suppress re-indexing warning
    faCount['long']= len(faTrials.loc[condLong])

    hitCount['short'] = len(hitTrials.loc[condShort])
    faCount['short'] = len(faTrials.loc[condShort])

    #adjust to avoid infinte d'/criterion
    for cond in conditions:
        #adjust hit count
        if hitCount[cond] == targetTrials[cond]:
            hitCount[cond] -= 1
            print(thisSubject, '- Hit count for condition =', cond, 'adjusted to', hitCount[cond])
        #adjust fa count
        if faCount[cond] == 0:
            faCount[cond] = 1
            print(thisSubject, '- FA count for condition =', cond, 'adjusted to 1')

    # compute hit and FA rates
    hitRate = {}
    faRate = {}

    for cond in conditions:
        hitRate[cond] = hitCount[cond]/targetTrials[cond]
        faRate[cond] = faCount[cond]/notTargetTrials[cond]

    ##3D. add d' and criterion to groupDataSDT [subject, condition, HR, FAR, d', criterion]

    #long condition
    groupDataSDT.loc[rowIndex] = [thisSubject, 
                                  'long',
                                  hitRate['long'],
                                  faRate['long'], 
                                  dPrime(hitRate['long'], faRate['long']),
                                  criterion(hitRate['long'], faRate['long'])]
    rowIndex += 1

    #short condition
    groupDataSDT.loc[rowIndex] = [thisSubject, 
                                  'short',
                                  hitRate['short'],
                                  faRate['short'], 
                                  dPrime(hitRate['short'], faRate['short']), 
                                  criterion(hitRate['short'], faRate['short'])]
    rowIndex += 1

#check that everything looks ok
print(groupDataSDT)
print('\n')

############
##4. STATS##
############

#variables for indexing
groupLongCond = groupDataSDT.cond == 'long'
groupShortCond = groupDataSDT.cond == 'short'

#select data
groupDataLong = groupDataSDT[groupLongCond]
groupDataShort = groupDataSDT[groupShortCond]

#print mean d's and criteria to console
print("Mean d' (long):", groupDataLong.dPrime.mean())
print("Mean d' (short):", groupDataShort.dPrime.mean())

print('Mean criterion (long):', groupDataLong.criterion.mean())
print('Mean criterion (short)', groupDataShort.criterion.mean())

print('\n')

#t-tests
dPrimeStats = ttest_ind(groupDataLong.dPrime, groupDataShort.dPrime , equal_var = False)
print("T-test (d'):", dPrimeStats)

criterionStats = ttest_ind(groupDataLong.criterion, groupDataShort.criterion, equal_var = False)
print("T-test (criterion):", criterionStats)

################
##5. VISUALIZE##
################

#note: could also use other graphs discussed for RTs (e.g. a bar graph between conditions for HR and FAR)

#box plot (d' in this example)
fig, ax = plt.subplots()

box = ax.boxplot([groupDataLong.dPrime, groupDataShort.dPrime])

ax.set_ylabel("d'")
ax.set_title("Sensitivity (d') by stimulus duration")
ax.set_xticklabels(['Long Duration', 'Short Duration'])

plt.show()

#################################
##6. BONUS: FORMAT FOR SDT KAMU##
#################################

#note: this step is just for formatting groupDataAll to use in the SDT Kamu app to check your pipeline
#you could do this before even starting step 3; all that's needed is groupDataAll from step 2

#new data frame with same data as groupDataAll
appData = groupDataAll.copy() #need to copy(), otherwise new name will still refer to old groupDataAll object

#replace values
appData = appData.replace({'left': 1, 'right': 0})

#in case you need to change the column type to int
appData= appData.astype({'side': int, 'resp': int})

#save as new .csv file
pathOut = './exported_data/'
fileOut = 'export1.csv'

appData.to_csv(pathOut + fileOut, sep=';')
print('Data exported to:', pathOut + fileOut)