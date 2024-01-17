import pandas as pd
from scipy.stats import norm

#prevent from truncating dataframes when printing to console
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

####################
##0. SDT FUNCTIONS##
####################

#d' function
def dPrime(HR, FAR):
    result = norm.ppf(HR) - norm.ppf(FAR)
    return result

#criterion function
def criterion(HR, FAR):
    result = -.5*(norm.ppf(HR) + norm.ppf(FAR))
    return result

##################
##1. IMPORT DATA##
##################

#path to file folder + file name
dataPath = './sample_data/SDT/'
fileName = 'S1_example_staircase_2023-11-06_14h47.09.923.csv'

#read file into pandas
rawData = pd.read_csv(dataPath + fileName)

##########################
##2. CLEAN UP DATA FRAME##
##########################

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

#check that everything looks good
print(trialData)

############################
##3. CALCULATE SDT METRICS##
############################

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

conditions = ['all', 'long', 'short'] #include all as a 'condition'
for cond in conditions:
    if targetTrials[cond] != expectedTargets[cond] or notTargetTrials[cond] != expectedNotTargets[cond]:
        print('Possible error with trial counts')
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
        print('Hit count for condition =', cond, 'adjusted to', hitCount[cond])
    #adjust fa count
    if faCount[cond] == 0:
        faCount[cond] = 1
        print('FA count for condition =', cond, 'adjusted to 1')

# compute hit and FA rates
hitRate = {}
faRate = {}

for cond in conditions:
    hitRate[cond] = hitCount[cond]/targetTrials[cond]
    faRate[cond] = faCount[cond]/notTargetTrials[cond]

#print HRs and FARs to console
for cond in conditions:
    print('HR', cond, '=', hitRate[cond])
    print('FAR', cond, '=', faRate[cond])
    print('\n')

#d' and criterion
for cond in conditions:
    print("d'", cond, '=', dPrime(hitRate[cond], faRate[cond]))
    print('criterion', cond, '=', criterion(hitRate[cond], faRate[cond]))
    print('\n')


#################################
##4. BONUS: FORMAT FOR SDT KAMU##
#################################

#note: this step is just for formatting trialData to use in the SDT Kamu app to check your pipeline
#you could do this before even starting step 3; all that's needed is trialData from step 2

#new data frame with same data as old trialData
appData = trialData.copy() #need to copy(), otherwise new name will still refer to old trialData object

#replace values
appData = appData.replace({'left': 1, 'right': 0})

#in case you need to change the column type to int
appData= appData.astype({'side': int, 'resp': int})

print(appData)

#save as new .csv file
pathOut = './exported_data/'
fileOut = 'S1.csv'

appData.to_csv(pathOut + fileOut, sep=';')