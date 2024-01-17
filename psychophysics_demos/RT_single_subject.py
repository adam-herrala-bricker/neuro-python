import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, sem

#prevent from truncating dataframes when printing to console
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

##################
##1. IMPORT DATA##
##################

#path to data
dataPath = './sample_data/RT/S1_example_experiment_2023-10-23_11h54.26.543.csv'

#read data into pandas df
rawData = pd.read_csv(dataPath)

##########################
##2. CLEAN UP DATA FRAME##
##########################

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

#remove first and last row (since those are not trials)
rawData = rawData.loc[1:44, :]

##################
##3. FILTER DATA##
##################

#condition variable for not a catch trial
notCatch = rawData.cond != 'catch'

#condition variable for response registered
response = rawData.resp == 'space'

#remove catch trials (notice how trial number is preserved)
filtData = rawData[notCatch & response]

#########################
##4. MEAN RTs AND STATS##
#########################

#condition variable for each experimental condition
condHigh = filtData.cond == 'high'
condLow = filtData.cond == 'low'

#variable for RTs for each condition
highRTs = filtData[condHigh].RT
lowRTs = filtData[condLow].RT

#plus RTs for all conditions
allRTs = filtData.RT

#print mean RTs
print('mean RT (high opacity): ', highRTs.mean())
print('mean RT (low opacity): ', lowRTs.mean())
print('mean RT (both conditions): ', allRTs.mean())

#t-test
statsResults = ttest_ind(highRTs, lowRTs, equal_var= False)
print(statsResults)

################
##5. Visualize##
################

#histogram
"""
plt.hist(allRTs, bins = 10)
plt.ylabel("Frequency")
plt.xlabel("RT (s)")
plt.title("Demo Histogram")
plt.show()
"""

#scatter plot
"""
plt.scatter(allRTs.index, allRTs)
plt.ylim(bottom=0, top = .6)
plt.ylabel("RT (s)")
plt.xlabel("Trial Number")
plt.title("Reaction Time Sequence")
plt.show()
"""

#bar graph
"""
fig, ax = plt.subplots()

bars = ax.bar([.5,1.5], [highRTs.mean(), lowRTs.mean()], width=.4)

ax.set_ylabel('RT (s)')
ax.set_ylim(top=.5)
ax.set_title('Mean Reaction Time by Stimulus Opacity')
ax.set_xticks([.5,1.5])
ax.set_xticklabels(["High Opacity", "Low Opacity"])

#standard error of the mean for error bars
highSEM = sem(highRTs)
lowSEM = sem(lowRTs)
ax.errorbar(x = [.5,1.5], 
            y = [highRTs.mean(), lowRTs.mean()], 
            yerr = [2*highSEM, 2*lowSEM], 
            capsize = 3, 
            ecolor = "black", 
            fmt = "none")


plt.show()
"""

#box plot

fig, ax = plt.subplots()

box = ax.boxplot([highRTs, lowRTs])

ax.set_ylabel('RT (s)')
ax.set_title('Reaction Time by Opacity')
ax.set_xticklabels(['High Opacity', 'Low Opacity'])

plt.show()