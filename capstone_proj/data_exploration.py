import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy

## Data exploration
os.chdir("capstone_proj/csv_files")
""" In this section, we will take an in-depth view of the dataset. We will see
what we have available and whether we can make some initial predictions"""
s0001 = pd.read_csv("csv_files/2000_2001.csv")
s0001.head()
colsToKeep = list(["Div","Date","HomeTeam","AwayTeam","FTHG","HG","FTAG","AG","FTR","Res","HTHG","HTAG","HTR","Attendance","Referee","HS","AS","HST","AST","HHW","AHW","HC","AC","HF","AF","HFKC","AFKC","HO","AO","HY","AY","HR","AR","HBP","ABP"])
allCols = s0001.columns
colsToDrop = set(allCols) - set(colsToKeep)
s0001 = s0001.drop(labels=colsToDrop,axis=1)
print(s0001.head())
### Data ###
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(s0001.head())
### Home side advantage ####
describe = (s0001.describe())

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print((s0001.describe()))
### PLOT BAR GRAPH #########
plt.figure()
pd.DataFrame.to_latex(s0001.describe())

# width of the bars
barWidth = 0.3

# Choose the height of the blue bars
bars1 = np.array(describe[1:2][(['FTHG','HTHG','HS','HST','HHW','HC','HF','HO','HY','HR','HBP'])]).flatten()

# Choose the height of the cyan bars
bars2 = np.array(describe[1:2][(['FTAG','HTAG','AS','AST','AHW','AC','AF','AO','AY','AR','ABP'])]).flatten()

# Choose the height of the error bars (bars1)
yer1 = np.array(describe[2:3][(['FTHG','HTHG','HS','HST','HHW','HC','HF','HO','HY','HR','HBP'])]).flatten()

# Choose the height of the error bars (bars2)
yer2 = np.array(describe[2:3][(['FTAG','HTAG','AS','AST','AHW','AC','AF','AO','AY','AR','ABP'])]).flatten()

# The x position of bars
r1 = np.arange(len(bars1.flatten()))
r2 = [x + barWidth for x in r1]

# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', yerr=yer1, capsize=7, label='Home_Side')

# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = 'red', edgecolor = 'black', yerr=yer2, capsize=7, label='Away_Side')

# general layout
plt.xticks([r + barWidth for r in range(len(bars1))], ['Goals Scored', 'Half-time Gaosl Scored', 'Shots','Shots on target','Hit Woodwork', 'Corners','Fouls','Offsides','Yellow Cards','Red Cards','Booking points'])
plt.ylabel('Average Value')
plt.ylim(0)
plt.legend()

# Show graphic
plt.show()
