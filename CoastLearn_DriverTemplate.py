#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:51:41 2024

@author: fmuir
"""

import os
import matplotlib.pyplot as plt
from Toolshed import Predictions

# Only use tensorflow in CPU mode
import tensorflow as tf
tf.config.set_visible_devices([],'GPU')
# %load_ext tensorboard
# import tensorboard

#%% Load Transect Data
# Name of site to save directory and files under
sitename = 'StAndrewsEastS2Full'
filepath = os.path.join(os.getcwd(), 'Data')

# Load in transect data with coastal change variables
TransectInterGDF, TransectInterGDFWater, TransectInterGDFTopo, TransectInterGDFWave = Predictions.LoadIntersections(filepath, sitename)

#%% Compile Transect Data
# Compile relevant coastal change metrics into one dataframe
CoastalDF = Predictions.CompileTransectData(TransectInterGDF, TransectInterGDFWater, TransectInterGDFTopo, TransectInterGDFWave)

#%% Interpolate
# Subset and interpolate timeseries to match up to same dates
# TransectIDs = [271]
TransectIDs = [1325]

for Tr in TransectIDs:
    TransectDF = Predictions.InterpVEWL(CoastalDF, Tr)
    
#%% Separate Training and Validation
TransectDFTrain = TransectDF.iloc[:271]
TransectDFTest = TransectDF.iloc[270:]

VarDFDayTest = Predictions.DailyInterp(TransectDFTest)

#%% Prepare Training Data
PredDict, VarDFDay = Predictions.PrepData(TransectDFTrain, 
                                          MLabels=['optimised'], 
                                          TestSizes=[0.2], 
                                          TSteps=[10])

#%% Compile the Recurrent Neural Network 
# with desired number of epochs and batch size (per model run)
PredDict = Predictions.CompileRNN(PredDict, 
                                  epochNums=[150], 
                                  batchSizes=[32],
                                  denseLayers=[64],
                                  dropoutRt=[0.2],
                                  learnRt=[0.001])

#%% Train Neural Network
# FIlepath and sitename are used to save pickle file of model runs under
PredDict = Predictions.TrainRNN(PredDict,filepath,sitename)

#%% Export Hyperparameter Test Data
Predictions.RunsToCSV(os.path.join(filepath, sitename, 'predictions', 'tuning', 'combi'),
                      'combi_history.csv')
#%%
AccuracyPath = os.path.join(filepath, sitename,'predictions','tuning','combi_CSVs')
FigPath = os.path.join(filepath, sitename, 'plots', sitename+'_tuning_accuracy.png')
AccuracyDF = Predictions.PlotAccuracy(AccuracyPath, FigPath)

#%%
OptStudy = Predictions.TrainRNN_Optuna(PredDict, 'test1')

#%% Make Predictions
# Using full list of variables from past portion as test/placeholder
# ForecastDF = PredDict['X_test'][0]
# VarDFDayTest = VarDFDay[-360:]

FutureOutputs = Predictions.FuturePredict(PredDict, VarDFDayTest)

#%%
mID = 0

plt.plot(VarDFDay['distances'], 'C2')
plt.plot(VarDFDay['wlcorrdist'], 'C0')
plt.plot(FutureOutputs['output'][mID]['futureVE'], 'C8', ls='--')
plt.plot(FutureOutputs['output'][mID]['futureWL'], 'C9', ls='--')


#%% Cluster Past Observations
# VarDF = Predictions.Cluster(TransectDF, ValPlots=True)

