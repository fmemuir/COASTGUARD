#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:51:41 2024

@author: fmuir
"""

import os
import pickle
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
TransectDFTrain = TransectDF.iloc[:263]
TransectDFTest = TransectDF.iloc[262:]

VarDFDayTest = Predictions.DailyInterp(TransectDFTest)
# Why is this returned scaled?

#%% Prepare Training Data
PredDict, VarDFDay = Predictions.PrepData(TransectDFTrain, 
                                          MLabels=['dir_runup_iri'], 
                                          TestSizes=[0.2], 
                                          TSteps=[10])
# Needs additional lines for TransectID

#%% Compile the Recurrent Neural Network 
# with desired number of epochs and batch size (per model run)
PredDict = Predictions.CompileRNN(PredDict, 
                                  epochNums=[150], 
                                  batchSizes=[32],
                                  denseLayers=[64],
                                  dropoutRt=[0.2],
                                  learnRt=[0.001],
                                  DynamicLR=False)

#%% Train Neural Network
# FIlepath and sitename are used to save pickle file of model runs under
PredDict = Predictions.TrainRNN(PredDict,filepath,sitename,EarlyStop=True)


#%% Load In Pre-trained Model
with open(os.path.join(filepath, sitename, 'predictions', '20250111-165139_optimised.pkl'), 'rb') as f:
    PredDict = pickle.load(f)

#%% Make WL and VE Predictions
# Using full list of variables from past portion as test/placeholder

FutureOutputs = Predictions.FuturePredict(PredDict, VarDFDayTest)

#%% Plot Future WL and VE
Predictions.PlotFuture(0, VarDFDay, TransectDFTest, FutureOutputs, filepath, sitename)


#%% Export Hyperparameter Test Data
Predictions.RunsToCSV(os.path.join(filepath, sitename, 'predictions', 'tuning', 'combi'),
                      'combi_history.csv')
#%% Plot Accuracy Over Epochs (for all training runs)
AccuracyPath = os.path.join(filepath, sitename,'predictions','tuning','combi_CSVs')
FigPath = os.path.join(filepath, sitename, 'plots', sitename+'_tuning_accuracy.png')
AccuracyDF = Predictions.PlotAccuracy(AccuracyPath, FigPath)

#%% Train Using Optuna Hyperparameterisation
OptStudy = Predictions.TrainRNN_Optuna(PredDict, 'test1')

#%% Cluster Past Observations
ClusterDF = Predictions.Cluster(TransectDFTrain[['distances','wlcorrdist']], ValPlots=True)

