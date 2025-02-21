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
sitename = 'StAndrewsEastS2Full2024'
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

#%% Plot Interpolation Methods
Predictions.PlotInterps(CoastalDF, TransectIDs[0], '/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year4/Outputs/Figures/'+sitename+'_InterpolationMethods.png')

for Tr in TransectIDs:
    TransectDF = Predictions.InterpVEWL(CoastalDF, Tr, IntpKind='pchip')

#%% Load In Pre-trained Model
with open(os.path.join(filepath, sitename, 'predictions', '20250221-100808_dailywaves_fullvars.pkl'), 'rb') as f:
    PredDict = pickle.load(f)

#%% Separate Training and Validation
# TransectDFTrain = TransectDF.iloc[:263]
# TransectDFTest = TransectDF.iloc[262:]
TransectDFTrain = TransectDF.iloc[:int(len(TransectDF)*0.9)]
TransectDFTest = TransectDF.iloc[int(len(TransectDF)*0.9):]

Predictions.PlotVarTS(TransectDF, TransectIDs[0], filepath, sitename)
    
#%% Prepare Training Data
PredDict, VarDFDayTrain, VarDFDayTest  = Predictions.PrepData(TransectDF, 
                                          MLabels=['dailywaves_fullvars'], 
                                          ValidSizes=[0.2], 
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

#%% Feature Importance
mID = 0
IntGradAttr = Predictions.FeatImportance(PredDict, mID)
Predictions.PlotIntGrads(PredDict, VarDFDayTrain, IntGradAttr, filepath, sitename, TransectIDs[0])

#%% Make WL and VE Predictions
# Using full list of variables from past portion as test/placeholder

FutureOutputs = Predictions.FuturePredict(PredDict, VarDFDayTest)

#%% Plot Future WL and VE
Predictions.PlotFuture(0, TransectDFTrain, TransectDFTest, FutureOutputs, filepath, sitename)

#%%
Predictions.PlotFutureVars(0, TransectDFTrain, TransectDFTest, VarDFDayTrain, FutureOutputs, filepath, sitename)

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

