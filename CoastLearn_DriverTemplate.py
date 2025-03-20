#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:51:41 2024

@author: fmuir
"""

import os
import numpy as np
import pickle
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd

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

# Define symbol disctionary for labelling
SymbolDict = {'VE':          r'$VE$',
              'WL':          r'$WL$',
              'tideelev':    r'$z_{tide,sat}$',
              'beachwidth':  r'$d_{VE,WL}$',
              'tideelevFD':  r'$\bar{z}_{tide}$',
              'tideelevMx':  r'$z^{*}_{tide}$',
              'WaveHsFD':    r'$H_{s}$',
              'WaveDirFD':   r'$\bar\theta$',
              'WaveTpFD':    r'$T_{p}$', 
              'WaveAlphaFD': r'$\alpha$',
              'Runups':      r'$R_{2}$',
              'Iribarren':   r'$\xi_{0}$', 
              'WL_u':        r'$WL_{u}$',
              'VE_u':        r'$VE_{u}$', 
              'WL_d':        r'$WL_{d}$',
              'VE_d':        r'$VE_{d}$'}

#%% Compile Transect Data
# Compile relevant coastal change metrics into one dataframe
CoastalDF = Predictions.CompileTransectData(TransectInterGDF, TransectInterGDFWater, TransectInterGDFTopo, TransectInterGDFWave)

#%% Interpolate
# Subset and interpolate timeseries to match up to same dates
# TransectIDs = [271]
TransectIDs = [1325]
# TransectIDs = [66]


#%% Interpolate variables to daily
# Plot Interpolation Methods
# Predictions.PlotInterps(CoastalDF, TransectIDs[0], '/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year4/Outputs/Figures/'+sitename+'_InterpolationMethods.png')

for Tr in TransectIDs:
    TransectDF = Predictions.InterpVEWLWv(CoastalDF, Tr, IntpKind='pchip')
    
#%% Plot VE, WL and wave height for storm
Predictions.PlotStormWaveHs(TransectDF, CoastalDF.iloc[TransectIDs[0]], filepath, sitename)

#%% Separate Training and Validation
TransectDF = TransectDF.loc[:datetime(2024,7,31)]
TransectDFTrain = TransectDF.loc[:datetime(2023,8,31)]
TransectDFTest = TransectDF.loc[datetime(2023,9,1):]



#%% Plot timeseries of variables
TrainFeatsPlotting = ['WaveHsFD', 'Runups', 'WaveDirFD', 'WaveTpFD']

Predictions.PlotVarTS(TransectDF, TransectIDs[0],TrainFeatsPlotting, filepath, sitename)
# Predictions.PlotChosenVarTS(TransectDFTrain, TransectDFTest, CoastalDF, TrainFeatsPlotting, SymbolDict, TransectIDs[0], filepath, sitename)
    
#%% Prepare Training Data
TrainFeats = ['WaveHsFD', 'Runups', 'WaveDirFD', 'WaveTpFD']#, 'tideelev']
TargFeats = ['VE', 'WL']

# OptParam = [32, 64, 96, 128, 160, 192, 224, 256]

PredDict, VarDFDayTrain, VarDFDayTest = Predictions.PrepData(TransectDF, 
                                                              MLabels=['daily_optimal_'], 
                                                              ValidSizes=[0.1], 
                                                              TSteps=[10],
                                                              TrainFeatCols=[TrainFeats],
                                                              TargFeatCols=[TargFeats],
                                                              TrainTestPortion=datetime(2023,9,1))

# PredDict, VarDFDayTrain, VarDFDayTest = Predictions.PrepData(TransectDF, 
#                                                               MLabels=['daily_batch_'+str(i) for i in OptParam], 
#                                                               ValidSizes=[0.2]*len(OptParam), 
#                                                               TSteps=[10]*len(OptParam),
#                                                               TrainFeatCols=[TrainFeats]*len(OptParam),
#                                                               TargFeatCols=[TargFeats]*len(OptParam),
#                                                               TrainTestPortion=datetime(2023,9,1))
# Needs additional lines for TransectID
#%% Load In Pre-trained Model
with open(os.path.join(filepath, sitename, 'predictions', '20250307-113621_daily_optimal_val10.pkl'), 'rb') as f:
    PredDict = pickle.load(f)


#%% Compile the Recurrent Neural Network 
# with desired number of epochs and batch size (per model run)
PredDict = Predictions.CompileRNN(PredDict, 
                                  epochNums=[150], 
                                  batchSizes=[64],
                                  denseLayers=[128],
                                  dropoutRt=[0.2],
                                  learnRt=[0.001],
                                  hiddenLscale=[6],
                                  DynamicLR=False)

# PredDict = Predictions.CompileRNN(PredDict, 
#                                   epochNums=[150]*len(OptParam), 
#                                   batchSizes=OptParam,
#                                   denseLayers=[128]*len(OptParam),
#                                   dropoutRt=[0.2]*len(OptParam),
#                                   learnRt=[0.001]*len(OptParam),
#                                   hiddenLscale=[5]*len(OptParam),
#                                   DynamicLR=False)

#%% Train Neural Network
# FIlepath and sitename are used to save pickle file of model runs under
PredDict = Predictions.TrainRNN(PredDict,filepath,sitename,EarlyStop=True)


#%% Parallel Coordinates
HPfiles = [
           '20250306-114952_daily_epochs_200.pkl',
           '20250306-121419_daily_batch_8_daily_batch_16.pkl',
           '20250306-170808_daily_batch_32_daily_batch_64_daily_batch_96_daily_batch_128_daily_batch_160_daily_batch_192_daily_batch_224_daily_batch_256.pkl',
           '20250306-130124_daily_LSTMscale_1_daily_LSTMscale_2_daily_LSTMscale_3_daily_LSTMscale_4_daily_LSTMscale_5_daily_LSTMscale_6_daily_LSTMscale_7_daily_LSTMscale_8_daily_LSTMscale_9_daily_LSTMscale_10.pkl',
           '20250306-121929_daily_dense_4_daily_dense_8_daily_dense_16_daily_dense_32_daily_dense_64_daily_dense_128.pkl',
           '20250306-123232_daily_dropout_0.1_daily_dropout_0.2_daily_dropout_0.3_daily_dropout_0.4_daily_dropout_0.5.pkl',
           '20250306-124336_daily_learnrt_0.0001_daily_learnrt_0.001_daily_learnrt_0.01_daily_learnrt_0.1_daily_learnrt_0.9.pkl']

with open(os.path.join(filepath, sitename, 'predictions', '20250306-113911_daily_epochs_25_daily_epochs_50_daily_epochs_75_daily_epochs_100_daily_epochs_125_daily_epochs_150_daily_epochs_175.pkl',), 'rb') as f:
    PredDictCombi = pickle.load(f)
if 'hiddenLscale' not in PredDictCombi.keys():
    PredDictCombi['hiddenLscale'] = [5]*len(PredDictCombi['mlabel'])
    
for i in range(len(HPfiles)):
    with open(os.path.join(filepath, sitename, 'predictions', HPfiles[i]), 'rb') as f:
        PredDictHP = pickle.load(f)
    if 'hiddenLscale' not in PredDictHP.keys():
        PredDictHP['hiddenLscale'] = [5]*len(PredDictHP['mlabel'])
    for key in PredDictCombi.keys():
        PredDictCombi[key].extend(PredDictHP[key])
        
#%%
# Predictions.PlotParaCoords(PredDictCombi)

Predictions.PlotHPScatter(PredDictCombi)

#%% Looped Feature Testing
TrainFeats = ['WaveHsFD', 'Runups', 'WaveDirFD', 'WaveAlphaFD', 'WaveTpFD']
TargFeats = ['VE', 'WL']

TrainFeatsComb = []
for r in range(1, len(TrainFeats)+1):
    comb = combinations(TrainFeats, r)
    for ft in comb:
        TrainFeatsComb.append(list(ft))
        
PredDict, VarDFDayTrain, VarDFDayTest = Predictions.PrepData(TransectDF, 
                                                             MLabels=[str(i) for i in range(len(TrainFeatsComb))], 
                                                             ValidSizes=[0.2]*len(TrainFeatsComb), 
                                                             TSteps=[10]*len(TrainFeatsComb),
                                                             TrainFeatCols=TrainFeatsComb,
                                                             TargFeatCols=[TargFeats]*len(TrainFeatsComb))
PredDict = Predictions.CompileRNN(PredDict, 
                                  epochNums=[150]*len(TrainFeatsComb), 
                                  batchSizes=[32]*len(TrainFeatsComb),
                                  denseLayers=[64]*len(TrainFeatsComb),
                                  dropoutRt=[0.3]*len(TrainFeatsComb),
                                  learnRt=[0.001]*len(TrainFeatsComb),
                                  DynamicLR=False)

PredDict = Predictions.TrainRNN(PredDict,filepath,sitename,EarlyStop=True)

#%% Ensemble Run
TrainFeats = ['WaveHsFD', 'Runups', 'WaveDirFD', 'WaveTpFD']
TargFeats = ['VE', 'WL']

EnsembleCount = 10
        
PredDict, VarDFDayTrain, VarDFDayTest = Predictions.PrepData(TransectDF, 
                                                             MLabels=['ensemble_'+str(i) for i in range(EnsembleCount)], 
                                                             ValidSizes=[0.2]*EnsembleCount, 
                                                             TSteps=[10]*EnsembleCount,
                                                             TrainFeatCols=[TrainFeats]*EnsembleCount,
                                                             TargFeatCols=[TargFeats]*EnsembleCount)
PredDict = Predictions.CompileRNN(PredDict, 
                                  epochNums=[150]*EnsembleCount, 
                                  batchSizes=[32]*EnsembleCount,
                                  denseLayers=[64]*EnsembleCount,
                                  dropoutRt=[0.3]*EnsembleCount,
                                  learnRt=[0.001]*EnsembleCount,
                                  DynamicLR=False)

PredDict = Predictions.TrainRNN(PredDict,filepath,sitename,EarlyStop=True)



#%% Plot Feature Sensitivity
Predictions.PlotFeatSensitivity(PredDict,filepath, sitename,TransectIDs[0])

#%% Feature Importance
mID = 0
IntGradAttr = Predictions.FeatImportance(PredDict, mID)
Predictions.PlotIntGrads(PredDict, VarDFDayTrain, IntGradAttr, SymbolDict, filepath, sitename, TransectIDs[0])

#%% Make WL and VE Predictions
# Using full list of variables from past portion as test/placeholder

FutureOutputs = Predictions.FuturePredict(PredDict, VarDFDayTest)
FullFutureOutputs = Predictions.FuturePredict(PredDict, pd.concat([VarDFDayTrain, VarDFDayTest]))

#%% Plot Future WL and VE
for mID in range(len(FutureOutputs['mlabel'])): 
    PlotDateRange = [datetime(2023,10,1), datetime(2023,11,5)] # Storm Babet
    # Predictions.PlotFuture(mID, TransectIDs[0], TransectDFTrain, TransectDFTest, FullFutureOutputs, filepath, sitename)
    Predictions.PlotFutureShort(mID, TransectIDs[0], TransectDFTrain, TransectDFTest, FullFutureOutputs, 
                                filepath, sitename, PlotDateRange, Storm=[datetime(2023,10,21), datetime(2023,10,18)])

#%%
Predictions.PlotFutureEnsemble(TransectDFTrain, TransectDFTest, FullFutureOutputs, filepath, sitename)


#%%
# Predictions.PlotFutureVars(0, TransectDFTrain, TransectDFTest, VarDFDayTrain, FutureOutputs, filepath, sitename)
FutureOutputs = Predictions.ShorelineRMSE(FutureOutputs, TransectDFTest)

#%%
mID=0
Predictions.PlotTestScatter(FutureOutputs, TransectDFTest, mID, TransectIDs[0], filepath, sitename)

#%% Violin plot of distance differences between predicted and actual VEs and WLs
Predictions.FutureDiffViolin(FutureOutputs, mID, TransectDFTest, filepath, sitename, TransectIDs[0])

#%%
Predictions.FutureViolinLinReg(FutureOutputs, mID, TransectDFTest, filepath, sitename, TransectIDs[0])

#%% Thresholding Past Observations

ImpactClass = Predictions.ClassifyImpact(TransectDFTrain,Method='combi')

# FutureImpacts = Predictions.ApplyImpactClasses(ImpactClass, FutureOutputs)
Predictions.PlotImpactClasses(ImpactClass, TransectDFTrain)

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

