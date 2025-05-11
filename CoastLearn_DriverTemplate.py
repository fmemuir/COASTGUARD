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

from Toolshed import Predictions, PredictionsPlotting

# Only use tensorflow in CPU mode
import tensorflow as tf
tf.config.set_visible_devices([],'GPU')
# %load_ext tensorboard
# import tensorboard




#%% ------------------------ MODELLING ------------------------ 

#%% Load Transect Data
# Name of site to save directory and files under
sitename = 'EXAMPLE'
filepath = os.path.join(os.getcwd(), 'Data')

# Load in transect data with coastal change variables
TransectInterGDF, TransectInterGDFWater, TransectInterGDFTopo, TransectInterGDFWave = Predictions.LoadIntersections(filepath, sitename)

# Define symbol dictionary for labelling
SymbolDict = {'VE':          r'$VE$',
              'WL':          r'$WL$',
              'tideelev':    r'$z_{tide,sat}$',
              'beachwidth':  r'$d_{VE,WL}$',
              'tideelevFD':  r'$\bar{z}_{tide}$',
              'tideelevMx':  r'$z^{*}_{tide}$',
              'WaveHsFD':    r'$H_{s}$',
              'WaveDirFD':   r'$\bar\theta$',
              'WaveDirsin':  r'$sin(\bar\theta)$',
              'WaveDircos':  r'$cos(\bar\theta)$',
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

#%% Subset Timeseries to Match Dates
TransectIDs = [1325]


#%% Interpolate Variables to Daily
# Plot Interpolation Methods
PredictionsPlotting.PlotInterpsWLVE(CoastalDF, TransectIDs[0], '/path/to/your/figures/'+sitename+'_InterpolationMethods_WLVE.png')

# You can choose the type of interpolation from scipy.interpolate.interp1d(), but pchip is recommended
for Tr in TransectIDs:
    TransectDF = Predictions.InterpVEWLWv(CoastalDF, Tr, IntpKind='pchip')
    


#%% Separate Training and Validation
TransectDF = TransectDF.loc[:datetime(2024,7,31)]
TransectDFTrain = TransectDF.loc[:datetime(2023,8,31)]
TransectDFTest = TransectDF.loc[datetime(2023,9,1):]


#%% Plot timeseries of variables
TrainFeatsPlotting = ['WaveHsFD', 'Runups', 'WaveDirFD', 'WaveTpFD']

PredictionsPlotting.PlotVarTS(TransectDF, TransectIDs[0],TrainFeatsPlotting, filepath, sitename)
# Predictions.PlotChosenVarTS(TransectDFTrain, TransectDFTest, CoastalDF, TrainFeatsPlotting, SymbolDict, TransectIDs[0], filepath, sitename)
    
#%% Prepare Training Data
# Use the recommended variables (full descriptions are in the code or CoasTrack_README.txt)
TrainFeats = ['WaveHsEW', 'Runups', 'WaveDirEW', 'WaveTpEW', 'WL_u-10', 'VE_u-10','WL_d-10', 'VE_d-10']
TargFeats = ['VE', 'WL']

PredDict, VarDFDayTrain, VarDFDayTest = Predictions.PrepData(TransectDF, 
                                                              MLabels=['modelname'], # unique model ID
                                                              ValidSizes=[0.1], # proportion of training to use for validation
                                                              TSteps=[10], # sequence length in days
                                                              TrainFeatCols=[TrainFeats], 
                                                              TargFeatCols=[TargFeats],
                                                              # You can use a date for separating out training/validation/test, or just a proportion (decimal)
                                                              TrainTestPortion=datetime(2023,9,1))


#%% OR Load In Pre-trained Model
with open(os.path.join(filepath, sitename, 'predictions', 'oldmodelname.pkl'), 'rb') as f:
    PredDict = pickle.load(f)


#%% Compile the Recurrent Neural Network 
# with desired number of epochs and batch size (per model run)
# These hyperparameters are recommended; DON'T CHANGE unless you know what you are doing
PredDict = Predictions.CompileRNN(PredDict, 
                                  epochNums=[150],      # number of epochs
                                  batchSizes=[64],      # batch size
                                  denseLayers=[128],    # number of dense layers
                                  dropoutRt=[0.2],      # data dropout rate
                                  learnRt=[0.001],      # LSTM learning rate
                                  hiddenLscale=[6],     # scaling factor for number of hidden layers
                                  LossFn='Shoreshop',   # loss function, 'mse', 'CostSensitive', 'Shoreshop'
                                  DynamicLR=False)      # Flag for whether learning rate is dynamically changed in training

#%% Train Neural Network
# FIlepath and sitename are used to save pickle file of model runs under
# Early stopping helpds avoid overtraining
PredDict = Predictions.TrainRNN(PredDict,filepath,sitename,EarlyStop=True)

#%% OR Train Using Optuna Hyperparameterisation
# OptStudy outputs can then be used in CompileRNN() and TrainRNN()
OptStudy = Predictions.TrainRNN_Optuna(PredDict, 'test1')

#%% OPTIONAL: Export Hyperparameter Test Data
# Make sure all desired training history files produced by TensorFlow are in one folder called 'combi'
Predictions.RunsToCSV(os.path.join(filepath, sitename, 'predictions', 'tuning', 'combi'),
                      'combi_history.csv')

#%% OPTIONAL: Ensemble Run for Probabilistic Predictions
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

#%% Feature Importance

IntGradAttr = Predictions.FeatImportance(PredDict)

#%% Make WL and VE Predictions
# Using full list of variables from past portion as test/placeholder

FutureOutputs = Predictions.FuturePredict(PredDict, VarDFDayTest)
FullFutureOutputs = Predictions.FuturePredict(PredDict, pd.concat([VarDFDayTrain, VarDFDayTest]))

predpath = os.path.join(filepath,sitename,'predictions','modelname'+'_Prediction.pkl')
with open(predpath, 'wb') as f:
    pickle.dump(FutureOutputs, f)

#%% OR Read In Trained and Predicted Test Data

pklpath = os.path.join(filepath,sitename,'predictions','oldmodelname_Prediction.pkl')
with open(pklpath, 'rb') as f:
    FutureOutputs = pickle.load(f)

#%% Calculate Root Mean Square Error 
# (between test data and predictions) 
mID=0
FutureOutputs = Predictions.ShorelineRMSE(FutureOutputs, TransectDFTest)
for SL in ['VE', 'WL']: 
    FullFutureOutputs['output'][mID]['future'+SL] = FullFutureOutputs['output'][mID]['future'+SL].loc[TransectDFTest.index[0]:]
FullFutureOutputs = Predictions.ShorelineRMSE(FullFutureOutputs, TransectDF)


#%% Thresholding Past Observations for Impact Classification

ImpactClass = Predictions.ClassifyImpact(TransectDF,Method='combi')
PredImpactClass = Predictions.ClassifyImpact(FullFutureOutputs['output'][0], Method='combi')




#%% ------------------------ BATCH RUN OVER SITE ------------------------ 

#%% Initialise storage dicts

CoastalDF = Predictions.CompileTransectData(TransectInterGDF, TransectInterGDFWater, TransectInterGDFTopo, TransectInterGDFWave)

PredDicts = dict.fromkeys(list(range(len(CoastalDF))),None)
FutureOutputs = dict.fromkeys(list(range(len(CoastalDF))),None)
TransectsDFTrain = dict.fromkeys(list(range(len(CoastalDF))),None)
TransectsDFTest = dict.fromkeys(list(range(len(CoastalDF))),None)

#%% Full Site Run (looped through transects)

for Tr in CoastalDF[::1].index: # every nth Transect
    print(f"{Tr}/{len(CoastalDF)}")
    # ignore Tr with less than 6 months of data (img every 10 days on average)
    if len(CoastalDF['VE'].iloc[Tr]) < 18 or len(CoastalDF['WL'].iloc[Tr]) < 18: 
        continue
    else:
        # Interpolate over transect data to get daily metrics
        TransectDF = Predictions.InterpVEWLWv(CoastalDF, Tr, IntpKind='pchip')
        TransectDFTrain = TransectDF.iloc[:int(0.8825*len(TransectDF))]
        TransectDFTest = TransectDF.iloc[int(0.8825*len(TransectDF)):]
        
        # Define training and target features
        TrainFeats = ['WaveHsEW', 'Runups', 'WaveDirEW', 'WaveTpEW', 'WL_u-10', 'VE_u-10','WL_d-10', 'VE_d-10']
        TargFeats = ['VE', 'WL']
        
        # Separate timeseries into training/validation and testing portions
        TransectsDFTrain[Tr] = TransectDFTrain
        TransectsDFTest[Tr] = TransectDFTest
        
        # Prep data
        PredDict, VarDFDayTrain, VarDFDayTest = Predictions.PrepData(TransectDF, 
                                                                      MLabels=['Tr'+str(Tr)], 
                                                                      ValidSizes=[0.1], 
                                                                      TSteps=[10],
                                                                      TrainFeatCols=[TrainFeats],
                                                                      TargFeatCols=[TargFeats],
                                                                      TrainTestPortion=0.8825)
        # Compile LSTM based on provided hyperparameters
        PredDict = Predictions.CompileRNN(PredDict, 
                                          epochNums=[150], 
                                          batchSizes=[64],
                                          denseLayers=[128],
                                          dropoutRt=[0.2],
                                          learnRt=[0.001],
                                          hiddenLscale=[6],
                                          LossFn='Shoreshop',
                                          DynamicLR=False)
        
        # Train LSTM to predict target features
        PredDict = Predictions.TrainRNN(PredDict,filepath,sitename,EarlyStop=True)
        PredDicts[Tr] = PredDict
        
        # Use the trained model to predict target features in timeseries over the test period
        FutureOutput = Predictions.FuturePredict(PredDict, VarDFDayTest)
        # Assess the performance of the predictions against unseen test data
        FutureOutput = Predictions.ShorelineRMSE(FutureOutput, TransectDFTest)
        FutureOutputs[Tr] = FutureOutput

PredDictsClean = {k: v for k, v in PredDicts.items() if v is not None}
FutureOutputsClean = {k: v for k, v in FutureOutputs.items() if v is not None}

# Save all outputs to pickle file
pklpath = os.path.join(filepath,sitename,'predictions',sitename+'_FullPredict_EW_neighbours.pkl')
with open(pklpath, 'wb') as f:
    pickle.dump(FutureOutputsClean, f)
    
    


#%% ------------------------ PLOTTING ------------------------ 

#%%Plot Pearson correlations
PredictionsPlotting.PlotCorrs(filepath, sitename, TransectIDs[0], VarDFDayTrain, ['VE', 'WL', 'beachwidth', 'tideelevFD', 'tideelevMx',
       'WaveHsFD', 'WaveDirFD', 'WaveTpFD', 'WaveAlphaFD', 'Runups',
       'Iribarren', 'WL_u', 'VE_u', 'WL_d', 'VE_d'], SymbolDict)

#%% Plot Feature Sensitivity
PredictionsPlotting.PlotFeatSensitivity(PredDict,filepath, sitename,TransectIDs[0])

#%% Plot Accuracy Over Epochs (for all training runs)
AccuracyPath = os.path.join(filepath, sitename,'predictions','tuning','combi_CSVs')
FigPath = os.path.join(filepath, sitename, 'plots', sitename+'_tuning_accuracy.png')
AccuracyDF = PredictionsPlotting.PlotAccuracy(AccuracyPath, FigPath)

#%%Plot VE, WL and Wave Height for Storm
PredictionsPlotting.PlotStormWaveHs(TransectDF, CoastalDF.iloc[TransectIDs[0]], filepath, sitename)

#%% Plot Feature Importance Results
PredictionsPlotting.PlotIntGrads(PredDict, VarDFDayTrain, IntGradAttr, SymbolDict, filepath, sitename, TransectIDs[0])

#%% Plot Future WL and VE
PlotDateRange = [datetime(2023,10,1), datetime(2023,11,5)] # specific date range

for mID in range(len(FutureOutputs['modelname'])): 
    PredictionsPlotting.PlotFuture(mID, TransectIDs[0], PredDict, TransectDFTrain, TransectDFTest, FullFutureOutputs, 
                            filepath, sitename)
    PredictionsPlotting.PlotFutureShort(mID, TransectIDs[0], TransectDFTrain, TransectDFTest, FullFutureOutputs, 
                                filepath, sitename, PlotDateRange, Storm=[datetime(2023,10,18), datetime(2023,10,21)])

#%% Plot Future Prediction Ensemble
#(if ensemble cell above was run)
PredictionsPlotting.PlotFutureEnsemble(TransectDFTrain, TransectDFTest, FullFutureOutputs, filepath, sitename)

#%% Plot Future Variables In Full
PredictionsPlotting.PlotFutureVars(TransectDFTrain, TransectDFTest, VarDFDayTrain, FutureOutputs, filepath, sitename)

#%% Violin and scatter plots of distance differences between predicted and actual VEs and WLs

# Predictions.PlotTestScatter(FutureOutputs, TransectDFTest, mID, TransectIDs[0], filepath, sitename)
# Predictions.FutureDiffViolin(FutureOutputs, mID, TransectDFTest, filepath, sitename, TransectIDs[0])
# PredictionsPlotting.FutureViolinLinReg(FutureOutputs, mID, TransectDFTest, filepath, sitename, TransectIDs[0])
PredictionsPlotting.FutureViolinLinReg(FullFutureOutputs, mID, TransectDFTest, filepath, sitename, TransectIDs[0])

#%% Plot Impact Classifications
PredictionsPlotting.PlotImpactClasses(filepath, sitename, TransectIDs[0], ImpactClass, TransectDF)

PlotDateRange = [datetime(2023,10,1), datetime(2023,11,5)] # specific date range
PredictionsPlotting.PlotFutureShort(mID, TransectIDs[0], TransectDFTrain, TransectDFTest, FullFutureOutputs, 
                            filepath, sitename, PlotDateRange, Storm=[datetime(2023,10,18), datetime(2023,10,21)],
                            ImpactClass=PredImpactClass)





#%% ------------------------ CLUSTERING (UNUSED) ------------------------ 
#%% Cluster Past Observations
ClusterDF = Predictions.Cluster(TransectDFTrain[['distances','wlcorrdist']], ValPlots=True)

