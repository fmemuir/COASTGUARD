#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:51:41 2024

@author: fmuir
"""

import os
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
    
#%% Prepare Training Data
PredDict = Predictions.PrepData(TransectDF, MLabels=['test1'], TestSizes=[0.2], TSteps=[30])

#%% Compile the Recurrent Neural Network 
# with desired number of epochs and batch size (per model run)
PredDict = Predictions.CompileRNN(PredDict, epochSizes=[50], batchSizes=[32])

#%% Train Neural Network
# FIlepath and sitename are used to save pickle file of model runs under
PredDict = Predictions.TrainRNN(PredDict,filepath,sitename)

#%% Make Predictions
# Using full list of variables from past portion as test/placeholder
ForecastDF = PredDict['X_test'][0]

VEPredict, WLPredict = Predictions.FuturePredict(PredDict, ForecastDF)

#%%
mID = 0
VETest = PredDict['scalings'][mID]['distances'].inverse_transform(PredDict['y_test'][mID][:,0].reshape(-1, 1))
WLTest = PredDict['scalings'][mID]['wlcorrdist'].inverse_transform(PredDict['y_test'][mID][:,1].reshape(-1, 1))
plt.plot(VETest, 'C2')
plt.plot(WLTest, 'C0')
plt.plot(VEPredict, 'C8', alpha=0.5)
plt.plot(WLPredict, 'C9', alpha=0.5)


#%% Cluster Past Observations
# VarDF = Predictions.Cluster(TransectDF, ValPlots=True)

