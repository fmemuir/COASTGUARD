#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:51:41 2024

@author: fmuir
"""

import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from Toolshed import Predictions

# Only use tensorflow in CPU mode
import tensorflow as tf
tf.config.set_visible_devices([],'GPU')
# %load_ext tensorboard
# import tensorboard

#%%
# Name of site to save directory and files under
sitename = 'StAndrewsEastS2Full'
filepath = os.path.join(os.getcwd(), 'Data')

# Load in transect data with coastal change variables
TransectInterGDF, TransectInterGDFWater, TransectInterGDFTopo, TransectInterGDFWave = Predictions.LoadIntersections(filepath, sitename)
# Compile relevant coastal change metrics into one dataframe
CoastalDF = Predictions.CompileTransectData(TransectInterGDF, TransectInterGDFWater, TransectInterGDFTopo, TransectInterGDFWave)

#%% Subset and interpolate timeseries to match up to same dates
# TransectIDs = [271]
TransectIDs = [1325]

for Tr in TransectIDs:
    TransectDF = Predictions.InterpWL(CoastalDF, Tr)
    
    
    
#%%
VarDF = Predictions.Cluster(TransectDF)

#%%
PredDict = Predictions.PrepData(VarDF, ['test1','test2'], [0.2,0.2], [1,1])

#%%
PredDict = Predictions.CompileRNN(PredDict)

#%%
PredDict = Predictions.TrainRNN(PredDict,filepath,sitename)


