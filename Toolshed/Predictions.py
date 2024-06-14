#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:57:18 2024

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

# Only use tensorflow in CPU mode
import tensorflow as tf
tf.config.set_visible_devices([],'GPU')



def LoadIntersections(filepath, sitename):
    
    with open(os.path.join
              (filepath , sitename, 'intersections', sitename + '_transect_intersects.pkl'), 'rb') as f:
        TransectInterGDF = pickle.load(f)
        
    with open(os.path.join
              (filepath , sitename, 'intersections', sitename + '_transect_water_intersects.pkl'), 'rb') as f:
        TransectInterGDFWater = pickle.load(f)

    with open(os.path.join
              (filepath , sitename, 'intersections', sitename + '_transect_topo_intersects.pkl'), 'rb') as f:
        TransectInterGDFTopo = pickle.load(f)

    with open(os.path.join
              (filepath , sitename, 'intersections', sitename + '_transect_wave_intersects.pkl'), 'rb') as f:
        TransectInterGDFWave = pickle.load(f)
        
    return TransectInterGDF, TransectInterGDFWater, TransectInterGDFTopo, TransectInterGDFWave
        

def CompileTransectData(TransectInterGDF, TransectInterGDFWater, TransectInterGDFTopo, TransectInterGDFWave):
    
    # Merge veg edge intersection data with waterline intersection data
    CoastalDF = pd.merge(TransectInterGDF[['TransectID','dates','distances']], 
                         TransectInterGDFWater[['TransectID','wlcorrdist', 'waterelev','beachwidth']],
                         how='inner', on='TransectID')
    # Merge combined dataframe with topographic info
    CoastalDF = pd.merge(CoastalDF, 
                         TransectInterGDFTopo[['TransectID','TZwidth', 'TZwidthMn', 'SlopeMax', 'SlopeMean']],
                         how='inner', on='TransectID')
    # Merge combined dataframe with wave info
    CoastalDF = pd.merge(CoastalDF, 
                         TransectInterGDFWave[['TransectID','WaveHs', 'WaveDir', 'WaveTp', 'WaveDiffus']],
                         how='inner', on='TransectID')
    
    return CoastalDF


def PreprocessTraining(CoastalDF):
    
    X = CoastalDF.drop(columns=['TransectID', 'labels'])
    y = CoastalDF['labels']
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)