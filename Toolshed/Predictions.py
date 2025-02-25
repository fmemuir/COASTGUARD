#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:57:18 2024

@author: fmuir
"""

import os
import timeit
import pickle
import datetime as dt
from datetime import datetime,timedelta
import time
import string
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch, Rectangle, ConnectionPatch
import matplotlib.dates as mdates
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
pd.options.mode.chained_assignment = None # suppress pandas warning about setting a value on a copy of a slice
from scipy.interpolate import interp1d, PchipInterpolator

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import SMOTE
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.hparams import api as hp


import optuna
import shap

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams.update({'font.size':7})

# Only use tensorflow in CPU mode
import tensorflow as tf
tf.config.set_visible_devices([],'GPU')


# ----------------------------------------------------------------------------------------
### DATA PREPPING FUNCTIONS ###

def LoadIntersections(filepath, sitename):
    """
    Load in transect intersection dataframes stored as pickle files. Generated 
    using Transects.GetIntersections(), Transects.SaveIntersections(), 
    Transects.GetBeachWidth(), Transects.SaveWaterIntersections(),
    Transects.TZIntersect(), Transects.SlopeIntersect(), 
    Transects.WavesIntersect().
    FM Jul 2024

    Parameters
    ----------
    filepath : str
        Path to 'Data' directory for chosen site.
    sitename : str
        Name of site chosen.

    Returns
    -------
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of cross-shore transects, intersected with vegetation edge lines.
    TransectInterGDFWater : GeoDataFrame
        GeoDataFrame of cross-shore transects, intersected with waterlines.
    TransectInterGDFTopo : GeoDataFrame
        GeoDataFrame of cross-shore transects, intersected with slope raster and vegetation transition zones.
    TransectInterGDFWave : GeoDataFrame
        GeoDataFrame of cross-shore transects, intersected with Copernicus hindcast wave data.

    """
    print('Loading transect intersections...')
    
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
    """
    Merge together transect geodataframes produced from COASTGUARD.VedgeSat and CoastSat. Each transect holds 
    timeseries of a range of satellite-derived metrics.
    FM Aug 2024

    Parameters
    ----------
    TransectInterGDF : GeoDataFrame
        DataFrame of cross-shore transects intersected with timeseries of veg edges.
    TransectInterGDFWater : GeoDataFrame
        DataFrame of cross-shore transects intersected with timeseries of waterlines.
    TransectInterGDFTopo : GeoDataFrame
        DataFrame of cross-shore transects intersected with timeseries of slopes at the veg edge.
    TransectInterGDFWave : GeoDataFrame
        DataFrame of cross-shore transects intersected with timeseries of wave conditions.

    Returns
    -------
    CoastalDF : DataFrame
        DataFrame process.

    """
    print('Merging transect-based data...')
    # Merge veg edge intersection data with waterline intersection data
    CoastalDF = pd.merge(TransectInterGDF[['TransectID',
                                           'dates','times','distances']], 
                         TransectInterGDFWater[['TransectID',
                                                'wldates','wltimes','wlcorrdist', 
                                                'tideelev','beachwidth',
                                                'tidedatesFD','tideelevFD', 'tideelevMx']],
                         how='inner', on='TransectID')
    # Merge combined dataframe with wave info
    # TransectInterGDFWave[['TransectID','WaveHs', 'WaveDir', 'WaveTp', 'WaveDiffus']]
    # CoastalDF = pd.merge(CoastalDF, 
    #                      TransectInterGDFWave[['TransectID','WaveDates','WaveHs', 'WaveDir', 'WaveTp', 'WaveAlpha', 'Runups','Iribarrens']],
    #                      how='inner', on='TransectID')
    CoastalDF = pd.merge(CoastalDF, 
                         TransectInterGDFWave[['TransectID',
                                               'WaveDates','WaveDatesFD',
                                               'WaveHsFD', 'WaveDirFD', 'WaveTpFD', 'WaveAlphaFD', 
                                               'Runups', 'Iribarren']],
                         how='inner', on='TransectID')
    
    print('Converting to datetimes...')
    veDTs = []
    for Tr in range(len(CoastalDF)):
        veDTs_Tr = []
        for i in range(len(CoastalDF['dates'].iloc[Tr])):
            veDTs_Tr.append(dt.datetime.strptime(CoastalDF['dates'].iloc[Tr][i]+' '+CoastalDF['times'].iloc[Tr][i], '%Y-%m-%d %H:%M:%S.%f'))
        veDTs.append(veDTs_Tr)
    CoastalDF['veDTs'] = veDTs
    
    wlDTs = []
    for Tr in range(len(CoastalDF)):
        wlDTs_Tr = []
        for i in range(len(CoastalDF['wldates'].iloc[Tr])):
            wlDTs_Tr.append(dt.datetime.strptime(CoastalDF['wldates'].iloc[Tr][i]+' '+CoastalDF['wltimes'].iloc[Tr][i], '%Y-%m-%d %H:%M:%S.%f'))
        wlDTs.append(wlDTs_Tr)
    CoastalDF['wlDTs'] = wlDTs
    
    
    return CoastalDF


def InterpWL(CoastalDF, Tr):
    """
    Interpolate over waterline associated timeseries so that dates 
    match vegetation associated ones.
    FM Aug 2024

    Parameters
    ----------
    CoastalDF : DataFrame
        DataFrame of cross-shore transects (rows) and intersected coastal 
        timeseries/metrics (columns).
    Tr : int
        Transect ID of choice.

    Returns
    -------
    TransectDF : DataFrame
        Subset row matching the requested transect ID (Tr), with interpolated
        values for 'wlcorrdist', 'waterelev' and 'beachwidth'.

    """
    TransectDF = CoastalDF.iloc[[Tr],:] # single-row dataframe
    # TransectDF = TransectDF.transpose()

    # Interpolate over waterline associated variables to match dates with veg edge dates
    wl_numdates = pd.to_datetime(TransectDF['wldates'][Tr]).values.astype(np.int64)
    ve_numdates = pd.to_datetime(TransectDF['dates'][Tr]).values.astype(np.int64)
    wl_interp_f = interp1d(wl_numdates, TransectDF['wlcorrdist'][Tr], kind='linear', fill_value='extrapolate')
    wl_interp = wl_interp_f(ve_numdates).tolist()
    welev_interp_f = interp1d(wl_numdates, TransectDF['tideelev'][Tr], kind='linear', fill_value='extrapolate')
    welev_interp = welev_interp_f(ve_numdates).tolist()
    TransectDF['wlcorrdist'] = [wl_interp]
    TransectDF['tideelev'] = [welev_interp]
    # Recalculate beachwidth
    beachwidth = [abs(wl_interp[i] - TransectDF['distances'][Tr][i]) for i in range(len(wl_interp))]
    TransectDF['beachwidth'] = [beachwidth]
    
    TransectDF.drop(columns=['wldates'], inplace=True)
    
    # Transpose to get columns of variables and rows of timesteps
    TransectDF = pd.DataFrame({col: pd.Series(val.iloc[0]) for col,val in TransectDF.items()})
    
    return TransectDF


def InterpWLWaves(CoastalDF, Tr):
    """
    Interpolate over waterline and wave associated timeseries so that dates 
    match vegetation associated ones.
    FM Aug 2024

    Parameters
    ----------
    CoastalDF : DataFrame
        DataFrame of cross-shore transects (rows) and intersected coastal 
        timeseries/metrics (columns).
    Tr : int
        Transect ID of choice.

    Returns
    -------
    TransectDF : DataFrame
        Subset row matching the requested transect ID (Tr), with interpolated
        values for 'wlcorrdist', 'waterelev' and 'beachwidth'.

    """
    TransectDF = CoastalDF.iloc[[Tr],:] # single-row dataframe
    # TransectDF = TransectDF.transpose()

    # Interpolate over waterline and wave associated variables
    wl_numdates = pd.to_datetime(TransectDF['wlDTs'][Tr]).values.astype(np.int64)
    ve_numdates = pd.to_datetime(TransectDF['veDTs'][Tr]).values.astype(np.int64)
    wv_numdates = pd.to_datetime(TransectDF['WaveDates'][Tr]).values.astype(np.int64)
    # Match dates with veg edge dates and append back to TransectDF
    for wlcol in ['wlcorrdist', 'tideelev','beachwidth']:
        wl_interp_f = interp1d(wl_numdates, TransectDF[wlcol][Tr], kind='linear', fill_value='extrapolate')
        wl_interp = wl_interp_f(ve_numdates).tolist()
        TransectDF[wlcol] = [wl_interp]
    for wvcol in ['WaveHs','WaveDir','WaveTp','Runups', 'Iribarrens']:
        wv_interp_f = interp1d(wv_numdates, TransectDF[wvcol][Tr], kind='linear', fill_value='extrapolate')
        wv_interp = wv_interp_f(ve_numdates).tolist()
        TransectDF[wvcol] = [wv_interp]
    
    # Recalculate beachwidth as values will now be mismatched
    beachwidth = [abs(wl_interp[i] - TransectDF['distances'][Tr][i]) for i in range(len(wl_interp))]
    TransectDF['beachwidth'] = [beachwidth]
    
    TransectDF.drop(columns=['WaveDates','wldates','wltimes', 'wlDTs','dates','times'], inplace=True)
    
    # Transpose to get columns of variables and rows of timesteps
    TransectDF = pd.DataFrame({col: pd.Series(val.iloc[0]) for col,val in TransectDF.items()})
    
    # Reset index for timeseries
    TransectDF.index = TransectDF['veDTs']
    TransectDF = TransectDF.drop(columns=['TransectID', 'veDTs'])

    return TransectDF


def InterpVEWL(CoastalDF, Tr, IntpKind='linear'):
    """
    Interpolate over waterline and vegetation associated timeseries so that dates 
    match wave associated (full timeseries) ones.
    FM Nov 2024

    Parameters
    ----------
    CoastalDF : DataFrame
        DataFrame of cross-shore transects (rows) and intersected coastal 
        timeseries/metrics (columns).
    Tr : int
        Transect ID of choice.

    Returns
    -------
    TransectDF : DataFrame
        Subset row matching the requested transect ID (Tr), with interpolated
        values for 'wlcorrdist', 'waterelev' and 'beachwidth'.

    """
    TransectDF = CoastalDF.iloc[[Tr],:] # single-row dataframe
    # TransectDF = TransectDF.transpose()

    # Convert all circular metrics to radians
    for WaveProp in ['WaveDirFD','WaveAlphaFD']:
        TransectDF[WaveProp] = [np.deg2rad(WaveDir) for WaveDir in TransectDF[WaveProp]]

    # Sort and remove duplicate dates
    def Preprocess(dates, values):
        df = pd.DataFrame({'dates': dates, 'values': values})
        # Group by unique dates and take the mean of duplicate values
        df_grouped = df.groupby('dates', as_index=False).mean()
        # If any nans still exist, fill them with PCHIP
        if np.isnan(df_grouped['values']).sum() > 0:
            df_grouped = df_grouped.interpolate(kind='pchip')
        # Extract clean dates and values
        unique_numdates = df_grouped['dates'].values
        unique_vals = df_grouped['values'].values
        
        return unique_numdates, unique_vals


    # Interpolate over waterline and wave associated variables
    wl_numdates = pd.to_datetime(TransectDF['wlDTs'][Tr]).values.astype(np.int64)
    ve_numdates = pd.to_datetime(TransectDF['veDTs'][Tr]).values.astype(np.int64)
    td_numdates = pd.to_datetime(TransectDF['tidedatesFD'][Tr]).values.astype(np.int64)
    wvsat_numdates = pd.to_datetime(TransectDF['WaveDates'][Tr]).values.astype(np.int64)
    wv_numdates = pd.to_datetime(TransectDF['WaveDatesFD'][Tr]).values.astype(np.int64)
    # Adjacent VE and WL positions (+1 is to the upcoast of current transect, -1 is to downcoast)
    wl_numdates_up = pd.to_datetime(CoastalDF.iloc[[Tr+1],:]['wlDTs'][Tr+1]).values.astype(np.int64)
    ve_numdates_up = pd.to_datetime(CoastalDF.iloc[[Tr+1],:]['veDTs'][Tr+1]).values.astype(np.int64)
    wl_numdates_down = pd.to_datetime(CoastalDF.iloc[[Tr-1],:]['wlDTs'][Tr-1]).values.astype(np.int64)
    ve_numdates_down = pd.to_datetime(CoastalDF.iloc[[Tr-1],:]['veDTs'][Tr-1]).values.astype(np.int64)
    
    for wvcol in ['WaveDirFD', 'WaveAlphaFD', 'WaveHsFD', 'WaveTpFD']:
        if IntpKind == 'pchip':
            wv_numdates_clean, wvcol_clean = Preprocess(wv_numdates, TransectDF[wvcol][Tr])
            wv_interp_f = PchipInterpolator(wv_numdates_clean, wvcol_clean)
        else:
            wv_interp_f = interp1d(wv_numdates, TransectDF[wvcol][Tr], kind=IntpKind, fill_value='extrapolate')
        wv_interp = wv_interp_f(wv_numdates).tolist()
        TransectDF[wvcol] = [wv_interp]
        
    # Match dates with veg edge dates and append back to TransectDF
    for wlcol in ['wlcorrdist', 'tideelev','beachwidth']:
        if IntpKind == 'pchip':
            wl_numdates_clean, wlcol_clean = Preprocess(wl_numdates, TransectDF[wlcol][Tr])
            wl_interp_f = PchipInterpolator(wl_numdates_clean, wlcol_clean)
        else:
            wl_interp_f = interp1d(wl_numdates, TransectDF[wlcol][Tr], kind=IntpKind, fill_value='extrapolate')
        wl_interp = wl_interp_f(wv_numdates).tolist()
        TransectDF[wlcol] = [wl_interp]
    for wvsatcol in ['Runups','Iribarren']:
        if IntpKind == 'pchip':
            wvsat_numdates_clean, wvsatcol_clean = Preprocess(wvsat_numdates, TransectDF[wvsatcol][Tr])
            wvsat_interp_f = PchipInterpolator(wvsat_numdates_clean, wvsatcol_clean)
        else:
            wvsat_interp_f = interp1d(wvsat_numdates, TransectDF[wvsatcol][Tr], kind=IntpKind, fill_value='extrapolate')
        wvsat_interp = wvsat_interp_f(wv_numdates).tolist()
        TransectDF[wvsatcol] = [wvsat_interp]
    for tdcol in ['tideelevFD','tideelevMx']:
        if IntpKind == 'pchip':
            td_numdates_clean, tdcol_clean = Preprocess(td_numdates, TransectDF[tdcol][Tr])
            td_interp_f = PchipInterpolator(td_numdates_clean, tdcol_clean)
        else:
            td_interp_f = interp1d(td_numdates, TransectDF[tdcol][Tr], kind=IntpKind, fill_value='extrapolate')
        td_interp = td_interp_f(wv_numdates).tolist()
        TransectDF[tdcol] = [td_interp]
    for vecol in ['distances']:#,'TZwidth']:
        if IntpKind == 'pchip':
            ve_numdates_clean, vecol_clean = Preprocess(ve_numdates, TransectDF[vecol][Tr])
            ve_interp_f = PchipInterpolator(ve_numdates_clean, vecol_clean)
        else:
            ve_interp_f = interp1d(ve_numdates, TransectDF[vecol][Tr], kind=IntpKind, fill_value='extrapolate')
        ve_interp = ve_interp_f(wv_numdates).tolist()
        TransectDF[vecol] = [ve_interp]
    
    # Adjacent VE and WL interpolation
    for wlcol in ['wlcorrdist']:
        if IntpKind == 'pchip':
            wl_numdates_clean, wlcol_clean = Preprocess(wl_numdates_up, CoastalDF.iloc[[Tr+1],:][wlcol][Tr+1])
            wl_interp_f = PchipInterpolator(wl_numdates_clean, wlcol_clean)
        else:
            wl_interp_f = interp1d(wl_numdates_up, CoastalDF.iloc[[Tr+1],:][wlcol][Tr+1], kind=IntpKind, fill_value='extrapolate')
        wl_interp = wl_interp_f(wv_numdates).tolist()
        TransectDF[wlcol+'_u'] = [wl_interp]
    for vecol in ['distances']:#,'TZwidth']:
        if IntpKind == 'pchip':
            ve_numdates_clean, vecol_clean = Preprocess(ve_numdates_up, CoastalDF.iloc[[Tr+1],:][vecol][Tr+1])
            ve_interp_f = PchipInterpolator(ve_numdates_clean, vecol_clean)
        else:
            ve_interp_f = interp1d(ve_numdates_up, CoastalDF.iloc[[Tr+1],:][vecol][Tr+1], kind=IntpKind, fill_value='extrapolate')
        ve_interp = ve_interp_f(wv_numdates).tolist()
        TransectDF[vecol+'_u'] = [ve_interp]
    for wlcol in ['wlcorrdist']:
        if IntpKind == 'pchip':
            wl_numdates_clean, wlcol_clean = Preprocess(wl_numdates_down, CoastalDF.iloc[[Tr-1],:][wlcol][Tr-1])
            wl_interp_f = PchipInterpolator(wl_numdates_clean, wlcol_clean)
        else:
            wl_interp_f = interp1d(wl_numdates_down, CoastalDF.iloc[[Tr-1],:][wlcol][Tr-1], kind=IntpKind, fill_value='extrapolate')
        wl_interp = wl_interp_f(wv_numdates).tolist()
        TransectDF[wlcol+'_d'] = [wl_interp]
    for vecol in ['distances']:#,'TZwidth']:
        if IntpKind == 'pchip':
            ve_numdates_clean, vecol_clean = Preprocess(ve_numdates_down, CoastalDF.iloc[[Tr-1],:][vecol][Tr-1])
            ve_interp_f = PchipInterpolator(ve_numdates_clean, vecol_clean)
        else:
            ve_interp_f = interp1d(ve_numdates_down, CoastalDF.iloc[[Tr-1],:][vecol][Tr-1], kind=IntpKind, fill_value='extrapolate')
        ve_interp = ve_interp_f(wv_numdates).tolist()
        TransectDF[vecol+'_d'] = [ve_interp]
    
    # Recalculate beachwidth as values will now be mismatched
    beachwidth = [abs(wl_interp[i] - TransectDF['distances'][Tr][i]) for i in range(len(wl_interp))]
    TransectDF['beachwidth'] = [beachwidth]
    
    TransectDF.drop(columns=['veDTs','wldates','wltimes', 'wlDTs','dates','times'], inplace=True)
    
    # Transpose to get columns of variables and rows of timesteps
    TransectDF = pd.DataFrame({col: pd.Series(val.iloc[0]) for col,val in TransectDF.items()})
    
    # Reset index for timeseries
    TransectDF.index = TransectDF['WaveDatesFD']
    TransectDF = TransectDF.drop(columns=['TransectID', 'WaveDates','WaveDatesFD', 'tidedatesFD'])

    # If any index rows (i.e. daily wave dates) are empty, remove them
    TransectDF = TransectDF[~pd.isnull(TransectDF.index)]
    
    

    return TransectDF


def DailyInterp(TransectDF):
    """
    Reindex and interpolate over timeseries to convert it to daily
    FM Nov 2024

    Parameters
    ----------
    VarDF : DataFrame
        Dataframe of per-transect coastal metrics/variables in (irregular) timeseries.

    Returns
    -------
    VarDFDay : DataFrame
        Dataframe of per-transect coastal metrics/variables in daily timesteps.
    """
    
    # Fill nans factoring in timesteps for interpolation
    TransectDF.replace([np.inf, -np.inf], np.nan, inplace=True)
    VarDF = TransectDF.interpolate(method='time', axis=0)
    # Strip time (just set to date timestamps), and take mean if any timesteps duplicated
    VarDF.index = VarDF.index.normalize()
    VarDF = VarDF.groupby(VarDF.index).mean()
    
    DailyInd = pd.date_range(start=VarDF.index.min(), end=VarDF.index.max(), freq='D')
    VarDFDay = VarDF.reindex(DailyInd)
    
    for col in VarDFDay.columns:
        if col == 'WaveDir' or col == 'WaveAlpha':
            # VarDFDay[col] = VarDFDay[col].interpolate(method='pchip')
            # Separate into components to allow interp over 0deg threshold
            # but need to preserve the offset between WaveDir and WaveAlpha
            VarDFsin = np.sin(VarDFDay[col])
            VarDFsininterp = VarDFsin.interpolate(method='pchip')
            VarDFcos = np.cos(VarDFDay[col])
            VarDFcosinterp = VarDFcos.interpolate(method='pchip')
            VarDFDay[col] = np.arctan2(VarDFsininterp, VarDFcosinterp) % (2 * np.pi) # keep values 0-2pi
        else:
            VarDFDay[col] = VarDFDay[col].interpolate(method='spline',order=1)
    
    return VarDFDay


def CreateSequences(X, y=None, time_steps=1):
    '''
    Function to create sequences (important for timeseries data where data point
    is temporally dependent on the one that came before it). Data sequences are
    needed for training RNNs, where temporal patterns are learned.
    FM June 2024

    Parameters
    ----------
    X : array
        Training data as array of feature vectors.
    y : array
        Training classes as array of binary labels.
    time_steps : int, optional
        Number of time steps over which to generate sequences. The default is 1.

    Returns
    -------
    array, array
        Numpy arrays of sequenced data

    '''
    Xs = []
    ys = []
    Ind = []
    if len(X) > time_steps:  # Check if there's enough data
        for i in range(len(X) - time_steps):
            # Slice feature set into sequences using moving window of size = number of timesteps
            Xs.append(X[i:(i + time_steps)])
            Ind.append(X.index[i + time_steps])
            if y is not None:
                ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys), np.array(Ind)
    else:
        # Not enough data to create a sequence
        print(f"Not enough data to create sequences with time_steps={time_steps}")
        return np.array([]), np.array([]), np.array([])


# ----------------------------------------------------------------------------------------
### MODEL INFRASTRUCTURE FUNCTIONS ###


def CostSensitiveLoss(falsepos_cost, falseneg_cost, binary_thresh):
    """
    Create a cost-sensitive loss function to implement within an NN model.compile() step.
    NN model is penalised based on cost matrix, which helps avoid mispredictions
    with disproportionate importance (i.e. a missed storm warning is more serious
    than a false alarm).
    FM June 2024

    Parameters
    ----------
    falsepos_cost : int
        Proportional weight towards false positive classification.
    falseneg_cost : int
        Proportional weight towards false negative classification.
    binary_thresh : float
        Value between 0 and 1 representing .

    Returns
    -------
    loss : function
        Calls the loss function when it is set within model.compile(loss=LossFn).

    """
    def loss(y_true, y_pred):
        # Flatten the arrays
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        
        # Convert predictions to binary class predictions
        y_pred_classes = K.cast(K.greater(y_pred, binary_thresh), tf.float32)
        
        # Calculate cost
        falsepos = K.sum(y_true * (1 - y_pred_classes) * falseneg_cost)
        falseneg = K.sum((1 - y_true) * y_pred_classes * falsepos_cost)
        
        bce = K.binary_crossentropy(y_true, y_pred)
        
        return bce + falsepos + falseneg
    
    return loss


def PrepData(TransectDF, MLabels, ValidSizes, TSteps, UseSMOTE=False):
    """
    Prepare features (X) and labels (y) for feeding into a NN for timeseries prediction.
    Timeseries is expanded and interpolated to get daily timesteps, then scaled across
    all variables, then split into model inputs and outputs before sequencing these
    
    FM Sept 2024
    
    Parameters
    ----------
    TransectDF : DataFrame
        Dataframe of per-transect coastal metrics/variables in timeseries.
    MLabels : list
        List of unique model run names.
    TestSizes : list, hyperparameter
        List of different proportions of data to separate out from training to validate model.
    TSteps : list, hyperparameter
        Number of timesteps (1 step = 1 day) to sequence training data over.
    UseSMOTE : bool, optional
        Flag for using SMOTE to oversample imbalanced data. The default is False.

    Returns
    -------
    PredDict : dict
        Dictionary to store all the NN model metadata.
    VarDFDay_scaled : DataFrame
        Scaled DataFrame of past data interpolated to daily timesteps (with temporal index), 
        for training and validation.
    VarDFDayTest_scaled : DataFrame
        Scaled DataFrame of past data interpolated to daily timesteps (with temporal index), 
        for testing (unseen by model).

    """
    
    # Interpolate over data gaps to get regular (daily) measurements
    # VarDFDay = DailyInterp(TransectDF)
    
    
    
    # Scale vectors to normalise them
    Scalings = {}
    TransectDF_scaled = TransectDF.copy()
    for col in TransectDF.columns:
        Scalings[col] = StandardScaler()
        TransectDF_scaled[col] = Scalings[col].fit_transform(TransectDF[[col]])
        
    VarDFDay_scaled = TransectDF_scaled.iloc[:int(len(TransectDF_scaled)*0.9)]
    VarDFDayTest_scaled = TransectDF_scaled.iloc[int(len(TransectDF_scaled)*0.9):]
    
    # Separate into training features (what will be learned from) and target features (what will be predicted)
    TrainFeat = VarDFDay_scaled[['tideelev', 'beachwidth', 'tideelevFD','tideelevMx',
                                 'WaveHsFD', 'WaveDirFD', 'WaveTpFD', 'WaveAlphaFD', 'Runups', 'Iribarren',
                                 'wlcorrdist_u', 'distances_u', 'wlcorrdist_d', 'distances_d']]
    # TrainFeat = VarDFDay_scaled[['WaveDir', 'Runups', 'Iribarren']]
    TargFeat = VarDFDay_scaled[['distances', 'wlcorrdist']] # vegetation edge and waterline positions
    
    # Define prediction dictionary for multiple runs/hyperparameterisation
    PredDict = {'mlabel':MLabels,   # name of the model run
                'model':[],         # compiled model (Sequential object)
                'history':[],       # training history to be written to later
                'loss':[],          # final loss value of run
                'accuracy':[],      # final accuracy value of run
                'train_time':[],    # time taken to train
                'seqlen':[],        # length of temporal sequence in timesteps to break data up into
                'X_train':[],       # training features/cross-shore values (scaled and filled to daily)
                'y_train':[],       # training target/cross-shore VE and WL (scaled and filled to daily)
                'X_val':[],         # validation features/cross-shore values
                'y_val':[],         # validation target/cross-shore VE and WL
                'scalings':[],      # scaling used on each feature (to transform them back to real values)
                'epochN':[],        # number of times the full training set is passed through the model
                'batchS':[],        # number of samples to use in one iteration of training
                'denselayers':[],   # number of dense layers in model construction
                'dropoutRt':[],     # percentage of nodes to randomly drop in training (avoids overfitting)
                'learnRt':[]}       # size of steps to adjust parameters by on each iteration
    
    for MLabel, ValidSize, TStep in zip(PredDict['mlabel'], ValidSizes, TSteps):
        
        # Add scaling relationships and sequence lengths to dict to convert back later
        PredDict['scalings'].append(Scalings)
        PredDict['seqlen'].append(TStep)
        
        # Create temporal sequences of data based on daily timesteps
        X, y, TrainInd = CreateSequences(TrainFeat, TargFeat, TStep)
        
        # Separate test and train data and add to prediction dict (can't stratify when y is multicolumn)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=ValidSize, random_state=0)#, stratify=y)
        PredDict['X_val'].append(X_val)
        PredDict['y_val'].append(y_val)
        
        # Use SMOTE for oversampling when dealing with imbalanced classification
        if UseSMOTE is True:
            smote = SMOTE()
            X_train_smote, y_train_smote = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
            X_train_smote = X_train_smote.reshape(X_train_smote.shape[0], X_train.shape[1], X_train.shape[2])
            # set back onto original sequenced features/labels
            PredDict['X_train'].append(X_train_smote)
            PredDict['y_train'].append(y_train_smote)
        else: # just use unsmoted sequenced data from above
            PredDict['X_train'].append(X_train)
            PredDict['y_train'].append(y_train)
            
    return PredDict, VarDFDay_scaled, VarDFDayTest_scaled


def CompileRNN(PredDict, epochNums, batchSizes, denseLayers, dropoutRt, learnRt, CostSensitive=False, DynamicLR=False):
    """
    Compile the NN using the settings and data stored in the NN dictionary.
    FM Sept 2024

    Parameters
    ----------
    PredDict : dict
        Dictionary to store all the NN model metadata.
    epochNums : list, hyperparameter
        List of different numbers of times a dataset passes through model in training.
    batchSizes : list, hyperparameter
        List of different numbers of samples to work through before internal model parameters are updated.
    CostSensitive : bool, optional
        Option for including a cost-sensitive loss function. The default is False.
    DynamicLR : bool, optional
        Option for including a dynamic learning rate in the model. The default is False.

    Returns
    -------
    PredDict : dict
        Dictionary to store all the NN model metadata, now with compiled models added.

    """
    for mlabel in PredDict['mlabel']:
        # Index of model setup
        mID = PredDict['mlabel'].index(mlabel)
        
        # Append hyperparameters to prediction dict
        PredDict['epochN'].append(epochNums[mID])
        PredDict['batchS'].append(batchSizes[mID])
        PredDict['denselayers'].append(denseLayers[mID])
        PredDict['dropoutRt'].append(dropoutRt[mID])
        PredDict['learnRt'].append(learnRt[mID])
        
        # inshape = (N_timesteps, N_features)
        inshape = (PredDict['X_train'][mID].shape[0], PredDict['X_train'][mID].shape[2])
        
        # GRU Model (3-layer)
        # Model = Sequential([
        #                        Input(shape=inshape), 
        #                        GRU(64, return_sequences=True),
        #                        Dropout(0.2),
        #                        GRU(64, return_sequences=True),
        #                        Dropout(0.2),
        #                        GRU(32),
        #                        Dropout(0.2),
        #                        Dense(1, activation='sigmoid')
        #                        ])
        
        # Number  of hidden layers can be decided by rule of thumb:
            # N_hidden = N_trainingsamples / (scaling * (N_input + N_output))
        N_out = 2
        N_hidden = round(inshape[0] / (5 * (inshape[1] + N_out)))
        
        # LSTM (1 layer)
        # Input() takes input shape, used for sequential models
        # LSTM() has dimension of (batchsize, timesteps, units) and only retains final timestep (return_sequences=False)
        # Dropout() randomly sets inputs to 0 during training to prevent overfitting
        # Dense() transforms output into 2 metrics (VE and WL)
        Model = Sequential([
                            Input(shape=inshape),
                            LSTM(units=N_hidden, activation='tanh', return_sequences=False),
                            Dense(PredDict['denselayers'][mID], activation='relu'),
                            Dropout(PredDict['dropoutRt'][mID]),
                            Dense(N_out) 
                            ])
        
        # Compile model and define loss function and metrics
        if DynamicLR:
            print('Using dynamic learning rate...')
            # Use a dynamic (exponentially decreasing) learning rate
            LRschedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.0001)
            Opt = tf.keras.optimizers.Adam(learning_rate=LRschedule)
        else:
            Opt = Adam(learning_rate=float(PredDict['learnRt'][mID]))
                       
        if CostSensitive:
            print('Using cost sensitive loss function...')
            # Define values for false +ve and -ve and create matrix
            falsepos_cost = 1   # Inconvenience of incorrect classification
            falseneg_cost = 100 # Risk to infrastructure by incorrect classification
            binary_thresh = 0.5
            LossFn = CostSensitiveLoss(falsepos_cost, falseneg_cost, binary_thresh)
        else:
            # Just use MSE loss fn and static learning rates
            LossFn = 'mse'
        
        Model.compile(optimizer=Opt, 
                         loss=LossFn, 
                         metrics=['accuracy','mae'])
        # Save model infrastructure to dictionary of model sruns
        PredDict['model'].append(Model)
    
    return PredDict
  

def TrainRNN(PredDict, filepath, sitename, EarlyStop=False):
    """
    Train the compiled NN based on the training data set aside for it. Results
    are written to PredDict which is saved to a pickle file. If TensorBoard is
    used as the callback, the training history is also written to log files for
    viewing within a TensorBoard dashboard.
    FM Sept 2024

    Parameters
    ----------
    PredDict : dict
        Dictionary to store all the NN model metadata.
    filepath : str
        Filepath to save the PredDict dictionary to (for reading the trained model back in).
    sitename : str
        Name of the site of interest.
    EarlyStop : bool, optional
        Flag to include early stopping to avoid overfitting. The default is False.

    Returns
    -------
    PredDict : dict
        Dictionary to store all the NN model metadata, now with trained NN models.

    """
    predictpath = os.path.join(filepath, sitename,'predictions')
    if os.path.isdir(predictpath) is False:
        os.mkdir(predictpath)
    tuningpath = os.path.join(predictpath,'tuning')
    if os.path.isdir(tuningpath) is False:
        os.mkdir(tuningpath)
    filedt = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    tuningdir = os.path.join(tuningpath, filedt)
    if os.path.isdir(tuningdir) is False:
        os.mkdir(tuningdir)
            
    # Define hyperparameters to log
    HP_EPOCHS = hp.HParam('epochs', hp.Discrete(PredDict['epochN']))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete(PredDict['batchS']))
    HP_DENSE_LAYERS = hp.HParam('dense_layers', hp.Discrete(PredDict['denselayers']))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete(PredDict['dropoutRt']))
    HP_LEARNRT = hp.HParam('learn_rate', hp.Discrete(PredDict['learnRt']))

    metric_loss = 'val_loss'
    metric_accuracy = 'val_accuracy'
    
    for MLabel in PredDict['mlabel']:
        print(f"Run: {MLabel}")
        # Index of model setup
        mID = PredDict['mlabel'].index(MLabel)
        Model = PredDict['model'][mID]
    
        # TensorBoard directory to save log files to
        rundir = os.path.join(tuningdir, MLabel)
        
        HPWriter = tf.summary.create_file_writer(rundir)

        if EarlyStop:
            # Implement early stopping to avoid overfitting
            ModelCallbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                              TensorBoard(log_dir=rundir)]
        else:
            # If no early stopping, just send log data to TensorBoard for analysis
            ModelCallbacks = [TensorBoard(log_dir=rundir)]
            
        start=time.time() # start timer
        # Train the model on the training data, writing results to TensorBoard
        History = Model.fit(PredDict['X_train'][mID], PredDict['y_train'][mID], 
                            epochs=PredDict['epochN'][mID], batch_size=PredDict['batchS'][mID],
                            validation_data=(PredDict['X_val'][mID],PredDict['y_val'][mID]),
                            callbacks=[ModelCallbacks],
                            verbose=0)
        end=time.time() # end time
        
        # Write evaluation metrics to TensorBoard for hyperparam tuning
        FinalLoss, FinalAccuracy, FinalMAE = Model.evaluate(PredDict['X_val'][mID], PredDict['y_val'][mID],verbose=0)

        PredDict['loss'].append(FinalLoss)
        PredDict['accuracy'].append(FinalAccuracy)
        
        print(f"Accuracy: {FinalAccuracy}")
        # Time taken to train model
        PredDict['train_time'].append(end-start)
        print(f"Train time: {end-start} seconds")
        
        PredDict['history'].append(History)
        
        # Log hyperparameters and metrics to TensorBoard
        with HPWriter.as_default():
            hp.hparams({
                HP_EPOCHS: PredDict['epochN'][mID],
                HP_BATCH_SIZE: PredDict['batchS'][mID],
                HP_DENSE_LAYERS:PredDict['denselayers'][mID],
                HP_DROPOUT: PredDict['dropoutRt'][mID],
                HP_LEARNRT: PredDict['learnRt'][mID]
            })
            tf.summary.scalar(metric_loss, FinalLoss, step=1)
            tf.summary.scalar(metric_accuracy, FinalAccuracy, step=1)
    
    # Save trained models in dictionary for posterity
    pklpath = os.path.join(predictpath, f"{filedt+'_'+'_'.join(PredDict['mlabel'])}.pkl")
    with open(pklpath, 'wb') as f:
        pickle.dump(PredDict, f)
            
    return PredDict


def FeatImportance(PredDict, mID):
    """
    Calculate feature importance from trained model, using Integrated Gradients.
    FM Feb 2025
    
    Parameters
    ----------
    PredDict : dict
        Dictionary to store all the NN model metadata, now with trained NN models.
    mID : int
        ID of the chosen model run stored in PredDict.

    Returns
    -------
    IntGradAttr : array
        Array of integrated gradient values for chosen sequence of training features.

    """
    
    def IntGrads(model, baseline, input_sample, steps=50):
        """Computes Integrated Gradients for a single input sample."""
        
        alphas = tf.linspace(0.0, 1.0, steps)  # Interpolation steps
    
        # Compute the interpolated inputs
        interp_inputs = baseline + alphas[:, tf.newaxis, tf.newaxis] * (input_sample - baseline)
        # Compute gradients at each interpolation step
        with tf.GradientTape() as tape:
            tape.watch(interp_inputs)
            preds = model(interp_inputs)
        gradients = tape.gradient(preds, interp_inputs)
    
        # Average the gradients and multiply by input difference
        avg_gradients = tf.reduce_mean(gradients, axis=0)
        return (input_sample - baseline) * avg_gradients
    
    # Define the variables
    Model = PredDict['model'][mID]
    # validation data used because goal is to explain what model learned, not robustness of model (yet)
    X_val = PredDict['X_val'][mID]

    # Define an all-zero baseline to start at
    Baseline = tf.zeros_like(X_val[-1:], dtype=tf.float32)
    # Feed the most recent validation data to test
    InSample = tf.convert_to_tensor(X_val[-1:], dtype=tf.float32)
    
    # Compute Integrated Gradients
    IntGradAttr = IntGrads(Model, Baseline, InSample)
    # Convert to NumPy for interpretation
    IntGradAttr = IntGradAttr.numpy()

    return IntGradAttr


def SHAPTest(PredDict, mID):
    """
    IN DEVELOPMENT
    FM Feb 2025

    Parameters
    ----------
    PredDict : dict
        Dictionary to store all the NN model metadata, now with trained NN models.
    mID : int
        ID of the chosen model run stored in PredDict.

    Returns
    -------
    None.

    """
    from tensorflow import keras
    
    
    model = keras.models.load_model(PredDict['model'][mID])
    X_train = PredDict['X_train'][mID]
    X_test = PredDict['X_val'][mID]

    shap.initjs()
    
    # # Select a smaller subset of training data for SHAP
    # sample_idx = np.random.choice(X_train.shape[0], 100, replace=False)  # 100 random samples
    # X_train_sample = X_train[sample_idx]  

    # Use SHAP's DeepExplainer with the function-wrapped model
    explainer = shap.GradientExplainer(model, X_train)
    
    # Compute SHAP values for the test set
    shap_values = explainer.shap_values(X_test)
    
    # Reshape X_test for SHAP summary plot (flatten time steps)
    shap.summary_plot(shap_values[0], X_test.reshape((607, -1)))


    # return shap_values


def TrainRNN_Optuna(PredDict, mlabel):
    # Custom RMSE metric function
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def set_seed(seed):
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Index of model setup
    mID = PredDict['mlabel'].index(mlabel)
    # inshape = (N_timesteps, N_features)
    inshape = (PredDict['X_train'][mID].shape[0], PredDict['X_train'][mID].shape[2])    

    # Define the LSTM model (Input No of layers should be atleast 2, including Dense layer, the inputs are initial values only)
    def CreateModel(learning_rate, batch_size, num_layers, num_nodes, dropout_rate, epochs, xtrain, ytrain, xtest, ytest):
        set_seed(42)
        min_delta = 0.001

        model = Sequential()
        model.add(Input(inshape))
        model.add(LSTM(num_nodes, activation='relu', return_sequences=True))
        model.add(Dropout(dropout_rate))
        
        for _ in range(num_layers-2):
            model.add(LSTM(num_nodes, return_sequences=True))
            model.add(Dropout(dropout_rate))
            
        model.add(LSTM(num_nodes))
        model.add(Dropout(dropout_rate)) 
        model.add(Dense(2))
                
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics=[root_mean_squared_error])
        
        # Train model with early stopping callback
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=min_delta, restore_best_weights=True)
        
        history = model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, 
                            validation_data=(xtest, ytest), verbose=1, shuffle = False, callbacks=[early_stopping_callback])
    
        return model, history

    # Set to store unique hyperparameter configurations
    hyperparameter_set = set()

    # For Optuna, it is required to specify a Objective function which it tries to minimise (finding the global(?) minima). Here it is loss (RMSE) on validation set
    def objective(trial):
        # Define hyperparameters to optimize with Optuna
        # Set seed for os env
        os.environ['PYTHONHASHSEED'] = str(42)
        # Set random seed for Python
        random.seed(42)
        # Set random seed for NumPy
        np.random.seed(42)
        # Set random seed for TensorFlow
        tf.random.set_seed(42)
        
        ## If you want to use a parameter space and pickup hyperpameter combos randomly. Bit time consuming
        # learning_rate = trial.suggest_float('learning_rate', 1e-4, 2e-3, log=True)
        # num_layers = trial.suggest_int('num_layers', 2, 4)
        # dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
        # num_nodes = trial.suggest_int('num_nodes', 100, 300)
        # batch_size = trial.suggest_int('batch_size', 24, 92)
        # epochs = trial.suggest_int('epochs', 20, 100)
    
        # If you want to specify certain values of parameters and provide as a grid of hyperparameters (optuna samplers will pickup hyperpameter combos randomly from the grid). Bit time consuming
        learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.005, 0.01])
        num_layers = trial.suggest_categorical('num_layers', [2, 3, 4])
        dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.2, 0.4])
        num_nodes = trial.suggest_categorical('num_nodes', [50, 100, 200, 300])
        batch_size = trial.suggest_categorical('batch_size', [24, 32, 64, 92])
        epochs = trial.suggest_categorical('epochs', [50, 80, 100, 200])
        
        # Create a tuple of the hyperparameters
        hyperparameters = (learning_rate, num_layers, dropout_rate, num_nodes, batch_size, epochs)
    
        # Check if this set of hyperparameters has already been tried
        if hyperparameters in hyperparameter_set:
            raise optuna.exceptions.TrialPruned()  # Skip this trial and suggest a new set of hyperparameters
    
        # Add the hyperparameters to the set
        hyperparameter_set.add(hyperparameters)
    
        # Make a prediction using the LSTM model for the validation set
        optimized_model, history_HPO = CreateModel(learning_rate, batch_size, num_layers, num_nodes, dropout_rate, epochs, 
                                                   PredDict['X_train'][mID], PredDict['y_train'][mID], 
                                                   PredDict['X_val'][mID], PredDict['y_val'][mID])

        ypredict = optimized_model.predict(PredDict['X_val'][mID])
        val_loss = root_mean_squared_error(PredDict['y_val'][mID], ypredict)
        
        return val_loss # Returning the Objective function value. This will be used to optimise, through creating a surrogate model for the number of trial runs
        
    # Define the pruning callback
    pruner = optuna.pruners.MedianPruner()
    
    # Create Optuna study with Bayesian optimization (TPE) and trial pruning. There is Gausian sampler (and so many other sampling options too)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(), pruner=pruner)
    
    # Optimize hyperparameters
    study.optimize(objective, n_trials=50)
    
    # Retrieve best hyperparameters
    print("Best Hyperparameters:", study.best_params)
    
    # Save the study object (If required. So that it can be loaded as in the next cell without needing to re-run this whole HPO)
    # filename = f'optuna_study_SLP_H_weekly-HsTp-LSTM{n_steps}.pkl'
    # joblib.dump(study, filename)

    return study



def RunsToCSV(tuningdir,outputCSV):
    
    AllData = []
    
    # Iterate through all subdirectories (each subdirectory corresponds to a run)
    for rundir in os.listdir(tuningdir):
        run_path = os.path.join(tuningdir, rundir, 'validation')
        if os.path.isdir(run_path):
            # Load the TensorBoard event data
            event_acc = EventAccumulator(run_path)
            event_acc.Reload()
            
            # Extract scalar data for each metric
            for tag in event_acc.Tags()["scalars"]:
                scalar_events = event_acc.Scalars(tag)
                for event in scalar_events:
                    AllData.append({
                        "run": rundir,
                        "tag": tag,
                        "step": event.step,
                        "value": event.value,
                        "wall_time": event.wall_time
                    })
    
    # Convert all collected data into a single DataFrame
    df = pd.DataFrame(AllData)
    
    outputPath = os.path.join(tuningdir, outputCSV)
    # Extract and save to CSV
    df.to_csv(outputPath, index=False)
    print(f"Data saved to {outputPath}")
    

def FuturePredict(PredDict, ForecastDF):
    """
    Make prediction of future vegetation edge and waterline positions for transect
    of choice, using forecast dataframe as forcing and model pre-trained on past 
    observations.
    FM Nov 2024

    Parameters
    ----------
    PredDict : dict
        Dictionary to store all the NN model metadata, now with trained NN models.
    ForecastDF : DataFrame
        Per-transect dataframe of future observations, with same columns as past training data.

    Returns
    -------
    FutureOutputs : dict
        Dict storing per-model dataframes of future cross-shore waterline and veg edge predictions

    """
    # Initialise for ulti-run outputs
    FutureOutputs = {'mlabel':PredDict['mlabel'],
                     'output':[]}
    
    # For each trained model/hyperparameter set in PredDict
    for MLabel in PredDict['mlabel']:
        # Index of model setup
        mID = PredDict['mlabel'].index(MLabel)
        Model = PredDict['model'][mID]
        
        # Scale forecast data using same relationships as past training data
        for col in ForecastDF.columns:
            ForecastDF[col] = PredDict['scalings'][mID][col].transform(ForecastDF[[col]])
        
        # Separate out forecast features
        # ForecastFeat = ForecastDF[['WaveHs', 'WaveDir', 'WaveTp', 'WaveAlpha', 'Runups', 'Iribarrens']]
        ForecastFeat = ForecastDF[['WaveDir', 'Runups', 'Iribarrens']]
        
        # Sequence forecast data to shape (samples, sequencelen, variables)
        ForecastArr, _, ForecastInd = CreateSequences(X=ForecastFeat, time_steps=PredDict['seqlen'][mID])
        
        # Make prediction based off forecast data and trained model
        Predictions = Model.predict(ForecastArr)
        
        # Reverse scaling to get outputs back to their original scale
        VEPredict = PredDict['scalings'][mID]['distances'].inverse_transform(Predictions[:,0].reshape(-1, 1))
        WLPredict = PredDict['scalings'][mID]['wlcorrdist'].inverse_transform(Predictions[:,1].reshape(-1, 1))
        
        FutureDF = pd.DataFrame(
                   {'futureVE': VEPredict.flatten(),
                    'futureWL': WLPredict.flatten()},
                   index=ForecastInd)
        
        FutureOutputs['output'].append(FutureDF)
        
    return FutureOutputs


# ----------------------------------------------------------------------------------------
### PLOTTING FUNCTIONS ###


def PlotInterps(CoastalDF, Tr, FigPath):
    """
    Plot results of different scipy interpolation methods.
    FM Feb 2025

    Parameters
    ----------
    TransectDF : DataFrame
        Dataframe of per-transect coastal metrics/variables in timeseries.
    FigPath : str
        Path to save figure to.

    """
    
    TransectDF = pd.DataFrame(CoastalDF['WaveHsFD'].iloc[Tr], 
                              index=CoastalDF['WaveDatesFD'].iloc[Tr],
                              columns=['WaveHsFD'])
    
    Mthds = ['nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial', 'piecewise_polynomial', 'spline', 'pchip', 'akima', 'cubicspline', 'from_derivatives']
    fig, axs = plt.subplots(4,3, sharex=True, figsize=(6.55,5),dpi=300)
    axs=axs.flatten()
    for i, Mthd in enumerate(Mthds):

        if Mthd in ['polynomial','spline']:
            # Ord = pd.isnull(TransectDF['WaveHsFD']).sum() - 1
            Ord = 5
            TransectDFInterp = TransectDF.interpolate(method=Mthd, order=Ord, axis=0)
            axs[i].set_title(Mthd+', order='+str(Ord), pad=1)
        else:
            TransectDFInterp = TransectDF.interpolate(method=Mthd, axis=0)
            axs[i].set_title(Mthd, pad=1)

        axs[i].plot(TransectDFInterp['WaveHsFD'][3280:], c='#46B0E1', lw=1.5)
        axs[i].plot(TransectDF['WaveHsFD'][3280:], c='#1559A0', lw=1.5)
        
        axs[i].xaxis.set_major_locator(mdates.MonthLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        
        if Mthd in ['polynomial','akima','cubicspline','quadratic', 'cubic']:
            axs[i].text(TransectDFInterp['WaveHsFD'][3280:][TransectDFInterp['WaveHsFD'][3280:] == TransectDFInterp['WaveHsFD'][3280:].min()].index-timedelta(days=5),
                        TransectDFInterp['WaveHsFD'][3280:].min()+0.05, 
                        'undershooting', color='r', ha='right', va='center')
        # axs[i].set_ylim((-1,1))
            
    fig.supxlabel('Date')
    fig.supylabel('Wave height (m)')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)     


def PlotAccuracy(CSVdir, FigPath):
    # List to store each CSV's DataFrame
    dfs = []
    
    # Loop through all files in the folder
    for filename in os.listdir(CSVdir):
        if filename.endswith('.csv'):
            # Full path to the CSV
            file_path = os.path.join(CSVdir, filename)
            
            # Read the CSV into a DataFrame
            df = pd.read_csv(file_path)
            df.drop(['Wall time'], axis=1, inplace=True)
            
            # Rename columns to standardize (x and y for this script)
            df.columns = ['x', 'y']
            # Set x as the index
            df.set_index('x', inplace=True)
            
            # Rename the y column to the filename (without extension)
            column_name = os.path.splitext(filename)[0]
            df.rename(columns={'y': column_name}, inplace=True)
            # Add the DataFrame to the list
            dfs.append(df)
    
    # Merge all DataFrames on the x-values index
    # This will align rows by index and fill missing values with NaN
    AccuracyDF = pd.concat(dfs, axis=1)
    
    plt.figure(figsize=(3.22,1.72))
    plt.plot(AccuracyDF, c='w', alpha=0.5)
    plt.plot(AccuracyDF['dense64_validation'], c='#283252')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=True)
    
    return AccuracyDF


def PlotVarTS(TransectDF, Tr, filepath, sitename):
    """
    Plot timeseries of training variables used.
    FM Feb 2025

    Parameters
    ----------
    Tr : int
        ID of the chosen cross-shore transect.
    TransectDF : DataFrame
        DataFrame of past data to use for training (and validating) the model.
    VarDFDay : DataFrame
        DataFrame of past data interpolated to daily timesteps (with temporal index).
    FutureOutputs : dict
        Dict storing per-model dataframes of future cross-shore waterline and veg edge predictions.
    filepath : str
        Local path to COASTGUARD Data folder.
    sitename : str
        Name of site of interest.


    """
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams.update({'font.size':7})
    
    # Scale vectors to normalise them
    Scalings = {}
    TransectDF_sc = TransectDF.copy()
    for col in TransectDF.columns:
        Scalings[col] = StandardScaler()
        TransectDF_sc[col] = Scalings[col].fit_transform(TransectDF_sc[[col]])
    
    TransectDFTrain = TransectDF_sc.iloc[:int(len(TransectDF_sc)*0.9)]
    TransectDFTest = TransectDF_sc.iloc[int(len(TransectDF_sc)*0.9):]
    
    # Clip train data down into train and validation (have to do now because it gets done later in PrepData())
    TransectDFVal = TransectDFTrain[len(TransectDFTrain)-round((len(TransectDFTrain)+len(TransectDFTest))*0.2):]
    TransectDFTrain = TransectDFTrain[:len(TransectDFTrain)-len(TransectDFVal)]
    
    # set subplot spacing with small break between start and end of training
    gridspec = dict(wspace=0.0, width_ratios=[1, 0.1, 1, 1])
    fig, axs = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(4.31, 1.50), dpi=300, gridspec_kw=gridspec)
    axs[1].set_visible(False)
    
    # TRAIN DATA (with split)
    # plot the same data on both axes
    axs[0].plot(TransectDFTrain, c='#0E2841', lw=0.5, alpha=0.3)
    axs[2].plot(TransectDFTrain, c='#0E2841', lw=0.5, alpha=0.3)
    #pd.concat([TransectDFTrain[:int(len(TransectDFTrain)/2)],TransectDFVal])
    # zoom-in / limit the view to different portions of the data
    axs[0].set_xlim(TransectDFTrain.index.min(), TransectDFTrain.index[700])
    axs[2].set_xlim(TransectDFTrain.index[len(TransectDFTrain)-700],TransectDFTrain.index.max())  # most of the data
    # set yearly labels and monthly ticks
    for ax in axs:
        ax.set_ylim(-2,6)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        # ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    # diagonal separator marks
    axs[0].text(axs[0].get_xlim()[1],axs[0].get_ylim()[0],'/', ha='center', va='center', fontweight='bold')
    axs[0].text(axs[0].get_xlim()[1],axs[0].get_ylim()[1],'/', ha='center',va='center', fontweight='bold')
    axs[2].text(axs[2].get_xlim()[0],axs[2].get_ylim()[0],'/', ha='center',va='center', fontweight='bold')
    axs[2].text(axs[2].get_xlim()[0],axs[2].get_ylim()[1],'/', ha='center',va='center', fontweight='bold')    
    # hide the spines between ax and ax2
    axs[0].spines['right'].set_visible(False)
    axs[2].spines['left'].set_visible(False)
    axs[0].tick_params(axis='y', which='both',left=False, labelleft=False)  # don't put tick labels
    axs[2].tick_params(axis='y', which='both',left=False, labelleft=False)  # don't put tick labels
    
    # VALIDATION DATA
    axs[3].plot(TransectDFVal, c='#0E2841', lw=0.5, alpha=0.3)
    axs[3].tick_params(axis='y', which='both',left=False, labelleft=False)  # don't put tick labels
    axs[3].set_xlim(TransectDFVal.index.min(), TransectDFVal.index.max())
    
    # plt.xlabel('Date (yyyy)')       
    plt.tight_layout()
    plt.show()
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_TrainValVars_Tr'+str(Tr)+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=True)
    
    
    # set subplot spacing with small break between start and end of training
    gridspec = dict(wspace=0.0, width_ratios=[0.75, 0.1, 0.75, 0.7, 0.35])
    fig, axs = plt.subplots(nrows=1, ncols=5, sharey=True, figsize=(2.61, 1.50), dpi=300, gridspec_kw=gridspec)
    axs[1].set_visible(False)
    
    SelVars = ['distances', 'wlcorrdist','WaveHsFD', 'WaveDirFD', 'WaveTpFD']
    # TRAIN DATA (with split)
    # plot the same data on both axes
    axs[0].plot(TransectDFTrain[SelVars], c='#0E2841', lw=0.5, alpha=0.3)
    axs[2].plot(TransectDFTrain[SelVars], c='#0E2841', lw=0.5, alpha=0.3)
    #pd.concat([TransectDFTrain[:int(len(TransectDFTrain)/2)],TransectDFVal])
    # zoom-in / limit the view to different portions of the data
    axs[0].set_xlim(TransectDFTrain.index.min(), TransectDFTrain.index[700])
    axs[2].set_xlim(TransectDFTrain.index[len(TransectDFTrain)-700],TransectDFTrain.index.max())  # most of the data
    # set yearly labels and monthly ticks
    for ax in axs:
        ax.set_ylim(-2,6)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        # ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    # diagonal separator marks
    axs[0].text(axs[0].get_xlim()[1],axs[0].get_ylim()[0],'/', ha='center', va='center', fontweight='bold')
    axs[0].text(axs[0].get_xlim()[1],axs[0].get_ylim()[1],'/', ha='center',va='center', fontweight='bold')
    axs[2].text(axs[2].get_xlim()[0],axs[2].get_ylim()[0],'/', ha='center',va='center', fontweight='bold')
    axs[2].text(axs[2].get_xlim()[0],axs[2].get_ylim()[1],'/', ha='center',va='center', fontweight='bold')    
    # hide the spines between ax and ax2
    axs[0].spines['right'].set_visible(False)
    axs[2].spines['left'].set_visible(False)
    axs[0].tick_params(axis='y', which='both',left=False, labelleft=False)  # don't put tick labels
    axs[2].tick_params(axis='y', which='both',left=False, labelleft=False)  # don't put tick labels
    
    # VALIDATION DATA
    axs[3].plot(TransectDFVal[SelVars], c='#0E2841', lw=0.5, alpha=0.3)
    axs[3].tick_params(axis='y', which='both',left=False, labelleft=False)  # don't put tick labels
    axs[3].set_xlim(TransectDFVal.index.min(), TransectDFVal.index.max())
    
    # TEST DATA
    axs[4].plot(TransectDFTest[SelVars], c='#0E2841', lw=0.5, alpha=0.3)
    axs[4].tick_params(axis='y', which='both',left=False, labelleft=False)  # don't put tick labels
    axs[4].set_xlim(TransectDFTest.index.min(), TransectDFTest.index.max())
    
    # plt.xlabel('Date (yyyy)')       
    plt.tight_layout()
    plt.show()
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_TrainValTestVars_Tr'+str(Tr)+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=True)


def PlotStormWaveHs(TransectDF, CoastalDFTr, filepath, sitename):
    
    BabetTransect = TransectDF.loc['2023-09-28 00:00:00':'2023-12-05 00:00:00']
    BabetVEs = pd.DataFrame({'veDTs':CoastalDFTr['veDTs'],
                             'distances':CoastalDFTr['distances']})
    BabetWLs = pd.DataFrame({'wlDTs':CoastalDFTr['wlDTs'],
                             'wlcorrdist':CoastalDFTr['wlcorrdist']})
    BabetVEs = BabetVEs.groupby('veDTs').mean()
    BabetWLs = BabetWLs.groupby('wlDTs').mean()
    
    fig, ax = plt.subplots(1,1, figsize=(3.35,1.94), dpi=300)

    rectwidth = mdates.date2num(datetime(2023,10,21)) - mdates.date2num(datetime(2023,10,18))
    rect = mpatches.Rectangle((mdates.date2num(datetime(2023,10,18)), -200), rectwidth, 1000, 
                              fc=[0.3,0.3,0.3], ec=None, alpha=0.2)
    ax.add_patch(rect)
    
    ax.plot(BabetTransect['distances'], c='#79C060', label='VE')
    ax.scatter(BabetVEs.index, BabetVEs['distances'], s=15, marker='x', c='#79C060')
    ax.plot(BabetTransect['wlcorrdist'], c='#3E74B3', label='WL')
    ax.scatter(BabetWLs.index, BabetWLs['wlcorrdist'], s=15, marker='x', c='#3E74B3')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.set_xlabel('Date (2023-mm-dd)')
    # ax.set_xticklabels([date.day for date in BabetTransect.index])
    ax.set_xlim(pd.Timestamp('2023-09-28 00:00:00'), pd.Timestamp('2023-12-02 00:00:00'))
    ax.set_ylim(0, 500)
    ax.legend(loc='upper left')
    ax.text(x=mdates.date2num(datetime(2023,10,18)), y=ax.get_ylim()[0]+((ax.get_ylim()[1]-ax.get_ylim()[0])/2),
            s='Storm Babet', rotation=90, ha='right', va='center', alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(BabetTransect['WaveHsFD']+BabetTransect['tideelevFD'], c='#163E64', label='Significant wave height')
    ax2.set_ylim(0,4)
    ax2.legend(loc='upper right')
    
    ax.set_ylabel('Cross-shore distance (m)', labelpad=1)
    ax2.set_ylabel('Wave height (m)')
    
    
    plt.tight_layout()
    plt.show()
    
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_StormWavesVEWL_Tr'+str(CoastalDFTr.name)+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=True)
    

def PlotIntGrads(PredDict, VarDFDayTrain, IntGradAttr, filepath, sitename, Tr, enddate=None):
    """
    lot integrated gradient values for feature importance analysis.
    FM Feb 2025

    Parameters
    ----------
    PredDict : dict
        Dictionary to store all the NN model metadata, now with trained NN models.
    VarDFDayTrain : DataFrame
        Scaled DataFrame of past data interpolated to daily timesteps (with temporal index), 
        for training and validation.
    IntGradAttr : array
        Array of integrated gradient values for chosen sequence of training features.
    filepath : str
        Local path to COASTGUARD Data folder.
    sitename : str
        Name of site of interest.
    Tr : int
        Transect ID of interest.

    """
    # Get the date index for the X_train and X_val data
    ValDates = VarDFDayTrain.index[len(VarDFDayTrain)-len(PredDict['X_val'][0]):]  
    if enddate is None:
        # Find the start date of the last sequence (most recent)
        start_idx = len(ValDates) - 10  # Last 10-day sequence starts 10 days before the end
    else:
        # TO DO: give different dates
        start_idx = enddate
    ValTimestamps = ValDates[start_idx:]  # Get the last 10 days
    
    # Plot importance for each timestep
    gridspec = dict(width_ratios=[0.6, 1, 0.6])
    
    fig, axs = plt.subplots(1,3, figsize=(6.55, 2.6), dpi=300, gridspec_kw=gridspec) 
    ylimits = (0, 0.45)
    cmap = plt.get_cmap('PuBuGn')
    
    # Create heatmap
    # Remove batch dimension
    IntGradAttr_indiv = np.abs(IntGradAttr[0])
    # Swap position of beachwidth (1 to 10)
    IntGradAttr_indiv = IntGradAttr_indiv[:, [0,2,3,4,6,5,7,8,9,1,10,11,12,13]]
    
    # Create line plot of global importance
    norm = plt.Normalize(ylimits[0],ylimits[1])
    linec = cmap(norm(np.mean(np.abs(IntGradAttr),axis=-1).flatten()))
    # Plot global importance line
    axs[0].plot(range(IntGradAttr.shape[1]), np.mean(np.abs(IntGradAttr),axis=-1).flatten(), 
                  c='k', zorder=0)
    axs[0].scatter(range(IntGradAttr.shape[1]), np.mean(np.abs(IntGradAttr),axis=-1).flatten(), 
                   c=linec, marker='o', edgecolors='k', zorder=1)
    axs[0].set_xlabel(f"Date ({ValTimestamps[0].strftime('%Y-%m-%d')[:8]+'dd'})")
    axs[0].set_xticks(range(10))
    axs[0].set_xticklabels([date.day for date in ValTimestamps])
    axs[0].set_ylim(ylimits)
    axs[0].yaxis.tick_right()
    axs[0].set_title('(a) Global Integrated Gradients', fontsize=7)
    ax0inset = inset_axes(axs[0], width='70%', height='45%', loc='upper left', borderpad=0)
    ax0inset.plot(range(IntGradAttr.shape[1])[:7], np.mean(np.abs(IntGradAttr),axis=-1).flatten()[:7], 
                  c='k', zorder=0)
    ax0inset.scatter(range(IntGradAttr.shape[1])[:7], np.mean(np.abs(IntGradAttr),axis=-1).flatten()[:7], 
                   c=linec[:7], marker='o', edgecolors='k', zorder=1)
    ax0inset.set_xticks(range(7))
    ax0inset.set_xticklabels([date.day for date in ValTimestamps[:7]])
    ax0inset.set_xlim(ax0inset.get_xlim()[0]-0.2, ax0inset.get_xlim()[1]+0.2)
    ax0inset.set_ylim(0,0.02)
    ax0inset.yaxis.tick_right()
    ax0inset.tick_params(axis='y', which='major', pad=0.2)
    ax0inset.set_yticks([0, 0.005, 0.01, 0.015, 0.02])
    ax0inset.set_yticklabels(['{:.2f}'.format(x) for x in ax0inset.get_yticks()])
    for label in [ax0inset.yaxis.get_ticklabels()[1], ax0inset.yaxis.get_ticklabels()[3], ax0inset.yaxis.get_ticklabels()[4]]:
        label.set_visible(False)
    
    # Plot heatmap
    cax = axs[1].imshow(IntGradAttr_indiv.T, aspect='auto', cmap=cmap,
                        vmin=ylimits[0], vmax=ylimits[1])
    cbar = fig.colorbar(cax, location='left',pad=0.22)
    cbar.set_label('Feature Importance')
    axs[1].set_title('(b) Individual Integrated Gradients', fontsize=7)
    axs[1].set_xlabel(f"Date ({ValTimestamps[0].strftime('%Y-%m-%d')[:8]+'dd'})")
    axs[1].set_xticks(range(10))
    axs[1].set_xticklabels([date.day for date in ValTimestamps])
    # FeatNames = list(VarDFDayTrain.columns[2:])
    # sat tides, mean daily tides, max daily tides
    # wave height, wave period, wave direction, wave alpha,
    # runup, iribarren, beach width
    # upcoast WL, upcoast VE, downcoast WL, downcoast VE
    FeatNames = [r'$z_{tide,sat}$', r'$\bar{z}_{tide}$', r'$z^{*}_{tide}$',
                 r'$H_{s}$', r'$T_{p}$', r'$\bar\theta$', r'$\alpha$',
                 r'$R_{2}$', r'$\xi_{0}$', r'$d_{VE,WL}$',
                 r'$WL_{u}$', r'$VE_{u}$', r'$WL_{d}$', r'$VE_{d}$']
    axs[1].set_yticks(range(len(FeatNames)))
    axs[1].set_yticklabels(FeatNames)
    
    # Add smaller heatmap of just wave data
    cax = axs[2].imshow(IntGradAttr_indiv[:,3:8].T, aspect='equal', cmap=cmap,
                        vmin=0, vmax=0.05)
    cbar = fig.colorbar(cax, location='bottom', pad=0.2, label='Feature Importance')
    axs[2].set_title('(c) Wave Integrated Gradients', fontsize=7)
    axs[2].set_yticks(range(len(FeatNames[3:8])))
    axs[2].set_yticklabels(FeatNames[3:8])
    axs[2].set_xlabel(f"Date ({ValTimestamps[0].strftime('%Y-%m-%d')[:8]+'dd'})")
    axs[2].set_xticks(range(10))
    axs[2].set_xticklabels([date.day for date in ValTimestamps])
    # axs[2].set_box_aspect(0.9)
    pos = axs[2].get_position()  # Get current position
    axs[2].set_position([pos.x0, pos.y0, pos.width, pos.height])# axs[2].set_position([0.8, 1, 0.1, 0.1])
    
    # add connecting lines to inset plot
    # top line
    fig.add_artist(ConnectionPatch(
    xyA=(axs[2].get_xlim()[0], axs[2].get_ylim()[1]), coordsA=axs[2].transData,
    xyB=(1,11/len(IntGradAttr_indiv.T)), coordsB=axs[1].transAxes,
    color='k', alpha=0.3, linewidth=0.5))   
    # bottom line
    fig.add_artist(ConnectionPatch(
    xyA=(axs[2].get_xlim()[0], axs[2].get_ylim()[0]), coordsA=axs[2].transData,
    xyB=(1,6/len(IntGradAttr_indiv.T)), coordsB=axs[1].transAxes,
    color='k', alpha=0.3, linewidth=0.5))   
    
    # ax_labels = list(string.ascii_lowercase[:axs.shape[0]])
    # for ax, lab in zip(axs.flat, ax_labels):
    #     if lab == 'c':
    #         ypos = 1.3
    #     else:
    #         ypos = 1.08
    #     if lab=='a':
    #         xpos = 1.04
    #     else:
    #         xpos=-0.13
    #     ax.text(xpos, ypos, '('+lab+')', transform=ax.transAxes,
    #         fontsize=6, va='top', bbox=dict(facecolor='w', edgecolor='k',pad=1.5))
    
    # plt.subplots_adjust(wspace=-0.5)
    plt.tight_layout()
    plt.show()

    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_FeatureImportance_Tr'+str(Tr)+'_'+str(ValTimestamps[0].date())+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=True)



def PlotFuture(mID, VarDFDay, TransectDFTest, FutureOutputs, filepath, sitename):
    """
    Plot future waterline (WL) and vegetation edge (VE) predictions for the 
    chosen cross-shore transect.
    FM Jan 2025

    Parameters
    ----------
    mID : int
        ID of the chosen model run stored in FutureOutputs.
    VarDFDay : DataFrame
        DataFrame of past data interpolated to daily timesteps (with temporal index).
    TransectDFTest : DataFrame
        DataFrame of past data sliced from most recent end of TransectDF to use for testing the model.
    FutureOutputs : dict
        Dict storing per-model dataframes of future cross-shore waterline and veg edge predictions.
    filepath : str
        Local path to COASTGUARD Data folder.
    sitename : str
        Name of site of interest.


    """
    fig, ax = plt.subplots(1,1, figsize=(3.3,3.35), dpi=300)
    
    TrainStart = mdates.date2num(VarDFDay.index[0])
    TrainEnd = mdates.date2num(VarDFDay.index[round(len(VarDFDay)-(len(VarDFDay)*0.2))])
    ValEnd = mdates.date2num(VarDFDay.index[-1])
    TestEnd = mdates.date2num(TransectDFTest.index[-1])
    TrainT = mpatches.Rectangle((TrainStart,-100), TrainEnd-TrainStart, 1000, fc=[0.8,0.8,0.8], ec=None)
    ValT = mpatches.Rectangle((TrainEnd,-100), ValEnd-TrainEnd, 1000, fc=[0.9,0.9,0.9], ec=None)
    # TestT = mpatches.Rectangle((0,0), 10, 10, fc='red', ec=None, alpha=0.3)
    ax.add_patch(TrainT) 
    ax.add_patch(ValT)
    
    plt.text(TrainStart+(TrainEnd-TrainStart)/2, 20, 'Training', ha='center')
    plt.text(TrainEnd+(ValEnd-TrainEnd)/2, 20, 'Validation', ha='center')
    plt.text(ValEnd+(TestEnd-ValEnd)/2, 20, 'Test', ha='center')
    
    lw = 1 # line width
    # Plot cross-shore distances through time for WL and VE past
    plt.plot(VarDFDay['distances'], 'C2', lw=lw, label='Past VE')
    plt.plot(VarDFDay['wlcorrdist'], 'C0', lw=lw, label='Past WL')
    # Plot cross-shore distances through time for test data
    plt.plot(TransectDFTest['distances'], 'C2', ls=(0, (1, 1)), lw=lw, label='Test VE')
    plt.plot(TransectDFTest['wlcorrdist'], 'C0', ls=(0, (1, 1)), lw=lw, label='Test WL')
    plt.plot(FutureOutputs['output'][mID]['futureVE'], 'C8', alpha=0.7, lw=lw, label='Pred. VE')
    plt.plot(FutureOutputs['output'][mID]['futureWL'], 'C9', alpha=0.7, lw=lw, label='Pred. WL')

    plt.xlabel('Date (yyyy)')
    plt.ylabel('Cross-shore distance (m)')
    plt.legend(loc='upper left', ncols=3)
    plt.ylim(0,600)
    ax.tick_params(axis='both',which='major',pad=2)
    ax.xaxis.labelpad=2
    ax.yaxis.labelpad=2

    plt.tight_layout()
    plt.show()
    
    StartTime = FutureOutputs['output'][mID].index[0].strftime('%Y-%m-%d')
    EndTime = FutureOutputs['output'][mID].index[-1].strftime('%Y-%m-%d')
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_predictedWLVE_'+StartTime+'_'+EndTime+'_'+FutureOutputs['mlabel'][mID]+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)

 
def PlotFutureVars(mID, TransectDFTrain, TransectDFTest, VarDFDay, FutureOutputs, filepath, sitename):
    """
    Plot future waterline (WL) and vegetation edge (VE) predictions for the 
    chosen cross-shore transect, alongside timeseries of training variables used.
    FM Jan 2025

    Parameters
    ----------
    mID : int
        ID of the chosen model run stored in FutureOutputs.
    TransectDFTest : DataFrame
        DataFrame of past data (not interpolated) to use for training the model.
    TransectDFTest : DataFrame
        DataFrame of past data sliced from most recent end of TransectDF to use for testing the model.
    VarDFDay : DataFrame
        DataFrame of past data interpolated to daily timesteps (with temporal index).
    FutureOutputs : dict
        Dict storing per-model dataframes of future cross-shore waterline and veg edge predictions.
    filepath : str
        Local path to COASTGUARD Data folder.
    sitename : str
        Name of site of interest.


    """
    fig, axs = plt.subplots(5,1, sharex=True, figsize=(6.55,6), dpi=150)
    plt.subplots_adjust(wspace=None,hspace=None)
    lw = 1 # line width
    
    for i, ax, yvar, c, ylabel in zip(range(len(axs)), axs, 
                                      ['WaveDir',
                                       'Runups',
                                       'Iribarrens',
                                       'distances',
                                       'wlcorrdist'],
                                      ['cornflowerblue',
                                       'darkorchid',
                                       'orange',
                                       'forestgreen',
                                       'blue'],
                                      ['Wave direction (deg)',
                                       'Runup (m)',
                                       'Iribarren (1)',
                                       'Cross-shore distance (m)',
                                       'Cross-shore distance (m)']):
        TrainStart = mdates.date2num(TransectDFTrain.index[0])
        TrainEnd = mdates.date2num(TransectDFTrain.index[round(len(TransectDFTrain)-(len(TransectDFTrain)*0.2))])
        ValEnd = mdates.date2num(TransectDFTrain.index[-1])
        TrainT = mpatches.Rectangle((TrainStart,-100), TrainEnd-TrainStart, 1000, fc=[0.8,0.8,0.8], ec=None)
        ValT = mpatches.Rectangle((TrainEnd,-100), ValEnd-TrainEnd, 1000, fc=[0.9,0.9,0.9], ec=None)
        
        ax.add_patch(TrainT) 
        ax.add_patch(ValT)
    
        ax.plot(TransectDFTrain[yvar], c=c, lw=lw)
        ax.plot(VarDFDay[yvar], c=c, lw=lw, alpha=0.3)
        
        if i == 3:
            ax.plot(TransectDFTest['distances'], 'C2', ls=(0, (1, 1)), lw=lw, label='Test VE')
            ax.plot(FutureOutputs['output'][mID]['futureVE'], 'C8', alpha=0.7, lw=lw, label='Pred. VE')
            ax.legend(loc='upper left', ncols=3)
        elif i == 4:
            ax.plot(TransectDFTest['wlcorrdist'], 'C0', ls=(0, (1, 1)), lw=lw, label='Test WL')
            ax.plot(FutureOutputs['output'][mID]['futureWL'], 'C9', alpha=0.7, lw=lw, label='Pred. WL')
            ax.legend(loc='upper left', ncols=3)
        else:
            ax.plot(TransectDFTest[yvar], c=c, lw=lw)

        ax.set_ylabel(ylabel)
        ax.set_ylim(min(VarDFDay[yvar])-(min(VarDFDay[yvar])*0.2), max(VarDFDay[yvar])+(min(VarDFDay[yvar])*0.2))
        ax.tick_params(axis='both',which='major',pad=2)
        ax.xaxis.labelpad=2
        ax.yaxis.labelpad=2
    
    plt.xlabel('Date (yyyy)')       
    plt.tight_layout()
    plt.show()
    
    StartTime = FutureOutputs['output'][mID].index[0].strftime('%Y-%m-%d')
    EndTime = FutureOutputs['output'][mID].index[-1].strftime('%Y-%m-%d')
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_predictedVars_'+StartTime+'_'+EndTime+'_'+FutureOutputs['mlabel'][mID]+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)


# ----------------------------------------------------------------------------------------
### CLUSTERING FUNCTIONS ###

def Cluster(TransectDF, ValPlots=False):
    """
    Classify coastal change indicator data into low, medium or high impact from hazards,
    using a SpectralCluster clustering routine.
    FM Sept 2024

    Parameters
    ----------
    TransectDF : DataFrame
        Dataframe of single cross-shore transect, with timeseries of satellite-derived metrics attached.
    ValPlots : bool, optional
        Plot validation plots of silhouette score and inertia. The default is False.

    Returns
    -------
    VarDFClust : DataFrame
        Dataframe of just coastal metrics/variables in timeseries, with cluster values attached to each timestep.

    """
    # Fill nans factoring in timesteps for interpolation
    TransectDF.replace([np.inf, -np.inf], np.nan, inplace=True)
    VarDF = TransectDF.interpolate(method='time', axis=0)
    
    # VarDF = VarDF[['distances', 'wlcorrdist','TZwidth','WaveHs']]
    
    VarDF_scaled = StandardScaler().fit_transform(VarDF)
    
    # Apply PCA to reduce the dimensions to 3D for visualization
    pca = PCA(n_components=2)
    pca_VarDF = pca.fit_transform(VarDF_scaled)
    eigenvectors = pca.components_
    variances = pca.explained_variance_ratio_
    
    # ClusterMods = {'':SpectralClustering(n_clusters=3, eigen_solver='arpack', random_state=42)}
    # for Mod in ClusterMods.keys():
        
    ClusterMod = SpectralClustering(n_clusters=3, 
                                    eigen_solver='arpack',
                                    n_components=len(VarDF.columns), 
                                    random_state=42)
    # Map labels to cluster IDs based on cluster centres and their distance to eigenvectors
    ClusterMod.fit(VarDF_scaled)
    VarDF['Cluster'] = ClusterMod.labels_
    # 
    # ClusterCentres = np.array([pca_VarDF[VarDF['Cluster'] == i].mean(axis=0) for i in range(3)])
    
    # # Define cluster labels using identified centres
    # HighImpact = np.argmax(ClusterCentres[:, 0])
    # LowImpact = np.argmax(ClusterCentres[:, 1])
    # MediumImpact = (set([0,1,2]) - {HighImpact, LowImpact}).pop()
    # # Map labels to cluster IDs
    # ClusterToImpact = {HighImpact:'High',
    #                    MediumImpact:'Medium',
    #                    LowImpact:'Low'}
    # ImpactLabels = [ClusterToImpact[Cluster] for Cluster in VarDF['Cluster']]
    # VarDFClust = VarDF.copy()
    # VarDFClust['Impact'] = ImpactLabels

    # Create a DataFrame for PCA results and add cluster labels
    pca_df = pd.DataFrame(data=pca_VarDF, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = ClusterMod.labels_

    # Visualization of clusters
    # Example clustered timeseries using one or two variables
    # if ValPlots is True:
    #     fig, ax = plt.subplots(figsize=(10, 5))
    #     bluecm = cm.get_cmap('cool')
    #     greencm = cm.get_cmap('summer')
    #     ax.scatter(VarDF.index, 
    #                VarDF['WaveHs'], 
    #                c=VarDF['Cluster'], marker='X', cmap=bluecm)
    #     ax2 = ax.twinx()
    #     ax2.scatter(VarDF.index, 
    #                VarDF['distances'], 
    #                c=VarDF['Cluster'], marker='^', cmap=greencm)  
    #     ax.set_xlabel('Time')
    #     ax.set_ylabel('Significant wave height (m)')
    #     ax2.set_ylabel('VE distance (m)')
    #     plt.show()
    
    scale_factor = 0.5
    # Plot the clusters in the PCA space
    fig, ax = plt.subplots(figsize=(5, 5))
    clusterDF = []
    for cluster, clustc in zip(pca_df['Cluster'].unique(), ['#BEE4A8','#FF8183','#F6C999']):
        cluster_data = pca_df[pca_df['Cluster'] == cluster]
        plt.scatter(
            cluster_data['PC1']*scale_factor, 
            cluster_data['PC2']*scale_factor, 
            label=f'Cluster {cluster}', 
            s=40,
            c=clustc,
            alpha=0.7
        )
        clusterDF.append(cluster_data)
    # Overlay eignevectors of each variable
    coeffs = np.transpose(eigenvectors[0:2, :])*2
    n_coeffs = coeffs.shape[0]
    for i in range(n_coeffs):
        plt.arrow(0, 0, coeffs[i,0], coeffs[i,1], color='k', alpha=0.5, head_width=0.02, zorder=5)
        plt.annotate(text=VarDF.columns[i], xy=(coeffs[i,0], coeffs[i,1]), 
                     xytext=(coeffs[i,0]*15,5), textcoords='offset points',
                     color='k', ha='center', va='center', zorder=5)
    plt.tight_layout()
    plt.show()
        
    # 3D scatter plot (to investigate clustering or patterns in PCs)
    if 'PC3' in cluster_data.columns:
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        # colourdict = {'Low':'green','Medium':'orange','High':'red'}
        for cluster in pca_df['Cluster'].unique():
            cluster_data = pca_df[pca_df['Cluster'] == cluster]
        #     ax.scatter(cluster_data['PC1'],cluster_data['PC2'],cluster_data['PC3'], 
        #                color=colourdict[ClusterToImpact[cluster]], label=ClusterToImpact[cluster])
            ax.scatter(cluster_data['PC1']*scale_factor,cluster_data['PC2']*scale_factor,cluster_data['PC3']*scale_factor)
        ax.set_xlabel(rf'PC1 [explains {round(variances[0]*100,1)}% of $\sigma^2$]')
        ax.set_ylabel(rf'PC2 [explains {round(variances[1]*100,1)}% of $\sigma^2$]')
        ax.set_zlabel(rf'PC3 [explains {round(variances[2]*100,1)}% of $\sigma^2$]')
        # Plot eigenvectors of each variable        
        coeffs = np.transpose(eigenvectors[0:3, :])*2
        n_coeffs = coeffs.shape[0]
        for i in range(n_coeffs):
            ax.quiver(0,0,0, 
                      coeffs[i, 0], coeffs[i, 1], coeffs[i, 2], 
                      color='k', alpha=0.5, linewidth=2, arrow_length_ratio=0.1)
            ax.text(coeffs[i, 0] * 1.5, coeffs[i, 1] * 1.5, coeffs[i, 2] * 1.5, 
                    VarDF.columns[i], color='k', ha='center', va='center')
        # legorder = ['Low','Medium','High']
        # handles,labels = plt.gca().get_legend_handles_labels()
        # plt.legend([handles[labels.index(lab)] for lab in legorder],[labels[labels.index(lab)] for lab in legorder])
        plt.tight_layout()
        plt.show()
        
        
    return VarDF


def ClusterKMeans(TransectDF, ValPlots=False):
    """
    Classify coastal change indicator data into low, medium or high impact from hazards,
    using a KMeans clustering routine.
    FM Sept 2024

    Parameters
    ----------
    TransectDF : DataFrame
        Dataframe of single cross-shore transect, with timeseries of satellite-derived metrics attached.
    ValPlots : bool, optional
        Plot validation plots of silhouette score and inertia. The default is False.

    Returns
    -------
    VarDFClust : DataFrame
        Dataframe of just coastal metrics/variables in timeseries, with cluster values attached to each timestep.

    """
    # Define variables dataframe from transect dataframe by removing dates and transposing
    VarDF = TransectDF.drop(columns=['TransectID', 'dates'])
    VarDF.interpolate(method='nearest', axis=0, inplace=True) # fill nans using nearest
    VarDF.interpolate(method='linear', axis=0, inplace=True) # if any nans left over at start or end, fill with linear
    VarDF_scaled = StandardScaler().fit_transform(VarDF)
    
    
    # Fit k-means clustering to data iteratively over different cluster sizes
    k_n = range(2,15)
    # Inertia = compactness of clusters i.e. total variance within a cluster
    # Silhouette score = how similar object is to its own cluster vs other clusters 
    inertia = []
    sil_scores = []
    
    for k in k_n:
        kmeansmod = KMeans(n_clusters=k, random_state=42)
        kmeansmod.fit(VarDF_scaled)
        inertia.append(kmeansmod.inertia_)
        sil_scores.append(silhouette_score(VarDF_scaled, kmeansmod.labels_))
    
    # Apply PCA to reduce the dimensions to 3D for visualization
    pca = PCA(n_components=3)
    pca_VarDF = pca.fit_transform(VarDF_scaled)
    eigenvectors = pca.components_

    # Create a DataFrame for PCA results and add cluster labels
    pca_df = pd.DataFrame(data=pca_VarDF, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Cluster'] = kmeansmod.labels_
    
    
    if ValPlots is True:
        # Optional: Plot an elbow graph to find the optimal number of clusters
        plt.figure(figsize=(10, 5))
        plt.plot(k_n, inertia, marker='o')
        plt.title('Elbow Method For Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.show()
        
        # Optional: Plot silhouette scores for further cluster evaluation
        plt.figure(figsize=(10, 5))
        plt.plot(k_n, sil_scores, marker='o')
        plt.title('Silhouette Scores For Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.show()
    
    
    # # Fit the KMeans model with the chosen number of clusters
    # # Clusters are informed by 'impact' levels low, medium and high
    # optimal_k = 3
    # tic = timeit.default_timer() # start timer
    # kmeansmod = KMeans(n_clusters=optimal_k, random_state=42)
    # kmeansmod.fit(VarDF_scaled)
    # toc = timeit.default_timer() # stop timer
    
    # # Analyze the clustering results
    # VarDF['Cluster'] = kmeansmod.labels_
    
    ClusterMods = {'kmeans':KMeans(n_clusters=3, random_state=42),
                   'spectral':SpectralClustering(n_clusters=3, eigen_solver='arpack', random_state=42)}
    for Mod in ClusterMods.keys():
        
        ClusterMods[Mod].fit(VarDF_scaled)
        VarDF[Mod+'Cluster'] = ClusterMods[Mod].labels_
        ClusterMeans = VarDF.groupby(Mod+'Cluster').mean()
        
        ClusterCentres = np.array([pca_VarDF[VarDF[Mod+'Cluster'] == i].mean(axis=0) for i in range(3)])

        HighImpact = np.argmax(ClusterCentres[:, 0])
        LowImpact = np.argmax(ClusterCentres[:, 1])
        MediumImpact = (set([0,1,2]) - {HighImpact, LowImpact}).pop()
        
        ClusterToImpact = {HighImpact:'High',
                           MediumImpact:'Medium',
                           LowImpact:'Low'}
        ImpactLabels = [ClusterToImpact[Cluster] for Cluster in VarDF[Mod+'Cluster']]
        VarDFClust = VarDF.copy()
        VarDFClust[Mod+'Impact'] = ImpactLabels
        
        # HighImpact = ClusterMeans[(ClusterMeans['distances'] == ClusterMeans['distances'].min()) & # landward VE
        #                           (ClusterMeans['wlcorrdist'] == ClusterMeans['wlcorrdist'].min()) & # landward WL
        #                           (ClusterMeans['waterelev'] == ClusterMeans['waterelev'].max()) & # high water
        #                           (ClusterMeans['beachwidth'] == ClusterMeans['beachwidth'].min()) & # narrow width
        #                           (ClusterMeans['TZwidth'] == ClusterMeans['TZwidth'].min()) & # narrow TZ
        #                           (ClusterMeans['WaveHs'] == ClusterMeans['WaveHs'].max()) & # high waves
        #                           (ClusterMeans['WaveTp'] == ClusterMeans['WaveTp'].max())].index[0] # long period
        
        # LowImpact = ClusterMeans[(ClusterMeans['distances'] == ClusterMeans['distances'].max()) & # seaward VE
        #                           (ClusterMeans['wlcorrdist'] == ClusterMeans['wlcorrdist'].max()) & # seaward WL
        #                           (ClusterMeans['waterelev'] == ClusterMeans['waterelev'].min()) & # low water
        #                           (ClusterMeans['beachwidth'] == ClusterMeans['beachwidth'].max()) & # wide width
        #                           (ClusterMeans['TZwidth'] == ClusterMeans['TZwidth'].max()) & # wide TZ
        #                           (ClusterMeans['WaveHs'] == ClusterMeans['WaveHs'].min()) & # low waves
        #                           (ClusterMeans['WaveTp'] == ClusterMeans['WaveTp'].min())].index[0] # short period
        # AllClusters = set([0,1,2])
        # MediumImpact = (AllClusters - set([HighImpact, LowImpact])).pop()

        # Cluster to impact
        # ClusterToImpact = {'High': HighImpact,
        #                    'Medium':MediumImpact,
        #                    'Low':LowImpact}
        # VarDF['Impact'] = VarDF[Mod+'Cluster'].map(ClusterToImpact)
        
        # inertia.append(ClusterMods[Mod].inertia_)
        # sil_scores.append(silhouette_score(VarDF_scaled, ClusterMods[Mod].labels_))
    
        # Create a DataFrame for PCA results and add cluster labels
        pca_df = pd.DataFrame(data=pca_VarDF, columns=['PC1', 'PC2', 'PC3'])
        pca_df['Cluster'] = ClusterMods[Mod].labels_
    
        # Optional: Visualization of clusters
        if ValPlots is True:
            fig, ax = plt.subplots(figsize=(10, 5))
            bluecm = cm.get_cmap('cool')
            greencm = cm.get_cmap('summer')
            ax.scatter(VarDF.index, 
                       VarDF['WaveHs'], 
                       c=VarDF[Mod+'Cluster'], marker='X', cmap=bluecm)
            ax2 = ax.twinx()
            ax2.scatter(VarDF.index, 
                       VarDF['distances'], 
                       c=VarDF[Mod+'Cluster'], marker='^', cmap=greencm)  # Example visualization using one variable
            plt.title(f'Clustering Method: {Mod}')
            ax.set_xlabel('Time')
            # ax.set_ylabel('Cross-shore VE position (m)')
            ax.set_ylabel('Significant wave height (m)')
            ax2.set_ylabel('VE distance (m)')
            plt.show()
        
        # Plot the clusters in the PCA space
        fig, ax = plt.subplots(figsize=(5, 5))
        clusterDF = []
        for cluster in pca_df['Cluster'].unique():
            cluster_data = pca_df[pca_df['Cluster'] == cluster]
            plt.scatter(
                cluster_data['PC1'], 
                cluster_data['PC2'], 
                label=f'Cluster {cluster}', 
                s=40,
                alpha=0.7
            )
            clusterDF.append(cluster_data)
        # Plot eignevectors of each variable
        coeffs = np.transpose(eigenvectors[0:2, :])*2
        n_coeffs = coeffs.shape[0]
        for i in range(n_coeffs):
            plt.arrow(0, 0, coeffs[i,0], coeffs[i,1], color='k', alpha=0.5, head_width=0.02, zorder=5)
            plt.annotate(text=VarDF.columns[i], xy=(coeffs[i,0], coeffs[i,1]), 
                         xytext=(coeffs[i,0]*15,5), textcoords='offset points',
                         color='k', ha='center', va='center', zorder=5)
        plt.tight_layout()
        plt.show()
            
        # 3D scatter plot (to investigate clustering or patterns in PCs)
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        for cluster in pca_df['Cluster'].unique():
            cluster_data = pca_df[pca_df['Cluster'] == cluster]
            ax.scatter(cluster_data['PC1'],cluster_data['PC2'],cluster_data['PC3'])
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
        # Plot eignevectors of each variable
        coeffs = np.transpose(eigenvectors[0:3, :])*2
        n_coeffs = coeffs.shape[0]
        for i in range(n_coeffs):
            ax.quiver(0,0,0, 
                      coeffs[i, 0], coeffs[i, 1], coeffs[i, 2], 
                      color='k', alpha=0.5, linewidth=2, arrow_length_ratio=0.1)
            ax.text(coeffs[i, 0] * 1.5, coeffs[i, 1] * 1.5, coeffs[i, 2] * 1.5, 
                    VarDF.columns[i], color='k', ha='center', va='center')
        plt.tight_layout()
        plt.show()
        
        
    return VarDFClust
