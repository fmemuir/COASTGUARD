#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:57:18 2024

@author: fmuir
"""

import os

from datetime import datetime,timedelta

import string
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch, Rectangle, ConnectionPatch
import matplotlib.dates as mdates
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

import pandas as pd
pd.options.mode.chained_assignment = None # suppress pandas warning about setting a value on a copy of a slice

from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import SMOTE

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams.update({'font.size':7})

# Only use tensorflow in CPU mode
import tensorflow as tf
tf.config.set_visible_devices([],'GPU')


# ----------------------------------------------------------------------------------------
#%% PLOTTING FUNCTIONS ###
# SCALING:
# Journal 2-column width: 224pt or 3.11in
# Journal 1-column width: 384pt or 5.33in
# Spacing between: 0.33in
# Journal 2-column page: 6.55in


def MovingAverage(series, windowsize):
    """
    Generate a moving window average trendline from a timeseries 
    FM Apr 2023

    Parameters
    ----------
    series : list
        Timeseries to be plotted (yvalues).
    windowsize : int
        Number of steps to smooth over.

    Returns
    -------
    mvav : list
        Timeseries of values smoothed over requested interval.

    """
    # moving average trendline
    window = np.ones(int(windowsize))/float(windowsize)
    mvav = np.convolve(series, window, 'same')
    # Brute force first and last step to match
    mvav[0] = series[0]
    mvav[-1] = series[-1]
    
    return mvav


def VarCorrelations(VarDFDayTrain, TrainFeats):
    """
    Generate correlations dataframe for plotting
    FM Mar 2025

    Parameters
    ----------
    VarDFDayTrain : DataFrame
        Scaled DataFrame of past data interpolated to daily timesteps (with temporal index), 
        for training and validation.
    TrainFeats : list
        Training features to plot correlations of.

    Returns
    -------
    CorrDF : DataFrame
        Correlations dataframe of Pearson r values with VE and WL.

    """
    # Correlating with VE or WL
    CorrDict = {'VE':[], 'WL':[]}
    for SL in CorrDict.keys():
        # For each training feature, calculate pearson r with VE and WL
        for Key in TrainFeats:
            rval, _ = pearsonr(VarDFDayTrain[SL], VarDFDayTrain[Key])
            CorrDict[SL].append(rval)
    CorrDF = pd.DataFrame(CorrDict, index=TrainFeats)
    return CorrDF
        


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
        # Need to set order for polynomial interpolation approaches
        if Mthd in ['polynomial','spline']:
            # Ord = pd.isnull(TransectDF['WaveHsFD']).sum() - 1
            Ord = 5
            TransectDFInterp = TransectDF.interpolate(method=Mthd, order=Ord, axis=0)
            axs[i].set_title(Mthd+', order='+str(Ord), pad=1)
        else:
            TransectDFInterp = TransectDF.interpolate(method=Mthd, axis=0)
            axs[i].set_title(Mthd, pad=1)
        
        # Plot specific region of wave data which is missing
        axs[i].plot(TransectDFInterp['WaveHsFD'][3280:], c='#46B0E1', lw=1.5)
        axs[i].plot(TransectDF['WaveHsFD'][3280:], c='#1559A0', lw=1.5)
        
        # Set formatting of axis labels as mon-year
        axs[i].xaxis.set_major_locator(mdates.MonthLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        
        # Label where methods have over or undershot
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
    
    
def PlotInterpsWLVE(CoastalDF, Tr, FigPath):
    """
    Plot results of different scipy interpolation methods (VE and WL).
    FM Feb 2025

    Parameters
    ----------
    TransectDF : DataFrame
        Dataframe of per-transect coastal metrics/variables in timeseries.
    FigPath : str
        Path to save figure to.

    """
    
    # Set up vegetation edge and waterline dataframes
    VEDF = pd.DataFrame(CoastalDF['VE'].iloc[Tr], 
                              index=pd.to_datetime(CoastalDF['dates'].iloc[Tr], format='%Y-%m-%d'),
                              columns=['VE'])
    WLDF = pd.DataFrame(CoastalDF['WL'].iloc[Tr], 
                              index=pd.to_datetime(CoastalDF['wldates'].iloc[Tr], format='%Y-%m-%d'),
                              columns=['WL'])
    VEDF = VEDF[~VEDF.index.duplicated(keep='first')]
    WLDF = WLDF[~WLDF.index.duplicated(keep='first')]
    
    # Set up methods for interpolation
    Mthds = ['nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial', 'piecewise_polynomial', 'spline', 'pchip', 'akima', 'cubicspline', 'from_derivatives']
    fig, axs = plt.subplots(4,3, sharex=True, figsize=(6.55,5))
    axs=axs.flatten()
    
    # Start and end date for plotting
    SD = '2021-03-01 00:00:00'
    ED = '2022-01-01 00:00:00'
    for TransectDF, clr, lab in zip([VEDF,WLDF], ['#79C060','#3E74B3'], ['$VE$','$WL$']):
        for i, Mthd in enumerate(Mthds):
        # Need to set order for polynomial interpolation approaches
            if Mthd in ['polynomial','spline']:
                # Ord = pd.isnull(TransectDF['WaveHsFD']).sum() - 1
                Ord = 5
                TransectDFInterp = TransectDF.resample('D').interpolate(method=Mthd, order=Ord, axis=0)
                axs[i].set_title(Mthd+', order='+str(Ord), pad=1)
            else:
                TransectDFInterp = TransectDF.resample('D').interpolate(method=Mthd, axis=0)
                axs[i].set_title(Mthd, pad=1)
            
            # Plot results of interpolation as line and original points as markers
            axs[i].plot(TransectDF.loc[SD:ED], c=clr, marker='o', ms=2, lw=0, label=lab)
            axs[i].plot(TransectDFInterp.loc[SD:ED], c=clr, lw=1.5, alpha=0.5, label=lab+' interpolated')
            
            # Set formatting of axis labels as mon-year
            axs[i].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            
            # Label where methods have over/undershot
            if Mthd in ['quadratic','cubic','polynomial','spline','cubicspline'] and clr=='#3E74B3':
                axs[i].text(TransectDFInterp[SD:ED].idxmin()-timedelta(days=5),
                            TransectDFInterp.loc[SD:ED].min()+50, 
                            'undershooting', color='r', ha='right', va='center')
            if Mthd in ['polynomial'] and clr=='#3E74B3':
                axs[i].text(TransectDFInterp[SD:ED].idxmax()+timedelta(days=5),
                            TransectDFInterp.loc[SD:ED].max()+50, 
                            'overshooting', color='r', ha='left', va='center')
    # Plot legend outside fig box
    plt.legend(bbox_to_anchor=(0.5,-0.6), ncols=4)
        # axs[i].set_ylim((-1,1))
      
    # Common x and y labels
    fig.supxlabel('Date')
    fig.supylabel('Wave height (m)')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)  


def PlotCorrs(filepath, sitename, Tr, VarDFDayTrain, TrainFeats, SymbolDict):
    """
    Plot bar chart of Pearson r values between each training feature and VE or WL.
    FM Apr 2025

    Parameters
    ----------
    filepath : str
        Local path to COASTGUARD Data folder.
    sitename : str
        Name of site of interest.
    Tr : int
        Transect ID to plot correlations between.
    VarDFDayTrain : DataFrame
        Scaled DataFrame of past data interpolated to daily timesteps (with temporal index), 
        for training and validation.
    TrainFeats : list
        Training features to plot correlations of.
    SymbolDict : dict
        Dict of math-style labels for each feature name.


    """
    CorrDF = VarCorrelations(VarDFDayTrain, TrainFeats)
    
    NewInd = []
    for Key in CorrDF.index:
        NewInd.append(SymbolDict[Key])
    CorrDF.index = NewInd
    
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6.55,2))
    CorrDF.plot(kind="bar", color=['#79C060','#3E74B3'], ax=ax)
    
    ax.set_xticks(range(len(CorrDF.index)),CorrDF.index, rotation=0)
    ax.set_ylabel(r'$r$')
    ax.grid(which='major', axis='y', c=[0.8,0.8,0.8], lw=0.5, alpha=0.5)
    
    ax_inset = ax.inset_axes([0.333,0.5,0.333,0.47])
    CorrDF.iloc[5:10].plot(kind="bar", color=['#79C060','#3E74B3'], ax=ax_inset)
    
    ax_inset.set_ylim(-0.2,0.2)
    ax_inset.set_yticks(np.arange(-0.2,0.3,0.1))
    ax_inset.get_legend().remove()
    ax_inset.set_xticks(range(len(CorrDF.iloc[5:10].index)),CorrDF.iloc[5:10].index, rotation=0)
    ax_inset.grid(which='major', axis='y', c=[0.8,0.8,0.8], lw=0.5, alpha=0.5)
    
    for axs, lab, xpos, ypos in zip([ax, ax_inset], list(string.ascii_lowercase[:2]), [0.0041, 0.01], [0.985, 0.96]):
        axs.text(xpos,ypos, '('+lab+')', transform=axs.transAxes,
                fontsize=6, va='top', bbox=dict(facecolor='w', edgecolor='k',pad=1.5))
    
    plt.tight_layout()
    plt.show()
    
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_Correlations_Tr'+str(Tr)+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)    

def PlotAccuracy(CSVdir, FigPath):
    """
    Plot accuracy metric over all training runs in combined CSV, generated from
    Predictions.RunsToCSV()
    FM Feb 2025

    Parameters
    ----------
    CSVdir : str
        Filepath to CSV of combined training histories over different runs.
    FigPath : str
        Filepath to save the figure to.

    Returns
    -------
    AccuracyDF : DataFrame
        DataFrame of training histories (i.e. accuracy over epoch).

    """
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
    plt.plot(AccuracyDF, c='#283252', alpha=0.5)
    # Best performer
    plt.plot(AccuracyDF['dense64_validation'], c='r')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=True)
    
    return AccuracyDF


def PlotFeatSensitivity(PredDict, filepath, sitename, Tr):
    """
    
    FM Feb 2025

    Parameters
    ----------
    PredDict : dict
        Dictionary to store all the NN model metadata, with trained NN models.
    filepath : str
        Local path to COASTGUARD Data folder.
    sitename : str
        Name of site of interest.
    Tr : int
        ID of the chosen cross-shore transect.

    """
    # Grid point size
    sz = 200
    
    # Plot all feature test results
    gridspecs = dict(wspace=0.0) #width_ratios=[1,0.9])
    fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(3.11, 9), dpi=300, gridspec_kw=gridspecs)
    
    # Matrix of feature combos
    for r in range(len(PredDict['trainfeats'])):
        fts = [PredDict['trainfeats'][-1].index(ft) for ft in PredDict['trainfeats'][r]]
        maxInd = PredDict['accuracy'].index(np.max(PredDict['accuracy']))
        minInd = PredDict['accuracy'].index(np.min(PredDict['accuracy']))
        # Plot matrix of feature combinations
        if r == maxInd or r == minInd:
            ax.scatter(fts, [r]*len(fts), marker='s', c='#163E64', s=sz)
        else:
            ax.scatter(fts, [r]*len(fts), marker='s', c='#B3BADD', s=sz)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5,len(PredDict['trainfeats'][-1])-0.5)
    xticklocs = [i+0.5 for i in range(len(PredDict['trainfeats'][-1]))]
    ax.set_xticks(ticks=range(len(PredDict['trainfeats'][-1])), 
                  labels=[r'$H_{s}$',r'$R_{2}$', r'$\bar\theta$', r'$\alpha$', r'$T_{p}$'], minor=True)
    ax.tick_params(which='minor',top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xlabel('Train Features')
    ax.xaxis.set_label_position('top')
    # Turn off major ticks and labels (only there for grid)
    ax.set_xticks(ticks=xticklocs, minor=False)
    ax.xaxis.set_tick_params(which='major', bottom=False, labelbottom=False)
    # yxis params
    ax.set_ylim(-0.5,len(PredDict['trainfeats'])-0.5)
    yticklocs = [i+0.5 for i in range(len(PredDict['trainfeats']))]
    ax.set_yticks(yticklocs)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    
    ax.grid(which='major', c=[0.8,0.8,0.8], lw=0.8)
    
    # Resulting accuracy of each
    barlims = (0.5,1)
    # normc = plt.Normalize(np.min(PredDict['accuracy']), np.max(PredDict['accuracy']))
    # colours = plt.cm.PuBuGn(normc(PredDict['accuracy']))
    for r in range(len(PredDict['trainfeats'])):
        fts = [PredDict['trainfeats'][-1].index(ft) for ft in PredDict['trainfeats'][r]]
        # Plot matrix of feature combinations
        if r == maxInd or r == minInd:
            ax2.barh([r]*len(fts), PredDict['accuracy'][r], color='#163E64')
            ax2.text(PredDict['accuracy'][r], r, s=f"{round(PredDict['accuracy'][r],2)}",
                     ha='left', va='center', c='#163E64')
        else:
            ax2.barh([r]*len(fts), PredDict['accuracy'][r], color='#B3BADD')#colours[r])
    ax2.tick_params(which='major', top=True, labeltop=True, bottom=False, labelbottom=False)
    ax2.set_xticks(ticks=np.arange(0.5, 1.1, 0.1))
    ax2.tick_params(axis='y', colors=[0.7,0.7,0.7])
    ax2.set_xlim(barlims)
    ax2.set_xlabel('Accuracy')
    ax2.xaxis.set_label_position('top')
    
    plt.tight_layout()
    plt.show()
    
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_FeatureSensitivityAll_Tr'+str(Tr)+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)
    
    
    #---------------------------------
    # Plot just the top and bottom 5
    Accuracies = pd.DataFrame({'trainfeats':PredDict['trainfeats'], 'accuracy':PredDict['accuracy']})
    Accuracies.sort_values(by='accuracy', axis=0, ascending=True, inplace=True)
    Bottom5 = Accuracies.iloc[-5:]
    Top5 = Accuracies.iloc[0:5]
    TopBottom = pd.concat([Top5, Bottom5])
    
    gridspecs = dict(wspace=0.0) #width_ratios=[1,0.9])
    fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(3.11, 3.11), dpi=300, gridspec_kw=gridspecs)
    
    # Matrix of feature combos
    for r in range(len(TopBottom)):
        fts = [PredDict['trainfeats'][-1].index(ft) for ft in TopBottom['trainfeats'].iloc[r]]
        maxInd = 0
        minInd = 9
        # Plot matrix of feature combinations
        if r == maxInd or r == minInd:
            ax.scatter(fts, [r]*len(fts), marker='s', c='#163E64', s=sz)
        else:
            ax.scatter(fts, [r]*len(fts), marker='s', c='#B3BADD', s=sz)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5,len(PredDict['trainfeats'][-1])-0.5)
    xticklocs = [i+0.5 for i in range(len(PredDict['trainfeats'][-1]))]
    ax.set_xticks(ticks=range(len(PredDict['trainfeats'][-1])), 
                  labels=[r'$H_{s}$',r'$R_{2}$', r'$\bar\theta$', r'$\alpha$', r'$T_{p}$'], minor=True)
    ax.tick_params(which='minor',top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xlabel('Train Features')
    ax.xaxis.set_label_position('top')
    # Turn off major ticks and labels (only there for grid)
    ax.set_xticks(ticks=xticklocs, minor=False)
    ax.xaxis.set_tick_params(which='major', bottom=False, labelbottom=False)
    # yxis params
    ax.set_ylim(-0.5,len(TopBottom['trainfeats'])-0.5)
    yticklocs = [i+0.5 for i in range(len(TopBottom['trainfeats']))]
    ax.set_yticks(yticklocs)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    
    ax.grid(which='major', c=[0.8,0.8,0.8], lw=0.8)
    
    # Resulting accuracy of each
    barlims = (0.5,1)
    # normc = plt.Normalize(np.min(PredDict['accuracy']), np.max(PredDict['accuracy']))
    # colours = plt.cm.PuBuGn(normc(PredDict['accuracy']))
    for r in range(len(TopBottom['trainfeats'])):
        fts = [PredDict['trainfeats'][-1].index(ft) for ft in TopBottom['trainfeats'].iloc[r]]
        # Plot matrix of feature combinations
        if r == maxInd or r == minInd:
            ax2.barh([r]*len(fts), TopBottom['accuracy'].iloc[r], color='#163E64')
            ax2.text(TopBottom['accuracy'].iloc[r], r, s=f"{round(TopBottom['accuracy'].iloc[r],2)}",
                     ha='left', va='center', c='#163E64')
        else:
            ax2.barh([r]*len(fts), TopBottom['accuracy'].iloc[r], color='#B3BADD')#colours[r])
    ax2.axhline(4.5, c=[0.7,0.7,0.7], lw=1, ls='--')
    ax2.tick_params(which='major', top=True, labeltop=True, bottom=False, labelbottom=False)
    ax2.tick_params(axis='y', colors=[0.7,0.7,0.7])
    ax2.set_xticks(ticks=np.arange(0.5, 1.1, 0.1))
    ax2.set_xlim(barlims)
    ax2.set_xlabel('Accuracy')
    ax2.xaxis.set_label_position('top')
    
    plt.tight_layout()
    plt.show()
    
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_FeatureSensitivityTopBottom_Tr'+str(Tr)+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)
    
    
def PlotVarTS(TransectDF, Tr,TrainFeatsPlotting, filepath, sitename):
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
    TransectDFVal = TransectDFTrain[len(TransectDFTrain)-round((len(TransectDFTrain)+len(TransectDFTest))*0.1):]
    TransectDFTrain = TransectDFTrain[:len(TransectDFTrain)-len(TransectDFVal)]
    
    # set subplot spacing with small break between start and end of training
    # gridspec = dict(wspace=0.0, width_ratios=[1, 0.1, 1, 1])
    gridspec = dict(wspace=0.0, width_ratios=[1.2, 0.1, 1.2, 0.5])
    fig, axs = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(4.27, 1.50), dpi=300, gridspec_kw=gridspec)
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
    # gridspec = dict(wspace=0.0, width_ratios=[0.75, 0.1, 0.75, 0.7, 0.35])
    gridspec = dict(wspace=0.0, width_ratios=[0.75, 0.1, 0.75, 0.35, 0.35])

    fig, axs = plt.subplots(nrows=1, ncols=5, sharey=True, figsize=(2.65, 1.50), dpi=300, gridspec_kw=gridspec)
    axs[1].set_visible(False)
    
    SelVars = list(['VE', 'WL'] + TrainFeatsPlotting)
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


def PlotChosenVarTS(TransectDFTrain, TransectDFTest, CoastalDF, TrainFeatsPlotting, SymbolDict, Tr, filepath, sitename):
    """
    
    FM Mar 2025

    Parameters
    ----------
    TransectDFTrain : DataFrame
        DataFrame of past data to use for training (and validating) the model.
    TransectDFTest : DataFrame
        DataFrame of past (unseen) data to use for testing the model.
    CoastalDF : DataFrame
        DataFrame of cross-shore transects (rows) and intersected coastal 
        timeseries/metrics (columns).
    TrainFeatsPlotting : str
        Name of training features to plot.
    SymbolDict : dict
        Dict of math-style labels for each feature name.
    Tr : int
        ID of the chosen cross-shore transect.
    filepath : str
        Local path to COASTGUARD Data folder.
    sitename : str
        Name of site of interest.

    """
    # Append VE and WL to training feats
    TrainTargFeats = TrainFeatsPlotting.copy()
    TrainTargFeats.append('VE')
    TrainTargFeats.append('WL')
    
    TransectDFPlot = pd.concat([TransectDFTrain,TransectDFTest])
    
    TrainStart = mdates.date2num(TransectDFTrain.index[0])
    TrainEnd = mdates.date2num(TransectDFTrain.index[round(len(TransectDFTrain)-(len(TransectDFTrain)*0.1))])
    ValEnd = mdates.date2num(TransectDFTrain.index[-1])
    TestEnd = mdates.date2num(TransectDFTest.index[-1])
    
    fig, axs = plt.subplots(len(TrainTargFeats), 1, sharex=True, figsize=(6.55,6), dpi=300)
    labs = list(string.ascii_lowercase[:axs.shape[0]])
    
    for (axID, ax), Feat, lab in zip(enumerate(axs), TrainTargFeats, labs):
        
        # Convert back to deg
        if Feat == 'WaveDirFD':
            ax.plot(np.rad2deg(TransectDFPlot[Feat]), c='#163E64', lw=0.7, alpha=0.5)
        else:
            ax.plot(TransectDFPlot[Feat], c='#163E64', lw=0.7, alpha=0.5)
        # Set datetimes for uninterpolated data
        if Feat == 'WL' or Feat == 'tideelev':
            FeatDT = CoastalDF.iloc[Tr]['wlDTs']
        elif Feat == 'VE':
            FeatDT = CoastalDF.iloc[Tr]['veDTs']
        elif Feat == 'Runups' or Feat == 'Iribarren':
            FeatDT = CoastalDF.iloc[Tr]['WaveDates']
        elif 'FD' in Feat:
            FeatDT = CoastalDF.iloc[Tr]['WaveDatesFD']
        
        if Feat == 'WaveDirFD' or Feat == 'WaveAlphaFD':
            Unit = r' ($\degree$)'
        elif Feat == 'WaveTpFD':
            Unit = ' (s)'
        else:
            Unit = ' (m)'
        
        ax.plot(FeatDT, CoastalDF.iloc[Tr][Feat], c='#163E64', marker='o', ms=0.5, lw=0)
        ax.set_ylabel(SymbolDict[Feat]+Unit)
        # ax.set_xlim(mdates.date2num(min(CoastalDF.iloc[Tr]['WaveDatesFD'])),
        #             mdates.date2num(max(CoastalDF.iloc[Tr]['WaveDatesFD'])))
        ax.set_xlim(TrainStart, TestEnd)
        # Add train/val/test rects
        # TrainT = mpatches.Rectangle((TrainStart,ax.get_ylim()[0]), 
        #                             TrainEnd-TrainStart, ax.get_ylim()[1]-ax.get_ylim()[0], 
        #                             fc=[0.8,0.8,0.8], ec=None)
        # ValT = mpatches.Rectangle((TrainEnd,ax.get_ylim()[0]), 
        #                           ValEnd-TrainEnd, ax.get_ylim()[1]-ax.get_ylim()[0], 
        #                           fc=[0.9,0.9,0.9], ec=None)
        # ax.add_patch(TrainT) 
        # ax.add_patch(ValT)
        ax.axvline(TrainEnd, c='k', alpha=0.5, lw=1, ls='--')
        ax.axvline(ValEnd, c='k', alpha=0.5, lw=1, ls='--')
        
        # Add train/val/test labels
        if axID == len(axs)-1: # last plot
            Text_y = ax.get_ylim()[0]+((ax.get_ylim()[1]-ax.get_ylim()[0])*0.1)
            ax.text(TrainStart+(TrainEnd-TrainStart)/2, Text_y, 'Training', ha='center')
            ax.text(TrainEnd+(ValEnd-TrainEnd)/2, Text_y, 'Validation', ha='center')
            ax.text(ValEnd+(TestEnd-ValEnd)/2, Text_y, 'Test', ha='center')
        
        ax.text(0.0039, 0.97, '('+lab+')', transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(facecolor='w', edgecolor='k',pad=1.5))
        
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    plt.xlabel('Date (yyyy)')
    plt.tight_layout()
    plt.show()
    
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_SubsetTimeseries_Tr'+str(Tr)+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)
    

def PlotStormWaveHs(TransectDF, CoastalDFTr, filepath, sitename):
    """
    
    FM Feb 2025

    Parameters
    ----------
    TransectDF : DataFrame
        Dataframe of per-transect coastal metrics/variables in timeseries.
    CoastalDFTr : Series
        Slice of CoastalDF dataframe to use for plotting original coastal 
        observations (not interpolated).
    filepath : str
        Local path to COASTGUARD Data folder.
    sitename : str
        Name of site of interest.

    """
    
    BabetTransect = TransectDF.loc['2023-09-28 00:00:00':'2023-12-05 00:00:00']
    BabetVEs = pd.DataFrame({'veDTs':CoastalDFTr['veDTs'],
                             'VE':CoastalDFTr['VE']})
    BabetWLs = pd.DataFrame({'wlDTs':CoastalDFTr['wlDTs'],
                             'WL':CoastalDFTr['WL']})
    BabetVEs = BabetVEs.groupby('veDTs').mean()
    BabetWLs = BabetWLs.groupby('wlDTs').mean()
    
    fig, ax = plt.subplots(1,1, figsize=(3.3,1.94), dpi=300)

    rectwidth = mdates.date2num(datetime(2023,10,21)) - mdates.date2num(datetime(2023,10,18))
    rect = mpatches.Rectangle((mdates.date2num(datetime(2023,10,18)), -200), rectwidth, 1000, 
                              fc=[0.3,0.3,0.3], ec=None, alpha=0.2)
    ax.add_patch(rect)
    
    # Plot with VE and WL change over course of storm
    ax.plot(BabetTransect['VE'], c='#79C060', label=r'$VE$')
    ax.scatter(BabetVEs.index, BabetVEs['VE'], s=15, marker='x', c='#79C060')
    ax.plot(BabetTransect['WL'], c='#3E74B3', label=r'$WL$')
    ax.scatter(BabetWLs.index, BabetWLs['WL'], s=15, marker='x', c='#3E74B3')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.set_xlabel('Date (2023-mm-dd)')
    # ax.set_xticklabels([date.day for date in BabetTransect.index])
    ax.set_xlim(pd.Timestamp('2023-09-28 00:00:00'), pd.Timestamp('2023-12-02 00:00:00'))
    ax.set_ylim(-50, 550)
    ax.legend(loc='upper left', handlelength=1, columnspacing=1, handletextpad=0.6)
    # ax.text(x=mdates.date2num(datetime(2023,10,18)), y=ax.get_ylim()[0]+((ax.get_ylim()[1]-ax.get_ylim()[0])/2),
    #         s='Storm Babet', rotation=90, ha='right', va='center', alpha=0.3)
    ax.text(x=mdates.date2num(datetime(2023,10,18)), y=ax.get_ylim()[1]-(ax.get_ylim()[1]*0.23),
            s='Storm\nBabet', rotation=0, ha='right', va='top', alpha=0.3)
    
    # Second axis with water elevations
    ax2 = ax.twinx()
    ax2.plot(BabetTransect['WaveHsFD'], c='#163E64', lw=1.2, label=r'$H_{s}$')
    ax2.plot(BabetTransect['tideelevFD'], c='#163E64',lw=1.2,  ls='--', label=r'$\bar{z}_{tide}$')
    ax2.plot(BabetTransect['tideelevMx'], c='#163E64', lw=1.2, ls=':', label=r'$z^{*}_{tide}$')
    
    # Marker where harbour failed
    ax.axvline(x=mdates.date2num(datetime(2023,10,29)), c=[0.3,0.3,0.3], ls='--', lw=0.8, alpha=0.2)
    ax.text(x=mdates.date2num(datetime(2023,10,29)), y=ax.get_ylim()[1]-(ax.get_ylim()[1]*0.23),
            s='slipway lost', ha='left', va='top', alpha=0.3)

    ax2.set_ylim(-0.2,4.)
    ax2.legend(loc='upper right', ncols=3, handlelength=1.5, columnspacing=0.8, handletextpad=0.6)
    
    ax.set_ylabel('Cross-shore distance (m)', labelpad=1)
    ax2.set_ylabel('Water elevation (m)')
    
    
    plt.tight_layout()
    plt.show()
    
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_StormWavesVEWL_Tr'+str(CoastalDFTr.name)+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=True)
    
    
def PlotParaCoords(PredDict, filepath, sitename): 
    """
    
    FM Mar 2025

    Parameters
    ----------
    PredDict : dict
        Dictionary to store all the NN model metadata, with trained NN models.
    filepath : str
        Local path to COASTGUARD Data folder.
    sitename : str
        Name of site of interest.

    """
    
    PredDF = pd.DataFrame(PredDict)
    HPScaler = MinMaxScaler()
    HPs = ['epochN', 'batchS', 'hiddenLscale', 'denselayers', 'dropoutRt', 'learnRt', 'accuracy', 'train_time']
    PredDFScaled = pd.DataFrame(HPScaler.fit_transform(PredDF[HPs]), columns=HPs)
    
    fig, axs = plt.subplots(1,2, figsize=(6.55,3.5), dpi=300)
    
    for ax, Metric, MetricLab in zip(axs, ['accuracy', 'train_time'], ['Accuracy', 'Training time (s)']):
        normc = plt.Normalize(np.min(PredDF[Metric]), np.max(PredDF[Metric]))
        if Metric == 'accuracy':
            colours = plt.cm.PuBuGn(normc(PredDF[Metric]))
        else:
            colours = plt.cm.PuBuGn_r(normc(PredDF[Metric]))
        for i in range(len(PredDFScaled)):
            ax.plot(PredDFScaled.iloc[i,:], color=colours[i], alpha=0.5)
        ax.set_xticks(range(len(PredDFScaled.columns)), PredDFScaled.columns, rotation=45)
        ax.grid(axis='x', lw=0.5, ls=':')
        ax.tick_params(axis='y', left=False, labelleft=False)
        ax.set_facecolor('#D5D5D5')
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.PuBuGn, norm=normc)
        cbar = plt.colorbar(sm,ax=ax)
        cbar.set_ticks([np.min(PredDF[Metric]), np.mean(PredDF[Metric]), np.max(PredDF[Metric])])
        if Metric == 'accuracy':
            cbar.set_ticklabels([f"{np.min(PredDF[Metric]):.2f}", f"{np.mean(PredDF[Metric]):.2f}", f"{np.max(PredDF[Metric]):.2f}"])
        else:
            cbar.set_ticklabels([f"{np.min(PredDF[Metric]):.0f}", f"{np.mean(PredDF[Metric]):.0f}", f"{np.max(PredDF[Metric]):.0f}"])
        cbar.set_label(MetricLab, rotation=270)
        
    
    plt.tight_layout()
    plt.show()
    
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_HP_Accuracy_Time.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)
        

def PlotHPScatter(PredDict):
    """
    
    FM Mar 2025

    Parameters
    ----------
    PredDict : dict
        Dictionary to store all the NN model metadata, with trained NN models.

    """
    
    PredDF = pd.DataFrame(PredDict)
    # HPScaler = MinMaxScaler()
    # HPs = ['epochN', 'batchS', 'hiddenLscale', 'denselayers', 'dropoutRt', 'learnRt', 'accuracy', 'train_time']
    # PredDFScaled = pd.DataFrame(HPScaler.fit_transform(PredDF[HPs]), columns=HPs)
    
    fig, ax = plt.subplots(1,1, figsize=(3.11,3.11), dpi=300)

    ax.scatter(PredDF['train_time'], PredDF['accuracy'], alpha=0.5)
    ax.set_xlim(0,200)
    ax.set_ylim(0.7,1)
    plt.tight_layout()
    plt.show()
    

def PlotIntGrads(PredDict, VarDFDayTrain, IntGradAttr, SymbolDict, filepath, sitename, Tr, enddate=None):
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
    
    axs[1].set_yticks(range(len(SymbolDict.keys())))
    axs[1].set_yticklabels(SymbolDict.values())
    
    # Add smaller heatmap of just wave data
    cax = axs[2].imshow(IntGradAttr_indiv[:,3:8].T, aspect='equal', cmap=cmap,
                        vmin=0, vmax=0.05)
    cbar = fig.colorbar(cax, location='bottom', pad=0.2, label='Feature Importance')
    axs[2].set_title('(c) Wave Integrated Gradients', fontsize=7)
    axs[2].set_yticks(range(len(SymbolDict.values()[3:8])))
    axs[2].set_yticklabels(SymbolDict.values()[3:8])
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



def PlotFuture(mID, Tr,  PredDict, TransectDFTrain, TransectDFTest, FutureOutputs, filepath, sitename, SurveyGDF=None):
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
    
    gridspec = dict(width_ratios=[1, 0.2])
    fig, ax = plt.subplots(2,2, figsize=(6.5,5.5), gridspec_kw=gridspec)
    # ax[0] is full timeseries, ax[2] is test subset timeseries
    
    TrainStart = mdates.date2num(TransectDFTrain.index[0])
    # TrainEnd = mdates.date2num(TransectDFTrain.index[round(len(TransectDFTrain)-(len(TransectDFTrain)*0.1))])
    TrainEnd = mdates.date2num(TransectDFTrain.index[
        round(len(TransectDFTrain)-(len(TransectDFTrain)*PredDict['validsize'][mID]))
        ])
    ValEnd = mdates.date2num(TransectDFTrain.index[-1])
    TestEnd = mdates.date2num(TransectDFTest.index[-1])

    ax[0,0].axvline(TrainEnd, c='k', alpha=0.5, lw=1, ls='--')
    ax[0,0].axvline(ValEnd, c='k', alpha=0.5, lw=1, ls='--')
    
    ax[0,0].text(TrainStart+(TrainEnd-TrainStart)/2, 20, 'Training', ha='center')
    ax[0,0].text(TrainEnd+(ValEnd-TrainEnd)/2, 20, 'Validation', ha='center')
    ax[0,0].text(ValEnd+(TestEnd-ValEnd)/2, 20, 'Test', ha='center')
    
    ax[0,0].set_xlim(TrainStart,TestEnd)
    ax[1,0].set_xlim(ValEnd,TestEnd)
        
    
    lw = 0.8 # line width
    Lims = (0,600)
    for a in [ax[0,0], ax[1,0]]:
        if SurveyGDF is not None:
            a.plot(mdates.date2num(SurveyGDF['Vdates'].iloc[Tr]), [i+70 for i in SurveyGDF['Vdists'].iloc[Tr]], 'ro')
        for SL, SLc in zip(['VE', 'WL'], ['#79C060','#3E74B3']):
            # Calculate smoothed version of predictions
            Smooth = MovingAverage(FutureOutputs['output'][mID]['future'+SL], 10)
            # Plot cross-shore distances through time for WL and VE past
            a.scatter(pd.concat([TransectDFTrain[SL], TransectDFTest[SL]]).index, 
                        pd.concat([TransectDFTrain[SL], TransectDFTest[SL]]), 
                        s=15, color='k', edgecolors=SLc, linewidths=0.3, marker='.', label=f"${SL}$")
            # Plot predicted WL and VE
            a.plot(FutureOutputs['output'][mID]['future'+SL], color=SLc, alpha=0.4, lw=lw, label=f"${{\\hat{{{SL}}}}}$")
            # Plot smoothed predicted WL and VE
            a.plot(FutureOutputs['output'][mID]['future'+SL].index, Smooth, color=SLc, alpha=0.8, lw=lw, label=f"${{ \\hat{{{SL}}}_{{t[i:i+10]}} }}$")
    # histograms on side to show distribution
    Bins = np.arange(-600,600,10)
    for a, ObsHist, PredHist  in zip([ax[0,1],ax[1,1]], 
                                     [pd.concat([TransectDFTrain, TransectDFTest]), TransectDFTest],
                                     [FutureOutputs['output'][mID], FutureOutputs['output'][mID].loc[TransectDFTrain.index[-1]:]]):
        for SL, SLc in zip(['VE', 'WL'], ['#79C060','#3E74B3']):
            a.hist(ObsHist[SL], bins=Bins, fc='k', orientation='horizontal')
            a.hist(PredHist['future'+SL], bins=Bins, fc=SLc, alpha=0.7, orientation='horizontal')
            a.set_ylim(Lims)
            a.yaxis.set_tick_params(which='major', labelleft=False)
    
    handles, labels = ax[0,0].get_legend_handles_labels() 
    legorder = [3,0,4,1,5,2]  
    
    ax[0,0].set_xlabel('Date (yyyy)')
    ax[1,0].set_xlabel('Date (yyyy-mm)')
    ax[0,0].set_ylabel('Cross-shore distance (m)')
    ax[0,0].legend(handles=[handles[i] for i in legorder],
                labels=[labels[i] for i in legorder], 
                loc='upper left', ncols=3,
                handlelength=1, columnspacing=1, handletextpad=0.6)
    ax[0,0].set_ylim(Lims)
    ax[0,0].tick_params(axis='both',which='major',pad=2)
    ax[0,0].xaxis.labelpad=2
    ax[0,0].yaxis.labelpad=2
    
    handles, labels = ax[1,0].get_legend_handles_labels() 
    legorder = [3,0,4,1,5,2]  
    ax[1,0].legend(handles=[handles[i] for i in legorder],
                labels=[labels[i] for i in legorder], 
                loc='upper left', ncols=3,
                handlelength=1, columnspacing=1, handletextpad=0.6)
    # top line
    fig.add_artist(ConnectionPatch(
    xyA=(ValEnd,0), coordsA=ax[0,0].transData,
    xyB=(0,1), coordsB=ax[1,0].transAxes,
    color='k', alpha=0.3, linewidth=0.5))   
    # bottom line
    fig.add_artist(ConnectionPatch(
    xyA=(TestEnd, 0), coordsA=ax[0,0].transData,
    xyB=(1,1), coordsB=ax[1,0].transAxes,
    color='k', alpha=0.3, linewidth=0.5))  

    plt.tight_layout()
    plt.show()
    
    # StartTime = FutureOutputs['output'][mID].index[0].strftime('%Y-%m-%d')
    # EndTime = FutureOutputs['output'][mID].index[-1].strftime('%Y-%m-%d')
    StartTime = mdates.num2date(ax[0,0].axis()[0]).strftime('%Y-%m-%d')
    EndTime = mdates.num2date(ax[0,0].axis()[1]).strftime('%Y-%m-%d')
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_predictedWLVE_'+StartTime+'_'+EndTime+'_'+FutureOutputs['mlabel'][mID]+'_Tr'+str(Tr)+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)


def PlotFutureShort(mID, Tr, TransectDFTrain, TransectDFTest, FutureOutputs, filepath, sitename, PlotDateRange, Storm=None, ImpactClass=None):
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
       
    # fig, ax = plt.subplots(1,1, figsize=(6.5,3.35))
    fig, ax = plt.subplots(1,1, figsize=(3.5,2.6))

    
    if Storm is not None:
        rectwidth = mdates.date2num(Storm[1]) - mdates.date2num(Storm[0])
        rect = mpatches.Rectangle((mdates.date2num(datetime(2023,10,18)), -200), rectwidth, 1000, 
                                  fc=[0.3,0.3,0.3], ec=None, alpha=0.2)
        ax.add_patch(rect)
        
    plt.xlim(mdates.date2num(PlotDateRange[0]),mdates.date2num(PlotDateRange[1]))

    lw = 1.2 # line width
    for SL, SLc in zip(['VE', 'WL'], ['#79C060','#3E74B3']):
        # Calculate smoothed version of predictions
        Smooth = MovingAverage(FutureOutputs['output'][mID]['future'+SL], 10)
        # Plot cross-shore distances through time for WL and VE past
        ax.scatter(pd.concat([TransectDFTrain[SL], TransectDFTest[SL]]).index, 
                    pd.concat([TransectDFTrain[SL], TransectDFTest[SL]]), 
                    s=30, color='k', edgecolors=SLc, linewidths=1, marker='.', label=f"${SL}$")
        # Plot predicted WL and VE
        ax.plot(FutureOutputs['output'][mID]['future'+SL], color=SLc, alpha=0.4, lw=lw, label=f"${{\\hat{{{SL}}}}}$")
        # Plot smoothed predicted WL and VE
        ax.plot(FutureOutputs['output'][mID]['future'+SL].index, Smooth, color=SLc, alpha=0.8, lw=lw, label=f"${{ \\hat{{{SL}}}_{{t[i:i+10]}} }}$")
        
        if ImpactClass is not None:
            ax2 = ax.twinx()
            Low, = ax2.plot(FutureOutputs['output'][mID]['future'+SL][ImpactClass['future'+SL]==1],
                     color='#E4FF5C', alpha=0.7, lw=0, 
                     marker='^', ms=4, markeredgewidth=0.8, markeredgecolor='k')
            Med, = ax2.plot(FutureOutputs['output'][mID]['future'+SL][ImpactClass['future'+SL]==2],
                     color='#FF990A', alpha=0.7, lw=0, 
                     marker='^', ms=4, markeredgewidth=0.8, markeredgecolor='k')
            High, = ax2.plot(FutureOutputs['output'][mID]['future'+SL][ImpactClass['future'+SL]==3],
                     color='#D52941', alpha=0.7, lw=0, 
                     marker='^', ms=4, markeredgewidth=0.8, markeredgecolor='k')
            ax2.set_ylim(0,600)
            ax2.tick_params(axis='y', right=False, labelright=False)
    
    if ImpactClass is not None:
        ax2.legend(handles=[Low, Med, High], labels=['Low impact', 'Medium impact', 'High impact'],
                   loc='upper left', handlelength=1, columnspacing=1, handletextpad=0.6)

    handles, labels = ax.get_legend_handles_labels() 
    legorder = [3,0,4,1,5,2]  
    
    ax.set_xlabel('Date (2023-mm-dd)')
    ax.set_ylabel('Cross-shore distance (m)')
    # plt.legend(handles=[handles[i] for i in legorder],
    #             labels=[labels[i] for i in legorder], 
    #             loc='upper left', ncols=3,
    #             handlelength=1, columnspacing=1, handletextpad=0.6)
    ax.legend(handles=[handles[i] for i in legorder],
                labels=[labels[i] for i in legorder], 
                loc='upper center', bbox_to_anchor=(0.5, -0.2), ncols=6,
                handlelength=1, columnspacing=1, handletextpad=0.6)
    ax.set_ylim(0,600)
    ax.tick_params(axis='both',which='major',pad=2)
    ax.xaxis.labelpad=2
    ax.yaxis.labelpad=2

    # Marker where storm occurred and harbour failed
    if Storm is not None:
        ax.text(x=mdates.date2num(Storm[0]), y=ax.get_ylim()[1]-(ax.get_ylim()[1]*0.05),
                s='Storm\nBabet', rotation=0, ha='right', va='top', alpha=0.3)
        ax.axvline(x=mdates.date2num(datetime(2023,10,29)), c=[0.3,0.3,0.3], ls='--', lw=0.8, alpha=0.2)
        ax.text(x=mdates.date2num(datetime(2023,10,29)), y=ax.get_ylim()[1]-(ax.get_ylim()[1]*0.05),
                s='slipway lost', ha='left', va='top', alpha=0.3)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    plt.tight_layout()
    plt.show()
    
    # StartTime = FutureOutputs['output'][mID].index[0].strftime('%Y-%m-%d')
    # EndTime = FutureOutputs['output'][mID].index[-1].strftime('%Y-%m-%d')
    StartTime = mdates.num2date(plt.axis()[0]).strftime('%Y-%m-%d')
    EndTime = mdates.num2date(plt.axis()[1]).strftime('%Y-%m-%d')
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_predictedWLVE_'+StartTime+'_'+EndTime+'_'+FutureOutputs['mlabel'][mID]+'_Tr'+str(Tr)+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)



def PlotFutureEnsemble(TransectDFTrain, TransectDFTest, FutureOutputs, filepath, sitename, PlotDateRange=None):
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
    
    fig, ax = plt.subplots(1,1, figsize=(6.5,3.35), dpi=300)
    
    for mID in range(len(FutureOutputs['mlabel'])):
        # Calculate smoothed version of predictions
        SmoothVE = MovingAverage(FutureOutputs['output'][mID]['futureVE'], 10)
        SmoothWL = MovingAverage(FutureOutputs['output'][mID]['futureWL'], 10)    
        
        if PlotDateRange is None:
            TrainStart = mdates.date2num(TransectDFTrain.index[0])
            TrainEnd = mdates.date2num(TransectDFTrain.index[round(len(TransectDFTrain)-(len(TransectDFTrain)*0.1))])
            ValEnd = mdates.date2num(TransectDFTrain.index[-1])
            TestEnd = mdates.date2num(TransectDFTest.index[-1])
            TrainT = mpatches.Rectangle((TrainStart,-100), TrainEnd-TrainStart, 1000, fc=[0.8,0.8,0.8], ec=None)
            ValT = mpatches.Rectangle((TrainEnd,-100), ValEnd-TrainEnd, 1000, fc=[0.9,0.9,0.9], ec=None)
            # TestT = mpatches.Rectangle((0,0), 10, 10, fc='red', ec=None, alpha=0.3)
            ax.add_patch(TrainT) 
            ax.add_patch(ValT)
            
            plt.text(TrainStart+(TrainEnd-TrainStart)/2, 20, 'Training', ha='center')
            plt.text(TrainEnd+(ValEnd-TrainEnd)/2, 20, 'Validation', ha='center')
            plt.text(ValEnd+(TestEnd-ValEnd)/2, 20, 'Test', ha='center')
            
            plt.xlim(TrainStart,TestEnd)
        else:
            plt.xlim(mdates.date2num(PlotDateRange[0]),mdates.date2num(PlotDateRange[1]))
        
        
        lw = 1 # line width
        # Plot cross-shore distances through time for WL and VE past
        plt.plot(pd.concat([TransectDFTrain['VE'], TransectDFTest['VE']]), '#79C060', lw=lw, label=r'${VE}$')
        plt.plot(pd.concat([TransectDFTrain['WL'], TransectDFTest['WL']]), '#3E74B3', lw=lw, label=r'${WL}$')
        # Plot cross-shore distances through time for test data
        # plt.plot(TransectDFTest['distances'], '#79C060', ls=(0, (1, 1)), lw=lw, label='Test VE')
        # plt.plot(TransectDFTest['wlcorrdist'], '#3E74B3', ls=(0, (1, 1)), lw=lw, label='Test WL')
        # Plot predicted WL and VE
        plt.plot(FutureOutputs['output'][mID]['futureVE'], 'C8', alpha=0.3, lw=lw, label=r'$\hat{VE}$')
        plt.plot(FutureOutputs['output'][mID]['futureWL'], 'C9', alpha=0.3, lw=lw, label=r'$\hat{WL}$')
        # Plot smoothed predicted WL and VE
        # plt.plot(FutureOutputs['output'][mID]['futureVE'].index, SmoothVE, 'C8', alpha=1, lw=lw, label=r'$\hat{VE}_{t[i:i+10]}$')
        # plt.plot(FutureOutputs['output'][mID]['futureWL'].index, SmoothWL, 'C9', alpha=1, lw=lw, label=r'$\hat{WL}_{t[i:i+10]}$')

    plt.xlabel('Date (yyyy)')
    plt.ylabel('Cross-shore distance (m)')
    # plt.legend(loc='upper left', ncols=3)
    plt.ylim(0,600)
    ax.tick_params(axis='both',which='major',pad=2)
    ax.xaxis.labelpad=2
    ax.yaxis.labelpad=2

    plt.tight_layout()
    plt.show()
    
    # StartTime = FutureOutputs['output'][mID].index[0].strftime('%Y-%m-%d')
    # EndTime = FutureOutputs['output'][mID].index[-1].strftime('%Y-%m-%d')
    StartTime = mdates.num2date(plt.axis()[0]).strftime('%Y-%m-%d')
    EndTime = mdates.num2date(plt.axis()[1]).strftime('%Y-%m-%d')
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
                                      ['WaveDirFD',
                                       'Runups',
                                       'Iribarren',
                                       'VE',
                                       'WL'],
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
        TrainEnd = mdates.date2num(TransectDFTrain.index[round(len(TransectDFTrain)-(len(TransectDFTrain)*0.1))])
        ValEnd = mdates.date2num(TransectDFTrain.index[-1])
        TrainT = mpatches.Rectangle((TrainStart,-100), TrainEnd-TrainStart, 1000, fc=[0.8,0.8,0.8], ec=None)
        ValT = mpatches.Rectangle((TrainEnd,-100), ValEnd-TrainEnd, 1000, fc=[0.9,0.9,0.9], ec=None)
        
        ax.add_patch(TrainT) 
        ax.add_patch(ValT)
    
        ax.plot(TransectDFTrain[yvar], c=c, lw=lw)
        ax.plot(VarDFDay[yvar], c=c, lw=lw, alpha=0.3)
        
        if i == 3:
            ax.plot(TransectDFTest['VE'], 'C2', ls=(0, (1, 1)), lw=lw, label='Test VE')
            ax.plot(FutureOutputs['output'][mID]['futureVE'], 'C8', alpha=0.7, lw=lw, label='Pred. VE')
            ax.legend(loc='upper left', ncols=3)
        elif i == 4:
            ax.plot(TransectDFTest['WL'], 'C0', ls=(0, (1, 1)), lw=lw, label='Test WL')
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


def PlotTestScatter(FutureOutputs, TransectDFTest, mID, Tr, filepath, sitename):
    """
    
    FM Mar 2025

    Parameters
    ----------
    FutureOutputs : dict
        Dict storing per-model dataframes of future cross-shore waterline and veg edge predictions.
    TransectDFTest : DataFrame
        Past data sliced from most recent (unseen) end of TransectDF to use for testing the model.
    mID : int
        ID of the chosen model run stored in FutureOutputs.
    Tr : int
        ID of the chosen cross-shore transect.
    filepath : str
        Local path to COASTGUARD Data folder.
    sitename : str
        Name of site of interest.

    """
    # fig, axs = plt.subplots(1,2, figsize=(6.55,3.23), dpi=300)
    fig, axs = plt.subplots(1,1, figsize=(3.11,3.11), dpi=300)

    BothSLs = []
    for SL, SLc in zip(['WL', 'VE'], ['#3E74B3','#79C060']):
        # Define actual and predicted VE and WL
        realVals = TransectDFTest[SL]
        predVals = FutureOutputs['output'][mID]['future'+SL]
        # Match indexes and remove NaNs from CreateSequence moving window
        ComboDF = pd.concat([realVals, predVals], axis=1)
        ComboDF.dropna(how='any', inplace=True)
        
        BothSLs.append(ComboDF)
        
        # Line through the origin
        axs.plot([-100,1000],[-100,1000],c=[0.5,0.5,0.5], lw=0.8, linestyle='-', alpha=0.35)
        # Scatter plot of observed vs predicted SL
        axs.scatter(ComboDF[SL], ComboDF['future'+SL], c=SLc, s=1, alpha=0.5)
        # Smooth version of predicted SL
        # SmoothPred = MovingAverage(ComboDF['future'+SL], 10)
        # axs.scatter(ComboDF[SL], SmoothPred, c='k', s=0.5)
        
        # SL linear regression
        m, b = np.polyfit(ComboDF[SL], ComboDF['future'+SL], 1)  # Fit a linear trend (y = mx + b)
        sorted_x = np.sort(ComboDF[SL])  
        sorted_y = m * sorted_x + b       
        # R squared and Pearson r
        r2 = r2_score(ComboDF[SL], ComboDF['future'+SL])
        rval, _ = pearsonr(ComboDF[SL], ComboDF['future'+SL])

        # Plot lines of best fit
        axs.plot(sorted_x, sorted_y, 
                c='w', lw=1.7, alpha=0.7, zorder=2)
        axs.plot(sorted_x, sorted_y, 
                  color=SLc, linestyle='--', lw=1, zorder=3,
                  label=f'{SL} $R^2$ = {round(r2,2)}, $r$ = {round(rval,2)}')
    
    # overall linear regression
    BothReal = (list(BothSLs[1]['VE'])) + list(BothSLs[0]['WL']) 
    BothPred = (list(BothSLs[1]['futureVE'])) + list(BothSLs[0]['futureWL'])
    m, b = np.polyfit(BothReal, BothPred, 1)  # Fit a linear trend (y = mx + b)
    Both_sorted_x = np.sort(BothReal)  
    Both_sorted_y = m * Both_sorted_x + b
    # R squared  and Pearson r
    Both_r2 = r2_score(BothReal, BothPred)
    Both_rval, _ = pearsonr(ComboDF[SL], ComboDF['future'+SL])

    # Plot lines of best fit
    axs.plot(Both_sorted_x, Both_sorted_y, 
            c='w', lw=1.7, alpha=0.7, zorder=2)
    axs.plot(Both_sorted_x, Both_sorted_y, 
              color='#163E64', linestyle='--', lw=1, zorder=3,
              label=f'$R^2$ = {round(Both_r2,2)}, $r$ = {round(Both_rval,2)}')
        
    axs.set_xlim(0,600)
    axs.set_ylim(0,600)
    axs.set_xlabel(r'$y$ cross-shore distance (m)')
    axs.set_ylabel(r'$\hat{y}$ cross-shore distance (m)')
    axs.legend()
    
        # # Plot histogram of residuals
        # axs[1].hist(ComboDF['future'+SL] - ComboDF[SL], bins=np.arange(-300,300,10), 
        #             fc=SLc, alpha=0.5,
        #             label=f'$\\hat{{{SL}}} - {SL}$')
        
        # axs[1].legend()
        
    plt.tight_layout()
    plt.show()
    
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_VEWLErrorLinReg_Tr'+str(Tr)+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)
    

def FutureDiffViolin(FutureOutputs, mID, TransectDF, filepath, sitename, Tr):
    """
    
    FM Mar 2025

    Parameters
    ----------
    FutureOutputs : dict
        Dict storing per-model dataframes of future cross-shore waterline and veg edge predictions.
    mID : int
        ID of the chosen model run stored in FutureOutputs.
    TransectDF : DataFrame
        Dataframe of per-transect coastal metrics/variables in timeseries.
    filepath : str
        Local path to COASTGUARD Data folder.
    sitename : str
        Name of site of interest.
    Tr : int
        ID of the chosen cross-shore transect.

    """
    fig, axs = plt.subplots(1,1, figsize=(3.11,3.11), dpi=300)
    fs = 7
    
    # Rename columns so seaborn legend works
    leglabs = {'VEdiff':None, 'WLdiff':None}
    for SL, SLreal in zip(leglabs.keys(), ['VE','WL']):
        SLmedian = FutureOutputs['XshoreDiff'][mID][SL].median()
        SLmedianPcnt = abs(SLmedian / (TransectDF[SLreal].max()-TransectDF[SLreal].min()))*100
        leglabs[SL] = f"""$\\eta_{{\\hat{{{SL[:-4]}}} - {SL[:-4]}}}$ = {round(SLmedian,1)}m\n$\\frac{{\\eta_{{\\hat{{{SL[:-4]}}} - {SL[:-4]}}}}}{{{SL[:-4]}_{{[min,max]}}}}$ = {round(SLmedianPcnt)}%"""

    pltDict = dict((leglabs[key], value) for (key, value) in FutureOutputs['XshoreDiff'][mID].items())
    # Swap so WL is on top
    pltDF = pd.DataFrame(pltDict)
    pltDF = pltDF.iloc[:,[1,0]]
    ax = sns.violinplot(data = pltDF, 
                        palette=['#3E74B3', '#79C060'], linewidth=0.0001, orient='h', 
                        cut=0, inner='quartile', density_norm='count',
                        legend='full')
    for l in ax.lines:
        # l.set_linestyle('--')
        l.set_linewidth(1)
        l.set_color('white')
        
    ax.set_xlabel(r'Cross-shore $\hat{y}-y$ (m)',fontsize=fs)
    ax.tick_params(axis='y', left=False, labelleft=False)
    # ax.set_yticklabels(['VE','WL'], rotation=90, va='center')
    # ax.tick_params(axis='y', labelrotation=90)
    ax.tick_params(labelsize=fs)
    ax.grid(which='major', axis='x', c=[0.8,0.8,0.8], lw=0.5, alpha=0.3)
    xlims = [round(lim,-1) for lim in ax.get_xlim()]
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_xticks(np.arange(xlims[0],xlims[1],10), minor=True)
    
    ax.axvline(0, c='#163E64', ls='-', alpha=0.5, lw=0.5)
        
    sns.move_legend(ax, 'lower left')
    plt.tight_layout()
    plt.show()
    
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_VEWLErrorViolin_Tr'+str(Tr)+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)


def FutureViolinLinReg(FutureOutputs, mID, TransectDF, filepath, sitename, Tr):
    """
    
    FM Mar 2025

    Parameters
    ----------
    FutureOutputs : dict
        Dict storing per-model dataframes of future cross-shore waterline and veg edge predictions.
    mID : int
        ID of the chosen model run stored in FutureOutputs.
    TransectDF : DataFrame
        Dataframe of per-transect coastal metrics/variables in timeseries.
    filepath : str
        Local path to COASTGUARD Data folder.
    sitename : str
        Name of site of interest.
    Tr : int
        ID of the chosen cross-shore transect.

    """
    fig, axs = plt.subplots(1,2, figsize=(6.55,3.23), dpi=300)
    fs = 7
    
    # VIOLIN PLOT
    # Rename columns so seaborn legend works
    leglabs = {'VEdiff':None, 'WLdiff':None}
    for SL, SLreal in zip(leglabs.keys(), ['VE','WL']):
        SLRMSEPct = abs(FutureOutputs['rmse'][mID]['future'+SLreal] / (TransectDF[SLreal].max()-TransectDF[SLreal].min()))*100
        SLmedian = FutureOutputs['XshoreDiff'][mID][SL].median()
        SLmedianPcnt = abs(SLmedian / (TransectDF[SLreal].max()-TransectDF[SLreal].min()))*100
        leglab1 = f"$RMSE_{{{SLreal}}}$ = {round(FutureOutputs['rmse'][mID]['future'+SLreal],1)} m\n"
        leglab2 = f"$\\frac{{RMSE_{{{SL[:-4]}}}}}{{{SL[:-4]}_{{[min,max]}}}}$ = {round(SLRMSEPct)}%\n"
        leglab3 = f"$\\eta_{{\\hat{{{SL[:-4]}}} - {SL[:-4]}}}$ = {round(SLmedian,1)} m\n"
        leglab4 = f"$\\frac{{\\eta_{{\\hat{{{SL[:-4]}}} - {SL[:-4]}}}}}{{{SL[:-4]}_{{[min,max]}}}}$ = {round(SLmedianPcnt)}%"
        leglabs[SL] = leglab1 + leglab2 + leglab3 + leglab4
        

    pltDict = dict((leglabs[key], value) for (key, value) in FutureOutputs['XshoreDiff'][mID].items())
    # Swap so WL is on top
    pltDF = pd.DataFrame(pltDict)
    pltDF = pltDF.iloc[:,[1,0]]
    sns.violinplot(data = pltDF, 
                        palette=['#3E74B3', '#79C060'], linewidth=0.0001, orient='h', 
                        cut=0, inner='quartile', density_norm='count',
                        ax=axs[0], legend='full')
    for l in axs[0].lines:
        # l.set_linestyle('--')
        l.set_linewidth(1)
        l.set_color('white')
        
    # axs[0].set_xlabel(r'Cross-shore $\hat{y}-y$ (m)',fontsize=fs)
    multicolour_axlabel(axs[0], ('land    ', r'Cross-shore $\hat{y}-y$ (m)', '    sea'), 
                               ('#C2B280', 'k', '#236E95'), 
                       (0.18, -0.14), axis='x')
    axs[0].tick_params(axis='y', left=False, labelleft=False)
    # ax.set_yticklabels(['VE','WL'], rotation=90, va='center')
    # ax.tick_params(axis='y', labelrotation=90)
    axs[0].tick_params(labelsize=fs)
    axs[0].grid(which='major', axis='x', c=[0.8,0.8,0.8], lw=0.5, alpha=0.3)
    xlims = [round(lim,-1) for lim in axs[0].get_xlim()]
    axs[0].set_xlim(xlims[0],xlims[1])
    axs[0].set_xticks(np.arange(xlims[0],xlims[1],10), minor=True)
    
    axs[0].axvline(0, c=[0.5,0.5,0.5], lw=0.9, linestyle='-', alpha=0.35)
        
    sns.move_legend(axs[0], 'lower left')
    
    
    # LIN REG PLOT
    BothSLs = []
    for SL, SLc in zip(['WL', 'VE'], ['#3E74B3','#79C060']):
        # Define actual and predicted VE and WL
        realVals = TransectDF[SL]
        predVals = FutureOutputs['output'][mID]['future'+SL]
        # Match indexes and remove NaNs from CreateSequence moving window
        ComboDF = pd.concat([realVals, predVals], axis=1)
        ComboDF.dropna(how='any', inplace=True)
        
        BothSLs.append(ComboDF)
        
        # Line through the origin
        axs[1].plot([-100,1000],[-100,1000],c=[0.5,0.5,0.5], lw=0.9, linestyle='-', alpha=0.35)
        # Scatter plot of observed vs predicted SL
        axs[1].scatter(ComboDF[SL], ComboDF['future'+SL], c=SLc, s=1, alpha=0.5)
        # Smooth version of predicted SL
        # SmoothPred = MovingAverage(ComboDF['future'+SL], 10)
        # axs[1].scatter(ComboDF[SL], SmoothPred, c='k', s=0.5)
        
        # SL linear regression
        m, b = np.polyfit(ComboDF[SL], ComboDF['future'+SL], 1)  # Fit a linear trend (y = mx + b)
        sorted_x = np.sort(ComboDF[SL])  
        sorted_y = m * sorted_x + b       
        # R squared and Pearson r
        r2 = r2_score(ComboDF[SL], ComboDF['future'+SL])
        rval, _ = pearsonr(ComboDF[SL], ComboDF['future'+SL])

        # Plot lines of best fit
        axs[1].plot(sorted_x, sorted_y, 
                c='#163E64', lw=1.7, alpha=0.6, zorder=2)
        axs[1].plot(sorted_x, sorted_y, 
                  color=SLc, linestyle='--', lw=1, zorder=3,
                  label=f'$R^2_{{{SL}}}$ = {round(r2,2)}, $r_{{{SL}}}$ = {round(rval,2)}')
    
    # overall linear regression
    BothReal = (list(BothSLs[1]['VE'])) + list(BothSLs[0]['WL']) 
    BothPred = (list(BothSLs[1]['futureVE'])) + list(BothSLs[0]['futureWL'])
    m, b = np.polyfit(BothReal, BothPred, 1)  # Fit a linear trend (y = mx + b)
    Both_sorted_x = np.sort(BothReal)
    Both_sorted_y = m * Both_sorted_x + b
    # R squared  and Pearson r
    Both_r2 = r2_score(BothReal, BothPred)
    Both_rval, _ = pearsonr(ComboDF[SL], ComboDF['future'+SL])

    # Plot lines of best fit
    axs[1].plot(Both_sorted_x, Both_sorted_y, 
            c='w', lw=1.7, alpha=0.7, zorder=2)
    axs[1].plot(Both_sorted_x, Both_sorted_y, 
              color='#163E64', linestyle='--', lw=1, zorder=3,
              label=f'$R^2$ = {round(Both_r2,2)}, $r$ = {round(Both_rval,2)}')
        
    axs[1].set_xlim(0,600)
    axs[1].set_ylim(0,600)
    axs[1].grid(which='major', axis='both', c=[0.8,0.8,0.8], lw=0.5, alpha=0.3)
    
    multicolour_axlabel(axs[1], ('land    ', r'$y$ cross-shore distance (m)', '    sea'), 
                               ('#C2B280', 'k', '#236E95'), 
                       (0.18, -0.14), axis='x')
    multicolour_axlabel(axs[1], ('land    ', r'$\hat{y}$ cross-shore distance (m)', '    sea'), 
                               ('#C2B280', 'k', '#236E95'), 
                       (-0.15, 0.14), axis='y')
    
    axs[1].legend(loc='lower right')
    
    labs = list(string.ascii_lowercase[:axs.shape[0]])
    for ax, lab in zip(axs, labs):
        ax.text(0.009, 0.991, '('+lab+')', transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(facecolor='w', edgecolor='k',pad=1.5))
        
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.show()
    
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_VEWLErrorViolinLinReg_'+FutureOutputs['mlabel'][mID]+'_Tr'+str(Tr)+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)
    


def PlotSiteRMSE(FutureOutputs, filepath, sitename, Subtitle=''):
    """
    
    FM Apr 2025

    Parameters
    ----------
    FutureOutputs : dict
        Dict storing per-model dataframes of future cross-shore waterline and veg edge predictions.
    filepath : str
        Local path to COASTGUARD Data folder.
    sitename : str
        Name of site of interest.
    Subtitle : str, optional
        Additional filename title to differentiate between full site model runs. The default is ''.
    """
    fig, axs = plt.subplots(1,2, figsize=(3.25,5))
    
    SLkeys = ['futureVE','futureWL']
    
    # Extract out RMSE values
    RMSEList = dict(zip(SLkeys, [[],[]]))
    for Tr in FutureOutputs.keys():
        for SL in SLkeys:
            RMSEList[SL].append(FutureOutputs[Tr]['rmse'][0][SL])
        
    for SL, SLc, ax, SLlab in zip(SLkeys, ['#79C060','#3E74B3'], axs, [r'$VE$',r'$WL$']):
        RMSEArray = np.array(RMSEList[SL])
        Pct = round(np.percentile(RMSEArray, 75))
        ax.scatter(RMSEArray[np.where(RMSEArray > Pct)], 
                   np.array(list(FutureOutputs.keys()))[np.where(RMSEArray > Pct)],
                   s=15, facecolor='w', marker='o', edgecolors=SLc, label=SLlab)
        ax.scatter(RMSEArray[np.where(RMSEArray < Pct)], 
                   np.array(list(FutureOutputs.keys()))[np.where(RMSEArray < Pct)],
                   s=15, facecolor=SLc, marker='o', edgecolors=None, label=SLlab+f' < 75$^{{th}}$%\n({Pct} m)')
        # handles, labels = ax.get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # ax.legend(by_label.values(), by_label.keys(), loc='center right')
        ax.legend(loc='center right')
        ax.set_xlim(0)
        ax.set_ylim(0,list(FutureOutputs.keys())[-1]+5)
    
    axs[1].yaxis.set_tick_params(labelleft=False)
    axs[0].set_ylabel('Transect ID')
    # Shared axis label
    fig.supxlabel('            RMSE (m)')
    
    for ax, lab in zip(axs, list(string.ascii_lowercase[:2])):
        ax.text(0.9,0.993, '('+lab+')', transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(facecolor='w', edgecolor='k',pad=1.5))
        
    plt.tight_layout()
    plt.show()
    
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_FullSiteRMSE'+Subtitle+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)
    
    
def PlotRMSE_Rt(CoastalGDF, filepath, sitename, Subtitle=''):
    """
    
    FM Apr 2025

    Parameters
    ----------
    CoastalGDF : GeoDataFrame
        DataFrame of cross-shore transects (rows) and intersected coastal 
        timeseries/metrics (columns), with RMSE stats as additional columns and 
        a geometry column from the Transect GDFs.
    filepath : str
        Local path to COASTGUARD Data folder.
    sitename : str
        Name of site of interest.
    Subtitle : str, optional
        Additional filename title to differentiate between full site model runs. The default is ''.

    """
    fig, axs = plt.subplots(1,2, figsize=(6.55,3.28))
    
    # Only use non-nan values across both columns
    badVE = ~np.logical_or(np.isnan(CoastalGDF['VE_RMSE']), np.isnan(CoastalGDF['oldyoungRt']))
    badWL = ~np.logical_or(np.isnan(CoastalGDF['WL_RMSE']), np.isnan(CoastalGDF['oldyungRtW']))
    
    VEr,_ = pearsonr(CoastalGDF['VE_RMSE'][badVE], np.abs(CoastalGDF['oldyoungRt'][badVE]))
    WLr,_ = pearsonr(CoastalGDF['WL_RMSE'][badWL], np.abs(CoastalGDF['oldyungRtW'][badWL]))
    
    # Plot scatters of both VE and WL RMSE against change rate
    axs[0].scatter(CoastalGDF['VE_RMSE'][badVE], np.abs(CoastalGDF['oldyoungRt'][badVE]), s=15, facecolor='#79C060',
                   label=f'$r_{{VE}}$ = {round(VEr,2)}')
    axs[1].scatter(CoastalGDF['WL_RMSE'][badWL], np.abs(CoastalGDF['oldyungRtW'][badWL]), s=15, facecolor='#3E74B3',
                   label=f'$r_{{WL}}$ = {round(WLr,2)}')
        
    axs[0].set_xlabel(r'$RMSE_{VE}$ (m)')
    axs[1].set_xlabel(r'$RMSE_{WL}$ (m)')
    axs[0].set_ylabel(r'$\Delta VE$ (m/yr)')
    axs[1].set_ylabel(r'$\Delta WL$ (m/yr)')
    
    axs[0].set_xlim(0)
    axs[1].set_xlim(0)
    axs[0].set_ylim(0)
    axs[1].set_ylim(0)
    
    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')
    
    for ax, lab, xpos, ypos in zip(axs, list(string.ascii_lowercase[:2]), [0.955, 0.955], [0.992, 0.992]):
        ax.text(xpos,ypos, '('+lab+')', transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(facecolor='w', edgecolor='k',pad=1.5))
    
    plt.tight_layout()
    plt.show()
    
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_FullSiteRMSE_Rt'+Subtitle+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)
    
    
def PlotImpactClasses(filepath, sitename, Tr, ImpactClass, TransectDF):
    """
    
    FM Apr 2025

    Parameters
    ----------
    filepath : str
        Local path to COASTGUARD Data folder.
    sitename : str
        Name of site of interest.
    Tr : int
        ID of the chosen cross-shore transect.
    ImpactClass : dict
        Dict of impact classifications for futureVE and futureWL, plus the bounds
        calculated to apply the classifications.
    TransectDF : DataFrame
        Dataframe of per-transect coastal metrics/variables in timeseries.

    """
    # fig, ax = plt.subplots(1,1, figsize=(6.5,3.25))
    fig, ax = plt.subplots(1,1, figsize=(4.7,2.))

    if 'future' in TransectDF.columns.any():
        SLkeys = ['futureVE','futureWL']
    else:
        SLkeys = ['VE', 'WL']
    msize = 10
    for SL, SLc in zip(SLkeys, ['#79C060','#3E74B3']):
        # Plot cross-shore distances through time for WL and VE past
        ax.plot(TransectDF[SL].index, TransectDF[SL], 
                   color=SLc, lw = 0.8, label=f"${SL}$")
        #['red','orange','green']
        for ImpID, ImpCl, ImpLab in zip([3,2,1], ['#D52941','#FF990A','#E4FF5C'], ['High', 'Medium', 'Low']):
            ax.scatter(TransectDF[SL].index[ImpactClass[SL]==ImpID], TransectDF[SL][ImpactClass[SL]==ImpID], 
                       s=msize, marker='^', linewidths=0.3, c=ImpCl, label=f'{ImpLab} impact')

    ax.set_xlabel('Date (yyyy)')
    ax.set_ylabel('Cross-shore distance (m)')
    
    ax.set_xlim(TransectDF.index.min(), TransectDF.index.max())
    ax.set_ylim(0,600)
    ax.tick_params(axis='both',which='major',pad=2)
    ax.xaxis.labelpad=2
    ax.yaxis.labelpad=2
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = {labels[0]: handles[0],
                labels[4]: handles[4],
                labels[1]: handles[1],
                labels[2]: handles[2],
                labels[3]: handles[3]}
    plt.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5,
               handlelength=1, handletextpad=0.5)
    
    plt.tight_layout()
    plt.show()
    
    FigPath = os.path.join(filepath, sitename, 'plots', 
                           sitename+'_ImpactClassTimeseries_Tr'+str(Tr)+'.png')
    plt.savefig(FigPath, dpi=300, bbox_inches='tight',transparent=False)
    
    
def multicolour_axlabel(ax, list_of_strings, list_of_colors, bboxes, axis='x', anchorpad=0,**kw):
    """
    Axis label which has different font colours for different components. Used
    for labelling 'land' and 'sea' as yellow and blue respectively.
    FM Apr 2025

    Parameters
    ----------
    ax : matplotlib.Axes
        Axis to apply multicoloured label to .
    list_of_strings : tuple
        Separate strings to use for axis label.
    list_of_colors : tuple
        Separate font colours to use for axis label (same length as list_of_strings).
    bboxes : tuple
        Bounding box of where to place label.
    axis : str, optional
        Label x or y axis. The default is 'x'.
    anchorpad : float, optional
        Amount of padding to add to label. The default is 0.
    **kw : Text properties
        Additioonal keyword arguments for font properties.

    """
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw)) 
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,
                                          bbox_to_anchor=bboxes,
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw)) 
                     for text,color in zip(list_of_strings[::-1],list_of_colors[::-1]) ]
        ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, 
                                          bbox_to_anchor=bboxes, 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)
      
        
        
      
        
      
        
# ----------------------------------------------------------------------------------------
#%% CLUSTERING FUNCTIONS ###

def PlotClusteredTS(VarDF):
    """
    
    FM Sep 2024

    Parameters
    ----------
    VarDF : DataFrame
        Dataframe of per-transect coastal metrics/variables in (irregular) timeseries.


    """
    fig, ax = plt.subplots(figsize=(10, 5))
    bluecm = cm.get_cmap('cool')
    greencm = cm.get_cmap('summer')
    ax.scatter(VarDF.index, 
                VarDF['WaveHs'], 
                c=VarDF['Cluster'], marker='X', cmap=bluecm)
    ax2 = ax.twinx()
    ax2.scatter(VarDF.index, 
                VarDF['distances'], 
                c=VarDF['Cluster'], marker='^', cmap=greencm)  
    ax.set_xlabel('Time')
    ax.set_ylabel('Significant wave height (m)')
    ax2.set_ylabel('VE distance (m)')
    plt.show()

    
def PlotCluster(VarDF, pca_df, scale_factor, eigenvectors, variances):
    """
    
    FM Sep 2024

    Parameters
    ----------
    VarDF : DataFrame
        Dataframe of per-transect coastal metrics/variables in (irregular) timeseries.
    pca_df : DataFrame
        Principal Component Analysis dataframe generated from Predictions.Cluster().
        Output of sklearn.decomposition.PCA().
    scale_factor : float
        Scaling factor to fit points on plot.
    eigenvectors : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of maximum variance in the data.
    variances : ndarray of shape (n_components,)
        Percentage of variances explained by each component.

    """
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
        
        
def PlotClusterElbowSil(k_n, inertia, sil_scores):
    """
    
    FM Sep 2024

    Parameters
    ----------
    k_n : int
        k-means cluster number.
    inertia : list of floats
        Sum of squared distances of samples to their closest cluster center, 
        weighted by the sample weights if provided..
    sil_scores : list of floats
        Mean Silhouette Coefficient for all samples, for assessing closeness
        of clusters.

    """
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
    
    
def PlotClusterVisuals(VarDF, Mod, pca_df, eigenvectors):
    """
    
    FM Sep 2024

    Parameters
    ----------
    VarDF : DataFrame
        Dataframe of per-transect coastal metrics/variables in (irregular) timeseries.
    Mod : str
        Clustering model type.
    pca_df : DataFrame
        Principal Component Analysis dataframe generated from Predictions.Cluster().
        Output of sklearn.decomposition.PCA().
    eigenvectors : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of maximum variance in the data.

    
    """
    # Optional: Visualization of clusters
    fig, ax = plt.subplots(figsize=(10, 5))
    bluecm = cm.get_cmap('cool')
    greencm = cm.get_cmap('summer')
    ax.scatter(VarDF.index, 
               VarDF['WaveHs'], 
               c=VarDF[Mod+'Cluster'], marker='X', cmap=bluecm)
    ax2 = ax.twinx()
    ax2.scatter(VarDF.index, 
               VarDF['VE'], 
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
    