#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 13:15:00 2022

@author: fmuir
"""

#%% Imports and Initialisation


import os
import sys
import numpy as np
import pickle
import string
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")
import pdb

import seaborn as sns
# sns.set(style='whitegrid') #sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
import matplotlib as mpl
from matplotlib import cm
import matplotlib.colors as pltcls
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import matplotlib.dates as mdates

plt.ion()

from shapely import geometry
from shapely.geometry import Point, LineString
import rasterio

from Toolshed import Toolbox, Transects, Image_Processing

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from scipy.stats import spearmanr



import geemap
import ee

import pandas as pd
import geopandas as gpd

import csv
import math

ee.Initialize()


# SCALING:
# Journal 2-column width: 224pt or 3.11in
# Journal 1-column width: 384pt or 5.33in
# Spacing between: 0.33in
# Journal 2-column page: 6.55in


#%%
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



def InterpNaN(listvals):
    """
    Interpolate over NaN values in a timeseries.
    FM Nov 2023

    Parameters
    ----------
    listvals : list
        Values with NaNs to interpolate over.

    Returns
    -------
    listinterp : list
        Filled timseries.

    """
    listnans = [x if (x > (np.mean(listvals) - 2*np.std(listvals)) 
                         and x < (np.mean(listvals) + 2*np.std(listvals))) 
                   else np.nan for x in listvals]
    listinterp = pd.Series(listnans).interpolate(limit=10, limit_direction='both').tolist()
    return listinterp




def ValidViolin(sitename, ValidationShp, DatesCol, ValidDF, TransectIDs):
    """
    Violin plot showing distances between validation and satellite, for each date of validation line.
    FM Oct 2022

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    ValidationShp : str
        Path to validation lines shapefile.
    DatesCol : str
        Name of dates column in shapefile.
    ValidDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with validation lines.
    TransectIDs : list
        Transect IDs to plot (range from ID1 to ID2).

    """
    
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)
    
    ValidGDF = gpd.read_file(ValidationShp)
    violin = []
    violindates = []
    Vdates = ValidGDF[DatesCol].unique()
    for Vdate in Vdates:
        valsatdist = []
        for Tr in range(TransectIDs[0],TransectIDs[1]): 
            if Tr > len(ValidDF['Vdates']): # for when transect values extend beyond what transects exist
                print("check your chosen transect values!")
                return
            if Vdate in ValidDF['Vdates'].iloc[Tr]:
                DateIndex = (ValidDF['Vdates'].iloc[Tr].index(Vdate))
                # rare occasion where transect intersects valid line but NOT sat line (i.e. no distance between them)
                if ValidDF['valsatdist'].iloc[Tr] != []:
                    valsatdist.append(ValidDF['valsatdist'].iloc[Tr][DateIndex])
                else:
                    continue
            else:
                continue
        # due to way dates are used, some transects might be missing validation dates so violin collection will be empty
        if valsatdist != []: 
            violin.append(valsatdist)
            violindates.append(Vdate)
    # sort both dates and list of values by date
    if len(violindates) > 1:
        violindatesrt, violinsrt = [list(d) for d in zip(*sorted(zip(violindates, violin), key=lambda x: x[0]))]
    else:
        violindatesrt = violindates
        violinsrt = violin
    df = pd.DataFrame(violinsrt)
    df = df.transpose()
    df.columns = violindatesrt
    
    f = plt.figure(figsize=(14, 6))
    if len(violindates) > 1:
        ax = sns.violinplot(data = df, linewidth=1, palette = 'magma_r', orient='h')
    else:
        ax = sns.violinplot(data = df, linewidth=1, orient='h',)
        
    ax.set(xlabel='Distance$_{satellite - validation}$ (m)', ylabel='Validation line date')
    ax.set_title('Accuracy of Transects ' + str(TransectIDs[0]) + ' to ' + str(TransectIDs[1]))
    
    # set axis limits to rounded maximum value of all violins (either +ve or -ve)
    axlim = round(np.max([abs(df.min().min()),abs(df.max().max())]),-1)
    ax.set_xlim(-axlim, axlim)
    ax.set_xticks([-30,-15,-10,10,15,30],minor=True)
    ax.xaxis.grid(b=True, which='minor',linestyle='--', alpha=0.5)
    median = ax.axvline(df.median().mean(), c='r', ls='-.')
    
    handles = [median]
    labels = ['median' + str(round(df.median().mean(),1)) + 'm']
    ax.legend(handles,labels)
    
    ax.set_axisbelow(False)
    plt.tight_layout()
    
    figpath = os.path.join(filepath,sitename+'_Validation_Satellite_Distances_Violin_'+str(TransectIDs[0])+'to'+str(TransectIDs[1])+'.png')
    plt.savefig(figpath)
    print('figure saved under '+figpath)
    

def SatViolin(sitename, SatGDF, DatesCol, ValidDF, TransectIDs, PlotTitle):
    """
    Violin plot showing distances between validation and satellite, for each date of validation line.
    FM Oct 2022

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    SatGDF : GeoDataFrame
        GeoDataFrame of satellite-derived lines to use for unique dates.
    DatesCol : str
        Name of dates column in shapefile.
    ValidDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with validation lines.
    TransectIDs : list
        Transect IDs to plot (range from ID1 to ID2).
    PlotTitle : str
        Alternative plot title for placename locations.

    """
    
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)
    
       
    violin = []
    violindates = []
    Sdates = SatGDF[DatesCol].unique()
    
    for Sdate in Sdates:
        valsatdist = []
        # for each transect in given range
        for Tr in range(TransectIDs[0],TransectIDs[1]): 
            if Tr > len(ValidDF['dates']): # for when transect values extend beyond what transects exist
                print("check your chosen transect values!")
                return
            if Sdate in ValidDF['dates'].iloc[Tr]:
                DateIndex = (ValidDF['dates'].iloc[Tr].index(Sdate))
                # rare occasion where transect intersects valid line but NOT sat line (i.e. no distance between them)
                if ValidDF['valsatdist'].iloc[Tr] != []:
                    try:
                        valsatdist.append(ValidDF['valsatdist'].iloc[Tr][DateIndex])
                    except:
                        pdb.set_trace()
                else:
                    continue
            else:
                continue
        # due to way dates are used, some transects might be missing validation dates so violin collection will be empty
        if valsatdist != []: 
            violin.append(valsatdist)
            violindates.append(Sdate)
    # sort both dates and list of values by date
    if len(violindates) > 1:
        violindatesrt, violinsrt = [list(d) for d in zip(*sorted(zip(violindates, violin), key=lambda x: x[0]))]
    else:
        violindatesrt = violindates
        violinsrt = violin
    df = pd.DataFrame(violinsrt)
    df = df.transpose()
    df.columns = violindatesrt
    
    # initialise matching list of sat names for labelling
    satnames = dict.fromkeys(violindatesrt)
    # for each date in sorted list
    for Sdate in violindatesrt:    
        satmatch = []
        for Tr in range(len(ValidDF['TransectID'])):
            # loop through transects to find matching date from which to find satname
            if Sdate not in ValidDF['dates'].iloc[Tr]:
                continue
            else:
                satmatch.append(ValidDF['satname'].iloc[Tr][ValidDF['dates'].iloc[Tr].index(Sdate)])
        # cycling through transects leads to list of repeating satnames; take the unique entry
        satnames[Sdate] = list(set(satmatch))[0]
    
    f = plt.figure(figsize=(2.6, 4.51), dpi=300)
    sns.set(font_scale=0.5)
    
    patches = []
    rect10 = mpatches.Rectangle((-10, -50), 20, 100)
    rect15 = mpatches.Rectangle((-15, -50), 30, 100)
    patches.append(rect10)
    patches.append(rect15)
    coll=PatchCollection(patches, facecolor="black", alpha=0.05, zorder=0)
    
    sns.set_style("ticks", {'axes.grid' : False})
    if len(violindates) > 1:
        # plot stacked violin plots
        ax = sns.violinplot(data = df, linewidth=0, palette = 'magma_r', orient='h', cut=0, inner='quartile')
        ax.add_collection(coll)        # set colour of inner quartiles to white dependent on colour ramp 
        for l in ax.lines:
            l.set_linestyle('-')
            l.set_linewidth(1)
            l.set_color('white')
            
        # cut away bottom halves of violins
        # for violin in ax.collections:
        #     bbox = violin.get_paths()[0].get_extents()
        #     x0, y0, width, height = bbox.bounds
        #     violin.set_clip_path(plt.Rectangle((x0, y0), width, height / 2, transform=ax.transData))
    else:
        ax = sns.violinplot(data = df, linewidth=1, orient='h',cut=0, inner='quartile')
        ax.add_collection(coll)
        
    ax.set(xlabel='Distance$_{satellite - validation}$ (m)', ylabel='Validation line date')
    ax.set_title(PlotTitle)
    
    # set axis limits to rounded maximum value of all violins (either +ve or -ve)
    # round UP to nearest 10
    try:
        axlim = math.ceil(np.max([abs(df.min().min()),abs(df.max().max())]) / 10) * 10
        if axlim < 100:
            ax.set_xlim(-axlim, axlim)
        else:
            ax.set_xlim(-100,100)
    except:
        ax.set_xlim(-100, 100)
    
    # create specific median lines for specific platforms
    medians = []
    labels = []
    # dataframe dates and matching satnames
    satdf = pd.DataFrame(satnames, index=[0])
    # for each platform name
    uniquesats = sorted(set(list(satnames.values())))
    colors = plt.cm.Blues(np.linspace(0.4, 1, len(uniquesats)))
    for satname, c in zip(uniquesats, colors):
        sats = satdf.apply(lambda row: row[row == satname].index, axis=1)
        sats = sats[0].tolist()
        # get dataframe column indices for each date that matches the sat name
        colind = [df.columns.get_loc(sat) for sat in sats]
        # set the date axis label for each date to corresponding satname colour
        [ax.get_yticklabels()[ind].set_color(c) for ind in colind]
        # get median of only the columns that match each sat name
        concatl = []
        for s in sats:
            concatl.append(df[s])
        concatpd = pd.concat(concatl)
        medians.append(ax.axvline(concatpd.median(), c=c, ls='--', lw=1))
        if 'PSScene4Band' in satname:
            satname = 'PS'
        labels.append(satname + ' median = ' + str(round(concatpd.median(),1)) + 'm')
    
    ax.axvline(0, c='k', ls='-', alpha=0.4, lw=0.5)
    ax.legend(medians,labels, loc='lower right')
    
    ax.set_axisbelow(False)
    plt.tight_layout()
    
    figpath = os.path.join(filepath,sitename+'_Validation_Satellite_Distances_Violin_'+str(TransectIDs[0])+'to'+str(TransectIDs[1])+'.png')
    plt.savefig(figpath, dpi=300)
    print('figure saved under '+figpath)
    
    plt.show()
    

def SatPDF(sitename, SatGDF, DatesCol, ValidInterGDF, TransectIDs, PlotTitle):
    """
    Prob density function plot showing distances between validation and satellite, for each date of validation line.
    FM Oct 2022

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    SatGDF : GeoDataFrame
        GeoDataFrame of satellite-derived lines to use for unique dates.
    DatesCol : str
        Name of dates column in shapefile.
    ValidInterGDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with validation lines.
    TransectIDs : list
        Transect IDs to plot (range from ID1 to ID2).
    PlotTitle : str
        Alternative plot title for placename locations.

    """
    
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)
        
    violin = []
    violindates = []
    Sdates = SatGDF[DatesCol].unique()
    
    for Sdate in Sdates:
        valsatdist = []
        # for each transect in given range
        for Tr in range(TransectIDs[0],TransectIDs[1]): 
            if Tr > len(ValidInterGDF['dates']): # for when transect values extend beyond what transects exist
                print("check your chosen transect values!")
                return
            if Sdate in ValidInterGDF['dates'].iloc[Tr]:
                DateIndex = (ValidInterGDF['dates'].iloc[Tr].index(Sdate))
                # rare occasion where transect intersects valid line but NOT sat line (i.e. no distance between them)
                if ValidInterGDF['valsatdist'].iloc[Tr] != []:
                    valsatdist.append(ValidInterGDF['valsatdist'].iloc[Tr][DateIndex])
                else:
                    continue
            else:
                continue
        # due to way dates are used, some transects might be missing validation dates so violin collection will be empty
        if valsatdist != []: 
            violin.append(valsatdist)
            violindates.append(Sdate)
    # sort both dates and list of values by date
    if len(violindates) > 1:
        violindatesrt, violinsrt = [list(d) for d in zip(*sorted(zip(violindates, violin), key=lambda x: x[0]))]
    else:
        violindatesrt = violindates
        violinsrt = violin
    df = pd.DataFrame(violinsrt)
    df = df.transpose()
    df.columns = violindatesrt
    
    # initialise matching list of sat names for labelling
    satnames = dict.fromkeys(violindatesrt)
    # for each date in sorted list
    for Sdate in violindatesrt:    
        satmatch = []
        for Tr in range(len(ValidInterGDF['TransectID'])):
            # loop through transects to find matching date from which to find satname
            if Sdate not in ValidInterGDF['dates'].iloc[Tr]:
                continue
            else:
                satmatch.append(ValidInterGDF['satname'].iloc[Tr][ValidInterGDF['dates'].iloc[Tr].index(Sdate)])
        # cycling through transects leads to list of repeating satnames; take the unique entry
        satnames[Sdate] = list(set(satmatch))[0]
    
    f = plt.figure(figsize=(2.6, 4.58), dpi=300)
    ax = f.add_subplot(111)
    sns.set(font_scale=0.6)
    
    patches = []
    rect10 = mpatches.Rectangle((-10, -50), 20, 100)
    rect15 = mpatches.Rectangle((-15, -50), 30, 100)
    patches.append(rect10)
    patches.append(rect15)
    coll=PatchCollection(patches, facecolor="black", alpha=0.1, zorder=0)
    
    sns.axes_style("darkgrid")
    sns.set_style({'axes.facecolor':'#E0E0E0', 'axes.grid' : False})
    if len(violindates) > 1:
                   
        kdecmap = cm.get_cmap('magma_r',len(violindates))
        for i in range(len(violindates)):
            if df.iloc[:,i].isnull().sum() == df.shape[0]:
                kdelabel = None
            else:
                # find name of column for legend labelling (sat date)
                kdelabel = df.columns[i]
            sns.kdeplot(data = df.iloc[:,i], color=kdecmap.colors[i], label=kdelabel, alpha=0.8)
            
        ax.add_collection(coll)
        leg1 = ax.legend(loc='upper left', facecolor='w', framealpha=0.4)
            
    ax.set(xlabel='Distance$_{satellite - validation}$ (m)', ylabel='')
    ax.set_title(PlotTitle)
    plt.yticks([])
    
    # set axis limits to rounded maximum value of all violins (either +ve or -ve)
    # round UP to nearest 10
    try:
        axlim = math.ceil(np.max([abs(df.min().min()),abs(df.max().max())]) / 10) * 10
        ax.set_xlim(-axlim, axlim)
    except:
        ax.set_xlim(-100, 100)
      
    # land and sea labels on x axis
    axland = ax.twiny()
    axsea = ax.twiny()
    for xax, xlab, xloc, xcol in zip([axland, axsea], ['land', 'sea'], ['left', 'right'], ['#C2B280', '#236E95']):
        xax.xaxis.set_label_position('bottom')
        xax.xaxis.set_ticks_position('bottom')
        try:
            axlim = math.ceil(np.max([abs(df.min().min()),abs(df.max().max())]) / 10) * 10
            xax.set_xlim(-axlim, axlim)
        except:
            xax.set_xlim(-100, 100)
            
        xax.set_xlabel(xlab, loc=xloc)
        xax.xaxis.label.set_color(xcol)
        
        xax.xaxis.set_tick_params(width=0.5)
        xax.grid(False)
    ax.xaxis.set_tick_params(width=0.5)
    ax.grid(False)
    
    # create specific median lines for specific platforms
    medians = []
    labels = []
    # dataframe dates and matching satnames
    satdf = pd.DataFrame(satnames, index=[0])
    # remove empty columns to make plotting/legends easier
    df = df.dropna(axis=1, how='all')
    commondates=[col for col in df.columns.intersection(satdf.columns)]
    satdf = satdf[commondates]
    
    # for each platform name
    uniquesats = sorted(set(list(satnames.values())))
    colors = plt.cm.Blues(np.linspace(0.4, 1, len(uniquesats)))
    for satname, c in zip(uniquesats, colors):
        try:
            sats = satdf.apply(lambda row: row[row == satname].index, axis=1)
        except:
            print("Can't plot empty Transects with no validation data!")
            continue
        sats = sats[0].tolist()
        # skip calculating satellite median if transects are empty for this satellite
        if sats == []:
            continue
        # get dataframe column indices for each date that matches the sat name
        colind = [df.columns.get_loc(sat) for sat in sats]
        # set the date legend label for each date to corresponding satname colour
        [leg1.get_texts()[ind].set_color(c) for ind in colind]
            
        # get median of only the columns that match each sat name
        concatl = []
        for s in sats:
            concatl.append(df[s])
        concatpd = pd.concat(concatl)
        medians.append(ax.axvline(concatpd.median(), c=c, ls='--', lw=1))
        if 'PSScene4Band' in satname:
            satname = 'PS'
        labels.append(satname + ' $\eta$ = ' + str(round(concatpd.median(),1)) + 'm')
    
    # Overall error as text
    totald = []
    for date in df.columns:
        d = df[date]
        for i,datum in enumerate(d):
            totald.append(datum)

    totald = np.array(totald)
    mse = np.mean(np.power(totald[~np.isnan(totald)], 2))
    # mae = np.mean(abs(totald[~np.isnan(totald)]))
    rmse = np.sqrt(mse)
    
    l = Line2D([],[], color='none')
    medians.append(l)
    labels.append('RMSE = ' + str(round(rmse,1)) +'m')
    
    # set legend for median lines  
    ax.axvline(0, c='k', ls='-', alpha=0.4, lw=0.5)
    medleg = ax.legend(medians,labels, loc='upper right',facecolor='w', framealpha=0.5)
    ax.add_artist(leg1)
    
    
    # plt.draw()
    # # get bounding box loc of legend to plot text underneath it
    # p = medleg.get_window_extent()
    sns.set(font_scale=0.6)

    ax.set_axisbelow(False)
    plt.tight_layout()
    
    figpath = os.path.join(filepath,sitename+'_Validation_Satellite_Distances_PDF_'+str(TransectIDs[0])+'to'+str(TransectIDs[1])+'.png')
    plt.savefig(figpath, dpi=300)
    print('figure saved under '+figpath)
    
    plt.show()
     
    
    
def SatPDFPoster(sitename, SatGDF, DatesCol, ValidDF, TransectIDs, PlotTitle):
    """
    Prob density function plot showing distances between validation and satellite, for each date of validation line.
    FM Oct 2022

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    SatGDF : GeoDataFrame
        GeoDataFrame of satellite-derived lines to use for unique dates.
    DatesCol : str
        Name of dates column in shapefile.
    ValidDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with validation lines.
    TransectIDs : list
        Transect IDs to plot (range from ID1 to ID2).
    PlotTitle : str
        Alternative plot title for placename locations.

    """
    
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)
        
    violin = []
    violindates = []
    Sdates = SatGDF[DatesCol].unique()
    
    for Sdate in Sdates:
        valsatdist = []
        # for each transect in given range
        for Tr in range(TransectIDs[0],TransectIDs[1]): 
            if Tr > len(ValidDF['dates']): # for when transect values extend beyond what transects exist
                print("check your chosen transect values!")
                return
            if Sdate in ValidDF['dates'].iloc[Tr]:
                DateIndex = (ValidDF['dates'].iloc[Tr].index(Sdate))
                # rare occasion where transect intersects valid line but NOT sat line (i.e. no distance between them)
                if ValidDF['valsatdist'].iloc[Tr] != []:
                    valsatdist.append(ValidDF['valsatdist'].iloc[Tr][DateIndex])
                else:
                    continue
            else:
                continue
        # due to way dates are used, some transects might be missing validation dates so violin collection will be empty
        if valsatdist != []: 
            violin.append(valsatdist)
            violindates.append(Sdate)
    # sort both dates and list of values by date
    if len(violindates) > 1:
        violindatesrt, violinsrt = [list(d) for d in zip(*sorted(zip(violindates, violin), key=lambda x: x[0]))]
    else:
        violindatesrt = violindates
        violinsrt = violin
    df = pd.DataFrame(violinsrt)
    df = df.transpose()
    df.columns = violindatesrt
    
    # initialise matching list of sat names for labelling
    satnames = dict.fromkeys(violindatesrt)
    # for each date in sorted list
    for Sdate in violindatesrt:    
        satmatch = []
        for Tr in range(len(ValidDF['TransectID'])):
            # loop through transects to find matching date from which to find satname
            if Sdate not in ValidDF['dates'].iloc[Tr]:
                continue
            else:
                satmatch.append(ValidDF['satname'].iloc[Tr][ValidDF['dates'].iloc[Tr].index(Sdate)])
        # cycling through transects leads to list of repeating satnames; take the unique entry
        satnames[Sdate] = list(set(satmatch))[0]
    
    
    # mpl.rcParams.update({'font.size':26})  
    mpl.rcParams.update({'font.size':8})       
     
    # fig,ax = plt.subplots(figsize=(7.5,6.5),dpi=300)
    fig, ax = plt.subplots(figsize=(6.1,2.5), dpi=300)


    sns.set(font='Avenir', font_scale=0.7)
    
    patches = []
    rect10 = mpatches.Rectangle((-10, -50), 20, 100)
    rect15 = mpatches.Rectangle((-15, -50), 30, 100)
    patches.append(rect10)
    patches.append(rect15)
    coll=PatchCollection(patches, facecolor="black", alpha=0.1, zorder=0)
    
    textcolor = '#0B2D32'
    sns.set_style("ticks", {'axes.facecolor':'#CDCDCD', 'axes.grid' : False, 
                            'text.color':textcolor,
                            'axes.labelcolor':textcolor,
                            'xtick.color':textcolor,
                            'ytick.color':textcolor,
                            'font.sans-serif':'Avenir LT Std'})
    

    if len(violindates) > 1:
                   
        # kdecmap = cm.get_cmap('Greens',len(violindates))
        kdecmap = plt.cm.Greens(np.linspace(0, 1, len(violindates)))
        for i in range(len(violindates)):
            if df.iloc[:,i].isnull().sum() == df.shape[0]:
                kdelabel = None
            else:
                # find name of column for legend labelling (sat date)
                kdelabel = df.columns[i]
            sns.kdeplot(data = df.iloc[:,i], color=kdecmap[i], label=kdelabel, alpha=0.8, linewidth=2)
            
        ax.add_collection(coll)
        # leg1 = ax.legend(loc='upper left', facecolor='w', framealpha=0.4, handlelength=1)
            
    ax.set(xlabel='X-shore distance$_{satellite - validation}$ (m)', ylabel='')
    ax.set_title(PlotTitle)
    plt.yticks([])
    if abs(df.max().max()) > 100:
        majort = np.arange(-150,200,50)
    else:
        majort = np.arange(-150,200,25)
    ax.set_xticks(majort)  
    
    # set axis limits to rounded maximum value of all violins (either +ve or -ve)
    # round UP to nearest 10
    try:
        axlim = math.ceil(np.max([abs(df.min().min()),abs(df.max().max())]) / 10) * 10
        if axlim > 150:
            axlim = 150
        ax.set_xlim(-axlim, axlim)
    except:
        ax.set_xlim(-100, 100)
      

    # create specific median lines for specific platforms
    medians = []
    labels = []
    # dataframe dates and matching satnames
    satdf = pd.DataFrame(satnames, index=[0])
    # remove empty columns to make plotting/legends easier
    df = df.dropna(axis=1, how='all')
    commondates=[col for col in df.columns.intersection(satdf.columns)]
    satdf = satdf[commondates]
    
    # for each platform name
    uniquesats = sorted(set(list(satnames.values())))
    colors = plt.cm.Blues(np.linspace(0.4, 1, len(uniquesats)))
    for satname, c in zip(uniquesats, colors):
        try:
            sats = satdf.apply(lambda row: row[row == satname].index, axis=1)
        except:
            print("Can't plot empty Transects with no validation data!")
            continue
        sats = sats[0].tolist()
        # skip calculating satellite median if transects are empty for this satellite
        if sats == []:
            continue
        # get dataframe column indices for each date that matches the sat name
        # colind = [df.columns.get_loc(sat) for sat in sats]
        # set the date legend label for each date to corresponding satname colour
        # [leg1.get_texts()[ind].set_color(c) for ind in colind]
            
        # get median of only the columns that match each sat name
        concatl = []
        for s in sats:
            concatl.append(df[s])
        concatpd = pd.concat(concatl)
        medians.append(ax.axvline(concatpd.median(), c=c, ls='--', lw=1.5))
        if 'PSScene4Band' in satname:
            satname = 'PS'
        labels.append(satname + ' $\eta$ = ' + str(round(concatpd.median(),1)) + 'm')
    
    # Overall error as text
    totald = []
    for date in df.columns:
        d = df[date]
        for i,datum in enumerate(d):
            totald.append(datum)

    totald = np.array(totald)
    mse = np.mean(np.power(totald[~np.isnan(totald)], 2))
    mae = np.mean(abs(totald[~np.isnan(totald)]))
    rmse = np.sqrt(mse)
    
    l = Line2D([],[], color='none')
    medians.append(l)
    labels.append('RMSE = ' + str(round(rmse,1)) +'m')
    
    # set legend for median lines  
    ax.axvline(0, c='k', ls='-', alpha=0.4, lw=1.5)
    medleg = ax.legend(medians,labels, loc='upper left',facecolor='w', framealpha=0.5, handlelength=0.8)
    # plt.gca().add_artist(leg1)
    
    
    # plt.draw()
    # # get bounding box loc of legend to plot text underneath it
    # p = medleg.get_window_extent()
    # ax.annotate('Hi', (p.p0[1], p.p1[0]), (p.p0[1], p.p1[0]), xycoords='figure pixels', zorder=9, ha='right')    
    ax.set_axisbelow(False)
    plt.tight_layout()
    
    figpath = os.path.join(filepath,sitename+'_Validation_Satellite_Distances_PDF_'+str(TransectIDs[0])+'to'+str(TransectIDs[1])+'_Large.png')
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    print('figure saved under '+figpath)
    
    plt.show()
    
    #mpl.rcParams.update(mpl.rcParamsDefault)    
    
    

def PlatformViolin(sitename, SatShp, SatCol, ValidDF, TransectIDs, PlotTitle=None):
    """
    Violin plot showing distances between validation and satellite, for each platform used.
    FM Oct 2022

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    SatShp : GeoDataFrame, str
        GeoDataFrame (or path to shapefile) of satellite-derived lines to use for unique dates.
    SatCol : str
        Name of satellite platform column (e.g. 'satname').
    ValidDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with validation lines.
    TransectIDs : list
        Transect IDs to plot (range from ID1 to ID2).
    PlotTitle : str, optional
        Alternative plot title for placename locations. The default is None.

    """
    
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)
    
    if type(SatShp) == str:
        SatGDF = gpd.read_file(SatShp)
    else:
        SatGDF = SatShp
        
    violin = []
    violinsats = []
    Snames = SatGDF[SatCol].unique()
    
    for Sname in Snames:
        valsatdist = []
        # for each transect in given range
        for Tr in range(TransectIDs[0],TransectIDs[1]): 
            if Tr > len(ValidDF[SatCol]): # for when transect values extend beyond what transects exist
                print("check your chosen transect values!")
                return
            if Sname in ValidDF[SatCol].iloc[Tr]:
                # need to build list instead of using .index(), as there are multiple occurrences of sat names per transect
                DateIndexes = [i for i, x in enumerate(ValidDF[SatCol].iloc[Tr]) if x == Sname]
                # rare occasion where transect intersects valid line but NOT sat line (i.e. no distance between them)
                if ValidDF['valsatdist'].iloc[Tr] != []:
                    for DateIndex in DateIndexes:
                        valsatdist.append(ValidDF['valsatdist'].iloc[Tr][DateIndex])
                else: # if ValidDF['valsatdist'][Tr] is empty
                    continue
            else: # if Sname isn't in ValidDF[Tr]
                continue
        # due to way dates are used, some transects might be missing validation dates so violin collection will be empty
        if valsatdist != []: 
            violin.append(valsatdist)
            violinsats.append(Sname)
    # sort both dates and list of values by date
    if len(violinsats) > 1:
        violinsatsrt, violinsrt = [list(d) for d in zip(*sorted(zip(violinsats, violin), key=lambda x: x[0]))]
    else:
        violinsatsrt = violinsats
        violinsrt = violin
    df = pd.DataFrame(violinsrt)
    df = df.transpose()
    df.columns = violinsatsrt
       
    f = plt.figure(figsize=(3.31,3.31),dpi=300)
    sns.set(font_scale=0.6)
    
    patches = []
    rect10 = mpatches.Rectangle((-10, -50), 20, 100)
    rect15 = mpatches.Rectangle((-15, -50), 30, 100)
    patches.append(rect10)
    patches.append(rect15)
    coll=PatchCollection(patches, facecolor="black", alpha=0.05, zorder=0)
    colors = plt.cm.Blues(np.linspace(0.4, 1, len(violinsatsrt)))
    textcolor = '#0B2D32'
    
    sns.set_style("ticks", {'axes.grid' : False, 'axes.linewidth':0.5})
    if len(violinsatsrt) > 1:
        # plot stacked violin plots
        ax = sns.violinplot(data = df, linewidth=0, 
                            palette = colors, orient='h', 
                            cut=0, inner='quartile', width=0.6,
                            density_norm='area')
        ax.add_collection(coll)        # set colour of inner quartiles to white dependent on colour ramp 
        for il, l in enumerate(ax.lines):
            l.set_linestyle('--')
            l.set_linewidth(0.7)
            l.set_color('white')
            # overwrite middle line (median) setting to a thicker white line
            for i in range(0,3*len(violinsatsrt))[1::3]:
                if i == il:
                    l.set_linestyle('-')
                    l.set_linewidth(1)
                    l.set_color('white')
    else:
        ax = sns.violinplot(data = df, linewidth=1, orient='h',cut=0, inner='quartile')
        ax.add_collection(coll)
        
    ax.set(xlabel='Distance$_{satellite - validation}$ (m)', ylabel='Satellite image platform')
    if 'PSScene4Band' in violinsatsrt:
        yticklabels = [item.get_text() for item in ax.get_yticklabels()]
        yticklabels[yticklabels.index('PSScene4Band')] = 'PS'
        ax.set_yticklabels(yticklabels)
    
    if PlotTitle != None:
        ax.set_title(PlotTitle)
    
    # set axis limits to rounded maximum value of all violins (either +ve or -ve)
    # round UP to nearest 10
    axlim = math.ceil(np.max([abs(df.min().min()),abs(df.max().max())]) / 10) * 10
    if axlim < 150:
        ax.set_xlim(-axlim, axlim)
    else:
        ax.set_xlim(-150, 150)
        
    majort = np.arange(-150,200,50)
    ax.set_xticks(majort) 
    
    # ax.xaxis.grid(b=True, which='minor',linestyle='--', alpha=0.5)
    
    # land and sea labels on x axis
    axland = ax.twiny()
    axsea = ax.twiny()
    for xax, xlab, xloc, xcol in zip([axland, axsea], ['land', 'sea'], ['left', 'right'], ['#C2B280', '#236E95']):
        xax.xaxis.set_label_position('bottom')
        xax.xaxis.set_ticks_position('bottom')
        if axlim < 150:
            xax.set_xlim(-axlim, axlim)
        else:
            xax.set_xlim(-150, 150)
            
        majort = np.arange(-150,200,50)
        xax.set_xticks(majort) 
        xax.set_xlabel(xlab, loc=xloc)
        xax.xaxis.label.set_color(xcol)
    
    # create specific median lines for specific platforms
    legend_elements = []
    ilines = list(range(0,3*len(violinsatsrt))[1::3])
    for i, (satname, iline) in enumerate(zip(violinsatsrt, ilines)):
        satmedian = df[satname].median()
        satMSE = np.mean(df[satname]**2)
        satMAE = np.mean(abs(df[satname]))
        satRMSE = np.sqrt(satMSE)
        leglabel = 'RMSE = '+str(round(satRMSE,1))+'m'
        medianlabel = '$\eta_{dist}$ = '+str(round(satmedian,1))+'m'
        LegPatch = Patch( facecolor=colors[i], label = leglabel)
        legend_elements.append(LegPatch)
        if axlim < 150:
            ax.text(-axlim-10, i, leglabel, va='center')
        else:
            ax.text(-148, i, leglabel, va='center')
        medianline = ax.lines[iline].get_data()[1][0]
        ax.text(satmedian, medianline-0.05, medianlabel,ha='left', va='bottom')
    
    ax.axvline(0, c='k', ls='-', alpha=0.4, lw=0.5)
    
    ax.set_axisbelow(False)
    plt.tight_layout()
    
    figpath = os.path.join(filepath,sitename+'_Validation_Satellite_PlatformDistances_Violin_'+str(TransectIDs[0])+'to'+str(TransectIDs[1])+'.png')
    plt.savefig(figpath, dpi=300)
    print('figure saved under '+figpath)
    
    plt.show()
    
    for i in df.columns:
        print('No. of transects for '+i+' with sub-pixel accuracy:')
        if i == 'L5' or i == 'L7' or i == 'L8' or i == 'L9':
            subpix = (df[i].between(-15,15).sum()/df[i].count())*100
        else:
            subpix = (df[i].between(-10,10).sum()/df[i].count())*100
        print(str(round(subpix,2))+'%')
        
        
        
        
def PlatformViolinPoster(sitename, SatShp, SatCol, ValidDF, TransectIDs, PlotTitle=None):
    """
    Violin plot showing distances between validation and satellite, for each platform used.
    FM Oct 2022

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    SatShp : GeoDataFrame, str
        GeoDataFrame (or path to shapefile) of satellite-derived lines to use for unique dates.
    SatCol : str
        Name of satellite platform column (e.g. 'satname').
    ValidDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with validation lines.
    TransectIDs : list
        Transect IDs to plot (range from ID1 to ID2).
    PlotTitle : str, optional
        Alternative plot title for placename locations. The default is None.

    """
    
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)
    
    if type(SatShp) == str:
        SatGDF = gpd.read_file(SatShp)
    else:
        SatGDF = SatShp
        
    violin = []
    violinsats = []
    Snames = SatGDF[SatCol].unique()
    
    for Sname in Snames:
        valsatdist = []
        # for each transect in given range
        for Tr in range(TransectIDs[0],TransectIDs[1]): 
            if Tr > len(ValidDF[SatCol]): # for when transect values extend beyond what transects exist
                print("check your chosen transect values!")
                return
            if Sname in ValidDF[SatCol].iloc[Tr]:
                # need to build list instead of using .index(), as there are multiple occurrences of sat names per transect
                DateIndexes = [i for i, x in enumerate(ValidDF[SatCol].iloc[Tr]) if x == Sname]
                # rare occasion where transect intersects valid line but NOT sat line (i.e. no distance between them)
                if ValidDF['valsatdist'].iloc[Tr] != []:
                    for DateIndex in DateIndexes:
                        valsatdist.append(ValidDF['valsatdist'].iloc[Tr][DateIndex])
                else: # if ValidDF['valsatdist'][Tr] is empty
                    continue
            else: # if Sname isn't in ValidDF[Tr]
                continue
        # due to way dates are used, some transects might be missing validation dates so violin collection will be empty
        if valsatdist != []: 
            violin.append(valsatdist)
            violinsats.append(Sname)
    # sort both dates and list of values by date
    if len(violinsats) > 1:
        violinsatsrt, violinsrt = [list(d) for d in zip(*sorted(zip(violinsats, violin), key=lambda x: x[0]))]
    else:
        violinsatsrt = violinsats
        violinsrt = violin
    df = pd.DataFrame(violinsrt)
    df = df.transpose()
    df.columns = violinsatsrt

    # mpl.rcParams.update({'font.size':26})    
    mpl.rcParams.update({'font.size':8})       

    # f = plt.figure(figsize=(7.5,9),dpi=300)
    f = plt.subplots(figsize=(6.1,2.5), dpi=300)

    # plot stacked violin plots (switch orientation with orient='h')
    vorient = 'v'
    
    sns.set(font='Avenir', font_scale=0.7)
    
    patches = []
    if vorient == 'h':
        rect10 = mpatches.Rectangle((-10, -50), 20, 100)
        rect15 = mpatches.Rectangle((-15, -50), 30, 100)
    else:
        rect10 = mpatches.Rectangle((-50, -10), 100, 20)
        rect15 = mpatches.Rectangle((-50, -15), 100, 30)
    patches.append(rect10)
    patches.append(rect15)
    coll=PatchCollection(patches, facecolor="black", alpha=0.05, zorder=0)
    # colors = plt.cm.GnBu(np.linspace(0.4, 1, len(violinsatsrt)))
    colors = plt.cm.Blues(np.linspace(0.4, 1, len(violinsatsrt)))

    # colors=np.array( [[170,201,201,255],[110,150,150,255],[71,116,116,255],[11,45,50,255]] ) / 255
    
    textcolor = '#0B2D32'
    sns.set_style("ticks", {'axes.grid' : False, 'text.color':textcolor,
                            'axes.labelcolor':textcolor,
                            'xtick.color':textcolor,
                            'ytick.color':textcolor,
                            'font.sans-serif':'Avenir LT Std'})
    
    if len(violinsatsrt) > 1:
        ax = sns.violinplot(data = df, linewidth=0, palette = colors, orient=vorient, cut=0, inner='quartile')
        ax.add_collection(coll)        # set colour of inner quartiles to white dependent on colour ramp 
        for il, l in enumerate(ax.lines):
            l.set_linestyle('--')
            l.set_linewidth(1)
            l.set_color('white')
            # overwrite middle line (median) setting to a thicker white line
            for i in range(0,3*len(violinsatsrt))[1::3]:
                if i == il:
                    l.set_linestyle('-')
                    l.set_linewidth(1)
                    l.set_color('white')
    else:
        ax = sns.violinplot(data = df, linewidth=1, orient='h',cut=0, inner='quartile')
        ax.add_collection(coll)
        
    
    if vorient == 'h':
        ax.set(xlabel='X-shore distance$_{satellite - validation}$ (m)', ylabel='Satellite image platform')
        
        if 'PSScene4Band' in violinsatsrt:
            ticklabels = [item.get_text() for item in ax.get_yticklabels()]
            ticklabels[ticklabels.index('PSScene4Band')] = 'PS'
            ax.set_yticklabels(ticklabels)
        
        # set axis limits to rounded maximum value of all violins (either +ve or -ve)
        # round UP to nearest 10
        axlim = math.ceil(np.max([abs(df.min().min()),abs(df.max().max())]) / 10) * 10
        if axlim < 150:
            ax.set_xlim(-axlim, axlim)
        else:
            ax.set_xlim(-150, 150)
        
        majort = np.arange(-150,200,50)
        ax.set_xticks(majort)    
        # ax.xaxis.grid(b=True, which='minor',linestyle='--', alpha=0.5)
        ax.axvline(0, c='k', ls='-', alpha=0.4, lw=1)
            
    else:
        ax.set(xlabel='Satellite image platform', ylabel='X-shore distance$_{satellite - validation}$ (m)')

        if 'PSScene4Band' in violinsatsrt:
            ticklabels = [item.get_text() for item in ax.get_xticklabels()]
            ticklabels[ticklabels.index('PSScene4Band')] = 'PS'
            ax.set_xticklabels(ticklabels)
                
        # set axis limits to rounded maximum value of all violins (either +ve or -ve)
        # round UP to nearest 10
        axlim = math.ceil(np.max([abs(df.min().min()),abs(df.max().max())]) / 10) * 10
        if axlim < 150:
            ax.set_ylim(-axlim, axlim)
        else:
            ax.set_ylim(-150, 150)
        
        majort = np.arange(-150,200,50)
        ax.set_yticks(majort)
        ax.axhline(0, c='k', ls='-', alpha=0.4, lw=1)


    if PlotTitle != None:
        ax.set_title(PlotTitle)

    
    # # create specific median lines for specific platforms
    legend_elements = []
    ilines = list(range(0,3*len(violinsatsrt))[1::3])
    for i, (satname, iline) in enumerate(zip(violinsatsrt, ilines)):
        satmedian = df[satname].median()
        satMSE = np.mean(df[satname]**2)
        satMAE = np.mean(abs(df[satname]))
        satRMSE = np.sqrt(satMSE)
        leglabel = '\nRMSE = '+str(round(satRMSE,1))+'m'
        medianlabel = '$\eta_{dist}$ =\n'+str(round(satmedian,1))+'m'
        LegPatch = Patch( facecolor=colors[i], label = leglabel)
        legend_elements.append(LegPatch)
        if vorient == 'h':
            if axlim < 150:
                ax.text(-axlim+1, i+0.02, leglabel, va='center')
            else:
                ax.text(-149, i+0.02, leglabel, va='center')
        else:
            if axlim < 150:
                ax.text(i, -140, leglabel, ha='center')
            else:
                ax.text(i, -140, leglabel, ha='center')
        medianline = ax.lines[iline].get_data()[1][0]
        if vorient == 'h':
            if satname == 'PSScene4Band' or satname == 'S2':
                ax.text(satmedian, medianline-0.05, medianlabel, ha='center')
            else:
                ax.text(satmedian, medianline-0.12, medianlabel, ha='center')
        else:
            mediantxt = ax.text(i+0.35, satmedian, medianlabel, va='center',ha='center')
            mediantxt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w', alpha=0.8)])
 
        
    ax.set_axisbelow(False)
    plt.tight_layout()
    
    figpath = os.path.join(filepath,sitename+'_Validation_Satellite_PlatformDistances_Violin_'+str(TransectIDs[0])+'to'+str(TransectIDs[1])+'_Large.png')
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    print('figure saved under '+figpath)
    
    plt.show()
    
    
    # print stats after fig
    for i in df.columns:
        print('No. of transects for '+i+' with sub-pixel accuracy:')
        if i == 'L5' or i == 'L7' or i == 'L8' or i == 'L9':
            subpix = (df[i].between(-15,15).sum()/df[i].count())*100
        else:
            subpix = (df[i].between(-10,10).sum()/df[i].count())*100
        print(str(round(subpix,2))+'%')
    
def ThresholdViolin(sitename, filepath, sites):
    """
    Violin plot of NDVI thresholds calculated throughout entire VE extraction run.
    Specific to St Andrews east (open coast) and west (estuarine).
    FM Apr 2023

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    filepath : str
        Path to COASTGUARD/Data folder.
    sites : list
        Separate sitenames to compare.


    """
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
      
    violindict = {}
    for site in sites:
        with open(os.path.join(filepath,site ,site+ '_output.pkl'), 'rb') as f:
            # outputdict = pickle.load(f)
            outputdict = pd.read_pickle(f)
        violindict[site] = outputdict['vthreshold']
    
    # Add Planet results to violin data
    violindictPl = {}
    for site in ['StAndrewsPlanetEastTOA', 'StAndrewsPlanetWestTOA']:
        with open(os.path.join(filepath,site ,site+ '_output.pkl'), 'rb') as f:
            # outputdict = pickle.load(f)
            outputdict = pd.read_pickle(f)

        violindictPl[site] = outputdict['vthreshold']
    
    violindict['StAndrewsEast'].extend(violindictPl['StAndrewsPlanetEastTOA'])
    violindict['StAndrewsWest'].extend(violindictPl['StAndrewsPlanetWestTOA'])
    
    # concat together threshold columns (even if different sizes; fills with nans)
    violinDF = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in violindict.items()]))
        
    # colors = ['#21A790','#1D37FB'] #West = Teal, East = Blue
    # colors = [pltcls.to_hex(plt.cm.YlGnBu(i)) for i in range(len(violinDF.keys()))]
    ylgnbu = mpl.colormaps['YlGnBu']
    
    colors = ylgnbu(np.linspace((1/(len(violinDF.keys())+1)), 1-(1/(len(violinDF.keys())+1)), len(violinDF.keys())))
    
    # fig = plt.figure(figsize=[1.89,2.64], tight_layout=True)
    fig = plt.figure(figsize=(3.31, 3.31), dpi=300, tight_layout=True)

    # sns.set(font="Arial", font_scale=0.55)
    sns.set(font="Arial", font_scale=0.6)
    sns.set_style("ticks", {'axes.grid' : False})
    
    ax = sns.violinplot(data = violinDF, linewidth=0, palette = 'YlGnBu', orient='v', cut=0, inner='quart')
    # change quartile line styles
    for il, l in enumerate(ax.lines):
        l.set_linestyle('--')
        l.set_linewidth(0.7)
        l.set_color('white')
        # overwrite middle line (median) setting to a thicker white line
        for i in range(0,3*len(violinDF.columns))[1::3]:
            if i == il:
                l.set_linestyle('-')
                l.set_linewidth(1)
    
    ax.set_xticklabels(['Inner Estuarine','Open Coast'])
    plt.ylabel('NDVI threshold value')
    plt.ylim(-0.1,0.55)
    
    # create legend with medians and data labels for each violin
    legend_elements = []
    for i, key in enumerate(violinDF.keys()):
        # find median value
        satmedian = violinDF[key].median()
        satmin = violinDF[key].min()
        satmax = violinDF[key].max()
        # pass median to legend object
        leglabel = '$\eta_{NDVI}$ = '+str(round(satmedian,2))
        LegPatch = Patch( facecolor=colors[i], label = leglabel)
        legend_elements.append(LegPatch)
        # define data labels for max and min values
        ypos = [satmin-0.018, satmax+0.003]
        textlabels = [str(round(satmin,2)), str(round(satmax,2))]
        # for each min and max value of each violin, label data
        for j in range(len(ypos)):
            ax.text(i, ypos[j], textlabels[j], ha='center')
        
    plt.legend(handles=legend_elements, loc='upper right')
  
    plt.tight_layout()
    outfilename = outfilepath+'/'
    for site in sites:
        outfilename += site+'_'
    outfilename += 'Thresholds_Violin.png'
    plt.savefig(outfilename, dpi=300)
    print('figure saved under '+outfilename)

    plt.show()
    
  
def QsViolin(Qs_Trs, WaveTime, StormsDF):

    Qs_Trs_Storm = pd.DataFrame(Qs_Trs, index=WaveTime)
    Qs_Trs_Storm['Storm'] = 0
    for _, row in StormsDF.iterrows():
        mask = (Qs_Trs_Storm.index >= row['starttime']) & (Qs_Trs_Storm.index <= row['endtime'])
        Qs_Trs_Storm.loc[mask, 'Storm'] = 1
        
    # fig, ax = plt.subplots(1,1)
    # for Tr, clr in zip([1325, 271], ['b','r']):
    #     ax.hist(Qs_Trs_Storm[Tr][ Qs_Trs_Storm['Storm']==0], bins=np.arange(-0.2,0.21,0.01), fc='k', alpha=0.5)
    #     ax.hist(Qs_Trs_Storm[Tr][ Qs_Trs_Storm['Storm']==1], bins=np.arange(-0.2,0.21,0.01), fc=clr, alpha=0.5)
    # plt.show()  

    Qs_Trs_Fair = Qs_Trs_Storm[[1325,271]][Qs_Trs_Storm['Storm']==0]
    Qs_Trs_Stormy = Qs_Trs_Storm[[1325,271]][Qs_Trs_Storm['Storm']==1]
    fig, ax = plt.subplots(1,1)
    # sns.violinplot(data = Qs_Trs_Fair, ax=ax)
    sns.violinplot(data = Qs_Trs_Stormy, ax=ax, palette='bwr_r')
    plt.show()
    
    fig, ax = plt.subplots(1,1)
    sns.boxplot(data = Qs_Trs_Fair, ax=ax, palette='bwr_r')
    # sns.violinplot(data = Qs_Trs_Stormy, ax=ax)
    plt.show()

    
def MultivariateMatrixClusteredWaves(sitename, MultivarGDF, Loc1=None, Loc2=None):
    """
    Create a multivariate matrix plot of vegetation edges, waterlines, topographic data and wave data.
    Each point on scatter is a single value on a cross-shore transect (i.e. mean value or rate over time).
    Scatter points are separated into clustered regions (N vs S, eroding vs accreting).
    This version includes non-parametric regression and density heatmaps.
    FM Nov 2024

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with veg edge lines.
    TransectInterGDFWater : GeoDataFrame
        GeoDataFrame of transects intersected with waterlines.
    TransectInterGDFTopo : GeoDataFrame
        GeoDataFrame of transects intersected with topographic data.
    Loc1 : list
        Transect IDs to slice array up for north location
    Loc2 : list
        Transect IDs to slice array up for south location

    """
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)
    
    # summer (pale) eroding = #F9C784 
    # summer (pale) accreting = #9DB4C0
        
    # Scale up diffusivity (mu) for nicer labelling
    # MultivarGDF['WaveDiffus'] = MultivarGDF['WaveDiffus']*1000
    
    # Extract desired columns to an array for plotting
    MultivarArray = np.array(MultivarGDF[['oldyoungRt','oldyungRtW','TZwidthMn','SlopeMax','WaveDiffus']])#, 'WaveStabil']])
    
    fs = 7
    # fs = 10 # PPT dimensions
    mpl.rcParams.update({'font.size':fs})

    fig, axs = plt.subplots(MultivarArray.shape[1],MultivarArray.shape[1], figsize=(6.55,6.55), dpi=300)
    # fig, axs = plt.subplots(MultivarArray.shape[1],MultivarArray.shape[1], figsize=(12.68,6), dpi=300) # PPT dimensions

    # if no location of transects is specified, split array by eroding and accreting
    if Loc1 is None:
        Arr1 = MultivarArray[0:int(len(MultivarArray)/2), :]
        Arr2 = MultivarArray[int(len(MultivarArray)/2):, :]
    else:   
        Arr1 = MultivarArray[0:Loc1[1]-Loc1[0], :]
        Arr2 = MultivarArray[Loc2[1]-Loc2[0]:, :]
    # Plot matrix of relationships
    lab = [r'$\Delta$VE (m/yr)',
           r'$\Delta$WL (m/yr)',
           r'TZwidth$_{\mu}$ (m)',
           r'$\theta_{max}$ ($\circ$)',
           r'$\mu_{net}$ (m/s$^{2}$)']
           # r'$\Gamma$ (1)']
    
    for row in range(MultivarArray.shape[1]):
        for col in range(MultivarArray.shape[1]): 
            for Arr, colour, strpos, leglabel in zip(
                                                    [Arr1, Arr2], 
                                                    ['#C51B2F','#5499DE'],
                                                    [0.4, 0.2],
                                                    ['Eroding ','Accreting ']):
                # if plot is same var on x and y, change plot to a histogram    
                if row == col:
                    binnum = round(np.sqrt(len(MultivarArray)))*2
                    bins = np.histogram(MultivarArray[:,row],bins=binnum)[1]
                    mn, var, skw, kurt = Toolbox.Moments(Arr[:,row])
                    stdv = np.sqrt(var)
                    moments_label = f'$\gamma_{1}$ = {round(skw,2)}\n$\gamma_{2}$ = {round(kurt,2)}'
                    axs[col,row].hist(Arr[:,row], bins, color=colour, alpha=0.5, label=moments_label)
                    # Plot the mean as a vertical line
                    axs[col,row].axvline(x=mn, c=colour, lw=0.5, ls='--')
                    # Plot the variance
                    axs[col,row].axvline(x=mn-stdv, c=colour, lw=0.5, ls=':')
                    axs[col,row].axvline(x=mn+stdv, c=colour, lw=0.5, ls=':')
                    axs[col,row].set_yticks([]) # turns off ticks and tick labels
                    if col < 4:
                        legloc = 'upper right'
                    else:
                        legloc = 'upper left'
                    axs[col,row].legend(loc=legloc, 
                                        prop={'size':5}, edgecolor='none', framealpha=0.5,
                                        borderpad=0.2, labelspacing=0.2, handlelength=0.5, handletextpad=0.2)               
                
                # otherwise plot scatter of each variable against one another
                else:
                                      
                    scatterPl = axs[col,row].scatter(Arr[:,row], Arr[:,col], s=20, alpha=0.4, marker='.', c=colour, edgecolors='none')
                    # scatterPl = axs[col,row].scatter(Arr[:,row][0:int(len(Arr[:,row])/2)], Arr[:,col][0:int(len(Arr[:,col])/2)], s=20, alpha=0.4, marker='.', c='C4', edgecolors='none')

                    # overall non-parametric reg line
                    Lowess = sm.nonparametric.lowess(list(MultivarArray[:,col]), list(MultivarArray[:,row]), frac=0.2)
                    regLn, = axs[col,row].plot(Lowess[:, 0], Lowess[:, 1], c='k', ls='--', lw=1.5, zorder=3)
                
                    # Calculate Spearman correlation
                    Spearman, _ = spearmanr(list(MultivarArray[:,row]), list(MultivarArray[:,col]))
                    SpearmanLab = f"$r_s$ = {Spearman:.2f}"
                    SpearmanLeg = Line2D([0], [0], color='none', label=SpearmanLab)
                    ScatterLeg = axs[col,row].legend(handles=[SpearmanLeg],loc='lower right', 
                                        prop={'size':5}, edgecolor='none', framealpha=0.5,
                                        borderpad=0.2, labelspacing=0, handlelength=0, handletextpad=0)
                    
                # Horizontal and vertical lines through 0    
                hLn = axs[col,row].axvline(x=0, c=[0.5,0.5,0.5], lw=0.5)
                vLn = axs[col,row].axhline(y=0, c=[0.5,0.5,0.5], lw=0.5)
                
                if row == MultivarArray.shape[1]-1: # set x axis labels on last row
                    axs[row,col].set_xlabel(lab[col])
                else:
                    axs[row,col].tick_params(labelbottom=False)
                if col == 0: # set y axis labels on first column
                    axs[row,col].set_ylabel(lab[row])
                else:
                    axs[row,col].tick_params(labelleft=False)
                
                # clear plots on RHS of hists, print heatmaps of point density instead
                for i in range(MultivarArray.shape[1]):
                    if col == i and row > i:
                        # axs[col,row].cla() # clears axis on each loop
                        for Ln in [regLn, scatterPl, hLn, vLn, ScatterLeg]:
                            Ln.remove()
                        # axs[col,row].hexbin(list(MultivarArray[:,col]), list(MultivarArray[:,row]), gridsize=5, cmap='Greys')
                        bins = 25  # Adjust based on the data density and plot size
                        axs[col,row].hist2d(list(MultivarArray[:,col]), list(MultivarArray[:,row]), bins=bins, cmap='Greys')
                        axs[col,row].set_xlim(axs[row,col].get_xlim())
                        axs[col,row].set_ylim(axs[row,col].get_ylim())
                        axs[col,row].set_xticks([])
                        axs[col,row].set_yticks([])

            
    # align all yaxis labels in first column
    fig.align_ylabels(axs[:,0])
    
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.6, hspace=0.5)
    if Loc1 is None:
        figpath = os.path.join(filepath,sitename+'_MultivariateClustered_VegWaterTopoWaves_Heatmaps_AutoSplit.png')
    else:
        figpath = os.path.join(filepath,sitename+'_MultivariateClustered_VegWaterTopoWaves_Heatmaps_%s-%s_%s-%s.png' % 
                               (Loc1[0],Loc1[1],Loc2[0],Loc2[1]))
    plt.savefig(figpath)
    print('figure saved under '+figpath)
    
    plt.show()
    
    return



def MultivariateMatrixClusteredFlux(sitename, MultivarGDF, ColNames, Loc1=None, Loc2=None):
    """
    Create a multivariate matrix plot of vegetation edges, waterlines, topographic data and wave data.
    Each point on scatter is a single value on a cross-shore transect (i.e. mean value or rate over time).
    Scatter points are separated into clustered regions (N vs S, eroding vs accreting).
    This version includes non-parametric regression and density heatmaps.
    FM Nov 2024

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with veg edge lines.
    TransectInterGDFWater : GeoDataFrame
        GeoDataFrame of transects intersected with waterlines.
    TransectInterGDFTopo : GeoDataFrame
        GeoDataFrame of transects intersected with topographic data.
    Loc1 : list
        Transect IDs to slice array up for north location
    Loc2 : list
        Transect IDs to slice array up for south location

    """
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)
    
    # summer (pale) eroding = #F9C784 
    # summer (pale) accreting = #9DB4C0
        
    # Scale up diffusivity (mu) for nicer labelling
    # MultivarGDF['WaveDiffus'] = MultivarGDF['WaveDiffus']*1000
    
    # Extract desired columns to an array for plotting
    MultivarArray = np.array(MultivarGDF[ColNames])
    
    fs = 7
    # fs = 10 # PPT dimensions
    mpl.rcParams.update({'font.size':fs})

    fig, axs = plt.subplots(MultivarArray.shape[1],MultivarArray.shape[1], figsize=(6.55,6.35), dpi=300)
    # fig, axs = plt.subplots(MultivarArray.shape[1],MultivarArray.shape[1], figsize=(12.68,6), dpi=300) # PPT dimensions

    # if no location of transects is specified, split array by eroding and accreting
    if Loc1 is None:
        Arr1 = MultivarArray[0:int(len(MultivarArray)/2), :]
        Arr2 = MultivarArray[int(len(MultivarArray)/2):, :]
    else:   
        Arr1 = MultivarArray[0:Loc1[1]-Loc1[0], :]
        Arr2 = MultivarArray[Loc2[1]-Loc2[0]:, :]
    # Plot matrix of relationships
    lab = [r'$\Delta$VE (m/yr)',
           r'$\Delta$WL (m/yr)',
           r'TZwidth$_{\mu}$ (m)',
           r'$\theta_{max}$ ($\circ$)',
           r'$Q_{s,net}$ (m$^{3}/s$)']
           # r'$\Gamma$ (1)']
    
    for row in range(MultivarArray.shape[1]):
        for col in range(MultivarArray.shape[1]): 
            for Arr, colour, strpos, leglabel in zip(
                                                    [Arr1, Arr2], 
                                                    ['#C51B2F','#5499DE'],
                                                    [0.4, 0.2],
                                                    ['Eroding ','Accreting ']):
                # if plot is same var on x and y, change plot to a histogram    
                if row == col:
                    binnum = round(np.sqrt(len(MultivarArray)))*2
                    bins = np.histogram(MultivarArray[:,row],bins=binnum)[1]
                    mn, var, skw, kurt = Toolbox.Moments(Arr[:,row])
                    stdv = np.sqrt(var)
                    moments_label = f'$\gamma_{1}$ = {round(skw,2)}\n$\gamma_{2}$ = {round(kurt,2)}'
                    axs[col,row].hist(Arr[:,row], bins, color=colour, alpha=0.5, label=moments_label)
                    # Plot the mean as a vertical line
                    axs[col,row].axvline(x=mn, c=colour, lw=0.5, ls='--')
                    # Plot the variance
                    axs[col,row].axvline(x=mn-stdv, c=colour, lw=0.5, ls=':')
                    axs[col,row].axvline(x=mn+stdv, c=colour, lw=0.5, ls=':')
                    axs[col,row].set_yticks([]) # turns off ticks and tick labels
                    if col < 4:
                        legloc = 'upper right'
                    else:
                        legloc = 'upper left'
                    axs[col,row].legend(loc=legloc, 
                                        prop={'size':5}, edgecolor='none', framealpha=0.5,
                                        borderpad=0.2, labelspacing=0.2, handlelength=0.5, handletextpad=0.2)               
                
                # otherwise plot scatter of each variable against one another
                else:
                                      
                    scatterPl = axs[col,row].scatter(Arr[:,row], Arr[:,col], s=20, alpha=0.4, marker='.', c=colour, edgecolors='none')
                    # scatterPl = axs[col,row].scatter(Arr[:,row][0:int(len(Arr[:,row])/2)], Arr[:,col][0:int(len(Arr[:,col])/2)], s=20, alpha=0.4, marker='.', c='C4', edgecolors='none')

                    # overall non-parametric reg line
                    Lowess = sm.nonparametric.lowess(list(MultivarArray[:,col]), list(MultivarArray[:,row]), frac=0.2)
                    regLn, = axs[col,row].plot(Lowess[:, 0], Lowess[:, 1], c='k', ls='--', lw=1.5, zorder=3)
                
                    # Calculate Spearman correlation
                    Spearman, _ = spearmanr(list(MultivarArray[:,row]), list(MultivarArray[:,col]))
                    SpearmanLab = f"$r_s$ = {Spearman:.2f}"
                    SpearmanLeg = Line2D([0], [0], color='none', label=SpearmanLab)
                    ScatterLeg = axs[col,row].legend(handles=[SpearmanLeg],loc='lower right', 
                                        prop={'size':5}, edgecolor='none', framealpha=0.5,
                                        borderpad=0.2, labelspacing=0, handlelength=0, handletextpad=0)
                    
                # Horizontal and vertical lines through 0    
                hLn = axs[col,row].axvline(x=0, c=[0.5,0.5,0.5], lw=0.5)
                vLn = axs[col,row].axhline(y=0, c=[0.5,0.5,0.5], lw=0.5)
                
                if row == MultivarArray.shape[1]-1: # set x axis labels on last row
                    axs[row,col].set_xlabel(lab[col])
                else:
                    axs[row,col].tick_params(labelbottom=False)
                if col == 0: # set y axis labels on first column
                    axs[row,col].set_ylabel(lab[row])
                else:
                    axs[row,col].tick_params(labelleft=False)
                
                # clear plots on RHS of hists, print heatmaps of point density instead
                for i in range(MultivarArray.shape[1]):
                    if col == i and row > i:
                        # axs[col,row].cla() # clears axis on each loop
                        for Ln in [regLn, scatterPl, hLn, vLn, ScatterLeg]:
                            Ln.remove()
                        # axs[col,row].hexbin(list(MultivarArray[:,col]), list(MultivarArray[:,row]), gridsize=5, cmap='Greys')
                        bins = 25  # Adjust based on the data density and plot size
                        axs[col,row].hist2d(list(MultivarArray[:,col]), list(MultivarArray[:,row]), bins=bins, cmap='Greys')
                        axs[col,row].set_xlim(axs[row,col].get_xlim())
                        axs[col,row].set_ylim(axs[row,col].get_ylim())
                        axs[col,row].set_xticks([])
                        axs[col,row].set_yticks([])

    ErodePatch = mpatches.Patch(color='#C51B2F', label='Eroding Veg (-$\Delta VE$)')
    AccretePatch = mpatches.Patch(color='#5499DE', label='Accreting Veg (+$\Delta VE$)')
    plt.figlegend(handles=[ErodePatch, AccretePatch], loc='lower center', ncols=2, bbox_to_anchor=(0.5, -0.03), 
                  framealpha=0.5, borderpad=0.2, labelspacing=0.2, handlelength=0.5, handletextpad=0.2) 
       
    # align all yaxis labels in first column
    fig.align_ylabels(axs[:,0])
    
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.6, hspace=0.5)
    if Loc1 is None:
        figpath = os.path.join(filepath,sitename+'_MultivariateClustered_VegWaterTopoFlux_Heatmaps_AutoSplit.png')
    else:
        figpath = os.path.join(filepath,sitename+'_MultivariateClustered_VegWaterTopoFlux_Heatmaps_%s-%s_%s-%s.png' % 
                               (Loc1[0],Loc1[1],Loc2[0],Loc2[1]))
    plt.savefig(figpath, bbox_inches='tight') # tight ensures extra artists (figlegend) are accounted for
    print('figure saved under '+figpath)
    
    plt.show()
    
    return






def PCAHeatmap(pca, MultivarGDF, colnames):
    """
    IN DEVELOPMENT/UNUSED
    Principal Component Analysis heatmap (similar to 
    https://medium.com/analytics-vidhya/principal-component-analysis-754781cfb30f)
    FM Aug 2024

    Parameters
    ----------
    pca : sklearn.decomposition PCA object
        Principal Component Analysis object created and fitted to data.
    MultivarGDF : GeoDataFrame
        Multivariate dataframe holding data to be used in PCA.
    colnames : list
        List of strings representing dataframe columns (variables) to be plotted.

    """
    # Get the loadings (components)
    loadings = pca.components_
    
    # Create a DataFrame for the loadings
    loading_df = pd.DataFrame(loadings, columns=colnames, index=[f'PC{i+1}' for i in range(loadings.shape[0])])
    
    # Plot the heatmap
    plt.figure(figsize=(6, 5), dpi=200)
    sns.heatmap(loading_df, annot=True, cmap='PuOr', cbar=True, center=0)
    plt.title('Heatmap of Principal Component Loadings')
    plt.xlabel('Original Variables')
    plt.ylabel('Principal Components')
    plt.show()
    

def WaveHeatmap(TransectInterGDFWater, TransectInterGDFWave, TransectIDs):
    """
    Plot a heatmap of wave power and wave energy against vegetation edge and 
    waterline positions through time, to identify any relationship between 
    instantaneous measurements of these metrics across a timeseries.
    FM Nov 2024

    Parameters
    ----------
    TransectInterGDFWater : GeoDataFrame
        GeoDataFrame of transects intersected with waterlines.
    TransectInterGDFWave : GeoDataFrame
        GeoDataFrame of transects intersected with wave hindcasts.
    TransectIDs : list
        List of desired transect IDs to plot (single plots, not range).


    """
    for Tr in TransectIDs:
        # Convert dates to datetime objects for easier comparison
        plotdates = [datetime.strptime(date, "%Y-%m-%d") for date in TransectInterGDFWater['dates'].iloc[Tr]]
        plotwldates = [datetime.strptime(date, "%Y-%m-%d") for date in TransectInterGDFWater['wldates'].iloc[Tr]]       
        plotwavedates = [datetime.strptime(date, "%Y-%m-%d") for date in TransectInterGDFWave['dates'].iloc[Tr]]

        plotdists = TransectInterGDFWater['distances'].iloc[Tr]
        plotwldists = TransectInterGDFWater['wlcorrdist'].iloc[Tr]
        # Calculate wave power [p g^2 / 64pi * H^2 * Tp]
        plotwaveP = ((1020*(9.8**2))/(64*np.pi)) * \
                    np.array(TransectInterGDFWave['WaveHs'].iloc[Tr])**2 * \
                    np.array(TransectInterGDFWave['WaveTp'].iloc[Tr])
        # Calculate wave energy [1/16 p g H^2]
        plotwaveE = 0.0625 * 1020 * 9.8 * (np.array(TransectInterGDFWave['WaveHs'].iloc[Tr])**2)
        
        # Remove duplicates in plotwldates and plotwldists, keeping the first occurrence
        plotwldates_uq = []
        plotwldists_uq = []
        seen_dates = set()
        for date, dist in zip(plotwldates, plotwldists):
            if date not in seen_dates:
                plotwldates_uq.append(date)
                plotwldists_uq.append(dist)
                seen_dates.add(date)
        # Find common dates
        common_dates = set(plotdates) & set(plotwldates_uq)
        # Filter plotdates and plotdists to include only common dates
        plotdates_f = [date for date, dist in zip(plotdates, plotdists) if date in common_dates]
        plotdists_f = [dist for date, dist in zip(plotdates, plotdists) if date in common_dates]

        # Filter plotwldates_uq and plotwldists_uq to include only common dates
        plotwldates_f = [date for date, dist in zip(plotwldates_uq, plotwldists_uq) if date in common_dates]
        plotwldists_f = [dist for date, dist in zip(plotwldates_uq, plotwldists_uq) if date in common_dates]

        # Filter plotdates and plotdists to include only common dates
        plotwavedates_f = [date for date, val in zip(plotwavedates, plotwaveP) if date in common_dates]
        plotwaveP_f = [val for date, val in zip(plotwavedates, plotwaveP) if date in common_dates]
        plotwaveE_f = [val for date, val in zip(plotwavedates, plotwaveE) if date in common_dates]

        plotDict = {'dates':plotdates_f,'vegdists':plotdists_f,
                    'wldates':plotwldates_f,'wldists':plotwldists_f,
                    'wavedates':plotwavedates_f,'wavePower':plotwaveP_f,
                    'waveEnergy':plotwaveE_f}
        plotDF = pd.DataFrame(data=plotDict)
        
        # Set up the figure and axes for two subplots (one for each heatmap)
        fig, axs = plt.subplots(2, 1, figsize=(5,10))
        for ax, clrs, yvals in zip(axs, ['Greens', 'Blues'], [plotDF['vegdists'],plotDF['wldists']]):
            # 1. Heatmap of Wave Height vs. Waterline Position
            sns.histplot(
                x=plotDF['wavePower'], 
                y=yvals, 
                bins=50, 
                cmap=clrs, 
                cbar=True,
                ax=ax
            )

            ax.set_xlabel('Wave Power (kW/m)')
        plt.tight_layout()
        plt.show()
        
        fig, axs = plt.subplots(2, 1, figsize=(5,10))
        for ax, clrs, yvals in zip(axs, ['Greens', 'Blues'], [plotDF['vegdists'],plotDF['wldists']]):
            # 1. Heatmap of Wave Height vs. Waterline Position
            sns.histplot(
                x=plotDF['waveEnergy'], 
                y=yvals, 
                bins=50, 
                cmap=clrs, 
                cbar=True,
                ax=ax
            )

            ax.set_xlabel('Wave Energy (J/m$^2$)')
        plt.tight_layout()
        plt.show()


