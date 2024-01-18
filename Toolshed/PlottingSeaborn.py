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
plt.ion()

from shapely import geometry
from shapely.geometry import Point, LineString
import rasterio

from Toolshed import Toolbox, Transects, Image_Processing

from sklearn.metrics import mean_squared_error, r2_score

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


def ValidViolin(sitename, ValidationShp,DatesCol,ValidDF,TransectIDs):
    """
    Violin plot showing distances between validation and satellite, for each date of validation line.
    FM Oct 2022

    Parameters
    ----------
    ValidationShp : str
        Path to validation lines shapefile.
    DatesCol : str
        Name of dates column in shapefile.
    ValidDF : GeoDataFrame
        Validation GDF created from ValidateIntersects().

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
    

def SatViolin(sitename, SatGDF, DatesCol,ValidDF,TransectIDs, PlotTitle):
    """
    Violin plot showing distances between validation and satellite, for each date of validation line.
    FM Oct 2022

    Parameters
    ----------
    ValidationShp : str
        Path to validation lines shapefile.
    DatesCol : str
        Name of dates column in shapefile.
    ValidDF : GeoDataFrame
        Validation dictionary created from ValidateIntersects().

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
    

def SatPDF(sitename, SatGDF,DatesCol,ValidInterGDF,TransectIDs, PlotTitle):
    """
    Prob density function plot showing distances between validation and satellite, for each date of validation line.
    FM Oct 2022

    Parameters
    ----------
    ValidationShp : str
        Path to validation lines shapefile.
    DatesCol : str
        Name of dates column in shapefile.
    ValidDF : GeoDataFrame
        Validation GDF created from ValidateIntersects().

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
    for xax, xlab, xloc, xcol in zip([axland, axsea], ['land', 'sea'], ['left', 'right'], ['#236E95', '#C2B280']):
        xax.xaxis.set_label_position('bottom')
        xax.xaxis.set_ticks_position('bottom')
        try:
            axlim = math.ceil(np.max([abs(df.min().min()),abs(df.max().max())]) / 10) * 10
            xax.set_xlim(-axlim, axlim)
        except:
            xax.set_xlim(-100, 100)
            
        xax.set_xlabel(xlab, loc=xloc)
        xax.xaxis.label.set_color(xcol)
    
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
    ax.set_axisbelow(False)
    plt.tight_layout()
    
    figpath = os.path.join(filepath,sitename+'_Validation_Satellite_Distances_PDF_'+str(TransectIDs[0])+'to'+str(TransectIDs[1])+'.png')
    plt.savefig(figpath, dpi=300)
    print('figure saved under '+figpath)
    
    plt.show()
     
    
    
def SatPDFPoster(sitename, SatGDF,DatesCol,ValidDF,TransectIDs, PlotTitle):
    """
    Prob density function plot showing distances between validation and satellite, for each date of validation line.
    FM Oct 2022

    Parameters
    ----------
    ValidationShp : str
        Path to validation lines shapefile.
    DatesCol : str
        Name of dates column in shapefile.
    ValidDF : GeoDataFrame
        Validation GDF created from ValidateIntersects().

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
    
    

def PlatformViolin(sitename, SatShp,SatCol,ValidDF,TransectIDs, PlotTitle=None):
    """
    Violin plot showing distances between validation and satellite, for each platform used.
    FM Oct 2022

    Parameters
    ----------
    ValidationShp : str
        Path to validation lines shapefile.
    DatesCol : str
        Name of sat column in shapefile.
    ValidDF : GeoDataFrame
        Validation GDF created from ValidateIntersects().

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
        ax = sns.violinplot(data = df, linewidth=0, palette = colors, orient='h', cut=0, inner='quartile')
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
    for xax, xlab, xloc, xcol in zip([axland, axsea], ['land', 'sea'], ['left', 'right'], ['#236E95', '#C2B280']):
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
            ax.text(-axlim+1, i, leglabel, va='center')
        else:
            ax.text(-145, i, leglabel, va='center')
        medianline = ax.lines[iline].get_data()[1][0]
        ax.text(satmedian, medianline-0.05, medianlabel,ha='center')
    
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
        
        
        
        
def PlatformViolinPoster(sitename, SatShp,SatCol,ValidDF,TransectIDs, PlotTitle=None):
    """
    Violin plot showing distances between validation and satellite, for each platform used.
    FM Oct 2022

    Parameters
    ----------
    ValidationShp : str
        Path to validation lines shapefile.
    DatesCol : str
        Name of sat column in shapefile.
    ValidDF : GeoDataFrame
        Validation GDF created from ValidateIntersects().

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
    
def ThresholdViolin(sitename,filepath,sites):
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
      
    violindict = {}
    for site in sites:
        with open(os.path.join(filepath,site ,site+ '_output.pkl'), 'rb') as f:
            outputdict = pickle.load(f)
        violindict[site] = outputdict['vthreshold']
    
    violindictPl = {}
    for site in ['StAndrewsPlanetEast', 'StAndrewsPlanetWest']:
        with open(os.path.join(filepath,site ,site+ '_output.pkl'), 'rb') as f:
            outputdict = pickle.load(f)
        violindictPl[site] = outputdict['vthreshold']
    
    violindict['StAndrewsEast'].extend(violindictPl['StAndrewsPlanetEast'])
    violindict['StAndrewsWest'].extend(violindictPl['StAndrewsPlanetWest'])
    
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
    plt.ylim(0.05,0.55)
    
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
    
  