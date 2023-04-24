#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:51:37 2023

@author: fmuir
"""
import os
import numpy as np
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

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
import matplotlib.dates as mdates
plt.ion()

import rasterio


# SCALING:
# Journal 2-column width: 224pt or 3.11in
# Journal 1-column width: 384pt or 5.33in
# Spacing between: 0.33in
# Journal 2-column page: 6.55in


#%%

def movingaverage(interval, windowsize):
    # moving average trendline
    window = np.ones(int(windowsize))/float(windowsize)
    return np.convolve(interval, window, 'same')

#%%

def SatGIF(metadata,settings,output):
    """
    Create animated GIF of sat images and their extracted shorelines.
    
    FM Jul 2022
    
    Parameters
    ----------
    Sat : list
        Image collection metadata


    Returns
    -------
    None.
    """
    

    polygon = settings['inputs']['polygon']
    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    dates = settings['inputs']['dates']

    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')
    if not os.path.exists(filepath_jpg):
            os.makedirs(filepath_jpg)
    # close all open figures
    plt.close('all')
    
    ims_ms = []
    ims_date = []
    
    
    # Loop through satellite list
    for satname in metadata.keys():

        # Get image metadata
        ## need to fix: get this from output not metadata as some images get skipped by user
        filenames = metadata[satname]['filenames']
        filedates = metadata[satname]['dates']
        
        
        # loop through the images
        for i in range(len(filenames)):

            print('\r%s:   %d%%' % (satname,int(((i+1)/len(filenames))*100)), end='')
            
            # TO DO: need to load in images from jpg_files folder
            # Append image array and dates to lists for plotting
            img = rasterio.open(filenames[i])
            im_RGB = img.read()
            
            ims_ms.append(im_RGB)
            ims_date.append(filedates[i])
            
    shorelineArr = output['shorelines']
    sl_date = output['dates']
    
    # shoreline dataframe back to array
    # TO DO: need to load in shorelines from shapefile and match up each date to corresponding image
    #shorelineArr = Toolbox.GStoArr(shoreline)
    # sl_pix=[]
    # for line in shorelineArr:
    #     sl_pix.append(Toolbox.convert_world2pix(shorelineArr, georef))
    
    # Sort image arrays and dates by date
    ims_date_sort, ims_ms_sort = (list(t) for t in zip(*sorted(zip(ims_date, ims_ms), key=lambda x: x[0])))
    
    # Set up figure for plotting
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.grid(False)
    # Set up function to be called repeatedly for FuncAnimation()
    def animate(n):
        ax.imshow(ims_ms_sort[n])
        ax.set_title(ims_date_sort[n])

    # Use FuncAnimation() which sets a figure and calls a function repeatedly for as many frames as you set
    anim = FuncAnimation(fig=fig, func=animate, frames=len(ims_ms), interval=1, repeat=False)
    # Save as GIF; fps controls the speed of refresh
    anim.save(os.path.join(filepath_jpg, sitename + '_AnimatedImages.gif'),fps=3)




def VegTimeseries(sitename, TransectDict, TransectID, daterange):
    """
    

    Parameters
    ----------
    ValidDict : TYPE
        DESCRIPTION.
    TransectID : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    
    plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectDict['dates'][TransectID][daterange[0]:daterange[1]]]
    plotsatdist = TransectDict['distances'][TransectID][daterange[0]:daterange[1]]
    plotsatdist = np.array(plotsatdist)[(np.array(plotsatdist) < np.mean(plotsatdist)+40) & (np.array(plotsatdist) > np.mean(plotsatdist)-40)]
    
    plotdate, plotsatdist = [list(d) for d in zip(*sorted(zip(plotdate, plotsatdist), key=lambda x: x[0]))]
    
    # linear regression line
    x = mpl.dates.date2num(plotdate)
    msat, csat = np.polyfit(x,plotsatdist,1)
    polysat = np.poly1d([msat, csat])
    xx = np.linspace(x.min(), x.max(), 100)
    dd = mpl.dates.num2date(xx)
    
    # scaling for single column A4 page
    mpl.rcParams.update({'font.size':8})
    fig, ax = plt.subplots(1,1,figsize=(6.55,3), dpi=300)
    
    ax.plot(plotdate, plotsatdist, linewidth=0, marker='.', c='k', markersize=6, markeredgecolor='k', label='Satellite VegEdge')
    plt.grid(color=[0.7,0.7,0.7], ls=':', lw=0.5)
    
    recjanlist = []
    recmarchlist = []
    for i in range(plotdate[0].year-1, plotdate[-1].year):
        recjan = mdates.date2num(datetime(i, 12, 1, 0, 0))
        recmarch = mdates.date2num(datetime(i+1, 3, 1, 0, 0))
        recwidth = recmarch - recjan
        rec = mpatches.Rectangle((recjan, -500), recwidth, 1000, fc=[0,0.3,1], ec=None, alpha=0.3)
        ax.add_patch(rec)
    
    # recstart= mdates.date2num(plotdate[0])
    # recend= mdates.date2num(plotdate[10])
    # recwidth= recend - recstart
    
    # rec = mpatches.Rectangle((recstart,0), recwidth, 50, color=[0.8,0.8,0.8])
    # ax.add_patch(rec)
    
    # plot trendlines
    yav = movingaverage(plotsatdist, 3)
    ax.plot(plotdate, yav, 'green', lw=1.5, label='3pt Moving Average')
    ax.plot(dd, polysat(xx), '--', color='C7', lw=1.5, label=str(round(msat*365.25,2))+'m/yr')

    plt.legend()
    plt.title('Transect '+str(TransectID))
    plt.xlabel('Date')
    plt.ylabel('Cross-shore distance')
    # plt.xlim(plotdate[0]-10, plotdate[-1]+10)
    plt.ylim(min(plotsatdist)-10, max(plotsatdist)+10)
    plt.tight_layout()
    
    plt.savefig(os.path.join(outfilepath,sitename + '_SatTimeseries_Transect'+str(TransectID)+'.png'))
    print('Plot saved under '+os.path.join(outfilepath,sitename + '_SatTimeseries_Transect'+str(TransectID)+'.png'))
    
    plt.show()
    
    

def ValidTimeseries(sitename, ValidDict, TransectID):
    """
    

    Parameters
    ----------
    ValidDict : TYPE
        DESCRIPTION.
    TransectID : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    
    plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in ValidDict['Vdates'][TransectID]]
    plotsatdist = ValidDict['distances'][TransectID]
    plotvaliddist = ValidDict['Vdists'][TransectID]
    
    plotdate, plotvaliddist = [list(d) for d in zip(*sorted(zip(plotdate, plotvaliddist), key=lambda x: x[0]))]
    plotdate, plotsatdist = [list(d) for d in zip(*sorted(zip(plotdate, plotsatdist), key=lambda x: x[0]))]
    
    magma = cm.get_cmap('magma')
    
    x = mpl.dates.date2num(plotdate)
    mvalid, cvalid = np.polyfit(x,plotvaliddist,1)
    msat, csat = np.polyfit(x,plotsatdist,1)
    
    polyvalid = np.poly1d([mvalid, cvalid])
    polysat = np.poly1d([msat, csat])
    
    xx = np.linspace(x.min(), x.max(), 100)
    dd = mpl.dates.num2date(xx)
    
    mpl.rcParams.update({'font.size':8})
    fig, ax = plt.subplots(1,1,figsize=(6.55,3), dpi=300)
    
    validlabels = ['Validation VegEdge','_nolegend_','_nolegend_','_nolegend_']
    satlabels = ['Satellite VegEdge','_nolegend_','_nolegend_','_nolegend_',]
    
    for i,c in enumerate([0.95,0.7,0.6,0.2]):
        ax.plot(plotdate[i], plotvaliddist[i], 'X', color=magma(c), markersize=10,markeredgecolor='k', label=validlabels[i])
        ax.plot(plotdate[i], plotsatdist[i], 'o', color=magma(c),markersize=10,markeredgecolor='k', label=satlabels[i])
    
    
    ax.plot(dd, polyvalid(xx), '--', color=[0.7,0.7,0.7], zorder=0, label=str(round(mvalid*365.25,2))+'m/yr')
    ax.plot(dd, polysat(xx), '-', color=[0.7,0.7,0.7], zorder=0, label=str(round(msat*365.25,2))+'m/yr')
    
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Cross-shore distance')
    plt.tight_layout()
    
    plt.savefig(os.path.join(outfilepath,sitename + '_ValidVsSatTimeseries_Transect'+str(TransectID)+'.png'))
    print('Plot saved under '+os.path.join(outfilepath,sitename + '_ValidVsSatTimeseries_Transect'+str(TransectID)+'.png'))
    
    plt.show()


def WidthTimeseries(sitename, TransectDict, TransectID, daterange):
    """
    

    Parameters
    ----------
    ValidDict : TYPE
        DESCRIPTION.
    TransectID : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    
    plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectDict['wldates'][TransectID][daterange[0]:daterange[1]]]

    plotwldate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectDict['wldates'][TransectID][daterange[0]:daterange[1]]]
    plotvegdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectDict['dates'][TransectID][daterange[0]:daterange[1]]]

    plotvegdist = TransectDict['distances'][TransectID][daterange[0]:daterange[1]]
    plotwldist = TransectDict['wlcorrdist'][TransectID][daterange[0]:daterange[1]]
    plotsatdist = TransectDict['beachwidth'][TransectID][daterange[0]:daterange[1]]

    plotvegdate, plotvegdist = [list(d) for d in zip(*sorted(zip(plotvegdate, plotvegdist), key=lambda x: x[0]))]
    plotwldate, plotwldist = [list(d) for d in zip(*sorted(zip(plotwldate, plotwldist), key=lambda x: x[0]))]
    plotdate, plotsatdist = [list(d) for d in zip(*sorted(zip(plotdate, plotsatdist), key=lambda x: x[0]))]

    # linear regression line
    x = mpl.dates.date2num(plotdate)
    msat, csat = np.polyfit(x,plotsatdist,1)
    polysat = np.poly1d([msat, csat])
    xx = np.linspace(x.min(), x.max(), 100)
    dd = mpl.dates.num2date(xx)
    
    # scaling for single column A4 page
    mpl.rcParams.update({'font.size':8})
    fig, ax = plt.subplots(1,1,figsize=(6.55,3), dpi=300)
    
    ax.plot(plotdate, plotsatdist, linewidth=0, marker='.', c='k', markersize=8, markeredgecolor='k', label='Upper Beach Width')
    # plt.plot(plotvegdate, plotvegdist, linewidth=0, marker='.', c='g', markersize=8, label='Upper Beach Width')
    # plt.plot(plotwldate, plotwldist, linewidth=0, marker='.', c='b', markersize=8,  label='Upper Beach Width')

    # plot trendlines
    yav = movingaverage(plotsatdist, 3)
    ax.plot(plotdate, yav, 'r', label='3pt Moving Average')
    ax.plot(dd, polysat(xx), '--', color=[0.7,0.7,0.7], zorder=0, label=str(round(msat*365.25,2))+'m/yr')

    
    plt.legend()
    plt.title('Transect '+str(TransectID))
    plt.xlabel('Date')
    plt.ylabel('Cross-shore distance')
    plt.ylim(-200,1000)
    plt.tight_layout()
    
    plt.savefig(os.path.join(outfilepath,sitename + '_SatTimeseries_Transect'+str(TransectID)+'.png'))
    print('Plot saved under '+os.path.join(outfilepath,sitename + '_SatTimeseries_Transect'+str(TransectID)+'.png'))
    
    plt.show()



def BeachWidthSeries(TransectID):
    
    f = plt.figure(figsize=(8, 3))
    
    
    plt.plot('.-', color='k')
    
    
    