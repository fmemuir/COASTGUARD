#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:51:37 2023

@author: fmuir
"""
import os
import glob
import pickle
import numpy as np
import string
import warnings
from datetime import datetime, timedelta
import calendar
warnings.filterwarnings("ignore")
import pdb

import matplotlib as mpl
from matplotlib import cm
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.patches import Patch, Rectangle, ConnectionPatch
import matplotlib.lines as mlines
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as PathEffects
import matplotlib.font_manager as mplfm
import matplotlib.colors as mcolors

plt.ion()

import rasterio
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import scipy.stats
from scipy.stats import circmean
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.sans-serif'] = 'Arial'

from Toolshed import Toolbox, Waves, PlottingSeaborn

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

#%%

def SatGIF(metadata,settings,output):
    """
    Create animated GIF of sat images and their extracted shorelines.
    
    FM Jul 2022
    
    Parameters
    ----------
    Sat : list
        Image collection metadata

    """
    

    polygon = settings['inputs']['polygon']
    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    dates = settings['inputs']['dates']

    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, 'img_files', 'detection')
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
            
            # TO DO: need to load in images from img_files folder
            # Append image array and dates to lists for plotting
            img = rasterio.open(filenames[i])
            im_RGB = img.read()
            
            ims_ms.append(im_RGB)
            ims_date.append(filedates[i])
            
 
    # shoreline dataframe back to array
    # TO DO: need to load in shorelines from shapefile and match up each date to corresponding image
    # shorelineArr = output['shorelines']
    # sl_date = output['dates']
    
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





def VegTimeseries(sitename, TransectInterGDF, TransectIDs, Hemisphere='N', Titles=None, ShowPlot=True):
    """
    Plot timeseries of cross-shore veg edge change for selected transect(s).
    If more than one transect is supplied in a list, create subplots for comparison.
    FM Nov 2022

    Parameters
    ----------
    sitename : str
        Name of site.
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with veg edge lines.
    TransectIDs : list
        List of transect IDs to plot.
    Hemisphere : str, optional
        Northern (N) or Southern (S) Hemisphere for marking 'winter' season. The default is 'N'.
    Titles : list, optional
        List of strings of same length as TransectIDs, denoting what alternative 
        title the plots should have. The default is None.
    ShowPlot : bool, optional
        Flag to turn plt.show() on or off (if plotting lots of transects). The default is True.


    """
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    figID = ''
    
    if ShowPlot is False:
        plt.ioff()
    
    # if more than one Transect ID is to be compared on a single plot
    if type(TransectIDs) == list:
        # scaling for single column A4 page
        mpl.rcParams.update({'font.size':7})
        # fig, axs = plt.subplots(len(TransectIDs),1,figsize=(11.6,5.9), dpi=300, sharex=True)
        fig, axs = plt.subplots(len(TransectIDs),1,figsize=(6.55,4), dpi=300, sharex=True)

    else:
        TransectIDs = [TransectIDs]
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':7})
        # use 2 subplots with one empty to be able to loop through them
        # fig, axs = plt.subplots(1,1,figsize=(11.6,5.9), dpi=300, sharex=True)
        fig, axs = plt.subplots(1,1,figsize=(6.55,4), dpi=300, sharex=True)
        axs = [axs] # to be able to loop through
        
    # common plot labels
    lab = fig.add_subplot(111,frameon=False)
    lab.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False, right=False)
    if type(TransectIDs) == list: 
        lab.set_xlabel('Date')#,labelpad=22)
    else:
        lab.set_xlabel('Date')
    lab.set_ylabel('Cross-shore distance (veg) (m)', color='#81A739')
    
    if Titles==None:
        Titles = ['Transect '+str(TransectID) for TransectID in TransectIDs]
        
    for TransectID, ax, Title in zip(TransectIDs,axs, Titles):
        daterange = [0,len(TransectInterGDF['dates'].iloc[TransectID])]
        plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDF['dates'].iloc[TransectID][daterange[0]:daterange[1]]]
        plotsatdist = TransectInterGDF['distances'].iloc[TransectID][daterange[0]:daterange[1]]
        # remove and interpolate outliers
        plotsatdistinterp = InterpNaN(plotsatdist)
        
        if len(plotdate) == 0:
            print('Transect %s is empty! No values to plot.' % (TransectID))
            return
        
        plotdate, plotsatdist = [list(d) for d in zip(*sorted(zip(plotdate, plotsatdist), key=lambda x: x[0]))]    
        ax.grid(color=[0.7,0.7,0.7], ls=':', lw=0.5, zorder=0)        
                
        ax.scatter(plotdate, plotsatdist, marker='o', c='#81A739', s=5, alpha=0.8, edgecolors='none', label='Sat. VegEdge')
        
        # xaxis ticks as year with interim Julys marked
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        
        # create error bar lines to fill between
        for axloop, errorRMSE, plotdist, col in zip([ax], [10.4], [plotsatdist], ['#81A739']):
            yerrorplus = [x + errorRMSE for x in plotdist]
            yerrorneg = [x - errorRMSE for x in plotdist]
            # axloop.fill_between(plotdate, yerrorneg, yerrorplus, color=col, alpha=0.3, edgecolor=None)
       
        # ax2.errorbar(plotdate, plotsatdist, yerr=errorRMSE, elinewidth=0.5, fmt='none', ecolor='#81A739')
            
        # create rectangles highlighting winter months (based on N or S hemisphere 'winter')
        for i in range(plotdate[0].year-1, plotdate[-1].year):
            if Hemisphere == 'N':
                rectWinterStart = mdates.date2num(datetime(i, 11, 1, 0, 0))
                rectWinterEnd = mdates.date2num(datetime(i+1, 3, 1, 0, 0))
            elif Hemisphere == 'S':
                rectWinterStart = mdates.date2num(datetime(i, 5, 1, 0, 0))
                rectWinterEnd = mdates.date2num(datetime(i, 9, 1, 0, 0))
            rectwidth = rectWinterEnd - rectWinterStart
            rect = mpatches.Rectangle((rectWinterStart, -2000), rectwidth, 4000, fc=[0.3,0.3,0.3], ec=None, alpha=0.2)
            ax.add_patch(rect)
          
        # plot trendlines
        vegav = MovingAverage(plotsatdistinterp, 3)
        if len(plotdate) >= 3:
            ax.plot(plotdate, vegav, color='#81A739', lw=1.5, label='3pt Mov. Av. VegEdge')
    
        # linear regression lines
        x = mpl.dates.date2num(plotdate)
        for y, pltax, clr in zip([plotsatdist], [ax], ['#3A4C1A']):
            m, c = np.polyfit(x,y,1)
            polysat = np.poly1d([m, c])
            xx = np.linspace(x.min(), x.max(), 100)
            dd = mpl.dates.num2date(xx)
            pltax.plot(dd, polysat(xx), '--', color=clr, lw=1.5, label=str(round(m*365.25,2))+' m/yr')
            
        ax.title.set_text(Title)
        
        # ax.set_xlabel('Date (yyyy-mm)')
        # ax2.set_ylabel('Cross-shore distance (veg) (m)', color='#81A739')
        # ax.set_ylabel('Cross-shore distance (water) (m)', color='#4056F4')
        # plt.xlim(plotdate[0]-10, plotdate[-1]+10)
        ax.set_ylim(np.nanmin(plotsatdistinterp)-10, np.nanmax(plotsatdistinterp)+20)
        ax.set_xlim(np.nanmin(plotdate)-timedelta(days=100),np.nanmax(plotdate)+timedelta(days=100))
        
        leg1 = ax.legend(loc=2, ncol=3, handlelength=2., handletextpad=0.5, columnspacing=1.5)
        # weird zorder with twinned axes; remove first axis legend and plot on top of second
        # leg1.remove()
        
        figID += '_'+str(TransectID)
        plt.tight_layout()
        
    figname = os.path.join(outfilepath,sitename + '_SatVegTimeseriesTrend_Transect'+figID+'.png')
    
    plt.tight_layout()
            
    plt.savefig(figname, bbox_inches='tight')
    print('Plot saved under '+figname)
    
    plt.show()
    
    
def VegTimeseriesNeon(sitename, TransectInterGDF, TransectIDs, Hemisphere='N', ShowPlot=True):
    """
    Plot timeseries of cross-shore veg edge change for selected transect(s) [NEON EFFECT]
    If more than one transect is supplied in a list, create subplots for comparison.
    FM Nov 2022

    Parameters
    ----------
    sitename : str
        Name of site.
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with veg edge lines.
    TransectIDs : list
        List of transect IDs to plot.
    Hemisphere : str, optional
        Northern (N) or Southern (S) Hemisphere for marking 'winter' season. The default is 'N'.
    ShowPlot : bool, optional
        Flag to turn plt.show() on or off (if plotting lots of transects). The default is True.


    """
    
    def neonplot(x,y,colour,ax=None):
        """
        Apply neon effect by plotting successively narrowing lines atop one another.
        """
        if ax is None:
            ax = plt.gca()
        line, = ax.plot(x, y, lw=1, color=colour, zorder=6, label='3pt Mov. Av. VegEdge')
        for cont in range(6,1,-1):
            ax.plot(x, y, lw=cont, color=colour, zorder=5, alpha=0.05)
        return ax
        
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    figID = ''
    
    if ShowPlot is False:
        plt.ioff()
    
    # neon style
    repo = "https://raw.githubusercontent.com/nicoguaro/matplotlib_styles/master"
    style = repo + "/styles/neon.mplstyle"
    plt.style.use(style)
    
    # if more than one Transect ID is to be compared on a single plot
    if type(TransectIDs) == list:
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':12})
        fig, axs = plt.subplots(len(TransectIDs),1,figsize=(11.6,5.9), dpi=300, sharex=True)
    else:
        TransectIDs = [TransectIDs]
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':12})
        # use 2 subplots with one empty to be able to loop through them
        fig, axs = plt.subplots(1,1,figsize=(11.6,5.9), dpi=300, sharex=True)
        axs = [axs] # to be able to loop through
        
    # common plot labels
    lab = fig.add_subplot(111,frameon=False)
    lab.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False, right=False)
    if type(TransectIDs) == list: 
        lab.set_xlabel('Date (yyyy-mm)', labelpad=22)
    else:
        lab.set_xlabel('Date (yyyy-mm)')
    lab.set_ylabel('Cross-shore distance (veg) (m)', color='#24FC0E')
    
    for TransectID, ax in zip(TransectIDs,axs):
        daterange = [0,len(TransectInterGDF['dates'].iloc[TransectID])]
        plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDF['dates'].iloc[TransectID][daterange[0]:daterange[1]]]
        plotsatdist = TransectInterGDF['distances'].iloc[TransectID][daterange[0]:daterange[1]]
        # remove and interpolate outliers
        plotsatdistinterp = InterpNaN(plotsatdist)
        
        if len(plotdate) == 0:
            print('Transect %s is empty! No values to plot.' % (TransectID))
            return
        
        plotdate, plotsatdist = [list(d) for d in zip(*sorted(zip(plotdate, plotsatdist), key=lambda x: x[0]))]    
        # ax.grid(color=[0.7,0.7,0.7], ls=':', lw=0.5, zorder=0)        
                
        # ax.scatter(plotdate, plotsatdist, marker='o', c='#81A739', s=6, alpha=0.8, edgecolors='none', label='Sat. VegEdge')
        
        # xaxis ticks as year with interim Julys marked
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        
        # create error bar lines to fill between
        for axloop, errorRMSE, plotdist, col in zip([ax], [10.4], [plotsatdist], ['#81A739']):
            yerrorplus = [x + errorRMSE for x in plotdist]
            yerrorneg = [x - errorRMSE for x in plotdist]
            # axloop.fill_between(plotdate, yerrorneg, yerrorplus, color=col, alpha=0.3, edgecolor=None)
       
        # ax2.errorbar(plotdate, plotsatdist, yerr=errorRMSE, elinewidth=0.5, fmt='none', ecolor='#81A739')
        
        # create rectangles highlighting winter months (based on N or S hemisphere 'winter')
        for i in range(plotdate[0].year-1, plotdate[-1].year):
            if Hemisphere == 'N':
                rectWinterStart = mdates.date2num(datetime(i, 11, 1, 0, 0))
                rectWinterEnd = mdates.date2num(datetime(i+1, 3, 1, 0, 0))
            elif Hemisphere == 'S':
                rectWinterStart = mdates.date2num(datetime(i, 5, 1, 0, 0))
                rectWinterEnd = mdates.date2num(datetime(i, 9, 1, 0, 0))
            rectwidth = rectWinterEnd - rectWinterStart
            rect = mpatches.Rectangle((rectWinterStart, -2000), rectwidth, 4000, fc=[0.3,0.3,0.3], ec=None, alpha=0.2)
            # ax.add_patch(rect)
          
        # plot trendlines
        vegav = MovingAverage(plotsatdistinterp, 3)
        if len(plotdate) >= 3:
            neonplot(plotdate, vegav, colour='#24FC0E',ax=ax)
    
        # linear regression lines
        x = mpl.dates.date2num(plotdate)
        for y, pltax, clr in zip([plotsatdist], [ax], ['#3A4C1A']):
            m, c = np.polyfit(x,y,1)
            polysat = np.poly1d([m, c])
            xx = np.linspace(x.min(), x.max(), 100)
            dd = mpl.dates.num2date(xx)
            # pltax.plot(dd, polysat(xx), '--', color=clr, lw=1, label=str(round(m*365.25,2))+' m/yr')
    
        ax.title.set_text('Transect '+str(TransectID))
            
        # ax.set_xlabel('Date (yyyy-mm)')
        # ax2.set_ylabel('Cross-shore distance (veg) (m)', color='#81A739')
        # ax.set_ylabel('Cross-shore distance (water) (m)', color='#4056F4')
        # plt.xlim(plotdate[0]-10, plotdate[-1]+10)
        ax.set_ylim(np.nanmin(plotsatdistinterp)-10, np.nanmax(plotsatdistinterp)+30)
        ax.set_xlim(np.nanmin(plotdate)-timedelta(days=100),np.nanmax(plotdate)+timedelta(days=100))
        
        leg1 = ax.legend(loc=2)
        # weird zorder with twinned axes; remove first axis legend and plot on top of second
        # leg1.remove()
        
        figID += '_'+str(TransectID)
        plt.tight_layout()
        
    
    figname = os.path.join(outfilepath,sitename + '_SatVegTimeseriesNeon_Transect'+figID+'.png')
    
    plt.tight_layout()
            
    plt.savefig(figname, bbox_inches='tight')
    print('Plot saved under '+figname)
    
    plt.show()
    
    # reset to default style after neo style setting
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    
def VegWaterTimeseries(sitename, TransectInterGDF, TransectIDs, Hemisphere='N', ShowPlot=True):
    """
    Plot timeseries of cross-shore veg edge and waterline change for selected transect(s).
    If more than one transect is supplied in a list, create subplots for comparison.
    FM Nov 2022

    Parameters
    ----------
    sitename : str
        Name of site.
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with veg edge lines.
    TransectID : list
        Transect ID(s) to plot.
        

    """
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    figID = ''
       
    if ShowPlot is False:
        plt.ioff()
        
    # if more than one Transect ID is to be compared on a single plot
    if type(TransectIDs) == list:
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':7})
        fig, axs = plt.subplots(len(TransectIDs),1,figsize=(6.55,4), dpi=300, sharex=True)
    else:
        TransectIDs = [TransectIDs]
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':7})
        fig, axs = plt.subplots(1,1,figsize=(6.55,4), dpi=300, sharex=True)
        axs = [axs] # to be able to loop through
        
    # common plot labels
    lab = fig.add_subplot(111,frameon=False)
    lab2 = lab.twinx()
    for loc in ['top','right','bottom','left']:
        lab2.spines[loc].set_visible(False)
    lab2.get_xaxis().set_ticks([])
    lab2.get_yaxis().set_ticks([])
    lab.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False, right=False)
    lab2.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False, right=False)
    if type(TransectIDs) == list: 
        lab.set_xlabel('Date (yyyy-mm)', labelpad=22)
    else:
        lab.set_xlabel('Date (yyyy-mm)')
    lab.set_ylabel('Cross-shore distance (water) (m)', labelpad=22, color='#4056F4')
    lab2.set_ylabel('Cross-shore distance (veg) (m)', color='#81A739')

    
    for TransectID, ax in zip(TransectIDs,axs):
        print('plotting transect',TransectID)
        daterange = [0,len(TransectInterGDF['dates'].iloc[TransectID])]
        plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDF['dates'].iloc[TransectID][daterange[0]:daterange[1]]]
        plotwldate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDF['wldates'].iloc[TransectID][daterange[0]:daterange[1]]]
        plotsatdist = TransectInterGDF['distances'].iloc[TransectID][daterange[0]:daterange[1]]
        plotwldist = TransectInterGDF['wlcorrdist'].iloc[TransectID][daterange[0]:daterange[1]]
        # remove and interpolate outliers
        plotsatdistinterp = InterpNaN(plotsatdist)
        plotwldistinterp = InterpNaN(plotwldist)
        
        if len(plotdate) == 0:
            print('no intersections on Transect',TransectID)
            return
        
        plotdate, plotsatdist, plotwldist = [list(d) for d in zip(*sorted(zip(plotdate, plotsatdist, plotwldist), key=lambda x: x[0]))]    
        ax.grid(color=[0.7,0.7,0.7], ls=':', lw=0.5, zorder=0)        
        
        ax2 = ax.twinx()
        
        ax.scatter(plotwldate, plotwldist, marker='o', c='#4056F4', s=4, alpha=0.8, edgecolors='none', label='Satellite waterline')
        ax2.scatter(plotdate, plotsatdist, marker='o', c='#81A739', s=4, alpha=0.8, edgecolors='none', label='Satellite veg edge')
        
        # xaxis ticks as year with interim Julys marked
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        
        # create error bar lines to fill between
        for axloop, errorRMSE, xaxis, plotdist, col in zip([ax, ax2], [7.2, 10.4], [plotwldate,plotdate], [plotwldist,plotsatdist], ['#4056F4','#81A739']):
            yerrorplus = [x + errorRMSE for x in plotdist]
            yerrorneg = [x - errorRMSE for x in plotdist]
            axloop.fill_between(xaxis, yerrorneg, yerrorplus, color=col, alpha=0.3, edgecolor=None)
       
        # ax2.errorbar(plotdate, plotsatdist, yerr=errorRMSE, elinewidth=0.5, fmt='none', ecolor='#81A739')
            
        # create rectangles highlighting winter months (based on N or S hemisphere 'winter')
        for i in range(plotdate[0].year-1, plotdate[-1].year):
            if Hemisphere == 'N':
                rectWinterStart = mdates.date2num(datetime(i, 11, 1, 0, 0))
                rectWinterEnd = mdates.date2num(datetime(i+1, 3, 1, 0, 0))
            elif Hemisphere == 'S':
                rectWinterStart = mdates.date2num(datetime(i, 5, 1, 0, 0))
                rectWinterEnd = mdates.date2num(datetime(i, 9, 1, 0, 0))
            rectwidth = rectWinterEnd - rectWinterStart
            rect = mpatches.Rectangle((rectWinterStart, -2000), rectwidth, 4000, fc=[0.3,0.3,0.3], ec=None, alpha=0.2)
            ax.add_patch(rect)
          
        # plot trendlines
        vegav = MovingAverage(plotsatdistinterp, 3)
        wlav = MovingAverage(plotwldistinterp, 3)
        if len(plotwldate) != len(wlav):
            print('inconsistent number of plot dates to water level moving average (3pts), Transect', TransectID)
            return
        if len(plotdate) != len(vegav):
            print('inconsistent number of plot dates to vegetation edge moving average (3pts), Transect', TransectID)
            return
        ax.plot(plotdate, wlav, color='#4056F4', lw=1, label='3pt Moving Average waterline')
        ax2.plot(plotdate, vegav, color='#81A739', lw=1, label='3pt Moving Average veg edge')
    
        # linear regression lines
        for xaxis, y, pltax, clr in zip([plotwldate,plotdate], [plotwldist,plotsatdist], [ax,ax2], ['#0A1DAE' ,'#3A4C1A']):
            x = mpl.dates.date2num(xaxis)
            m, c = np.polyfit(x,y,1)
            polysat = np.poly1d([m, c])
            xx = np.linspace(x.min(), x.max(), 100)
            dd = mpl.dates.num2date(xx)
            pltax.plot(dd, polysat(xx), '--', color=clr, lw=1, label=str(round(m*365.25,2))+' m/yr')
    
        ax.title.set_text('Transect '+str(TransectID))
            

        ax2.set_ylim(np.nanmin(plotsatdistinterp)-10, np.nanmax(plotsatdistinterp)+30)
        ax.set_ylim(np.nanmin(plotwldistinterp)-10, np.nanmax(plotwldistinterp)+30)
        if len(plotdate) > len(plotwldate): # set axis limits to longer timeframe
            ax.set_xlim(np.nanmin(plotdate)-timedelta(days=100),np.nanmax(plotdate)+timedelta(days=100))
        else:
            ax.set_xlim(np.nanmin(plotwldate)-timedelta(days=100),np.nanmax(plotwldate)+timedelta(days=100))

        
        leg1 = ax.legend(loc=2)
        leg2 = ax2.legend(loc=1)
        # weird zorder with twinned axes; remove first axis legend and plot on top of second
        leg1.remove()
        ax2.add_artist(leg1)
        
        figID += '_'+str(TransectID)
        plt.tight_layout()
        
    figname = os.path.join(outfilepath,sitename + '_SatVegWaterTimeseries_Transect'+figID+'.png')
    
    plt.tight_layout()
            
    plt.savefig(figname, bbox_inches='tight')
    print('Plot saved under '+figname)
    
    plt.show()

    
def VegWaterSeasonality(sitename, TransectInterGDF, TransectIDs, Titles=None, Hemisphere='N', Normal=False, P=None):
    '''
    Plot three stacked subplots of vegedge and waterline timeseries, decomposed trend, and seasonal signal.
    FM Oct 2023

    Parameters
    ----------
    sitename : str
        Name of site.
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with veg edge lines.
    TransectIDs : list
        List of transect IDs to plot.
    Hemisphere : str, optional
        Northern (N) or Southern (S) Hemisphere for marking 'winter' season. The default is 'N'.
    Normal : bool, optional
        Flag to normalise axes between veg and water. The default is False.
    P : bool, optional
        Flag for calculting seasonality period (observations per cycle) using N obs. The default is None.

    '''
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    figID = ''
        
    # if more than one Transect ID is to be compared on a single plot
    if type(TransectIDs) == list:
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':7})
        fig, axs = plt.subplots(3,len(TransectIDs),figsize=(6.55,3.5), dpi=300, sharex=True)
    else:
        TransectIDs = [TransectIDs]
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':7})
        fig, axs = plt.subplots(3,1,figsize=(6.55,3.5), dpi=300, sharex=True)
        axs = [axs] # to be able to loop through
            
    for TransectID, Title, col in zip(TransectIDs, Titles, range(axs.shape[1])):
        # Define variables for each subplot per column/Transect
        ax_TS = axs[0,col]
        ax_Trend = axs[1,col]
        ax_Season = axs[2,col]
        
        # Process plot data
        plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDF['dates'].iloc[TransectID]]
        plotsatdist = TransectInterGDF['distances'].iloc[TransectID]
        plotwldate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDF['wldates'].iloc[TransectID]]
        plotwldist = TransectInterGDF['wlcorrdist'].iloc[TransectID]
    
        plotdate, plotwldate, plotsatdist, plotwldist = [list(d) for d in zip(*sorted(zip(plotdate, plotwldate, plotsatdist, plotwldist), key=lambda x: x[0]))]    
        
        # Calculate period (observations per cycle) to use for seasonality
        # DateDiff = []
        # for i in range(1,len(plotdate)):
        #     DateDiff.append(plotdate[i]-plotdate[i-1])
        # MeanDateDiff = np.mean(DateDiff).days # days between each observation
        # if P is None:
        #     P = round(90 / MeanDateDiff) # 90 days = 3 month cycle
        
        # Set up common subplot design
        for ax in [ax_TS,ax_Trend,ax_Season]: 
            ax.grid(color=[0.7,0.7,0.7], ls=':', lw=0.5, zorder=0)        
            
            # create rectangles highlighting winter months (based on N or S hemisphere 'winter')
            for i in range(plotdate[0].year-1, plotdate[-1].year):
                if Hemisphere == 'N':
                    rectWinterStart = mdates.date2num(datetime(i, 11, 1, 0, 0))
                    rectWinterEnd = mdates.date2num(datetime(i+1, 3, 1, 0, 0))
                elif Hemisphere == 'S':
                    rectWinterStart = mdates.date2num(datetime(i, 5, 1, 0, 0))
                    rectWinterEnd = mdates.date2num(datetime(i, 9, 1, 0, 0))
                rectwidth = rectWinterEnd - rectWinterStart
                rect = mpatches.Rectangle((rectWinterStart, -2000), rectwidth, 4000, fc=[0.3,0.3,0.3], ec=None, alpha=0.2)
                ax.add_patch(rect)
        
        ax_TS_veg = ax_TS.twinx()
        ax_Trend_veg = ax_Trend.twinx()
        ax_Season_veg = ax_Season.twinx()
               
        # plot trendlines
        # vegav = MovingAverage(plotsatdist, 3)
        # wlav = MovingAverage(plotwldist, 3)
        # if len(plotdate) >= 3:
            # ax_TS.plot(plotdate, wlav, color='#4056F4', lw=1, label='3pt Moving Average waterline')
            # ax_TS2.plot(plotdate, vegav, color='#81A739', lw=1, label='3pt Moving Average veg edge')
        twin_TS_lim = []
        twin_Trend_lim = []
        twin_Season_lim = []        
        for twin_TS, twin_Trend, twin_Season, twin_lab, x, y, clr in zip([ax_TS, ax_TS_veg],
                                                            [ax_Trend, ax_Trend_veg],
                                                            [ax_Season, ax_Season_veg],
                                                            ['WL', 'VE'],
                                                            [plotwldate, plotdate],
                                                            [plotwldist, plotsatdist],
                                                            ['#0A1DAE', '#81A739']):
            
            # Extend and interpolate to create daily observations
            Timeseries = pd.Series(y, index=x)
            Timeseries = Timeseries.groupby(Timeseries.index).mean() # if any duplicates, take mean
            Timeseries = Timeseries.resample('1D')
            Timeseries = Timeseries.interpolate(method='time')
            
            # Calculate seasonal .trend, .seasonal and .resid, using a year as the detrending period
            Seasonality = seasonal_decompose(Timeseries, model='additive', period=365)
            Season = Seasonality.seasonal
            Resid = Seasonality.resid
            SSI = np.var(Season) / (np.var(Season) + np.var(Resid))
            
            print('Transect '+str(TransectID)+' '+twin_lab+' seasonality index: '+str(SSI))
            
            # PLOT 1: timeseries scatter plot     
            twin_TS.scatter(x, y, marker='o', c=clr, s=4, alpha=0.7, edgecolors='none', zorder=1)
            # linear regression lines
            numx = mpl.dates.date2num(x)
            m, c = np.polyfit(numx,y,1)
            polysat = np.poly1d([m, c])
            xx = np.linspace(numx.min(), numx.max(), 100)
            dd = mpl.dates.num2date(xx)
            twin_TS.plot(dd, polysat(xx), '--', color=clr, lw=1, label=r'$\Delta$'+twin_lab+' = '+str(round(m*365.25,2))+' m/yr', zorder=1)
            twin_TS.set_ylim(np.nanmin(y)-(np.nanmin(y)/10), np.nanmax(y)+(np.nanmax(y)/10))
            twin_TS_lim.append(twin_TS.get_ylim())

            # PLOT 2: De-seasonalised trend 
            twin_Trend.plot(Seasonality.trend, color=clr, lw=1)
            twin_Trend.set_ylim(np.nanmin(Seasonality.trend)-50, np.nanmax(Seasonality.trend)+50)
            twin_Trend_lim.append(twin_Trend.get_ylim())
            
            # PLOT 3: Seasonality line (moving average) with residuals as error window
            # twin_Season.plot(Seasonality.resid, color=clr, lw=0, marker='x', ms=4, alpha=0.3)
            # plot residuals as vertical lines
            # twin_Season.vlines(Seasonality.resid.index,0,np.array(Seasonality.resid),color=clr,lw=0.4,alpha=0.1,label=twin_lab+' Resid.')
            # plot seasonal signal over top of residuals 
            twin_Season.plot(Seasonality.seasonal, color=clr, lw=0.5, label=twin_lab+' SSI = '+str(round(SSI,2)))
            twin_Season.set_ylim(-1*np.nanmax(Seasonality.resid), np.nanmax(Seasonality.resid))
            # horizontal line marking 0
            twin_Season.axhline(0,0,1,color=[0.7,0.7,0.7], lw=0.5, zorder=3, alpha=0.7)
            twin_Season_lim.append(twin_Season.get_ylim())
            
            # twin_TS.set_ylabel('Cross-shore dist (m)', color=clr)
            # twin_Trend.set_ylabel('Overall trend (m)', color=clr) 
            # twin_Season.set_ylabel('Seasonal signal (m)', color=clr)

            if Normal:
                twin_TS.set_ylim(np.min(twin_TS_lim),np.max(twin_TS_lim))
                twin_Trend.set_ylim(np.min(twin_Trend_lim),np.max(twin_Trend_lim))
                twin_Season.set_ylim(np.min(twin_Season_lim),np.max(twin_Season_lim))
            
            
        ax_TS.title.set_text('Transect '+str(TransectID)+' - '+Title)
        ax_TS.set_xlim(min(plotdate)-timedelta(days=100),max(plotdate)+timedelta(days=100))

        for axleg in [[ax_TS, ax_TS_veg], [ax_Season,ax_Season_veg]]:
            leg1 = axleg[0].legend(loc=3, handlelength=1.5, handletextpad=0.1)
            leg2 = axleg[1].legend(loc=4, handlelength=1.5, handletextpad=0.1)
            # weird zorder with twinned axes; remove first axis legend and plot on top of second
            leg1.remove()
            axleg[1].add_artist(leg1)
                    
        figID += '_'+str(TransectID)
        plt.tight_layout()
        
    # Add alphabetical labels to corners of subplots
    ax_labels = list(string.ascii_lowercase[:3*axs.shape[1]])
    axticklabs = ['Cross-shore dist (m)','Cross-shore dist (m)',
                  'Overall trend (m)','Overall trend (m)',
                  'Seasonal signal (m)','Seasonal signal (m)']
    for ax, axID, lab, axticklab in zip(axs.flat, range(len(axs.flat)), ax_labels, axticklabs):
        ax.text(0.011, 0.97, '('+lab+')', transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(facecolor='w', edgecolor='k',pad=1.5))
        if axID % 2 == 0: # if ID is even i.e. on left hand side
            ax.set_ylabel(axticklab, color='#0A1DAE')
            # ax.set_ylabel(None)
        else:
            ax.set_ylabel(axticklab, color='#81A739')
            ax.yaxis.set_label_position('right')
            ax.yaxis.labelpad=25
            
    
    if Normal:  
        figname = os.path.join(outfilepath,sitename + '_SatVegWaterSeasonalNormal_Transect'+figID+'.png')
    else:
        figname = os.path.join(outfilepath,sitename + '_SatVegWaterSeasonal_Transect'+figID+'.png')

    
    plt.tight_layout()
            
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    print('Plot saved under '+figname)
    
    plt.show()
    
    
    
def VegWaterSeasonalitySimple(sitename, TransectInterGDF, TransectIDs, Titles=None, Hemisphere='N', Normal=False, P=None):
    '''
    Plot two stacked subplots of vegedge and waterline timeseries, decomposed trend, and seasonal signal.
    FM Oct 2024

    Parameters
    ----------
    sitename : str
        Name of site.
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with veg edge lines.
    TransectIDs : list
        List of transect IDs to plot.
    Hemisphere : str, optional
        Northern (N) or Southern (S) Hemisphere for marking 'winter' season. The default is 'N'.
    Normal : bool, optional
        Flag to normalise axes between veg and water. The default is False.
    P : bool, optional
        Flag for calculting seasonality period (observations per cycle) using N obs. The default is None.

    '''
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    figID = ''
        
    # if more than one Transect ID is to be compared on a single plot
    if type(TransectIDs) == list:
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':7})
        fig, axs = plt.subplots(3,len(TransectIDs),figsize=(6.55,5.3), dpi=300)
    else:
        TransectIDs = [TransectIDs]
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':7})
        fig, axs = plt.subplots(3,1,figsize=(6.55,5.3), dpi=300)
        axs = [axs] # to be able to loop through
         
    
    
    for TransectID, Title, col in zip(TransectIDs, Titles, range(axs.shape[1])):
        # Define variables for each subplot per column/Transect
        ax_TS = axs[0,col]
        ax_Season = axs[1,col]
        ax_Year = axs[2,col]
        ax_TS.sharex = ax_Season # only share x-axis of first two rows
        
        # Process plot data
        plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDF['dates'].iloc[TransectID]]
        plotsatdist = [x-np.mean(TransectInterGDF['distances'].iloc[TransectID]) for x in TransectInterGDF['distances'].iloc[TransectID]]
        plotwldate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDF['wldates'].iloc[TransectID]]
        plotwldist = [x-np.mean(TransectInterGDF['wlcorrdist'].iloc[TransectID]) for x in TransectInterGDF['wlcorrdist'].iloc[TransectID]]
    
        # plotdate, plotwldate, plotsatdist, plotwldist = [list(d) for d in zip(*sorted(zip(plotdate, plotwldate, plotsatdist, plotwldist), key=lambda x: x[0]))]    
        
        # Calculate period (observations per cycle) to use for seasonality
        # DateDiff = []
        # for i in range(1,len(plotdate)):
        #     DateDiff.append(plotdate[i]-plotdate[i-1])
        # MeanDateDiff = np.mean(DateDiff).days # days between each observation
        # if P is None:
        #     P = round(90 / MeanDateDiff) # 90 days = 3 month cycle
        
        # Set up common subplot design
        for ax in [ax_TS,ax_Season,ax_Year]: 
            # ax.grid(color=[0.7,0.7,0.7], ls=':', lw=0.5, zorder=0)
            # create rectangles highlighting winter months (based on N or S hemisphere 'winter')
            for i in range(plotdate[0].year-1, plotdate[-1].year+1):
                if Hemisphere == 'N':
                    rectWinterStart = mdates.date2num(datetime(i, 11, 1, 0, 0))
                    rectWinterEnd = mdates.date2num(datetime(i+1, 3, 1, 0, 0))
                elif Hemisphere == 'S':
                    rectWinterStart = mdates.date2num(datetime(i, 5, 1, 0, 0))
                    rectWinterEnd = mdates.date2num(datetime(i, 9, 1, 0, 0))
                rectwidth = rectWinterEnd - rectWinterStart
                rect = mpatches.Rectangle((rectWinterStart, -2000), rectwidth, 4000, fc=[0.3,0.3,0.3], ec=None, alpha=0.2)
                ax.add_patch(rect)
        
        ax_TS_veg = ax_TS.twinx()
        ax_Season_veg = ax_Season.twinx()
        ax_Year_veg = ax_Year.twinx()
               
        # plot trendlines
        # vegav = MovingAverage(plotsatdist, 3)
        # wlav = MovingAverage(plotwldist, 3)
        # if len(plotdate) >= 3:
            # ax_TS.plot(plotdate, wlav, color='#4056F4', lw=1, label='3pt Moving Average waterline')
            # ax_TS2.plot(plotdate, vegav, color='#81A739', lw=1, label='3pt Moving Average veg edge')
        twin_TS_lim = []
        twin_Trend_lim = []
        twin_Season_minlim = []    
        twin_Season_maxlim = []
        for twin_TS, twin_Season, twin_Year, twin_lab, x, y, clr in zip([ax_TS, ax_TS_veg],
                                                                        [ax_Season, ax_Season_veg],
                                                                        [ax_Year, ax_Year_veg],
                                                                        ['WL', 'VE'],
                                                                        [plotwldate, plotdate],
                                                                        [plotwldist, plotsatdist],
                                                                        ['#0A1DAE', '#81A739']):
            
            # Extend and interpolate to create daily observations
            Timeseries = pd.Series(y, index=x)
            Timeseries = Timeseries.groupby(Timeseries.index).mean() # if any duplicates, take mean
            Timeseries = Timeseries.resample('1D')
            Timeseries = Timeseries.interpolate(method='time')
            
            # Calculate seasonal .trend, .seasonal and .resid, using a year as the detrending period
            Seasonality = seasonal_decompose(Timeseries, model='additive', period=365)
            Season = Seasonality.seasonal
            Resid = Seasonality.resid
            SSI = np.var(Season) / (np.var(Season) + np.var(Resid))
            
            print('Transect '+str(TransectID)+' '+twin_lab+' annual seasonality index: '+str(SSI))
            
            # Tidal seasonality
            TideSeasonality = seasonal_decompose(Timeseries, model='additive', period=18)
            TideSeason = TideSeasonality.seasonal
            TideResid = TideSeasonality.resid
            TideSSI = np.var(TideSeason) / (np.var(TideSeason) + np.var(TideResid))
            print('Transect '+str(TransectID)+' '+twin_lab+' tidal seasonality index: '+str(TideSSI))
            
            
            # PLOT 1: timeseries scatter plot  
            twin_TS.axhline(y=0, color=clr, lw=0.6,alpha=0.3, zorder=0)
            twin_TS.scatter(x, y, marker='o', c=clr, s=2, alpha=0.5, edgecolors='none', zorder=1)
            # linear regression lines
            numx = mpl.dates.date2num(x)
            m, c = np.polyfit(numx,y,1)
            # polysat = np.poly1d([m, c])
            xx = np.linspace(numx.min(), numx.max(), 100)
            # dd = mpl.dates.num2date(xx)
            # twin_TS.plot(dd, polysat(xx), '--', color=clr, lw=1, label=r'$\Delta$'+twin_lab+' = '+str(round(m*365.25,2))+' m/yr', zorder=1)

            # get 5th smallest and largest as limits to avoid outliers
            if twin_lab == 'WL':
                twin_TS.set_ylim( np.sort(y)[-6]*-1,
                                  np.sort(y)[5]*-1 )
            else:
                twin_TS.set_ylim( np.sort(y)[15],
                                  np.sort(y)[-2] )

            # twin_TS.set_ylim(0,1000)
            # twin_TS_lim.append(twin_TS.get_ylim())

            twin_TS.plot(Seasonality.trend, '--', color=clr, lw=1.2, label=r'$\Delta$'+twin_lab+' = '+str(round(m*365.25,2))+' m/yr')
            # twin_TS_lim.append(twin_TS.get_ylim())
            twin_TS.tick_params(labelbottom=False)
            
            
            # PLOT 2: Seasonality line (moving average) with residuals as error window
            # twin_Season.plot(Seasonality.resid, color=clr, lw=0, marker='x', ms=4, alpha=0.3)
            # plot residuals as vertical lines
            twin_Season.vlines(Seasonality.resid.index, 0, np.array(Seasonality.resid),
                               color=clr,lw=0.1,alpha=0.3,
                               label=twin_lab+' $\sigma^{2}_{resid.}$ = '+str(round(np.var(Resid),2)))
            # plot seasonal signal over top of residuals 
            twin_Season.plot(Seasonality.seasonal, color=clr, lw=0.4, alpha=1)
            # horizontal line marking 0
            twin_Season.axhline(0,0,1,color=[0.7,0.7,0.7], lw=0.5, zorder=3, alpha=0.7)
            if TransectID == 1325:
                if twin_lab == 'WL':
                    twin_Season.set_ylim(-120, 120)
                else:
                    twin_Season.set_ylim(-40, 40)
            elif TransectID == 271:
                if twin_lab == 'WL':
                    twin_Season.set_ylim(-120, 120)
                else:
                    twin_Season.set_ylim(-40, 40)
            else:
                twin_Season.set_ylim(np.min(Seasonality.resid), np.min(Seasonality.resid)*-1)
            
            
            # PLOT 3: third row is inset of the second row with just one year of seasonality
            # get middle year to ensure full year of seasonality is plotted
            MidYear = Seasonality.resid.index[int(len(Seasonality.resid.index)/2)].year
            StartInd = np.where(Seasonality.resid.index == datetime(MidYear, 1, 1))[0][0]
            EndInd = np.where(Seasonality.resid.index == datetime(MidYear+1, 1, 1))[0][0]
            # twin_Year.vlines(Seasonality.resid.index[StartInd:EndInd], 0, np.array(Seasonality.resid[StartInd:EndInd]),
                             # color=clr,lw=0.3,alpha=0.05,
                             # label=twin_lab+' $\sigma^{2}_{resid.}$ = '+str(round(np.var(Resid),2))) 
            # plot seasonal signal over top of residuals 
            twin_Year.plot(Seasonality.seasonal[StartInd:EndInd], color=clr, lw=0.7, alpha=1,
                             label=twin_lab+' SSI = '+str(round(SSI,2)))
            # horizontal line marking 0
            twin_Year.axhline(0,0,1,color=[0.7,0.7,0.7], lw=0.5, zorder=3, alpha=0.7)
            if TransectID == 1325:
                if twin_lab == 'WL':
                    twin_Year.set_ylim(-120, 120)
                else:
                    twin_Year.set_ylim(-40, 40)
            elif TransectID == 271:
                if twin_lab == 'WL':
                    twin_Year.set_ylim(-120, 120)
                else:
                    twin_Year.set_ylim(-40, 40)
            else:
                twin_Year.set_ylim(np.min(Seasonality.resid), np.min(Seasonality.resid)*-1)
            
            
            twin_TS.tick_params(axis='y', colors=clr)
            twin_Season.tick_params(axis='y', colors=clr)
            twin_Year.tick_params(axis='y', colors=clr)
            
           
        ax_TS.title.set_text('Transect '+str(TransectID)+' - '+Title)
        ax_TS.set_xlim(min(plotdate)-timedelta(days=20),max(plotdate)+timedelta(days=40))
        ax_Season.set_xlim(min(plotdate)-timedelta(days=20),max(plotdate)+timedelta(days=40))
        ax_Year.set_xlim(Seasonality.resid.index[StartInd],Seasonality.resid.index[EndInd])
        # set x axis labels for year plot as month names
        ax_Year.xaxis.set_major_locator(mdates.MonthLocator())
        ax_Year.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        for axleg in [[ax_TS, ax_TS_veg], [ax_Season,ax_Season_veg], [ax_Year, ax_Year_veg]]:
            leg1 = axleg[0].legend(loc=3, handlelength=1.5, handletextpad=0.1)
            leg2 = axleg[1].legend(loc=4, handlelength=1.5, handletextpad=0.1)
            # weird zorder with twinned axes; remove first axis legend and plot on top of second
            leg1.remove()
            axleg[1].add_artist(leg1)
                    
        # add connecting lines to inset plot (3)
        fig.add_artist(ConnectionPatch(
        xyA=(ax_Year.get_xlim()[0],ax_Season.get_ylim()[0]), coordsA=ax_Season.transData,
        xyB=(0,1), coordsB=ax_Year.transAxes,
        color='k', alpha=0.3, linewidth=0.5))      
        fig.add_artist(ConnectionPatch(
        xyA=(ax_Year.get_xlim()[1],ax_Season.get_ylim()[0]), coordsA=ax_Season.transData,
        xyB=(1,1), coordsB=ax_Year.transAxes,
        color='k', alpha=0.3, linewidth=0.5))   
        
        figID += '_'+str(TransectID)
        plt.tight_layout()
        
    # Add alphabetical labels to corners of subplots
    ax_labels = list(string.ascii_lowercase[:3*axs.shape[1]])
    axticklabs = ['Cross-shore dist. (m)','Cross-shore dist. (m)',
                  'Seasonal signal (m)','Seasonal signal (m)',
                  'Seasonal signal (m)','Seasonal signal (m)']
    for ax, axID, lab, axticklab in zip(axs.flat, range(len(axs.flat)), ax_labels, axticklabs):
        ax.text(0.011, 0.98, '('+lab+')', transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(facecolor='w', edgecolor='k',pad=1.5))
        if axID % 2 == 0: # if ID is even i.e. on left hand side
            ax.set_ylabel(axticklab, color='#0A1DAE')
            # ax.set_ylabel(None)
        else:
            ax.set_ylabel(axticklab, color='#81A739')
            ax.yaxis.set_label_position('right')
            ax.yaxis.labelpad=25
    
    
    if Normal:  
        figname = os.path.join(outfilepath,sitename + '_SatVegWaterSeasonalSimpleNormal_Transect'+figID+'.png')
    else:
        figname = os.path.join(outfilepath,sitename + '_SatVegWaterSeasonalSimple_Transect'+figID+'.png')

    plt.subplots_adjust(hspace=0.)
    plt.tight_layout()
            
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    print('Plot saved under '+figname)
    
    plt.show()
        
    
    
def VegTZTimeseries(sitename, TransectInterGDFTopo, TransectIDs, Hemisphere='N', ShowPlot=True):
    """
    Plot timeseries of cross-shore veg edge and waterline change for selected transect(s),
    with TZ widths plotted over top.
    If more than one transect is supplied in a list, create subplots for comparison.
    FM Nov 2022

    Parameters
    ----------
    sitename : str
        Name of site.
    TransectInterGDFTopo : GeoDataFrame
        GeoDataFrame of transects intersected with topographic data.
    TransectIDs : list
        List of transect IDs to plot.
    Hemisphere : str, optional
        Northern (N) or Southern (S) Hemisphere for marking 'winter' season. The default is 'N'.
    ShowPlot : bool, optional
        Flag to turn plt.show() on or off (if plotting lots of transects). The default is True.


    """
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    figID = ''
    
    if ShowPlot is False:
        plt.ioff()
    
    # if more than one Transect ID is to be compared on a single plot
    if type(TransectIDs) == list:
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':12})
        fig, axs = plt.subplots(len(TransectIDs),1,figsize=(11.6,5.9), dpi=300, sharex=True)
    else:
        TransectIDs = [TransectIDs]
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':12})
        # use 2 subplots with one empty to be able to loop through them
        fig, axs = plt.subplots(1,1,figsize=(11.6,5.9), dpi=300, sharex=True)
        axs = [axs] # to be able to loop through
        
    # common plot labels
    lab = fig.add_subplot(111,frameon=False)
    lab.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False, right=False)
    if type(TransectIDs) == list: 
        lab.set_xlabel('Date', labelpad=22)
    else:
        lab.set_xlabel('Date')
    lab.set_ylabel('Cross-shore distance (veg) (m)', color='#81A739')
    
    for TransectID, ax in zip(TransectIDs,axs):
        daterange = [0,len(TransectInterGDFTopo['dates'].iloc[TransectID])]
        plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDFTopo['dates'].iloc[TransectID][daterange[0]:daterange[1]]]
        plotsatdist = TransectInterGDFTopo['distances'].iloc[TransectID][daterange[0]:daterange[1]]
        # remove and interpolate outliers
        plotsatdistinterp = InterpNaN(plotsatdist)
        plotTZ = TransectInterGDFTopo['TZwidth'].iloc[TransectID][daterange[0]:daterange[1]]
        plotTZMn = TransectInterGDFTopo['TZwidthMn'].iloc[TransectID]
                
        if len(plotdate) == 0:
            print('Transect %s is empty! No values to plot.' % (TransectID))
            return
        
        plotdate, plotsatdist = [list(d) for d in zip(*sorted(zip(plotdate, plotsatdist), key=lambda x: x[0]))]    
        ax.grid(color=[0.7,0.7,0.7], ls=':', lw=0.5, zorder=0)        
                
        satPl = ax.scatter(plotdate, plotsatdist, marker='o', c='#81A739', s=8, alpha=0.8, edgecolors='none', label='Sat. VegEdge')
        
        # xaxis ticks as year with interim Julys marked
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        
        # create TZ lines to fill between (per-image width as error bar, mean as fill between)
        yplus = []
        yneg = []
        for i in range(len(plotsatdist)):
            yplus.append(plotsatdist[i] + plotTZMn)
            yneg.append(plotsatdist[i] - plotTZMn)
            ax.errorbar(plotdate[i], plotsatdist[i], yerr=plotTZ[i], ecolor='#E7960D', elinewidth=0.7, capsize=1, capthick=0.5, label='TZwidth (m)')
        # single error bar plot for legend
        TZbar = mlines.Line2D([],[],linestyle='None', marker='|', ms=10, mec='#E7960D', mew=0.7, label='TZwidth (m)')
        TZfill = ax.fill_between(plotdate, yneg, yplus, color='#E7960D', alpha=0.3, edgecolor=None, zorder=0, label=r'$TZwidth_{\mu}$ ('+str(round(plotTZMn))+' m)')
                   
        # create rectangles highlighting winter months (based on N or S hemisphere 'winter')
        for i in range(plotdate[0].year-1, plotdate[-1].year):
            if Hemisphere == 'N':
                rectWinterStart = mdates.date2num(datetime(i, 11, 1, 0, 0))
                rectWinterEnd = mdates.date2num(datetime(i+1, 3, 1, 0, 0))
            elif Hemisphere == 'S':
                rectWinterStart = mdates.date2num(datetime(i, 5, 1, 0, 0))
                rectWinterEnd = mdates.date2num(datetime(i, 9, 1, 0, 0))
            rectwidth = rectWinterEnd - rectWinterStart
            rect = mpatches.Rectangle((rectWinterStart, -2000), rectwidth, 4000, fc=[0.3,0.3,0.3], ec=None, alpha=0.2)
            # ax.add_patch(rect)
          
        # plot trendlines
        vegav = MovingAverage(plotsatdistinterp, 3)
        if len(plotdate) >= 3:
            movavPl, = ax.plot(plotdate, vegav, color='#81A739', lw=2, label='3pt Mov. Av. VegEdge')
    
        # linear regression lines
        x = mpl.dates.date2num(plotdate)
        for y, pltax, clr in zip([plotsatdist], [ax], ['#3A4C1A']):
            m, c = np.polyfit(x,y,1)
            polysat = np.poly1d([m, c])
            xx = np.linspace(x.min(), x.max(), 100)
            dd = mpl.dates.num2date(xx)
            ltPl, = pltax.plot(dd, polysat(xx), '--', color=clr, lw=2, label=str(round(m*365.25,2))+' m/yr')
    
        ax.title.set_text('Transect '+str(TransectID))
            
        # ax.set_xlabel('Date (yyyy-mm)')
        # ax2.set_ylabel('Cross-shore distance (veg) (m)', color='#81A739')
        # ax.set_ylabel('Cross-shore distance (water) (m)', color='#4056F4')
        # plt.xlim(plotdate[0]-10, plotdate[-1]+10)
        ax.set_ylim(np.nanmin(plotsatdistinterp)-10, np.nanmax(plotsatdistinterp)+30)
        ax.set_xlim(np.nanmin(plotdate)-timedelta(days=100),np.nanmax(plotdate)+timedelta(days=100))
        
        leg1 = ax.legend(handles=[satPl, movavPl, ltPl], loc=2)
        ax.add_artist(leg1)
        leg2 = ax.legend(handles=[TZfill, TZbar], loc=1)
        # weird zorder with twinned axes; remove first axis legend and plot on top of second
        # leg1.remove()
        
        figID += '_'+str(TransectID)
        plt.tight_layout()
        
    figname = os.path.join(outfilepath,sitename + '_SatVegTZTimeseries_Transect'+figID+'.png')
    
    plt.tight_layout()
            
    plt.savefig(figname, bbox_inches='tight')
    print('Plot saved under '+figname)
    
    plt.show()
    
    
def TZTimeseries(sitename, TransectInterGDFTopo, TransectIDs, Titles=None, Hemisphere='N', ShowPlot=True):
    """
    Plot timeseries of cross-shore veg edge and waterline change for selected transect(s),
    with TZ widths plotted over top.
    If more than one transect is supplied in a list, create subplots for comparison.
    FM Nov 2022

    Parameters
    ----------
    sitename : str
        Name of site.
    TransectInterGDFTopo : GeoDataFrame
        GeoDataFrame of transects intersected with topographic data.
    TransectIDs : list
        List of transect IDs to plot.
    Hemisphere : str, optional
        Northern (N) or Southern (S) Hemisphere for marking 'winter' season. The default is 'N'.
    ShowPlot : bool, optional
        Flag to turn plt.show() on or off (if plotting lots of transects). The default is True.


    """
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    figID = ''
    
    if ShowPlot is False:
        plt.ioff()
    
    # if more than one Transect ID is to be compared on a single plot
    if type(TransectIDs) == list:
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':7})
        fig, axs = plt.subplots(len(TransectIDs),1,figsize=(2.02,2.91), dpi=300, sharex=True)
    else:
        TransectIDs = [TransectIDs]
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':7})
        # use 2 subplots with one empty to be able to loop through them
        fig, axs = plt.subplots(1,1,figsize=(2.02,1.45), dpi=300, sharex=True)
        axs = [axs] # to be able to loop through
        
    # common plot labels
    # lab = fig.add_subplot(111,frameon=False)
    # lab.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False, right=False)
    fig.text(0.5,-0.16,'Date',ha='center',va='center')
    fig.text(-0.23,0.5,'Cross-shore distance (m)', ha='center',va='center',rotation='vertical')
    
    for TransectID, Title, ax in zip(TransectIDs, Titles, axs):
        daterange = [0,len(TransectInterGDFTopo['dates'].iloc[TransectID])]
        plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDFTopo['dates'].iloc[TransectID][daterange[0]:daterange[1]]]
        plotsatdist = [x-np.mean(TransectInterGDFTopo['distances'].iloc[TransectID]) for x in TransectInterGDFTopo['distances'].iloc[TransectID][daterange[0]:daterange[1]]]
        # remove and interpolate outliers
        plotsatdistinterp = InterpNaN(plotsatdist)
        plotTZ = TransectInterGDFTopo['TZwidth'].iloc[TransectID][daterange[0]:daterange[1]]
        plotTZMn = TransectInterGDFTopo['TZwidthMn'].iloc[TransectID]
                
        if len(plotdate) == 0:
            print('Transect %s is empty! No values to plot.' % (TransectID))
            return
        
        plotdate, plotsatdist = [list(d) for d in zip(*sorted(zip(plotdate, plotsatdist), key=lambda x: x[0]))]    
        ax.grid(color=[0.7,0.7,0.7], ls=':', lw=0.2, zorder=0)   
        ax.axhline(y=0, color=[0.7,0.7,0.7], lw=0.5, zorder=0)
        ax.tick_params(length=2)
                        
        # xaxis ticks as year with interim Julys marked
        # ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
        # ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        
        # create TZ lines to fill between (per-image width as error bar, mean as fill between)
        yplus = []
        yneg = []
        for i in range(len(plotsatdist)):
            yplus.append(plotsatdist[i] + plotTZMn)
            yneg.append(plotsatdist[i] - plotTZMn)
            ax.errorbar(plotdate[i], plotsatdist[i], yerr=plotTZ[i], ecolor='#E7960D', elinewidth=0.5, capsize=0, capthick=0.5, label='TZwidth (m)')
        # single error bar plot for legend
        TZbar = mlines.Line2D([],[],linestyle='None', marker='|', ms=10, mec='#E7960D', mew=0.7, label='TZwidth (m)')
        TZfill = ax.fill_between(plotdate, yneg, yplus, color='#E7960D', alpha=0.5, edgecolor=None, zorder=0, label=r'$TZwidth_{\mu}$ ('+str(round(plotTZMn))+' m)')
                   
        # create rectangles highlighting winter months (based on N or S hemisphere 'winter')
        for i in range(plotdate[0].year-1, plotdate[-1].year+1):
            if Hemisphere == 'N':
                rectWinterStart = mdates.date2num(datetime(i, 11, 1, 0, 0))
                rectWinterEnd = mdates.date2num(datetime(i+1, 3, 1, 0, 0))
            elif Hemisphere == 'S':
                rectWinterStart = mdates.date2num(datetime(i, 5, 1, 0, 0))
                rectWinterEnd = mdates.date2num(datetime(i, 9, 1, 0, 0))
            rectwidth = rectWinterEnd - rectWinterStart
            rect = mpatches.Rectangle((rectWinterStart, -2000), rectwidth, 4000, fc=[0.3,0.3,0.3], ec=None, alpha=0.2)
            ax.add_patch(rect)
          
        # plot trendlines
        vegav = MovingAverage(plotsatdistinterp, 3)
        if len(plotdate) >= 3:
            movavPl, = ax.plot(plotdate, vegav, color='#81A739', lw=1, label='VE 3pt Mov. Av.')
    
        # linear regression lines
        x = mpl.dates.date2num(plotdate)
        for y, pltax, clr in zip([plotsatdist], [ax], ['#3A4C1A']):
            m, c = np.polyfit(x,y,1)
            polysat = np.poly1d([m, c])
            xx = np.linspace(x.min(), x.max(), 100)
            dd = mpl.dates.num2date(xx)
            # ltPl, = pltax.plot(dd, polysat(xx), '--', color=clr, lw=2, label=str(round(m*365.25,2))+' m/yr')
    
        ax.set_title('Transect '+str(TransectID)+' - '+Title, pad=1)
            
        # ax.set_ylim(np.nanmin(plotsatdistinterp)-10, np.nanmax(plotsatdistinterp)+30)
        ax.set_ylim(-200,180)
        ax.set_xlim(np.nanmin(plotdate)-timedelta(days=50),np.nanmax(plotdate)+timedelta(days=50))
        # ax.set_xlim(np.nanmin(plotdate),np.nanmax(plotdate))
        for tic in ax.xaxis.get_ticklabels()[::2]:
            tic.set_visible(False)
        
        # leg1 = ax.legend(handles=[satPl, movavPl, ltPl], loc=2)
        # ax.add_artist(leg1)
        leg2 = ax.legend(handles=[TZfill, TZbar, movavPl], loc=4)
        
        figID += '_'+str(TransectID)
        # plt.tight_layout()
        # ax.margins(-0.4)
        
    figname = os.path.join(outfilepath,sitename + '_TZTimeseries_Transect'+figID+'.png')
    
    plt.tight_layout()
    plt.subplots_adjust(left=-0.05, bottom=-0.08, right=1, top=1.08, wspace=0, hspace=0.1)
        
    plt.savefig(figname, bbox_inches='tight', pad_inches=0.05, dpi=300)
    print('Plot saved under '+figname)
    
    plt.show()
    
        


def ValidTimeseries(sitename, ValidInterGDF, TransectID):
    """
    

    Parameters
    ----------
    sitename : str
        Name of site.
    ValidDF : GeoDataFrame
        DataFrame of transects intersected with validation lines.
    TransectID : list
        Transect ID(s) to plot.

    """
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    
    plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in ValidInterGDF['Vdates'].iloc[TransectID]]
    plotsatdist = ValidInterGDF['distances'].iloc[TransectID]
    plotvaliddist = ValidInterGDF['Vdists'].iloc[TransectID]
    
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
    satlabels = ['Sat. VegEdge','_nolegend_','_nolegend_','_nolegend_',]
    
    for i,c in enumerate([0.95,0.7,0.6,0.2]):
        ax.plot(plotdate[i], plotvaliddist[i], 'X', color=magma(c), markersize=10,markeredgecolor='k', label=validlabels[i])
        ax.plot(plotdate[i], plotsatdist[i], 'o', color=magma(c),markersize=10,markeredgecolor='k', label=satlabels[i])
    
    
    ax.plot(dd, polyvalid(xx), '--', color=[0.7,0.7,0.7], zorder=0, label=str(round(mvalid*365.25,2))+'m/yr')
    ax.plot(dd, polysat(xx), '-', color=[0.7,0.7,0.7], zorder=0, label=str(round(msat*365.25,2))+'m/yr')
    
    plt.legend()
    plt.xlabel('Date (yyyy-mm)')
    plt.ylabel('Cross-shore distance (m)')
    plt.tight_layout()
    
    plt.savefig(os.path.join(outfilepath,sitename + '_ValidVsSatTimeseries_Transect'+str(TransectID)+'.png'))
    print('Plot saved under '+os.path.join(outfilepath,sitename + '_ValidVsSatTimeseries_Transect'+str(TransectID)+'.png'))
    
    plt.show()


def WidthTimeseries(sitename, TransectInterGDFWater, TransectIDs, Hemisphere = 'N'):
    """
    Plot timeseries of beach width between veg edge and waterline for selected transect(s).
    If more than one transect is supplied in a list, create subplots for comparison.
    FM Jul 2023

    Parameters
    ----------
    ValidDF : GeoDataFrame
        DataFrame of transects intersected with validation lines.
    TransectID : list
        Transect ID(s) to plot.


    """
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    figID = ''
        
    # if more than one Transect ID is to be compared on a single plot
    if type(TransectIDs) == list:
        # scaling for single column A4 page
        mpl.rcParams.update({'font.size':7})
        fig, axs = plt.subplots(len(TransectIDs),1,figsize=(6.55,6), dpi=300)
    else:
        TransectIDs = [TransectIDs]
        # scaling for single column A4 page
        mpl.rcParams.update({'font.size':7})
        # use 2 subplots with one empty to be able to loop through them
        fig, axs = plt.subplots(1,1,figsize=(6.55,3), dpi=300)
        axs = [axs] # to be able to loop through
        
    # common plot labels
    lab = fig.add_subplot(111,frameon=False)
    lab.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False, right=False)
    if type(TransectIDs) == list: 
        lab.set_xlabel('Date (yyyy-mm)', labelpad=22)
    else:
        lab.set_xlabel('Date (yyyy-mm)')
    lab.set_ylabel('Cross-shore beach width (m)')
    
    for TransectID, ax in zip(TransectIDs,axs):
        plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDFWater['wldates'].iloc[TransectID]]
    
        plotvegdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDFWater['dates'].iloc[TransectID]]
        plotwldate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDFWater['wldates'].iloc[TransectID]]
    
        plotvegdist = TransectInterGDFWater['distances'].iloc[TransectID]
        plotwldist = TransectInterGDFWater['wlcorrdist'].iloc[TransectID]
        plotbwdist = TransectInterGDFWater['beachwidth'].iloc[TransectID]
    
        plotvegdate, plotvegdist = [list(d) for d in zip(*sorted(zip(plotvegdate, plotvegdist), key=lambda x: x[0]))]
        plotwldate, plotwldist = [list(d) for d in zip(*sorted(zip(plotwldate, plotwldist), key=lambda x: x[0]))]
        plotdate, plotbwdist = [list(d) for d in zip(*sorted(zip(plotdate, plotbwdist), key=lambda x: x[0]))]
        
        ax.grid(color=[0.7,0.7,0.7], ls=':', lw=0.5, zorder=0)        

        # create rectangles highlighting winter months (based on N or S hemisphere 'winter')
        for i in range(plotdate[0].year-1, plotdate[-1].year):
            if Hemisphere == 'N':
                rectWinterStart = mdates.date2num(datetime(i, 11, 1, 0, 0))
                rectWinterEnd = mdates.date2num(datetime(i+1, 3, 1, 0, 0))
            elif Hemisphere == 'S':
                rectWinterStart = mdates.date2num(datetime(i, 5, 1, 0, 0))
                rectWinterEnd = mdates.date2num(datetime(i, 9, 1, 0, 0))
            rectwidth = rectWinterEnd - rectWinterStart
            rect = mpatches.Rectangle((rectWinterStart, -2000), rectwidth, 4000, fc=[0.3,0.3,0.3], ec=None, alpha=0.2)
            ax.add_patch(rect)
            
        # linear regression line
        x = mpl.dates.date2num(plotdate)
        msat, csat = np.polyfit(x,plotbwdist,1)
        polysat = np.poly1d([msat, csat])
        xx = np.linspace(x.min(), x.max(), 100)
        dd = mpl.dates.num2date(xx)
        
        ax.plot(plotdate, plotbwdist, linewidth=0, marker='.', c='k', markersize=5, markeredgecolor='k', label='Upper Beach Width')
        # plt.plot(plotvegdate, plotvegdist, linewidth=0, marker='.', c='g', markersize=8, label='Upper Beach Width')
        # plt.plot(plotwldate, plotwldist, linewidth=0, marker='.', c='b', markersize=8,  label='Upper Beach Width')
    
        # plot trendlines
        yav = MovingAverage(plotbwdist, 3)
        if len(plotdate) >= 3:
            ax.plot(plotdate, yav, 'r', label='3pt Moving Average')
            ax.plot(dd, polysat(xx), '--', color=[0.7,0.7,0.7], zorder=0, label=str(round(msat*365.25,2))+'m/yr')

    
        ax.title.set_text('Transect '+str(TransectID))
            
        # ax.set_xlabel('Date (yyyy-mm)')
        # ax2.set_ylabel('Cross-shore distance (veg) (m)', color='#81A739')
        # ax.set_ylabel('Cross-shore distance (water) (m)', color='#4056F4')
        # plt.xlim(plotdate[0]-10, plotdate[-1]+10)
        ax.set_ylim(min(plotbwdist)-10, max(plotbwdist)+30)
        ax.set_xlim(min(plotdate)-timedelta(days=100),max(plotdate)+timedelta(days=100))
        
        leg1 = ax.legend(loc=2)
        # weird zorder with twinned axes; remove first axis legend and plot on top of second
        # leg1.remove()
        
        figID += '_'+str(TransectID)
        plt.tight_layout()
    
    figname = os.path.join(outfilepath,sitename + '_SatBeachWidthTimeseries_Transect'+figID+'.png')
    plt.savefig(figname, bbox_inches='tight')
    print('Plot saved under '+figname)

    plt.show()

  
    

# def ResultsPlot(outfilepath, outfilename, sitename):
    
    
#     def formatAxes(fig):
#         for i, ax in enumerate(fig.axes):
#             ax.tick_params(labelbottom=False, labelleft=False)
    
#     fig = plt.figure(layout='constrained', figsize=(6.55,5))
    
#     gs = GridSpec(3,3, figure=fig)
#     ax1 = fig.add_subplot(gs[0,:])
#     ax2 = fig.add_subplot(gs[1,:-1])
#     ax3 = fig.add_subplot(gs[1:,-1])
#     ax4 = fig.add_subplot(gs[-1,0])
#     ax5 = fig.add_subplot(gs[-1,-2])
    
#     formatAxes(fig)
    
#     # # font size 8 and width of 6.55in fit 2-column journal formatting
#     # plt.rcParams['font.size'] = 8
#     # fig, ax = plt.subplots(3,2, figsize=(6.55,5), dpi=300, gridspec_kw={'height_ratios':[3,2,2]})
    
#     # # outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
#     # if os.path.isdir(outfilepath) is False:
#     #     os.mkdir(outfilepath)
    
#     plt.tight_layout()
#     #plt.savefig(os.path.join(outfilepath,outfilename), dpi=300)
#     print('Plot saved under '+os.path.join(outfilepath,outfilename))
    
#     plt.show()
    
    
def ValidPDF(sitename, VeglineGDF, DatesCol, ValidDF, TransectIDs, PlotTitle):    
    """
    Generate probability density function of validation vs sat lines.
    FM Jul 2023

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    ValidGDF : GeoDataFrame
        DataFrame of sat derived veg edges read in from shapefile.
    DatesCol : str
        Name of field containing dates in validation shapefile.
    ValidDF : GeoDataFrame
        DataFrame of transects intersected with validation lines.
    TransectIDs : list
        Index of requested transect IDs for plotting (min and max bounds).
    PlotTitle : str
        Plot title for placename locations.


    """
    # font size 8 and width of 6.55in fit 2-column journal formatting
    plt.rcParams['font.size'] = 8  
    
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)

    violin = []
    violindates = []
    Vdates = VeglineGDF[DatesCol].unique()
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
    
    
    # Above is violin stuff, below is KD
    x = np.array()
    x_d = np.linspace(0,1,1000)
    
    kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
    kde.fit(x[:,None])
    logprob = kde.score_samples(x_d[:,None])    
    
    
    fig, ax = plt.subplots(1,1, figsize=(2.48,4.51))
    if len(violindates) > 1:
        ax.plot(x_d, np.exp(logprob), linewidth=1)
    else:
        ax.plot(data = df, linewidth=1)
        
    ax.set(xlabel='Cross-shore distance of satellite-derived line from validation line (m)', ylabel='Validation line date')
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
    
    #plt.savefig(os.path.join(outfilepath,outfilename), dpi=300)
    figpath = os.path.join(filepath,sitename+'_Validation_Satellite_Distances_Violin_'+str(TransectIDs[0])+'to'+str(TransectIDs[1])+'.png')
    plt.savefig(figpath)
    print('figure saved under '+figpath)
    
    plt.show()
    
    
def SatRegress(sitename,SatGDF,DatesCol,ValidDF,TransectIDs,PlotTitle,Platform=None):
       
    
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)
        
    valdists = []
    satdists = []
    satplotdates = []
    # get unique sat dates
    if Platform is not None:
        SatGDF = SatGDF[SatGDF['satname']==Platform]
    Sdates = SatGDF[DatesCol].unique()
    # get unique validation dates
    Vdates = []
    for Tr in range(TransectIDs[0], TransectIDs[1]):
        for i in ValidDF['Vdates'].iloc[Tr]:
            if i != []:
                try:
                    Vdates.append(ValidDF['Vdates'].iloc[Tr][i]) 
                except:
                    Vdates.append(ValidDF['Vdates'].iloc[Tr][0])
    Vdates = set(Vdates)
    
    for Sdate in Sdates:
        satdist = []
        valdist = []
        # for each transect in given range
        for Tr in range(TransectIDs[0],TransectIDs[1]): 
            if Tr > len(ValidDF['dates']): # for when transect values extend beyond what transects exist
                print("check your chosen transect values!")
                return
            if Sdate in ValidDF['dates'].iloc[Tr]:
                DateIndex = (ValidDF['dates'].iloc[Tr].index(Sdate))
                # rare occasion where transect intersects valid line but NOT sat line (i.e. no distance between them)
                if ValidDF['valsatdist'].iloc[Tr] != []:
                    satdist.append(ValidDF['distances'].iloc[Tr][DateIndex])
                    # extract validation dists by performing difference calc back on sat dists
                    try:
                        valdist.append(ValidDF['distances'].iloc[Tr][DateIndex]-ValidDF['valsatdist'].iloc[Tr][DateIndex])
                    except:
                        pdb.set_trace()
                else:
                    continue
            else:
                continue
        # due to way dates are used, some transects might be missing validation dates so violin collection will be empty
        if satdist != []: 
            satdists.append(satdist)
            satplotdates.append(Sdate)
            valdists.append(valdist)
    # sort both dates and list of values by date
    if len(satplotdates) > 1:
        satplotdatesrt, satsrt, valsrt = [list(d) for d in zip(*sorted(zip(satplotdates, satdists, valdists), key=lambda x: x[0]))]
    else:
        satplotdatesrt = satplotdates
        satsrt = satdists
        valsrt = valdists

    if Platform is not None:
        f = plt.figure(figsize=(3.31, 3.76), dpi=300)
    else:
        f = plt.figure(figsize=(3.31, 4.41), dpi=300)
    mpl.rcParams.update({'font.size':7})
    ax = f.add_subplot(1,1,1)
    ax.set_facecolor('#E0E0E0')
    
    # line through the origin as a guide for error
    plt.plot([-100,1000],[-100,1000],c=[0.6,0.5,0.5], lw=0.8, linestyle='-', zorder=3, alpha=0.35)
    
    valsrtclean = []
    satsrtclean = []
    satdateclean = []
    # for each list of transects for a particular date
    for dat, vallist, satlist in zip(range(len(valsrt)), valsrt, satsrt):
        vallistclean = []
        satlistclean = []
        # for each transect obs
        for i in range(len(vallist)):
            if np.isnan(vallist[i]) == False: # if transect obs is not empty
                vallistclean.append(vallist[i])
                satlistclean.append(satlist[i])
        if vallistclean != []: # skip completely empty dates
            satdateclean.append(satplotdatesrt[dat])
            valsrtclean.append(vallistclean)
            satsrtclean.append(satlistclean)

    if Platform is not None:
        maxlim = max( max(max(satsrtclean)), max(max(valsrtclean)) )
    else:
        maxlim = max( max(max(satsrt)), max(max(valsrt)) )
    majort = np.arange(-100,maxlim+200,100)
    minort = np.arange(-100,maxlim+200,20)
    ax.set_xticks(majort)
    ax.set_yticks(majort)
    ax.set_xticks(minort, minor=True)
    ax.set_yticks(minort, minor=True)
    ax.grid(which='major', color='#BBB4BB', alpha=0.5, zorder=0)
    # ax.grid(which='minor', color='#BBB4BB', alpha=0.2, zorder=0)
    
    cmap = cm.get_cmap('magma_r',len(valsrtclean))
    for i in range(len(valsrtclean)): 
        # plot scatter of validation (observed) vs satellite (predicted) distances along each transect
        plt.scatter(valsrtclean[i], satsrtclean[i], color=cmap(i), s=2, alpha=0.2, edgecolors='none', zorder=2)
        # linear regression
        X = np.array(valsrtclean[i]).reshape((-1,1))
        y = np.array(satsrtclean[i])
        model = LinearRegression(fit_intercept=True).fit(X,y)
        R2 = model.score(X,y)
        r, pval = scipy.stats.pearsonr(X[:,0],y)
        
        valfit = np.linspace(0,round(np.max(valsrtclean[i])),len(valsrtclean[i])).reshape((-1,1))
        satfit = model.predict(valfit)

        plt.plot(valfit,satfit, c=cmap(i), alpha=0.8, linewidth=1.2, 
                 label=f"{satdateclean[i]}, $R^2$ = {R2:.2f}", zorder=3)
                 #label=f"{} $\\approx$", zorder=3)

    plt.legend(facecolor='#E0E0E0', framealpha=1, bbox_to_anchor=(0.5,0), loc='lower center', bbox_transform=f.transFigure, ncol=2)
    
    # overall linear regression
    valfull = [item for sublist in valsrtclean for item in sublist]
    satfull =[item for sublist in satsrtclean for item in sublist]
    X = np.array(valfull).reshape((-1,1))
    y = np.array(satfull)
    model = LinearRegression(fit_intercept=True).fit(X,y)
    R2 = model.score(X,y)
    r, pval = scipy.stats.pearsonr(X[:,0],y)
    
    valfit = np.linspace(0,round(np.max(valfull)),len(valfull)).reshape((-1,1))
    satfit = model.predict(valfit)

    # plot glowing background line for overall lin reg first
    plt.plot(valfit,satfit, c='w', linestyle='-', linewidth=1.6, alpha=0.7, zorder=3)
    plt.plot(valfit,satfit, c='#818C93', linestyle='--', linewidth=1.2, zorder=3)
    # textx = 7
    # texty = -5
    textx = 0
    texty = -20
    # place = 'top'
    place = 'bottom'
    # rotn = 41
    rotn = np.degrees(np.arctan2(satfit[-1]-satfit[0], valfit[-1]-valfit[0]))[0]
    
    if pval < 0.0001:
        R2text = plt.text(valfit[-1]+textx,satfit[-1]+texty,f"R$^2$ = {R2:.2f}\n$p$ < 0.0001", 
                 c='#818C93', zorder=3, ha='right', va=place, rotation=rotn)
    else:
        R2text = plt.text(valfit[-1]+textx,satfit[-1]+texty,f"R$^2$ = {R2:.2f}\n$p$ = {pval:.3f}", 
                 c='#818C93', zorder=3, ha='right', va=place, rotation=rotn)
    R2text.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w', alpha=0.8)])

    print(maxlim)
    plt.xlim(0,(np.ceil(maxlim / 10.0) * 10)+(maxlim/14))
    plt.ylim(0,(np.ceil(maxlim / 10.0) * 10)+(maxlim/14))
    
    plt.xlabel('Validation veg edge cross-shore distance (m)')
    plt.ylabel('Satellite veg edge cross-shore distance (m)')
    
    ax.set_aspect('equal')
    ax.set_anchor('N')
    plt.tight_layout()
    
    if Platform is not None:
        figpath = os.path.join(filepath,sitename+'_Validation_Satellite_Distances_LinReg_'+str(TransectIDs[0])+'to'+str(TransectIDs[1])+'_'+Platform+'.png')
    else:
        figpath = os.path.join(filepath,sitename+'_Validation_Satellite_Distances_LinReg_'+str(TransectIDs[0])+'to'+str(TransectIDs[1])+'_corr.png')
    plt.savefig(figpath)
    print('figure saved under '+figpath)

    plt.show()
    
    
    # # Print validation sample sizes for each date
    # print('date,valid N,sat N')
    # totalN = []
    # for i in range(len(valsrtclean)):
    #     totalN.append(len(valsrtclean[i]))
    #     print(satdateclean[i]+','+str(len(valsrtclean[i]))+','+str(len(satsrtclean[i])))
    # print('sum N: '+str(np.sum(totalN)))
    print(f"R-squared: {R2}, Pearson r = {r}, = p-value = {pval}")
    
    # # Print out unique dates and satnames    
    # SatGDFNames = SatGDF.groupby(['dates']).first()
    # SatNames = []
    
    # for d in satdateclean:
    #     SatNames.append(SatGDFNames.loc[d]['satname'])
    # SatNameList = sorted(set(SatNames))
    
    # for SatN in SatNameList:
    #     SatInd = []
    #     for i, e in enumerate(SatNames):
    #         if e == SatN:
    #             SatInd.append(i)
        
    #     valsrtN = []
    #     satsrtN = []
    #     for SatI in SatInd:
    #         valsrtN.append(valsrtclean[SatI])
    #         satsrtN.append(satsrtclean[SatI])

    #     valN = [item for sublist in valsrtN for item in sublist]
    #     satN =[item for sublist in satsrtN for item in sublist]
    #     X = np.array(valN).reshape((-1,1))
    #     y = np.array(satN)
    #     model = LinearRegression(fit_intercept=True).fit(X,y)
    #     r2 = model.score(X,y)
        
    #     print('Sat name: R^2')
    #     print(SatN, r2)
    
    
    
    
def SatRegressPoster(sitename,SatGDF,DatesCol,ValidDF,TransectIDs,PlotTitle):
       
    
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)
        
    valdists = []
    satdists = []
    satplotdates = []
    # get unique sat dates
    Sdates = SatGDF[DatesCol].unique()
    # get unique validation dates
    Vdates = []
    for Tr in range(TransectIDs[0], TransectIDs[1]):
        for i in ValidDF['Vdates'].iloc[Tr]:
            if i != []:
                try:
                    Vdates.append(ValidDF['Vdates'].iloc[Tr][i]) 
                except:
                    Vdates.append(ValidDF['Vdates'].iloc[Tr][0])
    Vdates = set(Vdates)
    
    for Sdate in Sdates:
        satdist = []
        valdist = []
        # for each transect in given range
        for Tr in range(TransectIDs[0],TransectIDs[1]): 
            if Tr > len(ValidDF['dates']): # for when transect values extend beyond what transects exist
                print("check your chosen transect values!")
                return
            if Sdate in ValidDF['dates'].iloc[Tr]:
                DateIndex = (ValidDF['dates'].iloc[Tr].index(Sdate))
                # rare occasion where transect intersects valid line but NOT sat line (i.e. no distance between them)
                if ValidDF['valsatdist'].iloc[Tr] != []:
                    satdist.append(ValidDF['distances'].iloc[Tr][DateIndex])
                    # extract validation dists by performing difference calc back on sat dists
                    try:
                        valdist.append(ValidDF['distances'].iloc[Tr][DateIndex]-ValidDF['valsatdist'].iloc[Tr][DateIndex])
                    except:
                        pdb.set_trace()
                else:
                    continue
            else:
                continue
        # due to way dates are used, some transects might be missing validation dates so violin collection will be empty
        if satdist != []: 
            satdists.append(satdist)
            satplotdates.append(Sdate)
            valdists.append(valdist)
    # sort both dates and list of values by date
    if len(satplotdates) > 1:
        satplotdatesrt, satsrt, valsrt = [list(d) for d in zip(*sorted(zip(satplotdates, satdists, valdists), key=lambda x: x[0]))]
    else:
        satplotdatesrt = satplotdates
        satsrt = satdists
        valsrt = valdists

    textcolor = '#0B2D32'
    mpl.rcParams.update({'font.size':8, 
                         'text.color':textcolor,
                         'axes.labelcolor':textcolor,
                         'xtick.color':textcolor,
                         'ytick.color':textcolor,
                         'font.sans-serif':'Avenir LT Std'})

    # fig, ax = plt.subplots(figsize=(7.5,9),dpi=300)
    fig, ax = plt.subplots(figsize=(6.1,2.5), dpi=300)

    ax.set_facecolor('#D5D5D5')
    
    valsrtclean = []
    satsrtclean = []
    satdateclean = []
    # for each list of transects for a particular date
    for dat, vallist, satlist in zip(range(len(valsrt)), valsrt, satsrt):
        vallistclean = []
        satlistclean = []
        # for each transect obs
        for i in range(len(vallist)):
            if np.isnan(vallist[i]) == False: # if transect obs is not empty
                vallistclean.append(vallist[i])
                satlistclean.append(satlist[i])
        if vallistclean != []: # skip completely empty dates
            satdateclean.append(satplotdatesrt[dat])
            valsrtclean.append(vallistclean)
            satsrtclean.append(satlistclean)

    # maxlim = max( max(max(satsrt)), max(max(valsrt)) )
    # minlim = min( min(min(satsrt)), min(min(valsrt)) )
    majort = np.arange(0,300,50)
    # minort = np.arange(0,270,20)
    ax.set_xticks(majort)
    ax.set_yticks(majort)
    
    # line through the origin as a guide for error
    plt.plot([0,250],[0,250],c=[0.6,0.5,0.5], lw=1, linestyle='-', zorder=3, alpha=0.4)

    ax.grid(which='major', color='#BBB4BB', alpha=0.5, zorder=0)
    # ax.grid(which='minor', color='#BBB4BB', alpha=0.2, zorder=0)
    
    r2s = []
    lines = []
    cmap = cm.get_cmap('Greens',len(valsrtclean))
    for i in range(len(valsrtclean)): 
        # plot scatter of validation (observed) vs satellite (predicted) distances along each transect
        plt.scatter(valsrtclean[i], satsrtclean[i], color=cmap(i), s=10, alpha=0.4, edgecolors='none', zorder=2)
        # linear regressions
        X = np.array(valsrtclean[i]).reshape((-1,1))
        y = np.array(satsrtclean[i])
        model = LinearRegression(fit_intercept=True).fit(X,y)
        r2 = model.score(X,y)
        r2s.append(r2)
        
        valfit = np.linspace(0,round(np.max(valsrtclean[i])),len(valsrtclean[i])).reshape((-1,1))
        satfit = model.predict(valfit)

        line = plt.plot(valfit,satfit, c=cmap(i), alpha=0.5, linewidth=2, label=(satdateclean[i]+' R$^2$ = '+str(round(r2,2))), zorder=3)
        lines.append(line)
        
    besti = r2s.index(max(r2s))
    worsti = r2s.index(min(r2s))
    
    # plt.text(valsrtclean[besti][-1], satsrtclean[besti][-1], satdateclean[besti]+' (R$^2$ = '+str(round(r2s[besti],2))+')', c=cmap(besti))
    # plt.text(valsrtclean[worsti][-1], satsrtclean[worsti][-1], satdateclean[worsti]+' (R$^2$ = '+str(round(r2s[worsti],2))+')', c=cmap(worsti))

    hands = [ lines[besti][0], lines[worsti][0] ]
    labs = [ satdateclean[besti]+'\nS2, R$^2$ = '+str(round(r2s[besti],2)), satdateclean[worsti]+'\nL5, R$^2$ = '+str(round(r2s[worsti],2)) ]
    plt.legend(hands,labs, loc='upper left',facecolor='#D5D5D5')
    
    # overall linear regression
    valfull = [item for sublist in valsrtclean for item in sublist]
    satfull =[item for sublist in satsrtclean for item in sublist]
    X = np.array(valfull).reshape((-1,1))
    y = np.array(satfull)
    model = LinearRegression(fit_intercept=True).fit(X,y)
    r2 = model.score(X,y)
    
    
    valfit = np.linspace(0,round(np.max(valfull)),len(valfull)).reshape((-1,1))
    satfit = model.predict(valfit)

    plt.plot(valfit,satfit, c='k', linestyle='--', linewidth=2, zorder=3)
    RMSEtxt = plt.text(valfit[int(len(valfit)*0.98)],satfit[int(len(valfit)*0.99)],'R$^2$ = '+str(round(r2,2))+'\nRMSE = 23 m', zorder=3, horizontalalignment='right')
    RMSEtxt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w', alpha=0.8)])

    plt.xlim(0,225)
    plt.ylim(0,250)

    plt.xlabel('X-shore distance$_{validation}$ (m)')
    plt.ylabel('X-shore distance$_{satellite}$ (m)')
    
    # ax.axis('equal')
    plt.tight_layout()
    
    figpath = os.path.join(filepath,sitename+'_Validation_Satellite_Distances_LinReg_'+str(TransectIDs[0])+'to'+str(TransectIDs[1])+'_Large.png')
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    print('figure saved under '+figpath)

    plt.show()
    
    # Get unique dates and satnames    
    SatGDFNames = SatGDF[['dates', 'times', 'filename','satname']].groupby(['dates']).max()
    SatNames = []
    
    for d in satdateclean:
        SatNames.append(SatGDFNames.loc[d]['satname'])
    SatNameList = sorted(set(SatNames))
    
    for SatN in SatNameList:
        SatInd = []
        for i, e in enumerate(SatNames):
            if e == SatN:
                SatInd.append(i)
        
        valsrtN = []
        satsrtN = []
        for SatI in SatInd:
            valsrtN.append(valsrtclean[SatI])
            satsrtN.append(satsrtclean[SatI])

        valN = [item for sublist in valsrtN for item in sublist]
        satN =[item for sublist in satsrtN for item in sublist]
        X = np.array(valN).reshape((-1,1))
        y = np.array(satN)
        model = LinearRegression(fit_intercept=True).fit(X,y)
        r2 = model.score(X,y)
        
        print(SatN, r2)
    
    
    
    
def ClusterRates(sitename, TransectInterGDF, Sloc, Nloc):
    
    ## Cluster Plot
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)
    
    
    mpl.rcParams.update({'font.size':7})
    
    # Create array of veg change rates vs shoreline change rates per transect
    MultivarArray = np.array([[ID,x, y] for ID, x, y in zip(TransectInterGDF['TransectID'],TransectInterGDF['oldyoungRt'],TransectInterGDF['oldyungRtW'])])
    # Remove outliers (set to nan then remove in one go below)
    MultivarArray[:,1] = np.where(MultivarArray[:,1] < 50, MultivarArray[:,1], np.nan)
    MultivarArray[:,1] = np.where(MultivarArray[:,1] > -50, MultivarArray[:,1], np.nan)
    MultivarArray[:,2] = np.where(MultivarArray[:,2] < 190, MultivarArray[:,2], np.nan)
    MultivarArray[:,2] = np.where(MultivarArray[:,2] > -190, MultivarArray[:,2], np.nan)
    # Remove any transects with nan values in either column
    MultivarArray = MultivarArray[~np.isnan(MultivarArray).any(axis=1)]
    # Fit k-means clustering to array of rates
    RateCluster = KMeans(n_clusters=8).fit_predict(MultivarArray[:,1:])
    
    fig, axs = plt.subplots(1,2, figsize=(5,5), dpi=200)
    # Plot array using clusters as colour map
    ax1 = axs[0].scatter(MultivarArray[:,1], MultivarArray[:,2], c=RateCluster, s=5, alpha=0.5, marker='.')
    ax2 = axs[1].scatter(MultivarArray[:,1], MultivarArray[:,2], c=MultivarArray[:,0], s=5, alpha=0.5, marker='.')
    
    # axs[0].set_aspect('equal')
    # axs[1].set_aspect('equal')
    axs[0].set_xlim(-25,25)
    axs[0].set_ylim(-100,100)
    axs[1].set_xlim(-25,25)
    axs[1].set_ylim(-100,100)
    axs[0].set_xlabel('Veg change rate (m/yr)')
    axs[0].set_ylabel('Shore change rate (m/yr)')
    axs[1].set_xlabel('Veg change rate (m/yr)')
    axs[0].set_title('Clustering')
    axs[1].set_title('TransectID')
    
    plt.colorbar(ax1, ax=axs[0])
    plt.colorbar(ax2, ax=axs[1])
    plt.tight_layout()
    plt.show()
    

def MultivariateMatrix(sitename, TransectInterGDF,  TransectInterGDFWater, TransectInterGDFTopo, Loc1, Loc2):
    """
    Create a multivariate matrix plot of vegetation edges, waterlines, topographic data and wave data.
    Each point on scatter is a single value on a cross-shore transect (i.e. mean value or rate over time).
    FM Aug 2023

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
        
    ## Multivariate Plot
    # Subset into south and north transects
    MultivarGDF1 = pd.concat([ TransectInterGDF['oldyoungRt'].iloc[Loc1[0]:Loc1[1]], 
                           TransectInterGDFWater['oldyungRtW'].iloc[Loc1[0]:Loc1[1]],
                           TransectInterGDFTopo[['TZwidthMn','SlopeMax']].iloc[Loc1[0]:Loc1[1]] ], axis=1)
    MultivarGDF2 = pd.concat([ TransectInterGDF['oldyoungRt'].iloc[Loc2[0]:Loc2[1]], 
                           TransectInterGDFWater['oldyungRtW'].iloc[Loc2[0]:Loc2[1]],
                           TransectInterGDFTopo[['TZwidthMn','SlopeMax']].iloc[Loc2[0]:Loc2[1]] ], axis=1)
    
    MultivarGDF = pd.concat([MultivarGDF1, MultivarGDF2], axis=0)
    
    # Extract desired columns to an array for plotting
    MultivarArray = np.array(MultivarGDF[['oldyoungRt','oldyungRtW','TZwidthMn','SlopeMax']])
    
    mpl.rcParams.update({'font.size':12})
    fig, axs = plt.subplots(MultivarArray.shape[1],MultivarArray.shape[1], figsize=(11.6,5.9), dpi=300)
    
    # Plot matrix of relationships
    lab = [r'$\Delta$veg (m/yr)',
           r'$\Delta$water (m/yr)',
           r'$TZwidth_{\mu}$ (m)',
           r'$slope_{max}$ ($\circ$)']
    
    for row in range(MultivarArray.shape[1]):
        for col in range(MultivarArray.shape[1]):
            
            # if plot is same var on x and y, change plot to a histogram    
            if row == col:
                binnum = round(np.sqrt(len(MultivarArray)))*2
                bins = np.histogram(MultivarArray[:,row],bins=binnum)[1]
                axs[col,row].hist(MultivarArray[:,row],bins, color='k', alpha=0.5)
                axs[col,row].set_yticks([]) # turns off ticks and tick labels

            # otherwise plot scatter of each variable against one another
            else:
                scatterPl = axs[col,row].scatter(MultivarArray[:,row], MultivarArray[:,col], s=20, alpha=0.4, marker='.', c='k', edgecolors='none')
                
                # overall linear reg line
                z = np.polyfit(list(MultivarArray[:,row]), list(MultivarArray[:,col]), 1)
                poly = np.poly1d(z)
                order = np.argsort(MultivarArray[:,row])
                xlr = MultivarArray[:,row][order]
                ylr = poly(MultivarArray[:,row][order])
                linregLn, = axs[col,row].plot(xlr, ylr, c='k', ls='--', lw=1.5)
                r, p = scipy.stats.pearsonr(list(MultivarArray[:,row]), list(MultivarArray[:,col]))
                statstr = 'r = %.2f' % (r)
                # label with stats [axes.text(x,y,str)]
                if r > 0: # +ve line slanting up
                    va = 'top'
                else: # -ve line slanting down
                    va = 'bottom'
                # rtxt = axs[col,row].text(xlr[0], ylr[0], statstr, c='k', fontsize=10, ha='left', va=va)#transform = axs[row,col].transAxes
                # rtxt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w', alpha=0.8)])
                
            vLn = axs[col,row].axvline(x=0, c=[0.5,0.5,0.5], lw=0.5)
            hLn = axs[col,row].axhline(y=0, c=[0.5,0.5,0.5], lw=0.5)
            
            if row == MultivarArray.shape[1]-1: # set x label for last row only
                axs[row,col].set_xlabel(lab[col])
            if col == 0: # set y label for first column only
                axs[row,col].set_ylabel(lab[row])
            
            # set veg vs water plots to equal axes to highlight orders of difference
            if lab[col] == r'$\Delta$veg (m/yr)' and lab[row] == r'$\Delta$water (m/yr)' :
                axs[row,col].axis('equal')
            if lab[row] == r'$\Delta$veg (m/yr)' and lab[col] == r'$\Delta$water (m/yr)' :
                axs[row,col].axis('equal')
            
            # clear plots on RHS of hists, print stats instead
            for i in range(MultivarArray.shape[1]):
                if col == i and row > i:
                    # axs[col,row].cla() # clears axis on each loop
                    for Ln in [linregLn, scatterPl, hLn, vLn]:
                        Ln.remove()
                    axs[col,row].set_xticks([])
                    axs[col,row].set_yticks([])
                    axs[col,row].text(0.5,0.75, statstr, c='k', fontsize=10, ha='center', transform = axs[col,row].transAxes)   
            
                 
    # align all yaxis labels in first column
    fig.align_ylabels(axs[:,0])
    
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.6, hspace=0.5)
    
    figpath = os.path.join(filepath,sitename+'_Multivariate_%s-%s_%s-%s.png' % 
                           (Loc1[0],Loc1[1],Loc2[0],Loc2[1]))
    plt.savefig(figpath)
    print('figure saved under '+figpath)
    
    plt.show()
    
    return


def MultivariateMatrixClustered(sitename, TransectInterGDF,  TransectInterGDFWater, TransectInterGDFTopo, Loc1, Loc2):
    """
    Create a multivariate matrix plot of vegetation edges, waterlines, topographic data and wave data.
    Each point on scatter is a single value on a cross-shore transect (i.e. mean value or rate over time).
    FM Aug 2023

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
        
    ## Multivariate Plot
    # Subset into south and north transects
    MultivarGDF1 = pd.concat([ TransectInterGDF['oldyoungRt'].iloc[Loc1[0]:Loc1[1]], 
                           TransectInterGDFWater['oldyungRtW'].iloc[Loc1[0]:Loc1[1]],
                           TransectInterGDFTopo[['TZwidthMn','SlopeMax']].iloc[Loc1[0]:Loc1[1]] ], axis=1)
    MultivarGDF2 = pd.concat([ TransectInterGDF['oldyoungRt'].iloc[Loc2[0]:Loc2[1]], 
                           TransectInterGDFWater['oldyungRtW'].iloc[Loc2[0]:Loc2[1]],
                           TransectInterGDFTopo[['TZwidthMn','SlopeMax']].iloc[Loc2[0]:Loc2[1]] ], axis=1)
    
    # summer (pale) eroding = #F9C784 
    # summer (pale) accreting = #9DB4C0
    
    MultivarGDF = pd.concat([MultivarGDF1, MultivarGDF2], axis=0)
    
    # Extract desired columns to an array for plotting
    MultivarArray = np.array(MultivarGDF[['oldyoungRt','oldyungRtW','TZwidthMn','SlopeMax']])
    
    mpl.rcParams.update({'font.size':12})
    fig, axs = plt.subplots(MultivarArray.shape[1],MultivarArray.shape[1], figsize=(11.6,5.9), dpi=300)
    
    # Plot matrix of relationships
    lab = [r'$\Delta$veg (m/yr)',
           r'$\Delta$water (m/yr)',
           r'$TZwidth_{\mu}$ (m)',
           r'$slope_{max}$ ($\circ$)']
    
    for row in range(MultivarArray.shape[1]):
        for col in range(MultivarArray.shape[1]): 
            for Arr, colour, strpos, leglabel in zip([ MultivarArray[0:Loc1[1]-Loc1[0],:], MultivarArray[Loc2[1]-Loc2[0]:,:] ], 
                                           ['#B2182B','#2166AC'],
                                           [0.5,0.25],
                                           ['Eroding ','Accreting ']):
                # if plot is same var on x and y, change plot to a histogram    
                if row == col:
                    binnum = round(np.sqrt(len(MultivarArray)))*2
                    bins = np.histogram(MultivarArray[:,row],bins=binnum)[1]
                    axs[col,row].hist(Arr[:,row],bins, color=colour, alpha=0.5, label=leglabel)
                    axs[col,row].set_yticks([]) # turns off ticks and tick labels
                    # axs[col,row].legend()
    
                # otherwise plot scatter of each variable against one another
                else:
                    scatterPl = axs[col,row].scatter(Arr[:,row], Arr[:,col], s=20, alpha=0.4, marker='.', c=colour, edgecolors='none')
                    
                    # overall linear reg line
                    z = np.polyfit(list(MultivarArray[:,row]), list(MultivarArray[:,col]), 1)
                    poly = np.poly1d(z)
                    order = np.argsort(MultivarArray[:,row])
                    xlr = MultivarArray[:,row][order]
                    ylr = poly(MultivarArray[:,row][order])
                    linregLn, = axs[col,row].plot(xlr, ylr, c='k', ls='--', lw=1.5)
                    r, p = scipy.stats.pearsonr(list(MultivarArray[:,row]), list(MultivarArray[:,col]))
                    statstr = 'r = %.2f' % (r)
                    # label with stats [axes.text(x,y,str)]
                    if r > 0: # +ve line slanting up
                        va = 'top'
                    else: # -ve line slanting down
                        va = 'bottom'
                    # rtxt = axs[col,row].text(xlr[0], ylr[0], statstr, c='k', fontsize=10, ha='left', va=va)#transform = axs[row,col].transAxes
                    # rtxt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w', alpha=0.8)])
                    
                    # clustered linear regression lines
                    zArr = np.polyfit(list(Arr[:,row]), list(Arr[:,col]), 1)
                    polyArr = np.poly1d(zArr)
                    orderArr = np.argsort(Arr[:,row])
                    # plot clustered linear reg line
                    clustLn, = axs[col,row].plot(Arr[:,row][orderArr], polyArr(Arr[:,row][orderArr]), c=colour, ls='--', lw=1.5)
                    rArr, pArr = scipy.stats.pearsonr(list(Arr[:,row]), list(Arr[:,col]))
                    statstrArr = 'r = %.2f' % (rArr)
                    
                hLn = axs[col,row].axvline(x=0, c=[0.5,0.5,0.5], lw=0.5)
                vLn = axs[col,row].axhline(y=0, c=[0.5,0.5,0.5], lw=0.5)
                
                if row == MultivarArray.shape[1]-1: # set x axis labels on last row
                    axs[row,col].set_xlabel(lab[col])
                if col == 0: # set y axis labels on first column
                    axs[row,col].set_ylabel(lab[row])
                
                # set veg vs water plots to equal axes to highlight orders of difference
                if lab[col] == r'$\Delta$veg (m/yr)' and lab[row] == r'$\Delta$water (m/yr)' :
                    axs[row,col].axis('equal')
                    
                # clear plots on RHS of hists, print stats instead
                for i in range(MultivarArray.shape[1]):
                    if col == i and row > i:
                        # axs[col,row].cla() # clears axis on each loop
                        for Ln in [linregLn,clustLn, scatterPl, hLn, vLn]:
                            Ln.remove()
                        axs[col,row].set_xticks([])
                        axs[col,row].set_yticks([])
                        axs[col,row].text(0.5,0.75, statstr, c='k', fontsize=10, ha='center', transform = axs[col,row].transAxes)   
                        axs[col,row].text(0.5,strpos, leglabel+statstrArr, c=colour, fontsize=10, ha='center', transform = axs[col,row].transAxes)   


            
    # align all yaxis labels in first column
    fig.align_ylabels(axs[:,0])
    
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.6, hspace=0.5)
    
    figpath = os.path.join(filepath,sitename+'_MultivariateClustered_%s-%s_%s-%s.png' % 
                           (Loc1[0],Loc1[1],Loc2[0],Loc2[1]))
    plt.savefig(figpath)
    print('figure saved under '+figpath)
    
    plt.show()
    
    return


def SplitGDF(TransectInterGDF,  TransectInterGDFWater, TransectInterGDFTopo, TransectInterGDFWave, ColNames, Loc1=None, Loc2=None):
        
    if Loc1 is None:
        # If no transect subset locations are provided, slice based on eroding/accreting VE
        # Accreting VE
        MultivarGDF1 = pd.concat([ TransectInterGDF['oldyoungRt'].iloc[TransectInterGDF.index[TransectInterGDF['oldyoungRt'] > 0]],
                               TransectInterGDFWater['oldyungRtW'].iloc[TransectInterGDFWater.index[TransectInterGDF['oldyoungRt'] > 0]],
                               TransectInterGDFTopo[['TZwidthMn','SlopeMax']].iloc[TransectInterGDFTopo.index[TransectInterGDF['oldyoungRt'] > 0]],
                               TransectInterGDFWave['WaveDiffus'].iloc[TransectInterGDFWave.index[TransectInterGDF['oldyoungRt'] > 0]],
                               TransectInterGDFWave['WaveQsNet'].iloc[TransectInterGDFWave.index[TransectInterGDF['oldyoungRt'] > 0]] ], axis=1)
        # Eroding VE
        MultivarGDF2 = pd.concat([ TransectInterGDF['oldyoungRt'].iloc[TransectInterGDF.index[TransectInterGDF['oldyoungRt'] < 0]],
                               TransectInterGDFWater['oldyungRtW'].iloc[TransectInterGDFWater.index[TransectInterGDF['oldyoungRt'] < 0]],
                               TransectInterGDFTopo[['TZwidthMn','SlopeMax']].iloc[TransectInterGDFTopo.index[TransectInterGDF['oldyoungRt'] < 0]],
                               TransectInterGDFWave['WaveDiffus'].iloc[TransectInterGDFWave.index[TransectInterGDF['oldyoungRt'] < 0]],
                               TransectInterGDFWave['WaveQsNet'].iloc[TransectInterGDFWave.index[TransectInterGDF['oldyoungRt'] < 0]]], axis=1)
    else:
        # Subset into south and north transects
        MultivarGDF1 = pd.concat([ TransectInterGDF['oldyoungRt'].iloc[Loc1[0]:Loc1[1]], 
                               TransectInterGDFWater['oldyungRtW'].iloc[Loc1[0]:Loc1[1]],
                               TransectInterGDFTopo[['TZwidthMn','SlopeMax']].iloc[Loc1[0]:Loc1[1]],
                               TransectInterGDFWave[['WaveDiffus']].iloc[Loc1[0]:Loc1[1]],
                               TransectInterGDFWave[['WaveQsNet']].iloc[Loc1[0]:Loc1[1]] ], axis=1)
                               # TransectInterGDFWave[['WaveStabil']].iloc[Loc1[0]:Loc1[1]]], axis=1)
    
        MultivarGDF2 = pd.concat([ TransectInterGDF['oldyoungRt'].iloc[Loc2[0]:Loc2[1]], 
                               TransectInterGDFWater['oldyungRtW'].iloc[Loc2[0]:Loc2[1]],
                               TransectInterGDFTopo[['TZwidthMn','SlopeMax']].iloc[Loc2[0]:Loc2[1]],
                               TransectInterGDFWave[['WaveDiffus']].iloc[Loc2[0]:Loc2[1]],
                               TransectInterGDFWave[['WaveQsNet']].iloc[Loc2[0]:Loc2[1]] ], axis=1)
                               # TransectInterGDFWave[['WaveStabil']].iloc[Loc2[0]:Loc2[1]]], axis=1)
    
    # Remove rows with NaNs
    MultivarGDF1.dropna(axis=0, inplace=True)
    MultivarGDF2.dropna(axis=0, inplace=True)
    
    # Make sure MultivarArray is split into equal halves by randomly sampling n entries
    # where n = len(smaller GDF)
    if len(MultivarGDF1) > len(MultivarGDF2):
        MultivarGDF1 = MultivarGDF1.sample(n=len(MultivarGDF2), random_state=1) # seed saved and reused
    elif len(MultivarGDF1) < len(MultivarGDF2):
        MultivarGDF2 = MultivarGDF2.sample(n=len(MultivarGDF1), random_state=1) # seed saved and reused
    
    MultivarGDF = pd.concat([MultivarGDF1, MultivarGDF2], axis=0)

    return MultivarGDF



def MultivariateMatrixClusteredWaves(sitename, MultivarGDF, Loc1=None, Loc2=None):
    """
    Create a multivariate matrix plot of vegetation edges, waterlines, topographic data and wave data.
    Each point on scatter is a single value on a cross-shore transect (i.e. mean value or rate over time).
    Scatter points are separated into clustered regions (N vs S, eroding vs accreting).
    FM March 2024

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
                    axs[col,row].hist(Arr[:,row],bins, color=colour, alpha=0.5, label=moments_label)
                    # Plot the mean as a vertical line
                    axs[col,row].axvline(x=mn, c=colour, lw=0.5, ls='--')
                    # Plot the variance
                    axs[col,row].axvline(x=mn-stdv, c=colour, lw=0.5, ls=':')
                    axs[col,row].axvline(x=mn+stdv, c=colour, lw=0.5, ls=':')
                    axs[col,row].set_yticks([]) # turns off ticks and tick labels
                    if col > 1:
                        legloc = 'upper right'
                    else:
                        legloc = 'upper left'
                    axs[col,row].legend(loc=legloc, 
                                        prop={'size':5}, edgecolor='none', framealpha=0.5,
                                        borderpad=0.2, labelspacing=0.2, handlelength=0.5, handletextpad=0.2)
    
                # otherwise plot scatter of each variable against one another
                else:
                    scatterPl = axs[col,row].scatter(Arr[:,row], Arr[:,col], s=20, alpha=0.4, marker='.', c=colour, edgecolors='none')
                    
                    # overall linear reg line
                    z = np.polyfit(list(MultivarArray[:,row]), list(MultivarArray[:,col]), 1)
                    poly = np.poly1d(z)
                    order = np.argsort(MultivarArray[:,row])
                    xlr = MultivarArray[:,row][order]
                    ylr = poly(MultivarArray[:,row][order])
                    linregLn, = axs[col,row].plot(xlr, ylr, c='k', ls='--', lw=1.5, zorder=3)
                    r, p = scipy.stats.pearsonr(list(MultivarArray[:,row]), list(MultivarArray[:,col]))
                    statstr = 'r = %.2f' % (r)
                    # label with stats [axes.text(x,y,str)]
                    if r > 0: # +ve line slanting up
                        va = 'top'
                    else: # -ve line slanting down
                        va = 'bottom'
                    # rtxt = axs[col,row].text(xlr[0], ylr[0], statstr, c='k', fontsize=10, ha='left', va=va)#transform = axs[row,col].transAxes
                    # rtxt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w', alpha=0.8)])
                    
                    # clustered linear regression lines
                    zArr = np.polyfit(list(Arr[:,row]), list(Arr[:,col]), 1)
                    polyArr = np.poly1d(zArr)
                    orderArr = np.argsort(Arr[:,row])
                    # plot clustered linear reg line
                    clustBuff, = axs[col,row].plot(Arr[:,row][orderArr], polyArr(Arr[:,row][orderArr]), c='w', ls='-', lw=1.9, alpha=0.8, zorder=1)
                    clustLn, = axs[col,row].plot(Arr[:,row][orderArr], polyArr(Arr[:,row][orderArr]), c=colour, ls='--', lw=1.5, zorder=2)
                    rArr, pArr = scipy.stats.pearsonr(list(Arr[:,row]), list(Arr[:,col]))
                    statstrArr = 'r = %.2f' % (rArr)
                    
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
                    
                
                # set veg vs water plots to equal axes to highlight orders of difference
                if lab[col] == r'$\Delta$VE (m/yr)' and lab[row] == r'$\Delta$WL (m/yr)' :
                    axs[row,col].axis('equal')
                    
                # clear plots on RHS of hists, print stats instead
                for i in range(MultivarArray.shape[1]):
                    if col == i and row > i:
                        # axs[col,row].cla() # clears axis on each loop
                        for Ln in [linregLn,clustBuff, clustLn, scatterPl, hLn, vLn]:
                            Ln.remove()
                        axs[col,row].set_xticks([])
                        axs[col,row].set_yticks([])
                        # fontsize 7 for paper, 10 for PPT
                        axs[col,row].text(0.5,0.8, lab[col].split(' (')[0]+' vs. '+lab[row].split(' (')[0], c='k', fontsize=fs, ha='center', transform = axs[col,row].transAxes)   
                        axs[col,row].text(0.5,0.6, statstr, c='k', fontsize=fs, ha='center', transform = axs[col,row].transAxes)   
                        axs[col,row].text(0.5,strpos, statstrArr, c=colour, fontsize=fs, ha='center', transform = axs[col,row].transAxes)   
                # Turn of top and right frame edges for tidiness
                axs[col,row].spines['right'].set_visible(False)
                axs[col,row].spines['top'].set_visible(False)


            
    # align all yaxis labels in first column
    fig.align_ylabels(axs[:,0])
    
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.6, hspace=0.5)
    if Loc1 is None:
        figpath = os.path.join(filepath,sitename+'_MultivariateClustered_VegWaterTopoWaves_AutoSplit.png')
    else:
        figpath = os.path.join(filepath,sitename+'_MultivariateClustered_VegWaterTopoWaves_%s-%s_%s-%s.png' % 
                               (Loc1[0],Loc1[1],Loc2[0],Loc2[1]))
    plt.savefig(figpath)
    print('figure saved under '+figpath)
    
    plt.show()
    
    return


def MultivariateMatrixClusteredSeason(sitename, TransectInterGDF,  TransectInterGDFWater, TransectInterGDFTopo, Loc1, Loc2):
    """
    Create a multivariate matrix plot of vegetation edges, waterlines, topographic data and wave data.
    Each point on scatter is a single value on a cross-shore transect (i.e. mean value or rate over time).
    FM Aug 2023

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
    
    # Subset into south and north transects
    MultivarGDF1 = pd.concat([ TransectInterGDF['dates'].iloc[Loc1[0]:Loc1[1]],
                           TransectInterGDF['distances'].iloc[Loc1[0]:Loc1[1]],
                           TransectInterGDFWater['wldates'].iloc[Loc1[0]:Loc1[1]],
                           TransectInterGDFWater['wlcorrdist'].iloc[Loc1[0]:Loc1[1]],
                           TransectInterGDFTopo[['TZwidth']].iloc[Loc1[0]:Loc1[1]] ], axis=1)
    MultivarGDF2 = pd.concat([ TransectInterGDF['dates'].iloc[Loc2[0]:Loc2[1]],
                           TransectInterGDF['distances'].iloc[Loc2[0]:Loc2[1]],
                           TransectInterGDFWater['wldates'].iloc[Loc2[0]:Loc2[1]],
                           TransectInterGDFWater['wlcorrdist'].iloc[Loc2[0]:Loc2[1]],
                           TransectInterGDFTopo[['TZwidth']].iloc[Loc2[0]:Loc2[1]] ], axis=1)
    
    
    # summer (pale) eroding = #F9C784 
    # summer (pale) accreting = #9DB4C0
    
    MultivarGDF = pd.concat([MultivarGDF1, MultivarGDF2], axis=0)
    
    # Seasonal summaries
    # For each transect, and each year in each transect's list of dates, compile summer and winter dists
    # and then perform lin reg on that subset.
    # After change rate calculated for each year, append to list and take mean of those rates.
    vegSummerRt = []
    vegWinterRt = []
    wlSummerRt = []
    wlWinterRt = []
    TZsummer = []
    TZwinter = []
    for Tr in range(len(MultivarGDF)): # for each transect
        vegTrdates = MultivarGDF['dates'].iloc[Tr]
        wlTrdates = MultivarGDF['wldates'].iloc[Tr]
        vegTrdists = MultivarGDF['distances'].iloc[Tr]
        wlTrdists = MultivarGDF['wlcorrdist'].iloc[Tr]
        TZwidths = MultivarGDF['TZwidth'].iloc[Tr]
        
        
        summerTrRt = []
        winterTrRt = []
        for Trdates, Trdists, SummerRt, WinterRt in zip([vegTrdates,wlTrdates,vegTrdates],
                                                        [vegTrdists,wlTrdists, TZwidths],
                                                        [vegSummerRt,wlSummerRt,TZsummer],
                                                        [vegWinterRt,wlWinterRt,TZwinter]):
        
            # initialise yearly list of dates/dists/slopes
            TrdatesYr = []
            TrdistsYr = []
            SSlopes = []
            WSlopes = []
            for Yr in range(int(min(Trdates)[:4]), int(max(Trdates)[:4])): # for each year in series
                # year search has to start from beginning of list each time (not ideal)
                for Date, Dist in zip(Trdates, vegTrdists): # for each date and dist in transect
                    if int(Date[:4]) == Yr: # if date in list is in matching year
                        # append dates and dists for that year
                        TrdatesYr.append(Date) 
                        TrdistsYr.append(Dist)
                STrdistsYr = []
                WTrdistsYr = []
                STrdatesYr = []
                WTrdatesYr = []
                for TrdatesYri in TrdatesYr:
                    # summer list (March to August)
                    if int(TrdatesYri[5:7]) >= 3 and int(TrdatesYri[5:7]) <= 8:
                        Ind = TrdatesYr.index(TrdatesYri)
                        STrdatesYr.append(TrdatesYr[Ind])
                        STrdistsYr.append(TrdistsYr[Ind])
                    # winter list (Sept to Feb)
                    else:
                        vegInd = TrdatesYr.index(TrdatesYri)
                        WTrdatesYr.append(TrdatesYr[vegInd])
                        WTrdistsYr.append(TrdistsYr[vegInd])  
                        
                for TrdatesYr, TrdistsYr, Slopes in zip([STrdatesYr,WTrdatesYr], 
                                                        [STrdistsYr,WTrdistsYr], 
                                                        [SSlopes, WSlopes]):
                    # convert dates to ordinals for linreg
                    OrdDates = [datetime.strptime(i,'%Y-%m-%d').toordinal() for i in TrdatesYr]
                    X = np.array(OrdDates[0:]).reshape((-1,1))
                    y = np.array(TrdistsYr[0:])
                    # change rate
                    model = LinearRegression(fit_intercept=True).fit(X,y)
                    Slope = round(model.coef_[0]*365.2425, 2) # convert to m/yr
                    Slopes.append(Slope)
                        
            summerTrRt.append(np.nanmean(SSlopes))
            winterTrRt.append(np.nanmean(WSlopes))
            
        SummerRt.append(summerTrRt)
        WinterRt.append(winterTrRt)
    
    # for some reason, values stored as sets of two items in list for veg and water;
    # take 1st element for veg and 2nd for water
    MultivarGDF['SummerVegRt'] = [Rt[0] for Rt in SummerRt]
    MultivarGDF['WinterVegRt'] = [Rt[0] for Rt in WinterRt]
    MultivarGDF['SummerWLRt'] = [Rt[1] for Rt in SummerRt]
    MultivarGDF['WinterWLRt'] = [Rt[1] for Rt in WinterRt]
    MultivarGDF['SummerTZ'] = [Rt[2] for Rt in SummerRt]
    MultivarGDF['WinterTZ'] = [Rt[2] for Rt in WinterRt]
    
 
    # Extract desired columns to an array for plotting
    # MultivarArray = np.array(MultivarGDF[['SummerVegRt','SummerWLRt','WinterVegRt','WinterWLRt','TZwidthMn']])
    MultivarArray = np.array(MultivarGDF[['SummerVegRt','SummerWLRt','WinterVegRt','WinterWLRt']])
    
    # VegMultivarArray = MultivarArray[:,[0,2]]
    # WLMultivarArray = MultivarArray[:,[1,3]]
    # MultivarArray = 
    
    mpl.rcParams.update({'font.size':10})
    fig, axs = plt.subplots(2, 2, figsize=(11.6,5.9), dpi=300)
    
    # Plot matrix of relationships
    lab = [r'$\Delta$veg (m/yr)',
           r'$\Delta$water (m/yr)',
           r'$TZwidth_{\mu}$ (m)']
    
    for row in range(2):
        for col in range(2): 
            for Arr in [MultivarArray[0:Loc1[1]-Loc1[0],:], MultivarArray[Loc2[1]-Loc2[0]:,:]]:
                if row == 0 or row == 2: # summer
                    for i,colour in enumerate(['#F67E4B','#6EA6CD']):
                        if row == col:
                            binnum = round(np.sqrt(len(MultivarArray)))*2
                            bins = np.histogram(MultivarArray[:,row],bins=binnum)[1]
                            axs[col,row].hist(Arr[:,row+i], bins, color=colour, alpha=0.5)
                        else:
                            scatterPl = axs[col,row].scatter(Arr[:,row+i], Arr[:,col+i], s=20, alpha=0.4, marker='.', c=colour, edgecolors='none')
                else: # winter
                    for i,colour in enumerate(['#A50026','#364B9A']):
                        if row == col:
                            binnum = round(np.sqrt(len(MultivarArray)))*2
                            bins = np.histogram(MultivarArray[:,row],bins=binnum)[1]
                            axs[col,row].hist(Arr[:,row+i], bins, color=colour, alpha=0.5)
                        else:
                            scatterPl = axs[col,row].scatter(Arr[:,row+i], Arr[:,col], s=20, alpha=0.4, marker='.', c=colour, edgecolors='none')
    
    
    
    # for row in range(2):
    #     for col in range(2): 
    #         # for top and bottom half of array (eroding and accreting), 
    #         for Arr, colour, strpos in zip([ VegMultivarArray[0:Loc1[1]-Loc1[0],0], VegMultivarArray[Loc2[1]-Loc2[0]:,1],
    #                                          WLMultivarArray[0:Loc1[1]-Loc1[0],0], WLMultivarArray[Loc2[1]-Loc2[0]:,1] ], 
    #                                        ['#F67E4B','#A50026','#6EA6CD','#364B9A'], # erode summer, erode winter, acc summer, acc winter
    #                                        [0.6,0.5,0.4,0.3]):
    #             # if plot is same var on x and y, change plot to a histogram    
    #             if row == col:
    #                 binnum = round(np.sqrt(len(MultivarArray)))*2
    #                 bins = np.histogram(MultivarArray[:,row],bins=binnum)[1]
    #                 axs[col,row].hist(Arr,bins, color=colour, alpha=0.5)
    #                 axs[col,row].set_yticks([]) # turns off ticks and tick labels
    
    #             # otherwise plot scatter of each variable against one another
    #             else:
    #                 scatterPl = axs[col,row].scatter(Arr[:,row], Arr[:,col], s=20, alpha=0.4, marker='.', c=colour, edgecolors='none')
                    
    #                 # overall linear reg line
    #                 z = np.polyfit(list(MultivarArray[:,row]), list(MultivarArray[:,col]), 1)
    #                 poly = np.poly1d(z)
    #                 order = np.argsort(MultivarArray[:,row])
    #                 xlr = MultivarArray[:,row][order]
    #                 ylr = poly(MultivarArray[:,row][order])
    #                 linregLn, = axs[col,row].plot(xlr, ylr, c='k', ls='--', lw=1.5)
    #                 r, p = scipy.stats.pearsonr(list(MultivarArray[:,row]), list(MultivarArray[:,col]))
    #                 statstr = 'r = %.2f' % (r)
    #                 # label with stats [axes.text(x,y,str)]
    #                 if r > 0: # +ve line slanting up
    #                     va = 'top'
    #                 else: # -ve line slanting down
    #                     va = 'bottom'
    #                 # rtxt = axs[col,row].text(xlr[0], ylr[0], statstr, c='k', fontsize=10, ha='left', va=va)#transform = axs[row,col].transAxes
    #                 # rtxt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w', alpha=0.8)])
                    
    #                 # clustered linear regression lines
    #                 zArr = np.polyfit(list(Arr[:,row]), list(Arr[:,col]), 1)
    #                 polyArr = np.poly1d(zArr)
    #                 orderArr = np.argsort(Arr[:,row])
    #                 # plot clustered linear reg line
    #                 clustLn, = axs[col,row].plot(Arr[:,row][orderArr], polyArr(Arr[:,row][orderArr]), c=colour, ls='--', lw=1.5)
    #                 rArr, pArr = scipy.stats.pearsonr(list(Arr[:,row]), list(Arr[:,col]))
    #                 statstrArr = 'r = %.2f' % (rArr)
                    
    #             hLn = axs[col,row].axvline(x=0, c=[0.5,0.5,0.5], lw=0.5)
    #             vLn = axs[col,row].axhline(y=0, c=[0.5,0.5,0.5], lw=0.5)
                
    #             if row == MultivarArray.shape[1]-1: # set x axis labels on last row
    #                 axs[row,col].set_xlabel(lab[col])
    #             if col == 0: # set y axis labels on first column
    #                 axs[row,col].set_ylabel(lab[row])
                
    #             # set veg vs water plots to equal axes to highlight orders of difference
    #             if lab[col] == r'$\Delta$veg (m/yr)' and lab[row] == r'$\Delta$water (m/yr)' :
    #                 axs[row,col].axis('equal')
                    
    #             # clear plots on RHS of hists, print stats instead
    #             for i in range(MultivarArray.shape[1]):
    #                 if col == i and row > i:
    #                     # axs[col,row].cla() # clears axis on each loop
    #                     for Ln in [linregLn,clustLn, scatterPl, hLn, vLn]:
    #                         Ln.remove()
                        
    #                     axs[col,row].text(0.5,0.75, statstr, c='k', fontsize=10, ha='center', transform = axs[col,row].transAxes)   
    #                     axs[col,row].text(0.5,strpos, statstrArr, c=colour, fontsize=10, ha='center', transform = axs[col,row].transAxes)   


            
    # align all yaxis labels in first column
    fig.align_ylabels(axs[:,0])
    
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.6, hspace=0.5)
    
    figpath = os.path.join(filepath,sitename+'_MultivariateClusteredSeason_%s-%s_%s-%s.png' % 
                           (Loc1[0],Loc1[1],Loc2[0],Loc2[1]))
    plt.savefig(figpath)
    print('figure saved under '+figpath)
    
    plt.show()
    
    return


def MultivariateMatrixWaves(sitename, TransectInterGDF,  TransectInterGDFWater, TransectInterGDFTopo, TransectInterGDFWave, Loc1, Loc2):
    """
    Create a multivariate matrix plot of vegetation edges, waterlines, topographic data and wave climate data.
    Each point on scatter is a single value on a cross-shore transect (i.e. mean value or rate over time).
    FM Aug 2023

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
    TransectInterGDFWave : GeoDataFrame
        GeoDataFrame of transects intersected with wave hindcasts.
    Loc1 : list
        Transect IDs to slice array up for north location
    Loc2 : list
        Transect IDs to slice array up for south location

    """
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)
        
    ## Multivariate Plot
    # Subset into south and north transects
    MultivarGDF1 = pd.concat([ TransectInterGDF['oldyoungRt'].iloc[Loc1[0]:Loc1[1]], 
                           TransectInterGDFWater['oldyungRtW'].iloc[Loc1[0]:Loc1[1]],
                           TransectInterGDFTopo[['TZwidthMn','SlopeMax']].iloc[Loc1[0]:Loc1[1]],
                           TransectInterGDFWave[['WaveDiffus']].iloc[Loc1[0]:Loc1[1]]], axis=1)#, 'WaveStabil']].iloc[Loc1[0]:Loc1[1]]], axis=1)

    MultivarGDF2 = pd.concat([ TransectInterGDF['oldyoungRt'].iloc[Loc2[0]:Loc2[1]], 
                           TransectInterGDFWater['oldyungRtW'].iloc[Loc2[0]:Loc2[1]],
                           TransectInterGDFTopo[['TZwidthMn','SlopeMax']].iloc[Loc2[0]:Loc2[1]],
                           TransectInterGDFWave[['WaveDiffus']].iloc[Loc2[0]:Loc2[1]]], axis=1)#, 'WaveStabil']].iloc[Loc2[0]:Loc2[1]]], axis=1)

    # summer (pale) eroding = #F9C784 
    # summer (pale) accreting = #9DB4C0
    
    MultivarGDF = pd.concat([MultivarGDF1, MultivarGDF2], axis=0)
    # Scale up diffusivity (mu) for nicer labelling
    MultivarGDF['WaveDiffus'] = MultivarGDF['WaveDiffus']*1000
    
    # Extract desired columns to an array for plotting
    MultivarArray = np.array(MultivarGDF[['oldyoungRt','oldyungRtW','TZwidthMn','SlopeMax','WaveDiffus']])#, 'WaveStabil']])
    
    mpl.rcParams.update({'font.size':10})
    fig, axs = plt.subplots(MultivarArray.shape[1],MultivarArray.shape[1], figsize=(6.55,8.33), dpi=300)
    # fig, axs = plt.subplots(MultivarArray.shape[1],MultivarArray.shape[1], figsize=(12.68,6), dpi=300) # PPT dimensions

    
    # Plot matrix of relationships
    lab = [r'$\Delta$veg (m/yr)',
           r'$\Delta$water (m/yr)',
           r'TZwidth$_{\mu}$ (m)',
           r'slope$_{max}$ ($\circ$)',
           r'$\mu_{net}$ (m/s$^{2}$)']
           #r'$\Gamma$ (1)']
    
    for row in range(MultivarArray.shape[1]):
        for col in range(MultivarArray.shape[1]): 
            # for Arr, colour, strpos, leglabel in zip([ MultivarArray[0:Loc1[1]-Loc1[0],:], MultivarArray[Loc2[1]-Loc2[0]:,:] ], 
            #                                ['#C51B2F','#5499DE'],
            #                                [0.5,0.25],
            #                                ['Eroding ','Accreting ']):
            # if plot is same var on x and y, change plot to a histogram    
            if row == col:
                binnum = round(np.sqrt(len(MultivarArray)))*2
                bins = np.histogram(MultivarArray[:,row],bins=binnum)[1]
                axs[col,row].hist(MultivarArray[:,row],bins, color='k', alpha=0.5)
                axs[col,row].set_yticks([]) # turns off ticks and tick labels
                # axs[col,row].legend()

            # otherwise plot scatter of each variable against one another
            else:
                scatterPl = axs[col,row].scatter(MultivarArray[:,row], MultivarArray[:,col], s=20, alpha=0.3, marker='.', c='k', edgecolors='none')
                
                # overall linear reg line
                z = np.polyfit(list(MultivarArray[:,row]), list(MultivarArray[:,col]), 1)
                poly = np.poly1d(z)
                order = np.argsort(MultivarArray[:,row])
                xlr = MultivarArray[:,row][order]
                ylr = poly(MultivarArray[:,row][order])
                linregBuff, = axs[col,row].plot(xlr, ylr, c='w', ls='-', lw=1.9, alpha=0.7, zorder=1)
                linregLn, = axs[col,row].plot(xlr, ylr, c='k', ls='--', lw=1.5, zorder=3)
                
                r, p = scipy.stats.pearsonr(list(MultivarArray[:,row]), list(MultivarArray[:,col]))
                statstr = 'r = %.2f' % (r)
                # label with stats [axes.text(x,y,str)]
                if r > 0: # +ve line slanting up
                    va = 'top'
                else: # -ve line slanting down
                    va = 'bottom'
                # rtxt = axs[col,row].text(xlr[0], ylr[0], statstr, c='k', fontsize=10, ha='left', va=va)#transform = axs[row,col].transAxes
                # rtxt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w', alpha=0.8)])
              
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
                
            
            # set veg vs water plots to equal axes to highlight orders of difference
            # if lab[col] == r'$\Delta$veg (m/yr)' and lab[row] == r'$\Delta$water (m/yr)' :
            #     axs[row,col].axis('equal')
                
            # clear plots on RHS of hists, print stats instead
            for i in range(MultivarArray.shape[1]):
                if col == i and row > i:
                    # axs[col,row].cla() # clears axis on each loop
                    for Ln in [linregLn,scatterPl, hLn, vLn]:
                        Ln.remove()
                    axs[col,row].set_xticks([])
                    axs[col,row].set_yticks([])
                    axs[col,row].text(0.5,0.75, statstr, c='k', fontsize=10, ha='center', transform = axs[col,row].transAxes)   
                    # axs[col,row].text(0.5,strpos, statstrArr, c=colour, fontsize=8, ha='center', transform = axs[col,row].transAxes)   
            # Turn of top and right frame edges for tidiness
            axs[col,row].spines['right'].set_visible(False)
            axs[col,row].spines['top'].set_visible(False)

            
    # align all yaxis labels in first column
    fig.align_ylabels(axs[:,0])
    
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.6, hspace=0.5)
    
    figpath = os.path.join(filepath,sitename+'_MultivariateAnalysis_VegWaterTopoWaves.png' )
    plt.savefig(figpath)
    print('figure saved under '+figpath)
    
    plt.show()
    
    return
    

def WPErrors(filepath, sitename, WPErrorPath, WPPath):
    """
    Generate plot error values associated with different Weighted Peaks thresholding values.
    FM Aug 2023

    Parameters
    ----------
    filepath : str
        Filepath to save figure to.
    sitename : str
        Name of site of interest.
    CSVpath : str
        Filepath to Weighted Peaks RMSE values stored in CSV.

    """
    
    mpl.rcParams.update({'font.size':7})

    fig, axs = plt.subplots(2,1,figsize=(3.31, 5), dpi=300, gridspec_kw={'height_ratios':[2,1]})  
    
    # First Plot (errors linked to different weights for each satellite)
    ax2 = axs[0].twiny()
    
    #read in CSV of errors
    errorDF = pd.read_csv(WPErrorPath)
    # sort sat names alphabetically
    errorDF = pd.concat([errorDF['veg'], errorDF['nonveg'], errorDF.iloc[:,2:].reindex(sorted(errorDF.columns[2:]), axis=1)], axis=1)
    
    # read in names of satellites from headings
    uniquesats = list(errorDF.columns[2:])
    colors = plt.cm.Blues(np.linspace(0.4, 1, len(uniquesats)))
    
    # for each satellite name
    for i,sat in enumerate(uniquesats):
        # plot graph of errors and max value of each sat as diamond
        ax2.plot(errorDF['nonveg'][errorDF[sat]==min(errorDF[sat])], errorDF[sat][errorDF[sat]==min(errorDF[sat])], marker='d', color=colors[i], markeredgecolor='r', markeredgewidth=0.5, markersize=5, zorder=5)
        axs[0].plot(errorDF['veg'], errorDF[sat], marker='o', markersize=2, color=colors[i], linewidth=1, label=sat)
    
    # set xticks using WP values
    axs[0].set_xticks(errorDF['veg'],minor=True)
    axs[0].set_xticks(errorDF['veg'].iloc[0::2], minor=False)
    ax2.set_xticks(errorDF['nonveg'],minor=True)
    ax2.set_xticks(errorDF['nonveg'].iloc[0::2], minor=False)
    # ax2.invert_axis()
    axs[0].set_xlim(min(errorDF['veg'])-0.05, max(errorDF['veg'])+0.05)
    ax2.set_xlim(max(errorDF['nonveg'])+0.05, min(errorDF['nonveg'])-0.05)
    axs[0].set_ylim(-10,215)
    
    axs[0].grid(which='major', color='#BBB4BB', alpha=0.5)
    axs[0].grid(which='minor', color='#BBB4BB', alpha=0.2)
    
    axs[0].set_xlabel('$\omega_{veg}$')
    ax2.set_xlabel('$\omega_{nonveg}$')
    axs[0].set_ylabel('RMSE (m)')
    
    axs[0].legend(loc='upper left',ncol=2)
    plt.tight_layout()
    mpl.rcParams.update({'font.size':7})
    
    
    # Second Plot (values)
    # read in arrays of NDVI pixel values for each class
    peaksDF = pd.read_csv(WPPath)
    int_veg = peaksDF['int_veg'].to_numpy()
    int_nonveg = peaksDF['int_nonveg'].to_numpy()
    
    # calculate WP threshold and TZ
    thresh, peaks = Toolbox.FindWPThresh(int_veg, int_nonveg)
    thresh = round(thresh,2)+0.03
    TZbuffer = Toolbox.TZValues(int_veg, int_nonveg)

    # define hist properties
    binwidth = 0.01
    bins = np.arange(-1, 1, binwidth)
    cmap = cm.get_cmap('Paired')
    # slice up colormap into desired colours
    vegc = cmap.colors[2]  # veg
    nonvegc = cmap.colors[8]  # non-veg
    threshc = cmap.colors[7] # threshold
    TZc = cmap.colors[6] # TZ
    vy, _, _ = axs[1].hist(int_veg, bins=bins, density=True, color=vegc, label='$I_{veg}$')
    nvy, _, _ = axs[1].hist(int_nonveg, bins=bins, density=True, color=nonvegc, alpha=0.75, label='$I_{nonveg}$') 
    
    # plot WP threshold and peaks as dashed vertical lines on PDF
    axs[1].plot([thresh,thresh], [0,max(nvy)+5], color=threshc, lw=1, ls='--', label='$I_{O}$')
    axs[1].plot([peaks[0],peaks[0]], [0,max(nvy)], color=cmap.colors[3], lw=1, ls='--', label='$\zeta_{veg}$') # veg
    axs[1].plot([peaks[1],peaks[1]], [0,max(nvy)], color=cmap.colors[9], lw=1, ls='--', label='$\zeta_{nonveg}$') # nonveg
    # plot TZ as transparent rectangle (xy, width, height, *)
    TZrec = mpatches.Rectangle((TZbuffer[0], 0), TZbuffer[1]-TZbuffer[0], max(nvy)+5, fc=TZc, ec=None, alpha=0.3, label='TZ')
    axs[1].add_patch(TZrec)    
    
    axs[1].set_xlim(-0.4, 1)
    axs[1].set_ylim(0, 6.1)
    axs[1].set_yticks([])
    axs[1].set_xlabel('NDVI')
    axs[1].set_ylabel('Density')
    
    # axs[1].legend(loc='upper left',ncol=1)   
    ax1hand, ax1lab = axs[1].get_legend_handles_labels()
    laborder = [0,3,1,4,2,5]
    axs[1].legend([ax1hand[idx] for idx in laborder], [ax1lab[idx] for idx in laborder], loc='upper left',ncol=1)  
    
    # subplot labels
    axs[0].text(1-0.012,215-3,'A', ha='right', va='top', 
             bbox=dict(boxstyle='square', fc='w', ec='k'), zorder=5)
    
    axs[1].text(1-0.015,6.1-0.15,'B', ha='right', va='top', 
             bbox=dict(boxstyle='square', fc='w', ec='k'), zorder=5)
    
    plt.tight_layout()
    plt.show()
    
    figpath = os.path.join(filepath,sitename+'_VedgeSat_WP_Errors_TZ.png')
    plt.savefig(figpath)
    print('figure saved under '+figpath)
        
    
    return


def TideHeights(figpath, sitename, VegGDF, CSVpath, cmapDates):
    """
    Generate plot of RMSE values vs tide heights for satellite veg edges in chosen transect range.
    FM Aug 2023

    Parameters
    ----------
    VegGDF : GeoDataFrame
        GeoDataFrame generated from reading in the sat-derived veg edge shapefile.
    CSVpath : str
        Filepath to errors CSV generated with Toolbox.QuantifyErrors().
    cmapDates : list
        List of date strings to be used to create a colour ramp.


    """
    mpl.rcParams.update({'font.size':7})
    
    # Read in errors CSV
    ErrorDF = pd.read_csv(CSVpath)
    # Remove 'Total' row
    ErrorDF.drop(ErrorDF[ErrorDF['Date'] == 'Total'].index, axis=0, inplace=True)
    
    # Take unique dates from veg edge shapefile
    VegLines = VegGDF[['dates','tideelev','satname']].groupby(['dates']).max()
    
    # Extract tide levels and sat names for each matching date in error CSV
    Tides = []
    SatNames = []
    for date in ErrorDF['Date']:
        Tides.append(VegLines.loc[date]['tideelev'])
        SatNames.append(VegLines.loc[date]['satname'])
    
    # Attach tides and satnames back to dataframe
    ErrorDF['Tides'] = Tides
    ErrorDF['satname'] = SatNames
    
    # Generate colour ramp based on full date range, then only keep dates that exist in error DF    
    cmap = cm.get_cmap('magma_r',len(cmapDates))
    cmc = []
    for i in range(len(ErrorDF['Date'])):
        cmi = cmapDates.index(ErrorDF['Date'].iloc[i])
        cmc.append(cmap.colors[cmi])
    
    # Create blue colour ramp for sat names
    Sats = sorted(set(SatNames))
    satcmap = plt.cm.Blues(np.linspace(0.4, 1, len(Sats)))
    satcm = { Sats[i] : satcmap[i] for i in range(len(Sats)) }
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(2.07,2.01), dpi=300)
    
    # Plot point for each tide height vs error value
    for i in range(len(ErrorDF['RMSE'])):
        plt.scatter(ErrorDF['RMSE'][i], ErrorDF['Tides'][i], s=25, marker='s', color=cmc[i], edgecolors=satcm[ErrorDF['satname'][i]], linewidth=1, label=ErrorDF['Date'][i])
    
    # Fit linear regression to scatter
    x = ErrorDF['RMSE']
    y = ErrorDF['Tides']
    msat, csat = np.polyfit(x,y,1)
    polysat = np.poly1d([msat, csat])
    xx = np.linspace(x.min(), x.max(), 100)
    plt.plot(xx, polysat(xx), c='#A5A5AF', linestyle='--', linewidth=1.2)
    
    X_sm = sm.add_constant(x)  # add intercept
    sm_model = sm.OLS(y, X_sm).fit()
    # print(sm_model.rsquared)
    # print(sm_model.pvalues[0])
    # print(dir(sm_model))
    # print(sm_model.summary())
    
    plt.text(xx[-1], polysat(xx)[-1]-0.1, 
             f"$R^2$ = {sm_model.rsquared:.2f}\n$p$ = {sm_model.pvalues[0]:.2f}", c='#A5A5AF',
             ha='right', va='top')
    
    # Plot properties
    plt.xticks(range(0,35,5))
    plt.yticks(np.arange(-2,2.5,0.5))
    plt.xlabel('RMSE (m)')
    plt.ylabel('Tide height (m)')
      
    # plt.legend(loc='lower left',ncol=2)
    plt.tight_layout()
    
    figname = os.path.join(figpath,sitename+'_VedgeSat_TideHeights_Errors.png')
    plt.savefig(figname, bbox_inches='tight')
    print('figure saved under '+figname)
    
    plt.show()
    
    return
    

def StormsTimeline(figpath, sitename, CSVpath):
    """
    Plot named storms as colour-coded bars in timeline (needs two plots if long).
    FM Sept 2023

    Parameters
    ----------
    figpath : str
        Path to folder to save figure in.
    sitename : str
        Name of site of interest.
    CSVpath : str
        Path (and filename) of CSV holding data to be plotted (laid out as 
        name, start date, end date, max wind gust).


    """
    
    # Read in errors CSV
    StormsDF = pd.read_csv(CSVpath)
    StormsDF = StormsDF.iloc[::-1]
    
    
    mpl.rcParams.update({'font.size':7})
    
    # Set up plot
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6.55,3), dpi=300)
    
    # format date fields and calculate length of storms
    StormsDF['StartDate'] = [datetime.strptime(i, '%d/%m/%Y') for i in StormsDF['Start']]
    StormsDF['EndDate'] = [datetime.strptime(i, '%d/%m/%Y') for i in StormsDF['End']]
    StormsDF['Duration'] = StormsDF['EndDate']-StormsDF['StartDate']
    
    
    inum = round(len(StormsDF) / 2)

    # cmap = plt.get_cmap("magma_r", len(StormsDF['WindGust']))
    
    # Plot gantt style timeline of storms where length of bar = duration of storm
    for ax, DFhalf in zip([ax1, ax2], [StormsDF.iloc[inum:],StormsDF.iloc[0:inum]]):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        # Approach for colormap is to plot a scatter, then access the colors from those objects for plotting Rectangles
        scatter = ax.scatter(x=DFhalf['StartDate'], y=DFhalf['Name'], c=DFhalf['WindGust'], cmap="Spectral_r", s=0.1, marker='.')
        
        # Plot Rectangle symbols where width = duration of storm and color = intensity
        for i in range(len(DFhalf['Name'])):
            ax.add_patch(Rectangle(
            xy=(DFhalf['StartDate'].iloc[i], i-0.75), width=DFhalf['Duration'].iloc[i], height=1.5, color=scatter.to_rgba(DFhalf['WindGust'])[i]))
            
        # Label most intense storms
        for i in range(len(ax.get_yticklabels())):
            if DFhalf['WindGust'].iloc[i] > 179:
                ax.get_yticklabels()[i].set_color('red')
                ax.text(DFhalf['EndDate'].iloc[i], i-0.2, str(DFhalf['WindGust'].iloc[i]), color='red', va='center')
    
    plt.tight_layout()
    cbax = inset_axes(ax2, width='30%', height='5%', loc=3)
    plt.colorbar(scatter, cax=cbax, ticks=range(80,max(StormsDF['WindGust']),20), orientation='horizontal') 
    cbax.xaxis.set_ticks_position('top')
    cbax.text(max(StormsDF['WindGust'])-min(StormsDF['WindGust']),5,'Maximum wind gust (km/h)', ha='center')
    # plt.gcf().autofmt_xdate()
    

    mpl.rcParams.update({'font.size':7})
    
    figname = os.path.join(figpath,sitename+'_VedgeSat_UKStorms.png')
    plt.savefig(figname, bbox_inches='tight',dpi=300)
    print('figure saved under '+figname)
    
    plt.show()
    
    return


def StormsTimelineSimple(figpath, sitename, CSVpath, StormsLim=None):
    """
    Plot named storms as simple bars in timeline (needs two plots if long).
    FM Sept 2023

    Parameters
    ----------
    figpath : str
        Path to folder to save figure in.
    sitename : str
        Name of site of interest.
    CSVpath : str
        Path (and filename) of CSV holding data to be plotted (laid out as 
        name, start date, end date, max wind gust).
    StormsLim : list
        Lower and upper limit of x axis (should match tide data timeframe limit)


    """
    
    # Read in errors CSV
    StormsDF = pd.read_csv(CSVpath)
    # StormsDF = StormsDF.iloc[::-1]
    
    mpl.rcParams.update({'font.size':7})
    
    # Set up plot
    fig, ax = plt.subplots(1,1, figsize=(3.05,1.72), dpi=300)
    
    # format date fields and calculate length of storms
    StormsDF['StartDate'] = [datetime.strptime(i, '%d/%m/%Y') for i in StormsDF['Start']]
    StormsDF['EndDate'] = [datetime.strptime(i, '%d/%m/%Y') for i in StormsDF['End']]
    StormsDF['Duration'] = StormsDF['EndDate']-StormsDF['StartDate']
    
    # Calculate storm season IDs
    SznCount = {}
    StormsDF['Season'] = StormsDF['StartDate'].apply(GetSznID, args=(SznCount,))
    
    def aggfn(x):
        d = {}
        d['StartDate_min'] = x['StartDate'].min()
        d['EndDate_max'] = x['EndDate'].max()
        d['WindGust_max'] = x['WindGust'].max()
        d['WindGust_min'] = x['WindGust'].min()
        d['WindGust_mn'] = x['WindGust'].mean()
        d['WindGust_md'] = x['WindGust'].median()
        return pd.Series(d, index=['StartDate_min',
                                   'EndDate_max',
                                   'WindGust_max',
                                   'WindGust_min',
                                   'WindGust_mn',
                                   'WindGust_md'])
        
    StormsGrp = StormsDF.groupby('Season').apply(aggfn)
    StormsGrp['Duration_szn'] = StormsGrp['EndDate_max'] - StormsGrp['StartDate_min']
    
    # Format date columns for x axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1,7)))
    
    scatter = ax.scatter(x=StormsDF['StartDate'], y=StormsDF['WindGust'], 
                           c='#CDCDCD', s=0, marker='.')
    # Plot tall rectangle symbols where width = duration of storm event
    # color = intensity: color=scatter.to_rgba(StormsDF['WindGust'])[i])
    for i in range(len(StormsDF['Name'])):
        ax.add_patch(Rectangle(
        xy=(StormsDF['StartDate'].iloc[i], 0), 
        width=StormsDF['Duration'].iloc[i], 
        height=StormsDF['WindGust'].max()+20, 
        color='#929292', lw=0.5, alpha=0.5))
    
    # Create colour ramp for storm years
    cmp = cm.get_cmap('Blues')
    colours = []
    # for i in range(len(StormsDF['Season'].unique())):
        # colours.append(cmp(i/len(StormsDF['Season'].unique())))
    # StormDiff = datetime.strptime(StormsLim[1],'%Y-%m-%d %H:%M:%S')-datetime.strptime(StormsLim[0],'%Y-%m-%d %H:%M:%S')
    # StormYears = round(StormDiff.total_seconds()/(365.2425*24*60*60))
    # for i in range(StormYears):
    #     colours.append(cmp(i/StormYears))
    for i in range(11):
        colours.append(cmp(i/11))
    colours = colours[1:]
        
    # Plot boxplot style rectangles over top of storm events, where:
    # width = duration of storm season (start of first storm to end of last storm)
    # and colormap = mean/median wind gust.
    # Approach for colormap is to plot a scatter, then access the colors from those objects for plotting Rectangles.
    scatter = ax.scatter(x=StormsGrp['StartDate_min'], y=StormsGrp['WindGust_max'], 
                           c='k', cmap='Spectral', s=0, marker='.') 
    for i in range(len(StormsGrp)):
        ax.add_patch(Rectangle(
        xy=(StormsGrp['StartDate_min'].iloc[i], StormsGrp['WindGust_min'].iloc[i]), 
        width=StormsGrp['Duration_szn'].iloc[i], 
        height=StormsGrp['WindGust_max'].iloc[i]-StormsGrp['WindGust_min'].iloc[i],
        edgecolor='k', facecolor=colours[i], lw=0.75))
        # Add median lines across pseudo boxplots
        ax.plot([StormsGrp['StartDate_min'].iloc[i],StormsGrp['EndDate_max'].iloc[i]], 
                [StormsGrp['WindGust_md'].iloc[i], StormsGrp['WindGust_md'].iloc[i]], 
                c='k', ls=':', lw=0.75)
    
    ax.set_yticks(np.arange(0, StormsDF['WindGust'].max()+20,20))
    ax.set_yticks(np.arange(0, StormsDF['WindGust'].max(),10), minor=True)
    ax.set_ylim(StormsDF['WindGust'].min()-40, StormsDF['WindGust'].max()+5)
    ax.set(ylabel='Wind gust (km/h)')
    ax.set_xlim(datetime.strptime(StormsLim[0],'%Y-%m-%d %H:%M:%S'),datetime.strptime(StormsLim[1],'%Y-%m-%d %H:%M:%S'))
    
    # cbax = inset_axes(ax, width='30%', height='5%', loc=3)
    # plt.colorbar(scatter, cax=cbax, ticks=range(80,max(StormsDF['WindGust'])+40,40), orientation='horizontal') 
    # cbax.xaxis.set_ticks_position('top')
    # cbax.text(max(StormsDF['WindGust'])-min(StormsDF['WindGust']),5,'Wind gust (km/h)', ha='center')
    # plt.gcf().autofmt_xdate()
    
    # Add single instances of each object in hidden corner for legend
    sznpl = ax.add_patch(Rectangle(xy=(0,0),width=0,height=0,edgecolor='k',facecolor='w',lw=0.75,label='Storm season'))    
    evtpl = ax.add_patch(Rectangle(xy=(0,0),width=0,height=0,facecolor='#929292',lw=0.75,label='Storm event'))   
    medpl, = ax.plot([StormsGrp['StartDate_min'].iloc[i],StormsGrp['EndDate_max'].iloc[i]], [0,1], 
            c='k', ls='--', dashes=(1,1), lw=0.75, label='Median wind gust')

    leg1 = ax.legend(handles=[sznpl,evtpl], loc='lower left', handlelength=0.5, handleheight=0.5, handletextpad=0.5)
    ax.add_artist(leg1)
    ax.legend(handles=[medpl], loc='lower right')
    # for legob in leg.legendHandles:
    #     legob.handle.setmarkersize(5)
    
    mpl.rcParams.update({'font.size':7})
    plt.tight_layout()

    figname = os.path.join(figpath,sitename+'_VedgeSat_UKStorms_Simple.png')
    plt.savefig(figname, bbox_inches='tight',dpi=300, transparent=True)
    print('figure saved under '+figname)
    
    plt.show()
    
    return

def VegStormsTimeSeries(figpath, sitename, CSVpath, TransectInterGDF, TransectIDs, Titles=None, Hemisphere='N', ShowPlot=True):
    """
    FM Oct 2023

    Parameters
    ----------
    figpath : str
        Path to folder to save plots.
    sitename : str
        Name of site of interest.
    CSVpath : str
        Path to CSV which stores storm timeline data.
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with veg edge lines.
    TransectIDs : list
        List of transect IDs to plot.
    Hemisphere : str, optional
        Northern (N) or Southern (S) Hemisphere for marking 'winter' season. The default is 'N'.
    ShowPlot : bool, optional
        Flag to turn plt.show() on or off (if plotting lots of transects). The default is True.


    """
    # Read in errors CSV
    StormsDF = pd.read_csv(CSVpath)
    StormsDF = StormsDF.iloc[::-1]
    StormsDF['Start'] = pd.to_datetime(StormsDF['Start'], format='%d/%m/%Y')
    StormsDF['End'] = pd.to_datetime(StormsDF['End'], format='%d/%m/%Y')
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    figID = ''
    
    if ShowPlot is False:
        plt.ioff()
    
    # if more than one Transect ID is to be compared on a single plot
    if type(TransectIDs) == list:
        # scaling for single column A4 page
        mpl.rcParams.update({'font.size':7})
        fig, axs = plt.subplots(len(TransectIDs),1,figsize=(6.55,4), dpi=300, sharex=True)
    else:
        TransectIDs = [TransectIDs]
        # scaling for single column A4 page
        mpl.rcParams.update({'font.size':7})
        # use 2 subplots with one empty to be able to loop through them
        fig, axs = plt.subplots(1,1,figsize=(6.55,4), dpi=300, sharex=True)
        axs = [axs] # to be able to loop through
        
    # common plot labels
    lab = fig.add_subplot(111,frameon=False)
    lab.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False, right=False)
    if type(TransectIDs) == list: 
        lab.set_xlabel('Date')#, labelpad=22)
    else:
        lab.set_xlabel('Date')
    lab.set_ylabel('Cross-shore distance (veg) (m)', color='#81A739')
    
    for TransectID, Title, ax in zip(TransectIDs, Titles, axs):
        daterange = [0,len(TransectInterGDF['dates'].iloc[TransectID])]
        plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDF['dates'].iloc[TransectID][daterange[0]:daterange[1]]]
        plotsatdist = TransectInterGDF['distances'].iloc[TransectID][daterange[0]:daterange[1]]
        # remove and interpolate outliers
        plotsatdistinterp = InterpNaN(plotsatdist)
        
        if len(plotdate) == 0:
            return
        
        plotdate, plotsatdist = [list(d) for d in zip(*sorted(zip(plotdate, plotsatdist), key=lambda x: x[0]))]    
        ax.grid(color=[0.7,0.7,0.7], ls=':', lw=0.5, zorder=0)        
                        
        # xaxis ticks as year with interim Julys marked
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        
        ax2 = ax.twinx()
        
        # scatter plot
        ax2.scatter(plotdate, plotsatdist, marker='o', c='#81A739', s=5, alpha=0.8, zorder=10, edgecolors='none', label=r'$VE$')

        # create error bar lines to fill between
        for axloop, errorRMSE, plotdist, col in zip([ax], [10.4], [plotsatdist], ['#81A739']):
            yerrorplus = [x + errorRMSE for x in plotdist]
            yerrorneg = [x - errorRMSE for x in plotdist]
            # axloop.fill_between(plotdate, yerrorneg, yerrorplus, color=col, alpha=0.3, edgecolor=None)
       
        # ax2.errorbar(plotdate, plotsatdist, yerr=errorRMSE, elinewidth=0.5, fmt='none', ecolor='#81A739')
        
        
        # create rectangles highlighting winter months (based on N or S hemisphere 'winter')
        for i in range(plotdate[0].year-1, plotdate[-1].year+1):
            if Hemisphere == 'N':
                rectWinterStart = mdates.date2num(datetime(i, 11, 1, 0, 0))
                rectWinterEnd = mdates.date2num(datetime(i+1, 3, 1, 0, 0))
            elif Hemisphere == 'S':
                rectWinterStart = mdates.date2num(datetime(i, 5, 1, 0, 0))
                rectWinterEnd = mdates.date2num(datetime(i, 9, 1, 0, 0))
            rectwidth = rectWinterEnd - rectWinterStart
            rect = mpatches.Rectangle((rectWinterStart, -2000), rectwidth, 4000, 
                                      fc=[0.3,0.3,0.3], ec=None, alpha=0.2)
            ax.add_patch(rect)
        winter = mpatches.Patch(fc=[0.3,0.3,0.3], ec=None, alpha=0.2, label='UK Winter')
          
        # plot trendlines (use interpolated version)
        vegav = MovingAverage(plotsatdistinterp, 3)
        if len(plotdate) >= 3:
            ax2.plot(plotdate, vegav, color='#81A739', lw=1.5, label=r'$VE$ 3pt Mov. Av.')
    
        # linear regression lines
        numx = mpl.dates.date2num(plotdate)
        for y, pltax, clr in zip([plotsatdist], [ax2], ['#3A4C1A']):
            m, c = np.polyfit(numx,y,1)
            polysat = np.poly1d([m, c])
            xx = np.linspace(numx.min(), numx.max(), 100)
            dd = mpl.dates.num2date(xx)
            pltax.plot(dd, polysat(xx), '--', color=clr, lw=1.5, zorder=10, label=r'$\Delta VE$ = '+str(round(m*365.25,2))+' m/yr')
    
            # Standard error for confidence intervals
            y_pred = polysat(numx)
            Resids = y - y_pred
            Resid_StErr = np.std(Resids)
            Conf = 0.99
            t_value = scipy.stats.t.ppf((1 + Conf) / 2, len(numx) - 2)
            mean_numx = np.mean(numx)
            n = len(numx)
            SE = Resid_StErr * np.sqrt(1/n + (numx - mean_numx)**2 / np.sum((numx - mean_numx)**2))
            # Confidence bands
            y_upper = y_pred + t_value * SE
            y_lower = y_pred - t_value * SE
            pltax.fill_between(numx, y_lower, y_upper, color=clr, edgecolor=None, alpha=0.3, label=r'$VE_{0.99CI}$')
    
    # Vertical lines marking storm events
        for Storm in range(len(StormsDF)):
            storm = ax.axvspan(xmin = StormsDF['Start'].iloc[Storm], xmax = StormsDF['End'].iloc[Storm], 
                       facecolor='#5B618A', alpha=0.7, label='UK Storm')
    
        ax2.title.set_text('Transect '+str(TransectID)+' - '+Title)
            
        ax2.set_ylim(min(plotsatdistinterp)-10, max(plotsatdistinterp)+30)
        ax2.set_xlim(min(plotdate)-timedelta(days=100),max(plotdate)+timedelta(days=100))
        
        leg1 = ax2.legend(loc=2, ncol=4)
        leg2 = ax.legend(handles=[winter,storm],loc=1, labelspacing=0.3, handletextpad=0)
        for patch, legwidth, legx in zip(leg2.get_patches(), [12,2], [0,6]):
            patch.set_width(legwidth)
            patch.set_x(legx)
        # weird zorder with twin axes; remove legend and plot on second axis
        leg1.remove()
        ax2.add_artist(leg1)
    
        ax.set_yticks([])
        ax2.yaxis.tick_left()
            
        
        figID += '_'+str(TransectID)
        plt.tight_layout()
    
    # Add alphabetical labels to corners of subplots
    ax_labels = list(string.ascii_lowercase[:3*axs.shape[0]])
    for ax, lab in zip(axs.flat, ax_labels):
        ax.text(0.0045, 0.071, '('+lab+')', transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(facecolor='w', edgecolor='k',pad=1.5))
        
    figname = os.path.join(outfilepath,sitename + '_SatVegStormTimeseries_Transect'+figID+'.png')
    
    plt.tight_layout()
            
    plt.savefig(figname, bbox_inches='tight')
    print('Plot saved under '+figname)
    
    plt.show()
    
    
    
def VEWLBWStormsTimeSeries(figpath, sitename, CSVpath, TransectInterGDFWater, TransectIDs, Titles=None, Hemisphere='N', ShowPlot=True):
    """
    FM Mar 2025

    Parameters
    ----------
    figpath : str
        Path to folder to save plots.
    sitename : str
        Name of site of interest.
    CSVpath : str
        Path to CSV which stores storm timeline data.
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with veg edge lines.
    TransectIDs : list
        List of transect IDs to plot.
    Hemisphere : str, optional
        Northern (N) or Southern (S) Hemisphere for marking 'winter' season. The default is 'N'.
    ShowPlot : bool, optional
        Flag to turn plt.show() on or off (if plotting lots of transects). The default is True.


    """
    # Read in errors CSV
    StormsDF = pd.read_csv(CSVpath)
    StormsDF = StormsDF.iloc[::-1]
    StormsDF['Start'] = pd.to_datetime(StormsDF['Start'], format='%d/%m/%Y')
    StormsDF['End'] = pd.to_datetime(StormsDF['End'], format='%d/%m/%Y')
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    figID = ''
    
    if ShowPlot is False:
        plt.ioff()
    
    # if more than one Transect ID is to be compared on a single plot
    if type(TransectIDs) == list:
        # scaling for single column A4 page
        mpl.rcParams.update({'font.size':7})
        fig, axs = plt.subplots(len(TransectIDs)*3,1,figsize=(6.55,7), sharex=True)
    else:
        TransectIDs = [TransectIDs]
        # scaling for single column A4 page
        mpl.rcParams.update({'font.size':7})
        # use 2 subplots with one empty to be able to loop through them
        fig, axs = plt.subplots(1,1,figsize=(6.55,4), sharex=True)
        axs = [axs] # to be able to loop through
    
    PlotVars = ['distances', 'wlcorrdist', 'beachwidth']
    PlotDateVars = ['dates','wldates','wldates']
    PlotColours = {'distances':'#81A739', 'wlcorrdist':'#0A1DAE', 'beachwidth':'#4F948A'}
    PlotLabels = {'distances':r'$VE$', 'wlcorrdist':r'$WL$', 'beachwidth':r'$BW$'}
    PlotLinRegColours = {'distances':'#3A4C1A','wlcorrdist':'#0A0D55', 'beachwidth':'#245151'}
    
    for TransectID, Title in zip(TransectIDs, Titles):
        for i, Var in enumerate(PlotVars):
            ax = axs[2 * i] if TransectID == TransectIDs[0] else axs[2 * i + 1]
            
            daterange = [0,len(TransectInterGDFWater[PlotDateVars[i]].iloc[TransectID])]
            plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDFWater[PlotDateVars[i]].iloc[TransectID][daterange[0]:daterange[1]]]
            plotsatdist = TransectInterGDFWater[Var].iloc[TransectID][daterange[0]:daterange[1]]
            # remove and interpolate outliers
            plotsatdistinterp = InterpNaN(plotsatdist)
            
            if len(plotdate) == 0:
                return
            
            plotdate, plotsatdist = [list(d) for d in zip(*sorted(zip(plotdate, plotsatdist), key=lambda x: x[0]))]    
            ax.grid(color=[0.7,0.7,0.7], ls=':', lw=0.5, zorder=0)        
                            
            # xaxis ticks as year with interim Julys marked
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            
            ax2 = ax.twinx()
            
            # scatter plot
            ax2.scatter(plotdate, plotsatdist, marker='o', c=PlotColours[Var], s=5, alpha=0.8, zorder=4, edgecolors='none', label=PlotLabels[Var])
    
            # create error bar lines to fill between
            for axloop, errorRMSE, plotdist, col in zip([ax], [10.4], [plotsatdist], [PlotColours[Var]]):
                yerrorplus = [x + errorRMSE for x in plotdist]
                yerrorneg = [x - errorRMSE for x in plotdist]
                # axloop.fill_between(plotdate, yerrorneg, yerrorplus, color=col, alpha=0.3, edgecolor=None)
           
            # ax2.errorbar(plotdate, plotsatdist, yerr=errorRMSE, elinewidth=0.5, fmt='none', ecolor='#81A739')
            
            
            # create rectangles highlighting winter months (based on N or S hemisphere 'winter')
            for i in range(plotdate[0].year-1, plotdate[-1].year+1):
                if Hemisphere == 'N':
                    rectWinterStart = mdates.date2num(datetime(i, 11, 1, 0, 0))
                    rectWinterEnd = mdates.date2num(datetime(i+1, 3, 1, 0, 0))
                elif Hemisphere == 'S':
                    rectWinterStart = mdates.date2num(datetime(i, 5, 1, 0, 0))
                    rectWinterEnd = mdates.date2num(datetime(i, 9, 1, 0, 0))
                rectwidth = rectWinterEnd - rectWinterStart
                rect = mpatches.Rectangle((rectWinterStart, -2000), rectwidth, 4000, 
                                          fc=[0.3,0.3,0.3], ec=None, alpha=0.2)
                ax.add_patch(rect)
            winter = mpatches.Patch(fc=[0.3,0.3,0.3], ec=None, alpha=0.2, label='UK Winter')
              
            # plot trendlines (use interpolated version)
            vegav = MovingAverage(plotsatdistinterp, 3)
            if len(plotdate) >= 3:
                ax2.plot(plotdate, vegav, color=PlotColours[Var], lw=1.5, label=PlotLabels[Var]+' 3pt Mov. Av.')
        
            # linear regression lines
            numx = mpl.dates.date2num(plotdate)
            for y, pltax, clr in zip([plotsatdist], [ax2], [PlotLinRegColours[Var]]):
                m, c = np.polyfit(numx,y,1)
                polysat = np.poly1d([m, c])
                xx = np.linspace(numx.min(), numx.max(), 100)
                dd = mpl.dates.num2date(xx)
                pltax.plot(dd, polysat(xx), '--', color=clr, lw=1.5, zorder=10, label=r'$\Delta$'+PlotLabels[Var]+' = '+str(round(m*365.25,2))+' m/yr')
        
                # Standard error for confidence intervals
                y_pred = polysat(numx)
                Resids = y - y_pred
                Resid_StErr = np.std(Resids)
                Conf = 0.99
                t_value = scipy.stats.t.ppf((1 + Conf) / 2, len(numx) - 2)
                mean_numx = np.mean(numx)
                n = len(numx)
                SE = Resid_StErr * np.sqrt(1/n + (numx - mean_numx)**2 / np.sum((numx - mean_numx)**2))
                # Confidence bands
                y_upper = y_pred + t_value * SE
                y_lower = y_pred - t_value * SE
                pltax.fill_between(numx, y_lower, y_upper, color=clr, edgecolor=None, alpha=0.3, label=PlotLabels[Var]+r'$_{0.99CI}$')
        
            # Vertical lines marking storm events
            for Storm in range(len(StormsDF)):
                storm = ax.axvspan(xmin = StormsDF['Start'].iloc[Storm], xmax = StormsDF['End'].iloc[Storm], 
                           facecolor='#5B618A', alpha=0.7, label='UK Storm')
        
            ax2.set_title('Transect '+str(TransectID)+' - '+Title, pad=2)
                
            ax2.set_ylim(min(plotsatdistinterp)-10, max(plotsatdistinterp)+30)
            ax2.set_xlim(min(plotdate)-timedelta(days=100),max(plotdate)+timedelta(days=100))
            
            leg1 = ax2.legend(loc=2, ncol=4)
            leg2 = ax.legend(handles=[winter,storm],loc=1, labelspacing=0.3, handletextpad=0)
            for patch, legwidth, legx in zip(leg2.get_patches(), [12,2], [0,6]):
                patch.set_width(legwidth)
                patch.set_x(legx)
            # weird zorder with twin axes; remove legend and plot on second axis
            leg1.remove()
            leg2.remove()
            ax2.add_artist(leg1)
            ax2.add_artist(leg2)
        
            ax.set_yticks([])
            ax2.yaxis.tick_left()
                
            
        figID += '_'+str(TransectID)
        plt.tight_layout()
    
    # common plot labels
    lab = fig.add_subplot(111,frameon=False)
    lab.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False, right=False)
    lab.set_xlabel('Date')
    for Var, Pos in zip(PlotVars,[0.83,0.52,0.23]):
        fig.text(0.04, Pos, 'Cross-shore distance (m)', va='center', ha='center', rotation='vertical', color=PlotColours[Var])
    
    # Add alphabetical labels to corners of subplots
    ax_labels = list(string.ascii_lowercase[:3*axs.shape[0]])
    for ax, lab in zip(axs.flat, ax_labels):
        ax.text(0.004, 0.12, '('+lab+')', transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(facecolor='w', edgecolor='k',pad=1.5))
        
    figname = os.path.join(outfilepath,sitename + '_SatVEWLBWStormTimeseries_Transect'+figID+'.png')
    
    plt.tight_layout()
            
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    print('Plot saved under '+figname)
    
    plt.show()


def WLStormsTimeSeriesErrors(figpath, sitename, CSVpath, TransectInterGDFWater, TransectIDs, BadDates, Titles=None, Hemisphere='N', ShowPlot=True):
    """
    FM Sep 2025

    Parameters
    ----------
    figpath : str
        Path to folder to save plots.
    sitename : str
        Name of site of interest.
    CSVpath : str
        Path to CSV which stores storm timeline data.
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of cross-shore transects intersected with veg edge lines.
    TransectIDs : list
        List of transect IDs to plot.
    Hemisphere : str, optional
        Northern (N) or Southern (S) Hemisphere for marking 'winter' season. The default is 'N'.
    ShowPlot : bool, optional
        Flag to turn plt.show() on or off (if plotting lots of transects). The default is True.


    """
    # Read in errors CSV
    StormsDF = pd.read_csv(CSVpath)
    StormsDF = StormsDF.iloc[::-1]
    StormsDF['Start'] = pd.to_datetime(StormsDF['Start'], format='%d/%m/%Y')
    StormsDF['End'] = pd.to_datetime(StormsDF['End'], format='%d/%m/%Y')
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    figID = ''
    
    if ShowPlot is False:
        plt.ioff()
    
    # if more than one Transect ID is to be compared on a single plot
    if type(TransectIDs) == list:
        # scaling for single column A4 page
        mpl.rcParams.update({'font.size':7})
        fig, axs = plt.subplots(len(TransectIDs)*2,1,figsize=(6.55,5.67), sharex=True)
    else:
        TransectIDs = [TransectIDs]
        # scaling for single column A4 page
        mpl.rcParams.update({'font.size':7})
        # use 2 subplots with one empty to be able to loop through them
        fig, axs = plt.subplots(1,1,figsize=(6.55,4), sharex=True)
        axs = [axs] # to be able to loop through
    
    PlotVars = ['distances','wlcorrdist']
    PlotDateVars = ['dates','wldates']
    PlotColours = {'distances':'#81A739','wlcorrdist':'#0A1DAE' }
    PlotLabels = {'distances':r'$VE$','wlcorrdist':r'$WL$'}
    PlotLinRegColours = {'distances':'#3A4C1A','wlcorrdist':'#0A0D55'}
    
    for TransectID, Title in zip(TransectIDs, Titles):
        for i, Var in enumerate(PlotVars):
            ax = axs[2 * i] if TransectID == TransectIDs[0] else axs[2 * i + 1]
            
            daterange = [0,len(TransectInterGDFWater[PlotDateVars[i]].iloc[TransectID])]
            # plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDFWater['dates'].iloc[TransectID][daterange[0]:daterange[1]]]
            plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDFWater[PlotDateVars[i]].iloc[TransectID][daterange[0]:daterange[1]]]
            plotsatdist = TransectInterGDFWater[Var].iloc[TransectID][daterange[0]:daterange[1]]
            # remove and interpolate outliers
            plotsatdistinterp = InterpNaN(plotsatdist)
            
            BadDTs = [datetime.strptime(x, '%Y-%m-%d') for x in BadDates[TransectID]]
            BadIDs = []
            for BadDT in BadDTs:
                try:
                    BadIDs.append(plotdate.index(BadDT))
                except:
                    continue
            
            if len(plotdate) == 0:
                return
            
            plotdate, plotsatdist = [list(d) for d in zip(*sorted(zip(plotdate, plotsatdist), key=lambda x: x[0]))]    
            ax.grid(color=[0.7,0.7,0.7], ls=':', lw=0.5, zorder=0)        
                            
            # xaxis ticks as year with interim Julys marked
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            
            ax2 = ax.twinx()
            
            # scatter plot
            ax2.scatter(plotdate, plotsatdist, 
                        marker='o', c=PlotColours[Var], s=5, alpha=0.8, zorder=4, edgecolors='none', 
                        label=PlotLabels[Var])
            # plot bad dates on scatter
            ax2.scatter([plotdate[BadID] for BadID in BadIDs],
                        [plotsatdist[BadID] for BadID in BadIDs],
                        marker='X', c='#A33530', s=10, alpha=0.8, zorder=4, edgecolors='none', 
                        label='Poor '+PlotLabels[Var])
            
            # create error bar lines to fill between
            for axloop, errorRMSE, plotdist, col in zip([ax], [10.4], [plotsatdist], [PlotColours[Var]]):
                yerrorplus = [x + errorRMSE for x in plotdist]
                yerrorneg = [x - errorRMSE for x in plotdist]
                # axloop.fill_between(plotdate, yerrorneg, yerrorplus, color=col, alpha=0.3, edgecolor=None)
           
            # ax2.errorbar(plotdate, plotsatdist, yerr=errorRMSE, elinewidth=0.5, fmt='none', ecolor='#81A739')
            
            
            # create rectangles highlighting winter months (based on N or S hemisphere 'winter')
            for i in range(plotdate[0].year-1, plotdate[-1].year+1):
                if Hemisphere == 'N':
                    rectWinterStart = mdates.date2num(datetime(i, 11, 1, 0, 0))
                    rectWinterEnd = mdates.date2num(datetime(i+1, 3, 1, 0, 0))
                elif Hemisphere == 'S':
                    rectWinterStart = mdates.date2num(datetime(i, 5, 1, 0, 0))
                    rectWinterEnd = mdates.date2num(datetime(i, 9, 1, 0, 0))
                rectwidth = rectWinterEnd - rectWinterStart
                rect = mpatches.Rectangle((rectWinterStart, -2000), rectwidth, 4000, 
                                          fc=[0.3,0.3,0.3], ec=None, alpha=0.2)
                ax.add_patch(rect)
            winter = mpatches.Patch(fc=[0.3,0.3,0.3], ec=None, alpha=0.2, label='UK Winter')
              
            # plot trendlines (use interpolated version)
            vegav = MovingAverage(plotsatdistinterp, 3)
            if len(plotdate) >= 3:
                ax2.plot(plotdate, vegav, color=PlotColours[Var], lw=1.5, label=PlotLabels[Var]+' 3pt Mov. Av.')
        
            # linear regression lines
            numx = mpl.dates.date2num(plotdate)
            for y, pltax, clr in zip([plotsatdist], [ax2], [PlotLinRegColours[Var]]):
                m, c = np.polyfit(numx,y,1)
                polysat = np.poly1d([m, c])
                xx = np.linspace(numx.min(), numx.max(), 100)
                dd = mpl.dates.num2date(xx)
                pltax.plot(dd, polysat(xx), '--', color=clr, lw=1.5, zorder=10, label=r'$\Delta$'+PlotLabels[Var]+' = '+str(round(m*365.25,2))+' m/yr')
        
                # Standard error for confidence intervals
                y_pred = polysat(numx)
                Resids = y - y_pred
                Resid_StErr = np.std(Resids)
                Conf = 0.99
                t_value = scipy.stats.t.ppf((1 + Conf) / 2, len(numx) - 2)
                mean_numx = np.mean(numx)
                n = len(numx)
                SE = Resid_StErr * np.sqrt(1/n + (numx - mean_numx)**2 / np.sum((numx - mean_numx)**2))
                # Confidence bands
                y_upper = y_pred + t_value * SE
                y_lower = y_pred - t_value * SE
                pltax.fill_between(numx, y_lower, y_upper, color=clr, edgecolor=None, alpha=0.3, label=PlotLabels[Var]+r'$_{0.99CI}$')
        
            # Vertical lines marking storm events
            for Storm in range(len(StormsDF)):
                storm = ax.axvspan(xmin = StormsDF['Start'].iloc[Storm], xmax = StormsDF['End'].iloc[Storm], 
                           facecolor='#5B618A', alpha=0.7, label='UK Storm')
        
            ax2.set_title('Transect '+str(TransectID)+' - '+Title, pad=2)
                
            ax2.set_ylim(min(plotsatdistinterp)-10, max(plotsatdistinterp)+30)
            ax2.set_xlim(min(plotdate)-timedelta(days=100),max(plotdate)+timedelta(days=100))
            
            leg1 = ax2.legend(loc=2, ncol=5)
            leg2 = ax.legend(handles=[winter,storm],loc=1, labelspacing=0.3, handletextpad=0)
            for patch, legwidth, legx in zip(leg2.get_patches(), [12,2], [0,6]):
                patch.set_width(legwidth)
                patch.set_x(legx)
            # weird zorder with twin axes; remove legend and plot on second axis
            leg1.remove()
            leg2.remove()
            ax2.add_artist(leg1)
            ax2.add_artist(leg2)
        
            ax.set_yticks([])
            ax2.yaxis.tick_left()
                
            
        figID += '_'+str(TransectID)
        plt.tight_layout()
    
    # common plot labels
    lab = fig.add_subplot(111,frameon=False)
    lab.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False, right=False)
    lab.set_xlabel('Date')
    for Var, Pos in zip(PlotVars,[0.83,0.52,0.23]):
        fig.text(0.04, Pos, 'Cross-shore distance (m)', va='center', ha='center', rotation='vertical', color=PlotColours[Var])
    
    # Add alphabetical labels to corners of subplots
    ax_labels = list(string.ascii_lowercase[:3*axs.shape[0]])
    for ax, lab in zip(axs.flat, ax_labels):
        ax.text(0.004, 0.12, '('+lab+')', transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(facecolor='w', edgecolor='k',pad=1.5))
        
    figname = os.path.join(outfilepath,sitename + '_SatWLStormTimeseriesErrors_Transect'+figID+'.png')
    
    plt.tight_layout()
            
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    print('Plot saved under '+figname)
    
    plt.show()

def CoregQuiver(sitename, coregstats):
    """
    Quiver plot of coregistration results from using AROSICS (Scheffler et al., 
    2017). Arrows are vectors of image shift calculated from the amount of 
    change in longitude (dX) and latitude (dY).
    FM Sept 2025

    Parameters
    ----------
    sitename : str
        Name of site.
    coregstats : dict
        Coregistration statistics after VedgeSat/CoastSat run, with datetime of 
        image acquisition, dX, dY, and reliability of shift calculated.

    Returns
    -------
    None.

    """
    
    coregDF = pd.DataFrame(coregstats)
    coregDF['date'] = pd.to_datetime(coregDF['datetime'], format='%Y-%m-%d %H:%M:%S.%f').dt.date
    coregDFsm = coregDF[['dX','dY','Reliability', 'date']].groupby(by='date').mean()
    coregDFsm['Shift'] = np.sqrt(np.array(coregDFsm['dX'])**2 + np.array(coregDFsm['dY']**2))
    coregDFsm.index = pd.to_datetime(coregDFsm.index)
    coregDFsm['month'] = coregDFsm.index.month

    coregMonth = coregDFsm[['Shift', 'dX','dY','Reliability', 'month']].groupby('month').mean()

    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)

    mpl.rcParams.update({'font.size':7})

    # Convert dates to numeric values for colormap
    date_nums = mdates.date2num(coregDFsm.index)
    norm = mcolors.Normalize(vmin=date_nums.min(), vmax=date_nums.max())
    cmap = cm.magma_r

    # Build the quiver plot
    fig, (ax, cax) = plt.subplots(nrows=2,figsize=(6.55,7.5), 
                      gridspec_kw={"height_ratios":[1, 0.05]})
    ax.quiver(
        np.zeros(len(coregDFsm)), np.zeros(len(coregDFsm)),  # all arrows start at origin
        coregDFsm['dX'], coregDFsm['dY'],
        color=cmap(norm(date_nums)),           # color by date
        angles='xy', scale_units='xy', scale=1, alpha=0.8
    )

    # Add inset axis (position relative to parent axis: [x0, y0, width, height])
    axins = inset_axes(ax, width="35%", height="35%", borderpad=4.5, loc="lower left")#borderpad=0.9
    axins2 = inset_axes(ax, width="100%", height="100%", borderpad=0, bbox_to_anchor=(0.58,0.575,0.35,0.25), bbox_transform=ax.transAxes)
    axbar = inset_axes(ax, width="100%", height="100%", borderpad=0, bbox_to_anchor=(0.58,0.075,0.35,0.35), bbox_transform=ax.transAxes)

    # Inset 1 (centre)
    axins.quiver(
        np.zeros(len(coregDFsm)), np.zeros(len(coregDFsm)),
        coregDFsm['dX'], coregDFsm['dY'],
        color=cmap(norm(date_nums)),
        angles='xy', scale_units='xy', scale=1, width=0.01, alpha=0.8
    )

    axins.axhline(0, color='k', linewidth=0.5)
    axins.axvline(0, color='k', linewidth=0.5)

    axins.set_xlim(-6, 6)
    axins.set_ylim(-6, 6)
    axins.set_xlabel("dX (m)")
    axins.set_ylabel("dY (m)")
    #axins.tick_params(bottom=False,top=True,left=False,right=True)
    #axins.tick_params(labelbottom=False,labeltop=True,labelleft=False,labelright=True)
    #axins.yaxis.set_label_position("right")
    #axins.xaxis.set_label_position("top")
    axins.set_aspect('equal')

    axins.tick_params(axis='both', which='both', labelsize=6)

    # Inset 2 (outliers)
    axins2.quiver(
        np.zeros(len(coregDFsm)), np.zeros(len(coregDFsm)),
        coregDFsm['dX'], coregDFsm['dY'],
        color=cmap(norm(date_nums)),
        angles='xy', scale_units='xy', scale=1, width=0.01, alpha=0.8
    )
    axins2.axhline(0, color='k', linewidth=0.5)
    axins2.axvline(0, color='k', linewidth=0.5)
    axins2.set_xlim(35, 65)
    axins2.set_ylim(-5, 5)
    axins2.set_xlabel("dX (m)")
    axins2.set_ylabel("dY (m)")
    axins2.set_aspect('equal')
    axins2.tick_params(axis='both', which='both', labelsize=6)

    # Add crosshairs at origin
    ax.axhline(0, color='k', linewidth=0.8)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.set_xlabel("dX (m)")
    ax.set_ylabel("dY (m)")
    ax.set_xlim(-65,65)
    ax.set_ylim(-65,65)

    #ax.set_facecolor('#E0E0E0')

    # Add colorbar with date formatting
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal', label="Acquisition Date")
    years = np.arange(coregDFsm.index.min().year, coregDFsm.index.max().year + 1)
    year_ticks = [mdates.date2num(np.datetime64(f"{y}-01-01")) for y in years]
    cbar.set_ticks(year_ticks)
    cbar.set_ticklabels(years)  # show just year

    # Bar chart inset
    axbar2 = axbar.twinx() # Create another axes that shares the same x-axis as ax.
    width = 0.4
    coregMonth['Shift'].plot(kind='bar', color='#5499DE', ax=axbar, width=width, position=1)
    coregMonth['Reliability'].plot(kind='bar', color='#C51B2F', ax=axbar2, width=width, position=0)
    axbar.set_xlim(-0.5,11.5)
    axbar.set_xlabel(None)
    axbar.set_ylabel(r'Mean Image Shift $\sqrt{dX^{2} + dY^{2}}$ (m)', color='#5499DE')
    axbar2.set_ylabel('Mean Reliability (%)', color='#C51B2F')
    axbar.set_xticklabels(list(calendar.month_abbr[1:]), rotation=0)

    ax.set_aspect('equal')
    plt.tight_layout()

    ax_labels = list(string.ascii_lowercase[:4])
    for axlab, lab, labloc in zip([ax,axins,axins2,axbar], ax_labels, ['upper left','upper left','upper right','upper right']):
        at = AnchoredText(f"({lab})",
                          prop=dict(size=6),
                          frameon=True,
                          loc=labloc,
                          pad=0.2,    # distance to the axes
                          borderpad=0.06)  # padding inside the box
        at.patch.set_facecolor('w')
        at.patch.set_edgecolor('k')
        axlab.add_artist(at)

    figname = os.path.join(outfilepath,sitename + '_CoregQuiver.png')
    plt.savefig(figname, bbox_inches='tight', dpi=300)
    print('Plot saved under '+figname)

    plt.show()
 
    
def CoregBar(sitename, coregstats):
    coregDF = pd.DataFrame(coregstats)
    coregDF['date'] = pd.to_datetime(coregDF['datetime'], format='%Y-%m-%d %H:%M:%S.%f').dt.date
    coregDFsm = coregDF[['dX','dY','Reliability', 'date']].groupby(by='date').mean()
    coregDFsm['Shift'] = np.sqrt(np.array(coregDFsm['dX'])**2 + np.array(coregDFsm['dY']**2))
    coregDFsm.index = pd.to_datetime(coregDFsm.index)
    coregDFsm['month'] = coregDFsm.index.month

    coregMonth = coregDFsm[['Shift', 'dX','dY','Reliability', 'month']].groupby('month').mean()

    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)

    mpl.rcParams.update({'font.size':7})

    fig, ax = plt.subplots(1,1,figsize=(3.31,3.5)) # Create matplotlib axes
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

    width = 0.4
    coregMonth['Shift'].plot(kind='bar', color='#5499DE', ax=ax, width=width, position=1)
    coregMonth['Reliability'].plot(kind='bar', color='#C51B2F', ax=ax2, width=width, position=0)

    ax.set_xlim(-0.5,11.5)
    ax.set_xlabel(None)
    ax.set_ylabel(r'Mean Image Shift $\sqrt{dX^{2} + dY^{2}}$ (m)')
    ax2.set_ylabel('Mean Reliability (%)')
    ax.set_xticklabels(list(calendar.month_abbr[1:]), rotation=0)

    plt.tight_layout()

    figname = os.path.join(outfilepath,sitename + '_CoregBar.png')
    plt.savefig(figname, bbox_inches='tight', dpi=300)
    print('Plot saved under '+figname)

    plt.show()
    

def muHist(sitename, muDF_path, Titles=None):
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    figID = ''

    # Read in CSV of timesteps and mu values for each transect to be plotted
    muDF = pd.read_csv(muDF_path)
    TransectIDs = muDF.columns[1:]
    # xmin = []
    # xmax = []
    # for TransectID in TransectIDs:
    #     xmin.append(muDF[TransectID].min())
    #     xmax.append(muDF[TransectID].max())
    # lims = (np.min(xmin), np.max(xmax))
    # scaling for single column A4 page: (6.55,6)
    mpl.rcParams.update({'font.size':7})
    # use 2 subplots with one empty to be able to loop through them
    # fig, axs = plt.subplots(1,1,figsize=(1.68,1.7), dpi=300, sharex=True)
    fig, axs = plt.subplots(len(TransectIDs),1,figsize=(1.88,3.1), dpi=300, sharex=True)
    # common x label
    fig.text(0.5,0.01,'$\mu$ (m/s$^2$)', ha='center', va='center')
    
    for ax, (Tr, TransectID) in zip(axs, enumerate(TransectIDs)):
        
        # Plot histogram with custom bins and colors
        # cbins = np.arange(-0.03,0.031,0.01)
        cbins=np.arange(-0.01,0.012,0.002)
        cm = plt.cm.get_cmap('seismic_r') # red-white-blue
        norm = mpl.colors.Normalize(vmin=cbins.min(), vmax=cbins.max())
        colors = [cm(norm(b)) for b in cbins[:-1]]
        
        # bins = np.arange(-0.02,0.02,0.001)
        bin_width = 0.001
        half_range = 0.02
        # Adjust bins to ensure 0 is in the center
        bins = np.arange(-half_range, half_range + bin_width, bin_width) - (bin_width / 2)        
        n, bins, patches = ax.hist(muDF[TransectID], bins=bins)

        for patch in patches:
            bin_mid = (patch.get_x() + patch.get_x() + patch.get_width()) / 2
            patch.set_facecolor(cm(norm(bin_mid)))  # Set color based on normalized bin position

        ax.set_xlim(-0.02,0.02)
        # ax.set_xlabel('Wave diffusivity (m/s$^2$)')
        if Titles is not None:
            ax.set_title('Transect '+str(TransectID)+' - '+Titles[Tr])
        else:
            ax.set_title('Transect '+str(TransectID))
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        
        figID += '_'+str(TransectID)
    
    plt.tight_layout()
    # plt.subplots_adjust(left=0.08,right=0.95, wspace=0, hspace=0.25)
    
    figname = os.path.join(outfilepath,sitename + '_WaveDiffusHist_Transect'+figID+'.png')
    plt.savefig(figname, bbox_inches='tight', dpi=300, transparent=True)
    print('Plot saved under '+figname)
        
    plt.show()
    
    
def QsHist(TransectInterGDFWave, Trs, sitename, Titles=None):
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    figID = ''

    # Read in CSV of timesteps and mu values for each transect to be plotted
    QsDF = TransectInterGDFWave[['WaveQs', 'WaveQsNet']].iloc[Trs]
    # xmin = []
    # xmax = []
    # for TransectID in TransectIDs:
    #     xmin.append(muDF[TransectID].min())
    #     xmax.append(muDF[TransectID].max())
    # lims = (np.min(xmin), np.max(xmax))
    # scaling for single column A4 page: (6.55,6)
    mpl.rcParams.update({'font.size':7})
    # use 2 subplots with one empty to be able to loop through them
    # fig, axs = plt.subplots(1,1,figsize=(1.68,1.7), dpi=300, sharex=True)
    # fig, axs = plt.subplots(len(Trs),1,figsize=(1.88,3.1), dpi=300, sharex=True)
    fig, axs = plt.subplots(len(Trs),1,figsize=(1.83,3.1), dpi=300, sharex=True)

    # common x label
    fig.text(0.5,0.01,'$Q_{s}$ (m$^3$/yr)', ha='center', va='center')
    
    for ax, (Tr, TransectID) in zip(axs, enumerate(Trs)):
        # Scale Qs_net to m3/yr for plotting
        QsDF['WaveQs'].iloc[Tr] = np.array(QsDF['WaveQs'].iloc[Tr])*31536000
        QsDF['WaveQsNet'].iloc[Tr] = QsDF['WaveQsNet'].iloc[Tr]*31536000
        # Plot histogram with custom bins and colors
        # cbins = np.arange(-0.03,0.031,0.01)
        # cbins=np.arange(-0.08,0.086,0.006)
        cbins=np.arange(-2600000,2800000,200000)
        cm = plt.cm.get_cmap('bwr_r') # red-white-blue
        norm = mpl.colors.Normalize(vmin=cbins.min(), vmax=cbins.max())
        colors = [cm(norm(b)) for b in cbins[:-1]]
        
        # bins = np.arange(-0.02,0.02,0.001)
        # bin_width = 0.01
        # half_range = 0.5
        bin_width = 0.01*31536000
        half_range = 0.5*31536000
        # Adjust bins to ensure 0 is in the center
        bins = np.arange(-half_range, half_range + bin_width, bin_width) - (bin_width / 2)        
        n, bins, patches = ax.hist(QsDF['WaveQs'].iloc[Tr], bins=bins)

        for patch in patches:
            bin_mid = (patch.get_x() + patch.get_x() + patch.get_width()) / 2
            patch.set_facecolor(cm(norm(bin_mid)))  # Set color based on normalized bin position

        # Plot Qs,net
        ax.axvline(QsDF['WaveQsNet'].iloc[Tr], c='k', alpha=0.3,
                   ls='--', lw=0.7)
        ax.text(x=QsDF['WaveQsNet'].iloc[Tr], y=np.max(n)-(0.1*np.max(n)),
                s=f"$Q_{{s,net}}$ = {QsDF['WaveQsNet'].iloc[Tr]:,.0f} m$^3$/yr",
                fontsize=6, ha='center')
        
        # ax.set_xticks(np.arange(-0.2,0.3,0.1))
        # ax.set_xticks(np.arange(-0.2,0.3,0.01), minor=True)
        ax.set_xticks(np.arange(-6000000,9000000,3000000))
        ax.set_xticks(np.arange(-6000000,9000000,300000), minor=True)
        # ax.set_yscale('log')
        ax.set_ylim(0,np.max(n))
        ax.set_xlim(-0.2*31536000,0.2*31536000)
        # ax.set_xlabel('Wave diffusivity (m/s$^2$)')
        if Titles is not None:
            ax.set_title('Transect '+str(TransectID)+' - '+Titles[Tr], pad=1)
        else:
            ax.set_title('Transect '+str(TransectID), pad=1)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        
        figID += '_'+str(TransectID)
    
    plt.tight_layout()
    # plt.subplots_adjust(left=0.08,right=0.95, wspace=0, hspace=0.25)
    
    figname = os.path.join(outfilepath,sitename + '_WaveSedFluxHist_Transect'+figID+'.png')
    plt.savefig(figname, bbox_inches='tight', dpi=300, transparent=True)
    print('Plot saved under '+figname)
        
    plt.show()
        
    
def TrWaveRose(sitename, TransectInterGDFWave, TransectIDs):
    """
    Wave rose plot (on polar projection) of wave direction, with wave height
    represented as a histogram.
    FM March 2024

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    TransectInterGDFWave : GeoDataFrame
        GeoDataFrame of transects intersected with wave hindcasts.
    TransectIDs : list
        List of transect IDs to plot.


    """
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    figID = ''

    # if more than one Transect ID is to be compared on a single plot
    if type(TransectIDs) == list:
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':6})
        fig, axs = plt.subplots(len(TransectIDs),1,figsize=(1.6,1.5), dpi=300, subplot_kw={'projection':'polar'})
        # fig, axs = plt.subplots(1, len(TransectIDs),figsize=(11.6,5.9), dpi=300, subplot_kw={'projection':'polar'})
    else:
        TransectIDs = [TransectIDs]
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':6})
        # use 2 subplots with one empty to be able to loop through them
        fig, axs = plt.subplots(1,1,figsize=(1.6,1.5), dpi=300, subplot_kw={'projection':'polar'})
        axs = [axs] # to be able to loop through
        
    for TransectID, ax in zip(TransectIDs,axs):
        plotwavedir = np.deg2rad(TransectInterGDFWave['WaveDir'].iloc[TransectID])
        plotwavehs = np.array(TransectInterGDFWave['WaveHs'].iloc[TransectID])
        
        # create stages for wave height breaks
        cmp = cm.get_cmap('YlGnBu')
        HsStages = [[0.00,0.25],
                    [0.25,0.50],
                    [0.50,0.75],
                    [0.75,1.00],
                    [1.00,100]]
        
        # initialise dict for dataframe
        plotL = {}
        colours = []
        for stepi, step in enumerate(HsStages):
            if stepi == 4:
                lab = f"> {step[0]:.2f}"
            else:
                lab = f"{step[0]:.2f} to {step[1]:.2f}"
            # mask each set of wave directions by the range of wave heights
            mask = [val > step[0] and val <= step[1] for val in plotwavehs]
            plotL[lab] = plotwavedir[mask]
            colours.append(cmp(step[0]))
        # to dataframe for plotting
        plotDF = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v, in plotL.items() ]))
        
        # For each wave height break, plot wave direction rose with 5deg bins
        # for i, c in zip(range(len(plotDF.columns)), np.arange(0,1,0.25)):
            # if i == 0: # first stage starts from 0 
        binsize = np.deg2rad(10)
        binset = np.arange(0,np.deg2rad(360)+binsize,binsize)
        WaveHist = ax.hist(plotDF, bins=binset, stacked=True, color=colours, label=plotDF.columns, linewidth=0.5)
        ax.set_ylim(ax.get_ylim()) # lock radius axis limit
        
        # Plot semicircle representing land/sea at shore orientation
        ShoreAngle = np.deg2rad(TransectInterGDFWave['ShoreAngle'].iloc[TransectID])
        ax.fill_between(np.linspace(ShoreAngle, ShoreAngle-np.pi, 100), 0, round(ax.get_ylim()[1],-1), 
                        fc='k', alpha=0.2, zorder=0)
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(c=[0.5,0.5,0.5], alpha=0.2, lw=0.5)
        ax.set_axisbelow(True)
        # Padding on tick labels is too big, N label plotted as text instead
        # Use max radius value as y loc, and then add 3% buffer on that
        ax.text(0, ax.get_ylim()[1]+(ax.get_ylim()[1]*0.03), 'N', c='k', ha='center')
        
        # ax.title.set_text('Transect '+str(TransectID)+' Wave Direction\n'+
                          # TransectInterGDFWave['dates'].iloc[TransectID][0]+' to '+
                          # TransectInterGDFWave['dates'].iloc[TransectID][-1])
        ax.title.set_text('Transect '+str(TransectID)) 
        
        figID += '_'+str(TransectID)
        plt.tight_layout()
        # TitleFont = mplfm.FontProperties(family='Arial', weight='bold', style='normal')
        leg = ax.legend(loc='center left', handlelength=1, handletextpad=0.5, title='Wave H$_s$ (m)', title_fontproperties={'weight':'bold'})
        
        # Adjust the location of the legend
        plt.draw()
        bb = leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted()) 
        xOffset = -0.2
        bb.x0 += xOffset
        bb.x1 += xOffset
        leg.set_bbox_to_anchor(bb, transform = ax.transAxes)
        
    figname = os.path.join(outfilepath,sitename + '_SatWaveDir_Transect'+figID+'.png')
    
    plt.tight_layout()

    plt.savefig(figname, bbox_inches='tight', dpi=300, transparent=True)
    print('Plot saved under '+figname)
    
    plt.show()
    

def FullWaveRose(sitename, outfilepath, WaveFilePath=None, PlotDates=None, XSlices=None,YSlices=None):
    """
    FM March 2024

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    outfilepath : str
        Filepath to save fig to.
    WaveFilePath : str, optional
        Filepath to load specific wave file from. The default is None.
    PlotDates : list, optional
        List of start and end datetimes for constraining plot. The default is None.


    """
    # outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    # if os.path.isdir(outfilepath) is False:
    #     os.mkdir(outfilepath)
    mpl.rcParams.update({'font.size':7})
   
    # Path to Copernicus wave file 
    if WaveFilePath is None:
        WavePath = os.path.join(os.getcwd(), 'Data', 'tides')
        WaveFilePath = glob.glob(WavePath+'/*'+sitename+'*.nc')
    
    
    WaveX, WaveY, SigWaveHeight, MeanWaveDir, PeakWavePer, WaveTime, StormEvents = Waves.ReadWaveFile(WaveFilePath)
    # Extended wave data so now need to slice back down for plotting
    if XSlices is not None or YSlices is not None:
        WaveX = WaveX[XSlices[0]:XSlices[1]]
        WaveY = WaveY[YSlices[0]:YSlices[1]]
        # clip down timeseries too
        if PlotDates is not None: 
            StartID = WaveTime.index(PlotDates[0])
            EndID = WaveTime.index(PlotDates[1])+1 # need to add 1 for including in slicing
            WaveTime = WaveTime[StartID:EndID]
            SigWaveHeight = SigWaveHeight[StartID:EndID, YSlices[0]:YSlices[1], XSlices[0]:XSlices[1]]
            MeanWaveDir = MeanWaveDir[StartID:EndID, YSlices[0]:YSlices[1], XSlices[0]:XSlices[1]]
            PeakWavePer = PeakWavePer[StartID:EndID, YSlices[0]:YSlices[1], XSlices[0]:XSlices[1]]
            StormEvents = StormEvents[StartID:EndID, YSlices[0]:YSlices[1], XSlices[0]:XSlices[1]]
        # Or just include whole timeseries and only clip down spatially
        else:
            SigWaveHeight = SigWaveHeight[:, YSlices[0]:YSlices[1], XSlices[0]:XSlices[1]]
            MeanWaveDir = MeanWaveDir[:, YSlices[0]:YSlices[1], XSlices[0]:XSlices[1]]
            PeakWavePer = PeakWavePer[:, YSlices[0]:YSlices[1], XSlices[0]:XSlices[1]]
            StormEvents = StormEvents[:, YSlices[0]:YSlices[1], XSlices[0]:XSlices[1]]
            
    
    # create stages for wave height breaks
    cmp = cm.get_cmap('YlGnBu')
    HsStages = [[0.00,0.25],
                [0.25,0.50],
                [0.50,0.75],
                [0.75,1.00],
                [1.00,2.00],
                [2.00,4.00]]
    # HsStages = [[0.0,0.5],
    #             [0.5,1.0],
    #             [1.0,1.5],
    #             [1.5,2.0],
    #             [2.0,2.5]]
    
    # scaling for single column A4 page: (6.55,6)
    # use 2 subplots with one empty to be able to loop through them
    fig, axs = plt.subplots(len(WaveY) - 1, len(WaveX), figsize=(1.8, 4.9), dpi=300, 
                            subplot_kw={'projection': 'polar'})
    # TO DO: Currently only works with one column, will need to figure out better way of looping
    for px in range(len(WaveX)):
        for ax, py in zip(axs, range(len(WaveY))):

            # Convert wave dirs to radians
            plotwavedir = np.deg2rad(MeanWaveDir[:,py,px])
            plotwavehs = np.array(SigWaveHeight[:,py,px])
        
            # initialise dict for dataframe
            plotL = {}
            colours = []
            for i, step in enumerate(HsStages):
                lab = f"{step[0]:.2f} to {step[1]:.2f}"
                # mask each set of wave directions by the range of wave heights
                mask = [val > step[0] and val <= step[1] for val in plotwavehs]
                plotL[lab] = plotwavedir[mask]
                colours.append(cmp(i/(len(HsStages)-1)))
            # to dataframe for plotting
            plotDF = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v, in plotL.items() ]))   
            
            # For each wave height break, plot wave direction rose with 5deg bins
            # for i, c in zip(range(len(plotDF.columns)), np.arange(0,1,0.25)):
                # if i == 0: # first stage starts from 0 
            binsize = np.deg2rad(10)
            binset = np.arange(0,np.deg2rad(360)+binsize,binsize)
            ax.hist(plotDF, bins=binset, stacked=True, color=colours, label=plotDF.columns)
                # else: # start each row with previous stage
                    # ax.bar(plotDF[plotDF.columns[i]], color=cmp(c), label=lab, bottom=plotDF[plotDF.columns[i-1]])
                
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_facecolor((1, 1, 1, 0.3))            
            ax.tick_params(axis='x',which='major', colors='w')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            for spine in ax.spines.values():
                spine.set_edgecolor('w')
            ax.grid(linestyle='-', lw=0.5, color='w')
            ax.set_axisbelow(True)
            # Padding on tick labels is too big, N label plotted as text instead
            # Use max radius value as y loc, and then add 3% buffer on that
            ax.text(0, ax.get_ylim()[1]+(ax.get_ylim()[1]*0.03), 'N', c='w', ha='center')    
                
    handles, labels = ax.get_legend_handles_labels()

    # TitleFont = mplfm.FontProperties(family='Arial', weight='bold', style='normal')
    leg = fig.legend(handles, labels, loc='center left', title='Wave H$_s$ (m)', title_fontproperties={'weight':'bold'})#, prop=TitleFont)
    # Adjust the location of the legend
    plt.draw()
    bb = leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted()) 
    xOffset = -0.3
    yOffset = -0.05
    bb.x0 += xOffset
    bb.x1 += xOffset
    bb.y0 += yOffset
    bb.y1 += yOffset
    leg.set_bbox_to_anchor(bb, transform = ax.transAxes)
    
    mpl.rcParams.update({'font.size':7})
    
    figname = os.path.join(outfilepath, sitename + '_CMEMSWaveDir_'+str(round(WaveY[0],3))+'_'+str(round(WaveX[0],3))+'.png')
    
    plt.tight_layout()
    # Set the figure background to fully transparent
    fig.patch.set_alpha(0)
    
    fig.savefig(figname, bbox_inches='tight')
    print('Plot saved under '+figname)
    
    plt.show()
    
    
def MeanWaveRose(sitename, outfilepath, WaveX, WaveY,
                 SigWaveHeight, MeanWaveDir,
                 WaveTime, StormEvents,
                 WaveFilePath=None, PlotDates=None, XSlices=None,YSlices=None):
    """
    FM Feb 2025

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    outfilepath : str
        Filepath to save fig to.
    WaveFilePath : str, optional
        Filepath to load specific wave file from. The default is None.
    PlotDates : list, optional
        List of start and end datetimes for constraining plot. The default is None.


    """
    # outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    # if os.path.isdir(outfilepath) is False:
    #     os.mkdir(outfilepath)
    mpl.rcParams.update({'font.size':7})
   
    
    # Extended wave data so now need to slice back down for plotting
    if XSlices is not None or YSlices is not None:
        WaveX = WaveX[XSlices[0]:XSlices[1]]
        WaveY = WaveY[YSlices[0]:YSlices[1]]
        # clip down timeseries too
        if PlotDates is not None: 
            StartID = WaveTime.index(PlotDates[0])
            EndID = WaveTime.index(PlotDates[1])+1 # need to add 1 for including in slicing
            WaveTime = WaveTime[StartID:EndID]
            SigWaveHeight = SigWaveHeight[StartID:EndID, YSlices[0]:YSlices[1], XSlices[0]:XSlices[1]]
            MeanWaveDir = MeanWaveDir[StartID:EndID, YSlices[0]:YSlices[1], XSlices[0]:XSlices[1]]
            StormEvents = StormEvents[StartID:EndID, YSlices[0]:YSlices[1], XSlices[0]:XSlices[1]]
        # Or just include whole timeseries and only clip down spatially
        else:
            SigWaveHeight = SigWaveHeight[:, YSlices[0]:YSlices[1], XSlices[0]:XSlices[1]]
            MeanWaveDir = MeanWaveDir[:, YSlices[0]:YSlices[1], XSlices[0]:XSlices[1]]
            StormEvents = StormEvents[:, YSlices[0]:YSlices[1], XSlices[0]:XSlices[1]]
    
    # Calculate mean values for all wave conds
    plotwavehs = np.nanmean(SigWaveHeight,axis=(1,2))
    plotwavedir = circmean(np.deg2rad(MeanWaveDir), axis=(1,2), nan_policy='omit')
    
    # create stages for wave height breaks
    cmp = cm.get_cmap('YlGnBu')
    HsStages = [[0.00,0.25],
                [0.25,0.50],
                [0.50,0.75],
                [0.75,1.00],
                [1.00,2.00],
                [2.00,4.00]]
    
    # scaling for single column A4 page: (6.55,6)
    
    fig, ax = plt.subplots(1,1, figsize=(1.26, 2.5), dpi=300, 
                            subplot_kw={'projection': 'polar'})
    
    # Convert wave dirs to radians
    # plotwavedir = np.deg2rad(MeanWaveDir_Mean)
    # plotwavehs = np.array(SigWaveHeight_Mean)

    # initialise dict for dataframe
    plotL = {}
    colours = []
    for i, step in enumerate(HsStages):
        lab = f"{step[0]:.2f} to {step[1]:.2f}"
        # mask each set of wave directions by the range of wave heights
        mask = [val > step[0] and val <= step[1] for val in plotwavehs]
        plotL[lab] = plotwavedir[mask]
        colours.append(cmp(i/(len(HsStages)-1)))
    # to dataframe for plotting
    plotDF = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v, in plotL.items() ]))   
    
    # For each wave height break, plot wave direction rose with 5deg bins
    # for i, c in zip(range(len(plotDF.columns)), np.arange(0,1,0.25)):
        # if i == 0: # first stage starts from 0 
    binsize = np.deg2rad(10)
    binset = np.arange(0,np.deg2rad(360)+binsize,binsize)
    ax.hist(plotDF, bins=binset, stacked=True, color=colours, label=plotDF.columns)
        # else: # start each row with previous stage
            # ax.bar(plotDF[plotDF.columns[i]], color=cmp(c), label=lab, bottom=plotDF[plotDF.columns[i-1]])
        
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_facecolor((1, 1, 1, 0.3))            
    ax.tick_params(axis='x',which='major', colors='w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_edgecolor('w')
    ax.grid(linestyle='-', lw=0.5, color='w')
    ax.set_axisbelow(True)
    # Padding on tick labels is too big, N label plotted as text instead
    # Use max radius value as y loc, and then add 3% buffer on that
    ax.text(0, ax.get_ylim()[1]+(ax.get_ylim()[1]*0.03), 'N', c='w', ha='center')    
                
    handles, labels = ax.get_legend_handles_labels()

    # TitleFont = mplfm.FontProperties(family='Arial', weight='bold', style='normal')
    leg = fig.legend(handles, labels, loc='lower center', title='Wave H$_s$ (m)', title_fontproperties={'weight':'bold'})#, prop=TitleFont)
    # Adjust the location of the legend
    plt.draw()
    bb = leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted()) 
    xOffset = 0
    yOffset = -1
    bb.x0 += xOffset
    bb.x1 += xOffset
    bb.y0 += yOffset
    bb.y1 += yOffset
    leg.set_bbox_to_anchor(bb, transform = ax.transAxes)
    
    mpl.rcParams.update({'font.size':7})
    
    figname = os.path.join(outfilepath, sitename + '_CMEMSWaveDir_Mean_'+str(round(WaveY[0],3))+'_'+str(round(WaveX[0],3))+'.png')
    
    plt.tight_layout()
    # Set the figure background to fully transparent
    fig.patch.set_alpha(0)
    
    fig.savefig(figname, bbox_inches='tight')
    print('Plot saved under '+figname)
    
    plt.show()
    
    
def FullWaveHsTimeseries(sitename, PlotDates=None):
    """
    IN DEVELOPMENT
    Plot wave height timeseries (with storm events marked).
    FM March 2024

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    PlotDates : list, optional
        List of two strings to limit the start and end date. The default is None.


    """
    # Path to Copernicus wave file 
    WavePath = os.path.join(os.getcwd(), 'Data', 'tides')
    WaveFilePath = glob.glob(WavePath+'/*'+sitename+'*.nc')
    
    if PlotDates is None:
        # If no plot start and end dates provided, plot whole timeseries from .nc file
        PlotDates = [WaveFilePath[0][-30:-20], WaveFilePath[0][-19:-9]] 
    
    WaveX, WaveY, SigWaveHeight, MeanWaveDir, PeakWavePer, WaveTime, StormEvents = Waves.ReadWaveFile(WaveFilePath)



    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    
    figname = os.path.join(outfilepath,sitename + '_SatWaveDir_'+PlotDates+'.png')
    
    plt.tight_layout()

    plt.savefig(figname, bbox_inches='tight')
    print('Plot saved under '+figname)
    
    plt.show()
    
    
def TidesSatPlot(sitename, output, dates, TidePath, OutFilePath):
    """
    Plot tidal elevations timeseries for requested site, with sat-derived 
    timings of tides on top.
    FM April 2024

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    output : dict
        Dictionary of sat-derived veg edges and waterlines.
    dates : list
        Start and end dates of study.
    TidePath : str
        Filepath to tide CSV file.
    OutFilePath : str
        Filepath to save fig to.


    """
    
    mpl.rcParams.update({'font.size':7})
    
    fig, ax = plt.subplots(1,1, figsize=(3.,1.72), dpi=300)
    
    TideData = pd.read_csv(TidePath)
    # date columns from string to datetime
    TideData['date'] = pd.to_datetime(TideData['date'])
    TideData['year'] = TideData['date'].dt.year
    
    # # For each year in list of unique years
    # for iyr, yr in enumerate(Yrs):
    #     # plot with different colours in ramp
    #     ax.plot(TideData['date'][TideData['year'] == yr], TideData['tide'][TideData['year'] == yr],
    #             c=colours[iyr], lw=0.5, label=None, zorder=1)
    
    # Initialise season year-to-ID dict
    SznCount = {}
    # apply season ID calculator to DataFrame
    TideData['season'] = TideData['date'].apply(GetSznID, args=(SznCount,))
        
    # Create colour map for storm years 
    cmp = cm.get_cmap('Blues')
    colours = []
    for i in range(len(TideData['season'].unique())):
        colours.append(cmp(i/len(TideData['season'].unique())))
        
    for SznID in TideData['season'].unique():
        # plot with different colours in ramp
        ax.plot(TideData['date'][TideData['season'] == SznID], TideData['tide'][TideData['season'] == SznID],
                c=colours[SznID], lw=0.5, label=None, zorder=1)
        
    SatDates = [datetime.strptime(output['dates'][isat]+' '+output['times'][isat],'%Y-%m-%d %H:%M:%S.%f')
                for isat in range(len(output['dates']))]
    SatData = pd.DataFrame({'dates':SatDates, 'tides':output['tideelev']})
    SatData['dates'] = pd.to_datetime(SatData['dates'])
    
    # Plot tide elevs coinciding with satellite images
    ax.scatter(SatData['dates'], SatData['tides'], s=1.5, c='k', zorder=2, label='Sat. image')
    
    # Upper annd lower limits of sat tides
    ax.axhline(SatData['tides'].min(), 0,1, c='k',ls='--',lw=1)
    ax.axhline(SatData['tides'].max(), 0,1, c='k',ls='--',lw=1)
    # Set ticks to yearly (and minor every half year)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1,7)))
    ax.set_yticks(np.arange(-3,3.5,1))
    ax.set_yticks(np.arange(-3,3.5,0.5), minor=True)
    ax.set(ylabel='Tidal elevation (m)')
    ax.set_xlim(TideData['date'].min(),TideData['date'].max())
    ax.set_ylim(-3.5,3.5)
    ax.set_facecolor('#B3B3B3')
    
    ax.legend(handletextpad=0)
    
    figname = os.path.join(OutFilePath, sitename + '_TidesSatTimeseries_'+dates[0]+'_'+dates[1]+'.png')
    
    plt.tight_layout()

    plt.savefig(figname, bbox_inches='tight',dpi=300,transparent=True)
    print('Plot saved under '+figname)
    
    plt.tight_layout()
    plt.show()
    
    
def TidesPlotAnnual(sitename, dates, TidePath, OutFilePath):
    """
    Plot tidal elevations timeseries for requested site, separating years out
    so they plot on top of one another.
    FM March 2024

    Parameters
    ----------
    sitename : str
        Name of site of interest.
    dates : list
        List of start and end dates as strings (from settings['inputs']['dates']).
    TidePath : str
        Path to where tides CSV is stored.
    OutFilePath : str
        Path to where the figure should be saved.


    """
    
    mpl.rcParams.update({'font.size':7})
    
    fig, ax = plt.subplots(1,1, figsize=(2.75,1.72), dpi=300)
    
    TideData = pd.read_csv(TidePath)
    # date column from string to datetime
    TideData['date'] = pd.to_datetime(TideData['date'])
    # # Plot individual years on top of one another
    # Parse out date and time into new column (year removed)
    TideData['time'] = TideData['date'].dt.time
    TideData['day'] = TideData['date'].dt.day
    TideData['month'] = TideData['date'].dt.month
    # New dates column with arbitrary year
    TideData['MDT'] = [datetime(1,
                                TideData['month'][itide], 
                                TideData['day'][itide], 
                                TideData['time'][itide].hour,
                                TideData['time'][itide].minute) for itide in range(len(TideData))]
    TideData['year'] = TideData['date'].dt.year
    
    cmp = cm.get_cmap('YlGnBu')
    colours = []
    Yrs = TideData['year'].unique()
    for i in range(len(Yrs)):
        colours.append(cmp(i/len(Yrs)))
    
    # For each year in list of unique years
    for iyr, yr in enumerate(Yrs):
        # Plot full years first (ignore first and last incomplete years)
        if len(TideData['MDT'][TideData['year'] == yr]) >= (24*365):
            ax.plot(TideData['MDT'][TideData['year'] == yr], TideData['tide'][TideData['year'] == yr], c=colours[iyr], lw=0.5, alpha=0.3, label=yr, zorder=iyr)
    # Then plot incomplete years
    for iyr, yr in enumerate(Yrs):
        if len(TideData['MDT'][TideData['year'] == yr]) < (24*365):
            ax.plot(TideData['MDT'][TideData['year'] == yr], TideData['tide'][TideData['year'] == yr], c=colours[iyr], lw=0.5, alpha=0.3, label=yr, zorder=iyr)
    
    # x axis ticks as months of one year
    # plt.xticks(ticks=range(0,(24*366),round((24*366)/12)),
    #            labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set(ylabel='Water elevation (m)')
    
    # Plotting incomplete years means legend order needs reset
    handles, labels = plt.gca().get_legend_handles_labels()
    legord = sorted(range(len(labels)), key=lambda k: labels[k])
    # Need to set zorder of actual legend as well (bring to front)
    plt.legend([handles[o] for o in legord], [labels[o] for o in legord]).set_zorder(len(Yrs)+1)
    
    figname = os.path.join(OutFilePath, sitename + '_TidesTimeseries_'+dates[0]+'_'+dates[1]+'.png')
    
    plt.tight_layout()

    plt.savefig(figname, bbox_inches='tight')
    print('Plot saved under '+figname)
    
    plt.show()
    
    
def GetSznID(Date, SznCount):
    """
    Calculate storm year and unique ID from list of dates, where storm year 
    runs from 01/09 to 31/08 the following year. Run using:
        SznCount = {}
        data_DF['date_column'].apply(GetSznID, args=(SznCount,))
    where SznCount is initialised outside of fn.
    FM Apr 2024

    Parameters
    ----------
    Date : Timestamp
        Pandas row in DataFrame representing date and time.
    SznCount : dict
        Empty dictitonary to be populated with storm year and matching season ID.

    Returns
    -------
    SznID : int
        Unique ID to be assigned to each row of DF depending on storm year date.

    """
    Year = Date.year
    # if date is after Sept, set season start to that year e.g. 01-09-2024 = 2024
    if Date.month >= 9:
        SznStart = Year
    # otherwise set season start to year before e.g. 31-08-2024 = 2023
    else:
        SznStart = Year - 1
    # Create season start year-to-ID dict
    SznID = SznCount.get(SznStart, None)
    if SznID is None:
        # Populate dict with season IDs matching each year
        SznCount[SznStart] = len(SznCount)
        SznID = SznCount[SznStart]
    # Return the ID calculated for each tide date
    return SznID


def ImageDateHist(OutFilePath, settings, sitename, output, metadata, satname='S2'):
     
    
    # Compile output dates into months
    DatesDF = pd.DataFrame(output['dates'], columns=['dates'])
    DatesDF['dates_dt'] = [datetime.strptime(x, '%Y-%m-%d') for x in DatesDF['dates']]
    DatesDF['month'] = DatesDF['dates_dt'].dt.month
    
    # Compile full image list of dates into months
    FullDatesDF = pd.DataFrame(metadata[satname]['dates'], columns=['dates'])
    FullDatesDF['dates_dt'] = [datetime.strptime(x, '%Y-%m-%d') for x in FullDatesDF['dates']]
    FullDatesDF['month'] = FullDatesDF['dates_dt'].dt.month
    
    # Counts of unsuccessful images due to cloud
    # Check for skipstats file
    if os.path.isfile(os.path.join(settings['inputs']['filepath'], sitename, sitename+'_skip_stats.pkl')):
        # if exists, get list of datetimes from cloudy key in pkl file
        with open(os.path.join(settings['inputs']['filepath'], sitename, sitename+'_skip_stats.pkl'),'rb',) as f:
            skipstats = pickle.load(f)
        cloudydates = [] 
        for cloudim in skipstats['cloudy']:
            cloudydates.append(cloudim[2][:10])
        ExceedDF = pd.DataFrame({'dates':cloudydates})
    else:
        # if no skipstats file exists, generate binary labels in metadata to 
        # mark whether cloud threshold was exceeded
        cloud_exceeded, cloud_score, empty_file = Toolbox.PercentageCloudy(metadata, settings)
        # Re-save metadata with cloud exceedance attached
        for satn in settings['satname']:
            metadata[satn]['cloud_exceed'] = cloud_exceeded
        with open(os.path.join(settings['inputs']['filepath'], sitename + '_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        CloudDF = pd.DataFrame({'dates':metadata[satname]['dates'], 
                                'cloud_exceed':metadata[satname]['cloud_exceed']})
        ExceedDF = CloudDF[CloudDF['cloud_exceed']==1]
        
    ExceedDF['dates_dt'] = [datetime.strptime(x, '%Y-%m-%d') for x in ExceedDF['dates']]
    ExceedDF['month'] = ExceedDF['dates_dt'].dt.month
    
    # Calculate image numbers per month
    counts = DatesDF['month'].value_counts().sort_index()
    fullcounts = FullDatesDF['month'].value_counts().sort_index()
    cloudcounts = ExceedDF['month'].value_counts().sort_index()
    # exceedcounts = cloudcounts + counts # add to successful images to stack bars
    
    f = plt.figure(figsize=(3.31, 3.31), dpi=300)
    mpl.rcParams.update({'font.size':7})
    ax = f.add_subplot(1,1,1)
    # ax.set_facecolor('#ECEAEC')
    
    ax.bar(fullcounts.index, fullcounts.values, 
           width=1, color='#FFFFFF', edgecolor='#6C8EBF', zorder=8, 
           label='Full S2 catalogue')
    cloudbar = ax.bar(cloudcounts.index, cloudcounts.values, 
           width=1, color=[0.75,0.75,0.8], edgecolor='#6C8EBF', zorder=9, 
           bottom=counts.values, label='Cloudy images')
    ax.bar(counts.index, counts.values, 
           width=1, color='#c4d8ff', edgecolor='#6C8EBF', zorder=10, #E8EFFC
           label='Suitable images')
    
    # label cloudy image bars
    cloudpcts = (cloudcounts.values / fullcounts.values)*100
    cloudlabels = [f"{cloudpct:.0f}%" for cloudpct in cloudpcts]
    ax.bar_label(cloudbar, labels=cloudlabels, label_type='center', color=[0.4,0.4,0.45], zorder=10)
    
    # create rectangles highlighting winter months (based on N or S hemisphere 'winter')
    rect1 = mpatches.Rectangle((10.5,0), 2, max(fullcounts.values)+5, fc=[0.3,0.3,0.3], ec=None, alpha=0.2, zorder=1)
    rect2 = mpatches.Rectangle((0.5,0), 2, max(fullcounts.values)+5, fc=[0.3,0.3,0.3], ec=None, alpha=0.2, zorder=1, label='UK Winter')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    ax.set_xticks(range(1,13))
    ax.set_xticklabels([calendar.month_abbr[i] for i in np.arange(1,13)])
    ax.set_xlim(0,13)
    ax.set_ylim(0,max(fullcounts.values)+5)
    ax.set_ylabel('Number of satellite images')
    
    ax.legend()

    plt.tight_layout()
    
    figpath = os.path.join(OutFilePath,sitename+'_SatImageMonth_Histogram.png')
    plt.savefig(figpath)
    print('figure saved under '+figpath)

    plt.show()
    
    
def AnnualStackTimeseries(sitename, TransectInterGDF, TransectInterGDFWater, TransectIDs, Titles):
    
    outfilepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(outfilepath) is False:
        os.mkdir(outfilepath)
    figID = ''
        
    # if more than one Transect ID is to be compared on a single plot
    if type(TransectIDs) == list:
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':7})
        fig, axs = plt.subplots(1,len(TransectIDs),figsize=(6.55,4), dpi=300, sharex=True)
    else:
        TransectIDs = [TransectIDs]
        # scaling for single column A4 page: (6.55,6)
        mpl.rcParams.update({'font.size':7})
        fig, axs = plt.subplots(1,1,figsize=(6.55,4), dpi=300, sharex=True)
        # axs = [axs] # to be able to loop through
            
    for TransectID, Title, axID in zip(TransectIDs, Titles, axs):
        # Define variables for each subplot per column/Transect
        # ax_WL = ax[0]
        # ax_VE = ax_WL.twinx()
        ax_VE = axs[axID]
        
        # Process plot data
        plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDF['dates'].iloc[TransectID]]
        plotsatdist = TransectInterGDF['distances'].iloc[TransectID]
        plotwldate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDFWater['wldates'].iloc[TransectID]]
        plotwldist = TransectInterGDFWater['wlcorrdist'].iloc[TransectID]
        
        plotdate, plotwldate, plotsatdist, plotwldist = [list(d) for d in zip(*sorted(zip(plotdate, plotwldate, plotsatdist, plotwldist), key=lambda x: x[0]))]    

        
        # for ax, plotT, plotD, in zip([ax_WL, ax_VE], 
        #                              [plotwldate, plotdate], 
        #                              [plotwldist,plotsatdist]):
        for ax, plotT, plotD, in zip([ax_VE], 
                                     [plotdate], 
                                     [plotsatdist]):
            
            plotDF = pd.DataFrame({'date':plotT, 'dist':plotD})
            
            plotDF.set_index('date', inplace=True)
            
            plotDF['year'] = plotDF.index.year
            plotDF['day_of_year'] = plotDF.index.dayofyear
            
            years = plotDF['year'].unique()
            Nyears = len(years)
            
            grouped = plotDF.groupby('year')
                        
            cmap = plt.get_cmap('Greens')
            colours = [cmap(i/Nyears) for i in range(Nyears)]
            
            for ic, pltyear in enumerate(years):
                plt.plot(plotDF['day_of_year'][plotDF['year']==pltyear],plotDF['dist'][plotDF['year']==pltyear],
                         c=colours[ic])

            
            # for (year, group), colour in zip(grouped, colours):
            #     ax.plot(group['day_of_year'], group['value'], label=str(year), color=colour)
        
        plt.grid(True)
        plt.show()
        
        
def PCAPlots(OutFilePath, sitename, MultivarGDF, ClassLabs=None, InsetLoc='upper left'):

    mpl.rcParams.update({'font.size':7})
    
    if ClassLabs is not None:
        MultivarGDF = MultivarGDF[ClassLabs.keys()]
    MultivarGDF.sort_index(inplace=True)
    # Tr indexes for figure name
    Loc1 = (MultivarGDF.index[0], MultivarGDF.index[int(len(MultivarGDF)/2)-1])
    Loc2 = (MultivarGDF.index[int(len(MultivarGDF)/2)], MultivarGDF.index[-1])
    
    MultivarGDF.reset_index(drop=True,inplace=True)
    # Standardise data columns
    MultivarGDFStd = StandardScaler().fit_transform(MultivarGDF)
    # Define and execute PCA
    pca = PCA(n_components=len(MultivarGDF.columns))
    PComps = pca.fit_transform(MultivarGDFStd)
    # Calculate variance of each component
    Var = pca.explained_variance_ratio_

    print('Explained Variance:')
    print(Var)

    
    # Find optimum number of components using a threshold for the explained variance
    CumVar = np.cumsum(Var)
    CompNum = np.argmax(CumVar >= 0.9) + 1
    # Rerun PCA with new optimum number of components
    pca = PCA(n_components=CompNum)
    PComps = pca.fit_transform(MultivarGDFStd)
    New_Var = pca.explained_variance_ratio_
    

    # Create dataframe of PCs
    PCA_DF = pd.DataFrame(data=PComps, columns=[f'PC{i+1}' for i in range(CompNum)])
    print('Principal Components:')
    print(PCA_DF)

    
    # Biplot
    # Classify plot based on eroding vs accreting veg
    if ClassLabs is not None:
        MultivarGDFbiplot = pd.DataFrame(data=MultivarGDF, columns=ClassLabs.values())
    else:
        MultivarGDFbiplot = pd.DataFrame(data=MultivarGDF, columns=MultivarGDF.columns)
    MultivarGDFbiplot['Class'] = 1
    MultivarGDFbiplot['Class'].iloc[:int(len(MultivarGDFbiplot)/2)] = 0 # First half contains eroding Trs
    # Separate features and labels
    labels = MultivarGDFbiplot['Class']
    # Classify the first two PCs
    biplotDF = pd.DataFrame(data=PCA_DF, columns=['PC1', 'PC2'])
    biplotDF['Class'] = labels
    # Plot the data and eigenvectors in PC space
    fig, ax1 = plt.subplots(figsize=(3.11, 3.11), dpi=300)
    # Horizontal and vertical lines through origin
    ax1.axvline(0, c=[0.5,0.5,0.5], lw=0.5, alpha=0.8, zorder=1)
    ax1.axhline(0, c=[0.5,0.5,0.5], lw=0.5, alpha=0.8, zorder=1)
    ax1.grid(c=[0.5,0.5,0.5], alpha=0.2, lw=0.5)
    # Scaling for coefficient vectors
    coeffs = np.transpose(pca.components_[0:2, :])
    n_coeffs = coeffs.shape[0]
    scalex = 0.9/(biplotDF['PC1'].max() - biplotDF['PC1'].min())
    scaley = 0.9/(biplotDF['PC2'].max() - biplotDF['PC2'].min())
    # Plot observations in principal component
    scatterlab = ['Eroding VE', 'Accreting VE']
    for clusterID, colour in enumerate(['#C51B2F','#5499DE']): # 0=eroding, 1=accreting
        ax1.scatter(biplotDF['PC1'][biplotDF['Class']==clusterID]*scalex, 
                    biplotDF['PC2'][biplotDF['Class']==clusterID]*scaley, 
                    s=10, c=colour, alpha=0.5, label=scatterlab[clusterID])
    # Plot eignevectors of each variable
    for i in range(n_coeffs):
        ax1.arrow(0, 0, coeffs[i,0], coeffs[i,1], color='k', alpha=0.5, head_width=0.03, zorder=5)
        # Use vector math to plot annotations along the direction of each arrow
        xtext = 2 + (coeffs[i,0] - 0) * 7
        ytext = 0 + (coeffs[i,1] - 0) * 7
        ax1.annotate(text=MultivarGDFbiplot.columns[i], xy=(coeffs[i,0], coeffs[i,1]), 
                     xytext=(xtext,ytext), textcoords='offset points',
                     color='k', ha='left', va='center', zorder=5,
                     path_effects=[PathEffects.withStroke(linewidth=1, foreground='w', alpha=0.7)])
    ax1.set_xlim(-1,1)
    ax1.set_ylim(-1,1)
    if 'left' in InsetLoc:
        ax1.yaxis.set_label_position('right')
        ax1.yaxis.set_ticks_position('right')
    ax1.set_xticks(np.arange(-1,1.5,0.5))
    ax1.set_yticks(np.arange(-1,1.5,0.5))
    # ax1.axis('equal')
    ax1.set_xlabel(f'PC1 [explains {round(Var[0]*100,1)}% of $\sigma^2$]')
    ax1.set_ylabel(f'PC2 [explains {round(Var[1]*100,1)}% of $\sigma^2$]')
    ax1.legend(loc='upper right', handletextpad=0.1)
    
    # Inset scree plot
    ax2 = inset_axes(ax1, width='45%', height='30%', loc=InsetLoc, borderpad=0)
    ax2.patch.set_alpha(0.5)
    varbar = ax2.bar(range(1, len(Var)+1), Var, facecolor=[0.5,0.5,0.5], alpha=0.5, align='center')
    for i in range(len(Var)):
        if i == 0:
            ax2.text(x=i+1, y=Var[i]-0.13, s=f'PC{i+1}', ha='center')
        else:
            ax2.text(x=i+1, y=Var[i]+0.03, s=f'PC{i+1}', ha='center')
    # ax2.bar_label(varbar, labels=[f'PC{i+1}' for i in range(len(Var))], label_type=labelloc)
    # ax2.step(range(1, len(Var)+1), np.cumsum(Var), where='mid', c='k', lw=1)
    ax2.plot(range(1, len(Var)+1), np.cumsum(Var), c='k', lw=1, marker='o', markersize=1.5)
    ax2.axhline(y=0.95, c='r', ls='--', lw=0.5)
    # plt.axhline(y=0.05, c='r', ls='--', lw=0.5)
    ax2.grid(axis='y', c=[0.5,0.5,0.5], alpha=0.2, lw=0.5)
    ax2.set_ylabel(r'Explained $\sigma^2$ (%)', labelpad=0.5, path_effects=[PathEffects.withStroke(linewidth=1, foreground='w', alpha=0.7)])
    if 'left' in InsetLoc:
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.set_ticks_position('right')
    ax2.set_yticks(np.arange(0,1,0.2))
    ax2.set_yticklabels(np.arange(0,100,20), path_effects=[PathEffects.withStroke(linewidth=1, foreground='w', alpha=0.7)])
    ax2.tick_params(axis='y', which='major', length=2, pad=2)
    ax2.set_xticks([])
    # for tic in ax2.yaxis.get_major_ticks():
    #     tic.tick1line.set_visible(False)
    #     tic.tick2line.set_visible(False)
    #     tic.label1.set_visible(False)
    #     tic.label2.set_visible(False)
    ax2.text(x=5, y=0.85, s='95%', c='r', ha='center')
    ax2.set_ylim(0,1)
    # Save full biplot fig
    plt.tight_layout()
    figpath = os.path.join(OutFilePath,sitename+'_PCABiplot_%s-%s_%s-%s.png' % 
                           (Loc1[0],Loc1[1],Loc2[0],Loc2[1]))
    plt.savefig(figpath)
    print('figure saved under '+figpath)
    plt.show()
    
    # Heatmap (using seaborn)
    # PlottingSeaborn.PCAHeatmap(pca, MultivarGDF, colnames)

    return PCA_DF


def WavesVsStorms(settings, CSVpath, WaveOutFile):
    
    # Get storms data from CSV
    StormsDF = pd.read_csv(CSVpath)
    StormsDF['starttime'] = pd.to_datetime(StormsDF['Start'], format='%d/%m/%Y')
    StormsDF['endtime'] = pd.to_datetime(StormsDF['End'], format='%d/%m/%Y')

    # Extract wave data from wave file
    WaveX, WaveY, SigWaveHeight, MeanWaveDir, PeakWavePer, WaveTime, StormEvents = Waves.ReadWaveFile(os.path.join(settings['inputs']['filepath'],
                                                                                                      'tides',
                                                                                                      WaveOutFile))
    # Wave data to dataframe to be paired with storm event (based on matching timing)
    WaveDF = pd.DataFrame({'time':WaveTime, 'hs':SigWaveHeight[:,1,0], 'dir':MeanWaveDir[:,1,0], 'tp':PeakWavePer[:,1,0]})
    WaveDF['storm'] = False
    WaveDF['storm_name'] = None
    WaveDF['storm_gust'] = None
    for _, storm in StormsDF.iterrows():
        mask = (WaveDF['time'] >= storm['starttime']) & (WaveDF['time'] <= storm['endtime'])
        WaveDF.loc[mask, 'storm'] = True
        WaveDF.loc[mask, 'storm_name'] = storm['Name']
        WaveDF.loc[mask, 'storm_gust'] = storm['WindGust']

    # Calculate 95th percentile of wave height for 'storm' limit
    pct = np.percentile(WaveDF['hs'], 95)
    fig, ax = plt.subplots()
    plt.plot(WaveDF['time'],WaveDF['hs'])
    plt.axhline(pct, 'k--')

    # Plot wave direction, wave period and wave height for normal wave conditions vs storm wave conditions
    fig, ax = plt.subplots()
    ax.hist(WaveDF['hs'], bins=np.arange(WaveDF['hs'].min(), WaveDF['hs'].max(), 0.1), 
            facecolor='b', label='Normal')
    ax2 = ax.twinx()
    ax2.hist(WaveDF[WaveDF['storm']==True]['hs'], bins=np.arange(WaveDF['hs'].min(), WaveDF['hs'].max(), 0.1),
             facecolor='r', alpha=0.5, label='Storm')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.hist(WaveDF['dir'], bins=np.arange(WaveDF['dir'].min(), WaveDF['dir'].max(), 5), 
            facecolor='b', label='Normal')
    ax2 = ax.twinx()
    ax2.hist(WaveDF[WaveDF['storm']==True]['dir'], bins=np.arange(WaveDF['dir'].min(), WaveDF['dir'].max(), 5),
             facecolor='r', alpha=0.5, label='Storm')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.hist(WaveDF['tp'], bins=np.arange(WaveDF['tp'].min(), WaveDF['tp'].max(), 0.5), 
            facecolor='b', label='Normal')
    ax2 = ax.twinx()
    ax2.hist(WaveDF[WaveDF['storm']==True]['tp'], bins=np.arange(WaveDF['tp'].min(), WaveDF['tp'].max(), 0.5),
             facecolor='r', alpha=0.5, label='Storm')
    plt.legend()
    plt.show()
    
    return WaveDF


def WaveHsHists(SigWaveHeight):
    
    
    cmp = cm.get_cmap('viridis')
    
    if SigWaveHeight.shape[1] > 1 and SigWaveHeight.shape[2] > 1:
        fig, axs = plt.subplots(nrows=SigWaveHeight.shape[1], ncols=SigWaveHeight.shape[2], figsize=(10,8), sharex=True)
        
        for y in range(SigWaveHeight.shape[1]):
            for x in range(SigWaveHeight.shape[2]):
                axs[y,x].hist(SigWaveHeight[:,y,x], bins=np.arange(0,4,0.1), color=cmp((y+1)/SigWaveHeight.shape[1]), alpha=0.5)
                axs[y,x].axvline(np.percentile(SigWaveHeight, 95, axis=0)[y, x], c=cmp((y+1)/SigWaveHeight.shape[1]), lw=2)
                axs[y,x].axvline(np.max(SigWaveHeight[:,y,x]), c=cmp((y+1)/SigWaveHeight.shape[1]), lw=2, ls='--')
        plt.xlabel('Significant wave height (m)')
        plt.tight_layout()
        plt.show()
        
    elif SigWaveHeight.shape[2] == 1: # only one column
        fig, axs = plt.subplots(nrows=SigWaveHeight.shape[1], ncols=SigWaveHeight.shape[2], figsize=(10,8), sharex=True)

        for y in range(SigWaveHeight.shape[1]):
            axs[y].hist(SigWaveHeight[:,y,0], bins=np.arange(0,4,0.1), color=cmp((y+1)/SigWaveHeight.shape[1]), alpha=0.5)
            axs[y].axvline(np.percentile(SigWaveHeight, 95, axis=0)[y], c=cmp((y+1)/SigWaveHeight.shape[1]), lw=2)
            axs[y].axvline(np.max(SigWaveHeight[:,y,0]), c=cmp((y+1)/SigWaveHeight.shape[1]), lw=2, ls='--')
        plt.xlabel('Significant wave height (m)')
        plt.tight_layout()
        plt.show()
        
    elif SigWaveHeight.shape[1] == 1: # only one row
        fig, axs = plt.subplots(nrows=SigWaveHeight.shape[1], ncols=SigWaveHeight.shape[2], figsize=(10,8), sharex=True)

        for x in range(SigWaveHeight.shape[2]):
            axs[x].hist(SigWaveHeight[:,0,x], bins=np.arange(0,4,0.1), color=cmp((x+1)/SigWaveHeight.shape[2]), alpha=0.5)
            axs[x].axvline(np.percentile(SigWaveHeight, 95, axis=0)[x], c=cmp((x+1)/SigWaveHeight.shape[2]), lw=2)
            axs[x].axvline(np.max(SigWaveHeight[:,0,x]), c=cmp((x+1)/SigWaveHeight.shape[2]), lw=2, ls='--')
        plt.xlabel('Significant wave height (m)')
        plt.tight_layout()
        plt.show()
