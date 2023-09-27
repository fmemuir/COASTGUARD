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
import pdb

import matplotlib as mpl
from matplotlib import cm
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.patches import Patch, Rectangle
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.ion()

import rasterio
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import scipy.stats

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.sans-serif'] = 'Arial'

from Toolshed import Toolbox

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




def VegTimeseries(sitename, TransectDict, TransectID, daterange):
    """
    

    Parameters
    ----------
    ValidDF : TYPE
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
    plt.xlabel('Date (yyyy-mm)')
    plt.ylabel('Cross-shore distance (m)')
    # plt.xlim(plotdate[0]-10, plotdate[-1]+10)
    plt.ylim(min(plotsatdist)-10, max(plotsatdist)+10)
    plt.tight_layout()
    
    plt.savefig(os.path.join(outfilepath,sitename + '_SatTimeseries_Transect'+str(TransectID)+'.png'))
    print('Plot saved under '+os.path.join(outfilepath,sitename + '_SatTimeseries_Transect'+str(TransectID)+'.png'))
    
    plt.show()
    
    
def VegWaterTimeseries(sitename, TransectInterGDF, TransectIDs, Hemisphere='N'):
    """
    

    Parameters
    ----------
    ValidDF : TYPE
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
    figID = ''
        
    if len(TransectIDs) > 1:
        # scaling for single column A4 page
        mpl.rcParams.update({'font.size':7})
        fig, axs = plt.subplots(len(TransectIDs),1,figsize=(6.55,6), dpi=300)
    else:
        # scaling for single column A4 page
        mpl.rcParams.update({'font.size':7})
        fig, axs = plt.subplots(2,1,figsize=(6.55,3), dpi=300, gridspec_kw={'height_ratios':[100,1]})
        
    for TransectID, ax in zip(TransectIDs,axs):
        daterange = [0,len(TransectInterGDF['dates'].iloc[TransectID])]
        plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDF['dates'].iloc[TransectID][daterange[0]:daterange[1]]]
        plotsatdist = TransectInterGDF['distances'].iloc[TransectID][daterange[0]:daterange[1]]
        plotwldist = TransectInterGDF['wldists'].iloc[TransectID][daterange[0]:daterange[1]]
        plotsatdist = np.array(plotsatdist)[(np.array(plotsatdist) < np.mean(plotsatdist)+40) & (np.array(plotsatdist) > np.mean(plotsatdist)-40)]
        
        plotdate, plotsatdist, plotwldist = [list(d) for d in zip(*sorted(zip(plotdate, plotsatdist, plotwldist), key=lambda x: x[0]))]    
        ax.grid(color=[0.7,0.7,0.7], ls=':', lw=0.5, zorder=0)        
        
        ax2 = ax.twinx()
        
        ax.scatter(plotdate, plotwldist, marker='o', c='#4056F4', s=6, alpha=0.8, edgecolors='none', label='Satellite Shoreline')
        ax2.scatter(plotdate, plotsatdist, marker='o', c='#81A739', s=6, alpha=0.8, edgecolors='none', label='Satellite VegEdge')
        
        # create error bar lines to fill between
        for axloop, errorRMSE, plotdist, col in zip([ax, ax2], [7.2, 10.4], [plotwldist,plotsatdist], ['#4056F4','#81A739']):
            yerrorplus = [x + errorRMSE for x in plotdist]
            yerrorneg = [x - errorRMSE for x in plotdist]
            axloop.fill_between(plotdate, yerrorneg, yerrorplus, color=col, alpha=0.3, edgecolor=None)
       
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
        vegav = movingaverage(plotsatdist, 3)
        wlav = movingaverage(plotwldist, 3)
        ax.plot(plotdate, wlav, color='#4056F4', lw=1, label='3pt Moving Average Shoreline')
        ax2.plot(plotdate, vegav, color='#81A739', lw=1, label='3pt Moving Average VegEdge')
    
        # linear regression lines
        x = mpl.dates.date2num(plotdate)
        for y, pltax, clr in zip([plotwldist,plotsatdist], [ax,ax2], ['#0A1DAE' ,'#3A4C1A']):
            m, c = np.polyfit(x,y,1)
            polysat = np.poly1d([m, c])
            xx = np.linspace(x.min(), x.max(), 100)
            dd = mpl.dates.num2date(xx)
            pltax.plot(dd, polysat(xx), '--', color=clr, lw=1, label=str(round(m*365.25,2))+' m/yr')
    
        if TransectID == 309:
            plt.title('Transect '+str(TransectID)+', Out Head')
        elif TransectID == 1575:
            plt.title('Transect '+str(TransectID)+', Reres Wood')
        else:
            plt.title('Transect '+str(TransectID))
        ax.set_xlabel('Date (yyyy-mm)')
        ax2.set_ylabel('Cross-shore distance (veg) (m)', color='#81A739')
        ax.set_ylabel('Cross-shore distance (water) (m)', color='#4056F4')
        # plt.xlim(plotdate[0]-10, plotdate[-1]+10)
        ax2.set_ylim(min(plotsatdist)-10, max(plotsatdist)+30)
        ax.set_ylim(min(plotwldist)-10, max(plotwldist)+30)
        ax.set_xlim(min(plotdate)-timedelta(days=100),max(plotdate)+timedelta(days=100))
        
        leg1 = ax.legend(loc=2)
        leg2 = ax2.legend(loc=1)
        # weird zorder with twinned axes; remove first axis legend and plot on top of second
        leg1.remove()
        ax2.add_artist(leg1)
        
        figID += '_'+str(TransectID)
        
    figname = os.path.join(outfilepath,sitename + '_SatVegWaterTimeseries_Transect'+figID+'.png')
    
    if not axs[1].lines:
        fig.delaxes(axs[1])
    plt.tight_layout()
            
    plt.savefig(figname)
    print('Plot saved under '+figname)
    
    plt.show()
    

def ValidTimeseries(sitename, ValidInterGDF, TransectID):
    """
    

    Parameters
    ----------
    ValidDF : TYPE
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
    satlabels = ['Satellite VegEdge','_nolegend_','_nolegend_','_nolegend_',]
    
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


def WidthTimeseries(sitename, TransectInterGDF, TransectID, daterange):
    """
    

    Parameters
    ----------
    ValidDF : TYPE
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
    
    plotdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDF['wldates'].iloc[TransectID][daterange[0]:daterange[1]]]

    plotwldate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDF['wldates'].iloc[TransectID][daterange[0]:daterange[1]]]
    plotvegdate = [datetime.strptime(x, '%Y-%m-%d') for x in TransectInterGDF['dates'].iloc[TransectID][daterange[0]:daterange[1]]]

    plotvegdist = TransectInterGDF['distances'].iloc[TransectID][daterange[0]:daterange[1]]
    plotwldist = TransectInterGDF['wlcorrdist'].iloc[TransectID][daterange[0]:daterange[1]]
    plotsatdist = TransectInterGDF['beachwidth'].iloc[TransectID][daterange[0]:daterange[1]]

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
    plt.xlabel('Date (yyyy-mm)')
    plt.ylabel('Cross-shore distance (m)')
    plt.ylim(-200,1000)
    plt.tight_layout()
    
    plt.savefig(os.path.join(outfilepath,sitename + '_SatTimeseries_Transect'+str(TransectID)+'.png'))
    print('Plot saved under '+os.path.join(outfilepath,sitename + '_SatTimeseries_Transect'+str(TransectID)+'.png'))
    
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
    
    
def ValidPDF(sitename, ValidGDF,DatesCol,ValidDF,TransectIDs,PlotTitle):    
    """
    Generate probability density function of validation vs sat lines
    FM 2023

    Parameters
    ----------
    sitename : TYPE
        DESCRIPTION.
    ValidationShp : TYPE
        DESCRIPTION.
    DatesCol : TYPE
        DESCRIPTION.
    ValidDF : TYPE
        DESCRIPTION.
    TransectIDs : TYPE
        DESCRIPTION.
    PlotTitle : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # font size 8 and width of 6.55in fit 2-column journal formatting
    plt.rcParams['font.size'] = 8  
    
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)

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
    
    
def SatRegress(sitename,SatGDF,DatesCol,ValidDF,TransectIDs,PlotTitle):
       
    
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


    f = plt.figure(figsize=(3.31, 4.31), dpi=300)
    mpl.rcParams.update({'font.size':7})
    ax = f.add_subplot(1,1,1)
    ax.set_facecolor('#ECEAEC')
    
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
        plt.scatter(valsrtclean[i], satsrtclean[i], color=cmap(i), s=2, alpha=0.4, edgecolors='none', zorder=2)
        # linear regression
        X = np.array(valsrtclean[i]).reshape((-1,1))
        y = np.array(satsrtclean[i])
        model = LinearRegression(fit_intercept=True).fit(X,y)
        r2 = model.score(X,y)
        
        valfit = np.linspace(0,round(np.max(valsrtclean[i])),len(valsrtclean[i])).reshape((-1,1))
        satfit = model.predict(valfit)

        plt.plot(valfit,satfit, c=cmap(i), alpha=0.8, linewidth=1.2, label=(satdateclean[i]+' R$^2$ = '+str(round(r2,2))), zorder=3)

    plt.legend(facecolor='#ECEAEC', framealpha=1, bbox_to_anchor=(0.5,0), loc='lower center', bbox_transform=f.transFigure, ncol=2)
    
    # overall linear regression
    valfull = [item for sublist in valsrtclean for item in sublist]
    satfull =[item for sublist in satsrtclean for item in sublist]
    X = np.array(valfull).reshape((-1,1))
    y = np.array(satfull)
    model = LinearRegression(fit_intercept=True).fit(X,y)
    r2 = model.score(X,y)
    
    valfit = np.linspace(0,round(np.max(valfull)),len(valfull)).reshape((-1,1))
    satfit = model.predict(valfit)

    # plot glowing background line for overall lin reg first
    plt.plot(valfit,satfit, c='w', linestyle='-', linewidth=1.6, alpha=0.7, zorder=3)
    plt.plot(valfit,satfit, c='#818C93', linestyle='--', linewidth=1.2, zorder=3)
    plt.text(valfit[-1],satfit[-1],'R$^2$ = '+str(round(r2,2)), c='#818C93', zorder=3, ha='right')

    plt.xlim(0,220)
    plt.ylim(0,220)
    
    plt.xlabel('Validation Veg Edge cross-shore distance (m)')
    plt.ylabel('Satellite Veg Edge cross-shore distance (m)')
    
    ax.set_aspect('equal')
    ax.set_anchor('N')
    plt.tight_layout()
    
    figpath = os.path.join(filepath,sitename+'_Validation_Satellite_Distances_LinReg_'+str(TransectIDs[0])+'to'+str(TransectIDs[1])+'.png')
    plt.savefig(figpath)
    print('figure saved under '+figpath)

    plt.show()
    
    
    # Print out unique dates and satnames    
    SatGDFNames = SatGDF.groupby(['dates']).max()
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
        
        print('Sat name: R^2')
        print(SatN, r2)
    
    
    
    
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
    mpl.rcParams.update({'font.size':26, 
                         'text.color':textcolor,
                         'axes.labelcolor':textcolor,
                         'xtick.color':textcolor,
                         'ytick.color':textcolor,
                         'font.sans-serif':'Avenir LT Std'})

    fig, ax = plt.subplots(figsize=(7.5,9),dpi=300)

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
    plt.plot([0,250],[0,250],c=[0.6,0.5,0.5], lw=2, linestyle='-', zorder=3, alpha=0.4)

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

        line = plt.plot(valfit,satfit, c=cmap(i), alpha=0.5, linewidth=3, label=(satdateclean[i]+' R$^2$ = '+str(round(r2,2))), zorder=3)
        lines.append(line)
        
    besti = r2s.index(max(r2s))
    worsti = r2s.index(min(r2s))
    
    # plt.text(valsrtclean[besti][-1], satsrtclean[besti][-1], satdateclean[besti]+' (R$^2$ = '+str(round(r2s[besti],2))+')', c=cmap(besti))
    # plt.text(valsrtclean[worsti][-1], satsrtclean[worsti][-1], satdateclean[worsti]+' (R$^2$ = '+str(round(r2s[worsti],2))+')', c=cmap(worsti))

    hands = [ lines[besti][0], lines[worsti][0] ]
    labs = [ satdateclean[besti]+'\nS2, R$^2$ = '+str(round(r2s[besti],2)), satdateclean[worsti]+'\nL5, R$^2$ = '+str(round(r2s[worsti],2)) ]
    plt.legend(hands,labs, loc='lower right',facecolor='#D5D5D5')
    
    # overall linear regression
    valfull = [item for sublist in valsrtclean for item in sublist]
    satfull =[item for sublist in satsrtclean for item in sublist]
    X = np.array(valfull).reshape((-1,1))
    y = np.array(satfull)
    model = LinearRegression(fit_intercept=True).fit(X,y)
    r2 = model.score(X,y)
    
    
    valfit = np.linspace(0,round(np.max(valfull)),len(valfull)).reshape((-1,1))
    satfit = model.predict(valfit)

    plt.plot(valfit,satfit, c='k', linestyle='--', linewidth=3, zorder=3)
    plt.text(valfit[-1],satfit[-1],'R$^2$ = '+str(round(r2,2))+'\nRMSE = 23 m', zorder=3, horizontalalignment='right')

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
    SatGDFNames = SatGDF.groupby(['dates']).max()
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
    RateArray = np.array([[ID,x, y] for ID, x, y in zip(TransectInterGDF['TransectID'],TransectInterGDF['oldyoungRt'],TransectInterGDF['oldyungRtW'])])
    # Remove outliers (set to nan then remove in one go below)
    RateArray[:,1] = np.where(RateArray[:,1] < 50, RateArray[:,1], np.nan)
    RateArray[:,1] = np.where(RateArray[:,1] > -50, RateArray[:,1], np.nan)
    RateArray[:,2] = np.where(RateArray[:,2] < 190, RateArray[:,2], np.nan)
    RateArray[:,2] = np.where(RateArray[:,2] > -190, RateArray[:,2], np.nan)
    # Remove any transects with nan values in either column
    RateArray = RateArray[~np.isnan(RateArray).any(axis=1)]
    # Fit k-means clustering to array of rates
    RateCluster = KMeans(n_clusters=8).fit_predict(RateArray[:,1:])
    
    fig, axs = plt.subplots(1,2, figsize=(5,5), dpi=200)
    # Plot array using clusters as colour map
    ax1 = axs[0].scatter(RateArray[:,1], RateArray[:,2], c=RateCluster, s=5, alpha=0.5, marker='.')
    ax2 = axs[1].scatter(RateArray[:,1], RateArray[:,2], c=RateArray[:,0], s=5, alpha=0.5, marker='.')
    
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
    

def MultivariateMatrix(sitename, TransectInterGDF,  TransectInterGDFWater, TransectInterGDFTopo, TransectInterGDFWave, Sloc, Nloc):
    
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'plots')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)
        
    ## Multivariate Plot
    # Subset into south and north transects
    RateArrayS = TransectInterGDF.iloc[Sloc[0]:Sloc[1]]
    RateArrayS = pd.concat([RateArrayS, 
                           TransectInterGDFWater.iloc[Sloc[0]:Sloc[1]],
                           TransectInterGDFTopo.iloc[Sloc[0]:Sloc[1]],
                           TransectInterGDFWave.iloc[Sloc[0]:Sloc[1]]])
    RateArrayS['LocLabel'] = 'blue'
    
    RateArrayN = TransectInterGDF.iloc[Nloc[0]:Nloc[1]]
    RateArrayN = pd.concat([RateArrayN, 
                           TransectInterGDFWater.iloc[Nloc[0]:Nloc[1]],
                           TransectInterGDFTopo.iloc[Nloc[0]:Nloc[1]],
                           TransectInterGDFWave.iloc[Nloc[0]:Nloc[1]]])
    RateArrayN['LocLabel'] = 'red'
    
    RateArray = pd.concat([RateArrayS, RateArrayN], axis=0)
    # Extract desired columns to an array for plotting
    RateArray = np.array(RateArray[['oldyoungRt','oldyungRtW','TZwidthmed','maxslope','LocLabel']])
    # = np.array([[ID, vrate, wrate, tz] for ID, vrate, wrate, tz in zip(RateArray['TransectID'],RateArray['oldyoungRt'],RateArray['oldyungRtW'],RateArray['TZwidthmed'])])

    fig, axs = plt.subplots(RateArray.shape[1]-1,RateArray.shape[1]-1, figsize=(6.55,6.55), dpi=300)
    
    # Plot matrix of relationships
    lab = [r'$\Delta$veg (m/yr)',r'$\Delta$shore (m/yr)',r'$TZwidth_{\eta}$ (m)',r'$slope_{max}$ ($\circ$)']
    for row in range(RateArray.shape[1]-1):
        for col in range(RateArray.shape[1]-1):
            # remove repeated plots on right hand side
            # for i in range(RateArray.shape[1]-1):
                # if row == i and col > i:
                    # fig.delaxes(axs[row,col])
            
            # if plot is same var on x and y, change plot to a histogram    
            if row == col:
                binnum = round(np.sqrt(len(RateArray)))+4
                axs[row,col].hist(RateArray[:int(len(RateArray)/2),row],binnum, color='blue', alpha=0.7,label='S')
                axs[row,col].hist(RateArray[int(len(RateArray)/2):,row],binnum, color='red', alpha=0.7,label='N')
                axs[row,col].legend(loc=2,fontsize=6)
            # otherwise plot scatter of each variable against one another
            else:
                axs[row,col].scatter(RateArray[:,row], RateArray[:,col], s=12, alpha=0.3, marker='.', c=RateArray[:,-1], edgecolors='none')
                axs[row,col].scatter(RateArray[:,row], RateArray[:,col], s=12, alpha=0.3, marker='.', c=RateArray[:,-1], edgecolors='none')
                
                # overall linear reg line
                z = np.polyfit(list(RateArray[:,row]), list(RateArray[:,col]), 1)
                poly = np.poly1d(z)
                order = np.argsort(RateArray[:,row])
                axs[row,col].plot(RateArray[:,row][order], poly(RateArray[:,row][order]), c='k', ls='--', lw=0.8)
                r, p = scipy.stats.pearsonr(list(RateArray[:,row]), list(RateArray[:,col]))
                stats = 'r = %.2f' % (r)
                # axs[row,col].text( RateArray[:,row][order][-1], poly(RateArray[:,row][order])[-1], stats, c='k', fontsize=5, ha='center')
                axs[row,col].text(0.2, 0.05, stats, c='k', fontsize=6, ha='center', transform = axs[row,col].transAxes)

                # linear regression lines
                S, N = [RateArray[:len(RateArrayS),row], RateArray[:len(RateArrayS),col]], [RateArray[len(RateArrayN):,row], RateArray[len(RateArrayN):,col]]
                for pos, Arr, regc in zip([0.3,0.6], [S,N], ['blue','red']):
                    zArr = np.polyfit(list(Arr[0]), list(Arr[1]), 1)
                    polyArr = np.poly1d(zArr)
                    orderArr = np.argsort(Arr[0])
                    # linear reg line
                    axs[row,col].plot(Arr[0][orderArr], polyArr(Arr[0][orderArr]), c=regc, ls='--', lw=0.8)
                    for i in range(RateArray.shape[1]-1):
                        if row == i and col > i:
                            # clear plots on RHS
                            axs[row,col].cla() 
                for pos, Arr, regc in zip([0.3,0.6], [S,N], ['blue','red']):
                    for i in range(RateArray.shape[1]-1):
                        if row == i and col > i:      
                            rArr, pArr = scipy.stats.pearsonr(list(Arr[0]), list(Arr[1]))
                            statsArr = 'r = %.2f , p = %.2f' % (rArr,pArr)
                            axs[row,col].text(0.5, pos, statsArr, c=regc, fontsize=6, ha='center')
                    
                        

            axs[row,col].set_xlabel(lab[row])
            axs[row,col].set_ylabel(lab[col])
            axs[row,col].axvline(x=0, c=[0.5,0.5,0.5], lw=0.5)
            axs[row,col].axhline(y=0, c=[0.5,0.5,0.5], lw=0.5)
            
            if lab[col] == r'$\Delta$veg (m/yr)' and lab[row] == r'$\Delta$shore (m/yr)' :
                axs[row,col].axis('equal')
            
            # turn off axes to tighten up layout
            # if col != 0 and row != RateArray.shape[1]-1: # first col and last row
            #     axs[row,col].set_xlabel(None)
            #     axs[row,col].set_ylabel(None)
                
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.6, hspace=0.5)
    
    figpath = os.path.join(filepath,sitename+'_MultivariateAnalysis.png')
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

    Returns
    -------
    None.

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
    axs[0].set_xticks(list(errorDF['veg'])[0::2], major=True)
    ax2.set_xticks(errorDF['nonveg'],minor=True)
    ax2.set_xticks(list(errorDF['nonveg'])[0::2], major=True)
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
    vy, _, _ = axs[1].hist(int_veg, bins=bins, density=True, color=vegc)
    nvy, _, _ = axs[1].hist(int_nonveg, bins=bins, density=True, color=nonvegc, alpha=0.75) 
    
    # plot WP threshold and peaks as dashed vertical lines on PDF
    axs[1].plot([thresh,thresh], [0,max(nvy)+5], color=threshc, lw=1, ls='--', label='$I_{0}$')
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
    
    axs[1].legend(loc='upper left',ncol=1)   
    
    
    # subplot labels
    axs[0].text(1-0.012,215-3,'A', ha='right', va='top', 
             bbox=dict(boxstyle='square', fc='w', ec='k'), zorder=5)
    
    axs[1].text(1-0.015,6.1-0.15,'B', ha='right', va='top', 
             bbox=dict(boxstyle='square', fc='w', ec='k'), zorder=5)
    
    plt.tight_layout()
    plt.show()
    
    figpath = os.path.join(filepath,sitename+'_VedgeSat_WP_Errors.png')
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

    Returns
    -------
    None.

    """
    mpl.rcParams.update({'font.size':7})
    
    # Read in errors CSV
    ErrorDF = pd.read_csv(CSVpath)
    # Remove 'Total' row
    ErrorDF.drop(ErrorDF[ErrorDF['Date'] == 'Total'].index, axis=0, inplace=True)
    
    # Take unique dates from veg edge shapefile
    VegLines = VegGDF.groupby(['dates']).max()
    
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
    msat, csat = np.polyfit(x,ErrorDF['Tides'],1)
    polysat = np.poly1d([msat, csat])
    xx = np.linspace(x.min(), x.max(), 100)
    plt.plot(xx, polysat(xx), c='#A5A5AF', linestyle='--', linewidth=1.2)
    
    # Plot properties
    plt.xticks(range(0,35,5))
    plt.yticks(np.arange(-2,2.5,0.5))
    plt.xlabel('RMSE (m)')
    plt.ylabel('Tide height (m)')
      
    # plt.legend(loc='lower left',ncol=2)
    plt.tight_layout()
    
    figname = os.path.join(figpath,sitename+'_VedgeSat_TideHeights_Errors.png')
    plt.savefig(figname)
    print('figure saved under '+figname)
    
    plt.show()
    
    return
    

def StormsTimeline(figpath, sitename, CSVpath):
    
    
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
    plt.savefig(figname)
    print('figure saved under '+figname)
    
    plt.show()
    
    return

