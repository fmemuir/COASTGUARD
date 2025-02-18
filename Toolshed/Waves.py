#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains functions specific to wave data extraction and hydrodynamic calculations.
Freya Muir  - University of Glasgow
"""

import os
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pickle

import netCDF4

from Toolshed import Toolbox, Download




def GetHindcastWaveData(settings, output, lonmin, lonmax, latmin, latmax):
    """
    Download command for CMEMS wave hindcast data. User supplies date range, AOI, username and password.
    
    FM, Oct 2021 (updated Aug 2023)

    Parameters
    ----------
    settings : dict
        Veg edge extraction tool settings (including user inputs).
    output : dict
        Output veg edges produced by model.
    lonmin, lonmax, latmin, latmax : float
        Bounding box coords.

    """
    
    print('Downloading wave data from CMEMS ...')   
    WavePath = os.path.join(settings['inputs']['filepath'],'tides')   
    
    # Buffer dates from output by 3 months
    DateMin = datetime.strftime(datetime.strptime(min(output['dates']), '%Y-%m-%d')-timedelta(days=90), '%Y-%m-%d %H:%M:%S')
    DateMax = datetime.strftime(datetime.strptime(max(output['dates']), '%Y-%m-%d'), '%Y-%m-%d %H:%M:%S')
    
    # NetCDF file will be a set of rasters at different times with different wave params
    # params get pulled out further down after downloading
    WaveOutFile = settings['inputs']['sitename']+'_'+DateMin[:10]+'_'+DateMax[:10]+'_waves.nc'
    
    # Add 2-cell buffer to bounding box
    latmin = latmin - 0.05
    lonmin = lonmin - 0.05
    latmax = latmax + 0.05
    lonmax = lonmax + 0.05
    # if file already exists, just return filepath to existing file
    if os.path.isfile(os.path.join(WavePath,WaveOutFile)):
        print('Wave data file already exists.')
    else:
        CMScmd = {'hind_fore':'hind',
                  'lonmin':lonmin, 'lonmax':lonmax, 'latmin':latmin, 'latmax':latmax, 
                  'DateMin':DateMin, 'DateMax':DateMax,
                  'WavePath':WavePath,'WaveOutFile':WaveOutFile}
        Download.CMSDownload(CMScmd)
        
    return WavePath, WaveOutFile    



def GetForecastWaveData(settings, DateMin, DateMax):
    """
    Download command for CMEMS wave forecast data. User supplies date range, AOI, username and password.
    
    FM, Oct 2021 (updated Aug 2023)

    Parameters
    ----------
    settings : dict
        Veg edge extraction tool settings (including user inputs).

    """
    
    print('Downloading wave data from CMEMS ...')   
    WavePath = os.path.join(os.getcwd(),'Data','tides')   
    
    # Get metadata from settings file
    sitename = settings['inputs']['sitename']
    lonmin, lonmax, latmin, latmax = Toolbox.GetBounds(settings)
    
    # NetCDF file will be a set of rasters at different times with different wave params
    # params get pulled out further down after downloading
    WaveOutFile = sitename+'_'+DateMin[:10]+'_'+DateMax[:10]+'_waves.nc'
    
    if os.path.isfile(os.path.join(WavePath, WaveOutFile)):
        print('Wave data file already exists.')
    else:
        CMScmd = {'hind_fore':'fore',
                  'lonmin':lonmin, 'lonmax':lonmax, 'latmin':latmin, 'latmax':latmax,
                  'DateMin':DateMin, 'DateMax':DateMax,
                  'WavePath':WavePath,'WaveOutFile':WaveOutFile}
        Download.CMSDownload(CMScmd)
    
    return WavePath, WaveOutFile


def ReadWaveFile(WaveFilePath):
    """
    Read wave data stored in NetCDF file (and fill any empty cells).
    FM March 2024

    Parameters
    -------
    WaveFilePath : str
        Path to wave timeseries NetCDF file.

    Returns
    -------
    WaveX : float
        Upper left x coord of wave grid cell.
    WaveY : TYPE
        Upper left y coord of wave grid cell.
    SigWaveHeight : list
        Timeseries of significant wave height (in metres).
    MeanWaveDir : list
        Timeseries of mean wave direction (in degrees from).
    PeakWavePer : list
        Timeseries of peak wave period (in seconds).
    WaveTime : list
        Timestamps of timeseries to match wave conditions.
    StormEvents : list
        Timeseries of bools with same length as timeseries (1 where wave heights
        exceed the 95th percentile for each individual cell's timeseries.)

    """
    # open the raster dataset to work on
    with netCDF4.Dataset(WaveFilePath) as WaveData:
    
        # spatial coords returned as arrays of lat and long representing boundaries of raster axis
        # can be rectangular, resulting in differently sized arrays, so transforming as two coordinate arrays doesn't work
        WaveX  = WaveData.variables['longitude'][:]
        WaveY  = WaveData.variables['latitude'][:]

        SigWaveHeight = WaveData.variables['VHM0'][:,:,:]  # Spectral significant wave height (Hs)
        MeanWaveDir = WaveData.variables['VMDR'][:,:,:] # Mean wave direction from (Dir)
        PeakWavePer = WaveData.variables['VTPK'][:,:,:] # Wave period at spectral peak (Tp)
        WaveSeconds = WaveData.variables['time'][:]
        
        #  Fill empty cells of each raster in stack using interpolation
        for t in range(SigWaveHeight.shape[0]):
            SigWaveHeight[t] = Toolbox.InterpolateRaster(SigWaveHeight[t])
            PeakWavePer[t] = Toolbox.InterpolateRaster(PeakWavePer[t])
            # catch empty wave data and fill with nan
            if MeanWaveDir[t].count()/MeanWaveDir[t].size > 0:
                MeanWaveDir[t] = Toolbox.InterpolateCircRaster(MeanWaveDir[t]) # Needs circular interpolation instead
            
        WaveTime = []
        for i in range(0,len(WaveSeconds)):
            if 'UK' in WaveData.institution:
                # European NW Shelf stored as 'seconds since 1970-01-01 00:00:00'
                WaveTime.append(datetime(1970,1,1,0,0,0)+timedelta(seconds=int(WaveSeconds[i])))
                # OLD # WaveTime.append(datetime.strptime(datetime.fromtimestamp(WaveSeconds.astype(int)[i]).strftime('%Y-%m-%d %H:%M:%S'),'%Y-%m-%d %H:%M:%S'))
            else:
                # Global Wave Reanalysis is stored as 'number of hours since 1950-01-01 00:00:00'
                WaveTime.append(datetime(1950,1,1,0,0,0)+timedelta(hours=int(WaveSeconds[i])))
                
        StormEvents = CalcStorms(WaveTime, SigWaveHeight)
        
    return WaveX, WaveY, SigWaveHeight, MeanWaveDir, PeakWavePer, WaveTime, StormEvents


def SampleWaves(settings, output, TransectInterGDF, WaveFilePath):
    """
    Function to extract wave information from Copernicus NWS data
    
    FM, Oct 2021 (updated Aug 2023)

    Parameters
    ----------
    settings : dict
        Veg edge extraction tool settings (including user inputs).
    output : dict
        Output veg edges (and waterlines) produced by VedgeSat.
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of transects with veg edge intersection info assigned.
    WaveFilePath : str
        Path to wave timeseries NetCDF file.

    Returns
    -------
    WaveDates : list of lists
        Datetimes of wave data observations (matched with unique satellite capture
        dates from output['dates']).
    WaveHs : list of lists
        Significant wave height at the time of each satellite image processed in run 
        (in metres).
    WaveDir : list of lists
        Mean wave direction (from) at the time of each satellite image processed in run 
        (in degrees clockwise from N).
    WaveTp : list of lists
        Peak wave period at the time of each satellite image processed in run 
        (in seconds).
    NormWaveHs : list of lists
        Mean/normalised significant wave height at the time of each satellite image processed in run
        (in metres).
    NormWaveDir : list of lists
        Mean/normalised wave direction (from) at the time of each satellite image processed in run
        (in degrees clockwise from N).
    NormWaveTp : list of lists
        Mean/normalised wave period at the time of each satellite image processed in run
        (in seconds).
    StDevWaveHs : list of lists
        Standard deviation of significant wave height at the time of each satellite image processed in run
        (in metres).
    StDevWaveDir : list of lists
        Standard deviation of mean wave direction (from) at the time of each satellite image processed in run
        (in degrees clockwise from N).
    StDevWaveTp : list of lists
        Standard deviation of peak wave period at the time of each satellite image processed in run
        (in seconds).
    WaveDiffusivity : list of floats
        Wave diffusivity calculated from entire wave timeseries available (in m/s^2). 
        From Ashton and Murray 2006: https://doi.org/10.1029/2005JF000423
    WaveStability : list of floats
        Wave stability calculated from entire wave timeseries available (dimensionless). 
        From Ashton and Murray 2006: https://doi.org/10.1029/2005JF000423
    ShoreAngles : list of floats
        Angle of shoreline at particular cross-shore transect 
        (in degrees clockwise from N).

    """
    
    print('Extracting wave data to transects ...')

    WaveX, WaveY, SigWaveHeight, MeanWaveDir, PeakWavePer, WaveTime, StormEvents = ReadWaveFile(WaveFilePath)

    # Calculate time step used for interpolating data between
    TimeStep = (WaveTime[1]-WaveTime[0]).total_seconds()/(60*60)    
    
    WaveDates = []
    WaveHs = []
    WaveDir = []
    WaveTp = []
    NormWaveHs = []
    NormWaveDir = []
    NormWaveTp = []
    StDevWaveHs = []
    StDevWaveDir = []
    StDevWaveTp = []
    WaveDiffusivity = []
    WaveStability = []
    ShoreAngles = []


    # more efficient to define centroids outside loop
    Centroids = TransectInterGDF.to_crs('4326').centroid
    # loop through transects and sample
    for Tr in range(len(TransectInterGDF)):
        print('\r %i / %i transects processed' % ( Tr, len(TransectInterGDF) ), end='')

        MidPnt = Centroids.iloc[Tr].coords[0] # midpoint of each transect
        
        # get index of closest matching grid square of wave data
        IDLat = (np.abs(WaveY - MidPnt[1])).argmin() 
        IDLong = (np.abs(WaveX - MidPnt[0])).argmin()
        
        # NEEDS WORK TO IMPLEMENT PROPERLY (11/07/24)
        # Is grid cell empty? Look for closest grid cell with data
        # if np.ma.count_masked(SigWaveHeight[:,IDLat,IDLong]) > 0:
        #     try: # catch if goes outside of lat long bounds
        #         # if grid square successfully found, save as new grid cell to use
        #         if np.ma.count_masked(SigWaveHeight[:,IDLat,IDLong+1]) == 0:
        #             IDLong = IDLong+1 
        #         elif np.ma.count_masked(SigWaveHeight[:,IDLat+1,IDLong]) == 0:
        #             IDLat = IDLat+1
        #         elif np.ma.count_masked(SigWaveHeight[:,IDLat+1,IDLong+1]) == 0:
        #             IDLat = IDLat+1
        #             IDLong = IDLong+1
        #     except:
        #         pass
        
        # Calculate shore angle at current transect using angle of transect (start and end points)
        ShoreAngle = CalcShoreAngle(TransectInterGDF, Tr)
        ShoreAngles.append(ShoreAngle)
        
        # Calculate wave climate indicators per transect over timeframe of provided date range
        # TrWaveDiffusivity, TrWaveStability = WaveClimate(ShoreAngle, SigWaveHeight[:,IDLat,IDLong], MeanWaveDir[:,IDLat,IDLong], PeakWavePer[:,IDLat,IDLong], WaveTime)
        # Above is old version; this one uses radians
        TrWaveDiffusivity, TrWaveStability = WaveClimateSimple(ShoreAngle, SigWaveHeight[:,IDLat,IDLong], MeanWaveDir[:,IDLat,IDLong], PeakWavePer[:,IDLat,IDLong], WaveTime)

        # Don't need below condition if waves are just using the same output['dates'] list each transect
        # InterPnts = TransectInterGDF['interpnt'].iloc[Tr] # line intersections on each transect
        # if transect intersect is empty i.e. no veg lines intersected, can't grab matching waves per sat image
        # if InterPnts == []: 
            # TrWaveHs, TrWaveDir, TrWaveTp, TrNormWaveHs, TrNormWaveDir,TrNormWaveTp, TrStDevWaveHs, TrStDevWaveDir, TrStDevWaveTp = (np.nan for i in range(9))
        
        # else:                       
            # per-transect wave data
        # TrWaveDates = []
        TrWaveHs = []
        TrWaveDir = []
        TrWaveTp = []
        TrNormWaveHs = []
        TrNormWaveDir = []
        TrNormWaveTp = []
        TrStDevWaveHs = []
        TrStDevWaveDir = []
        TrStDevWaveTp = []
                

        # for i in range(len(TransectInterGDF['dates'].iloc[Tr])): # for each VE date on each Transect
        DateTimeSat = [
            datetime.strptime(f"{date} {time}", '%Y-%m-%d %H:%M:%S.%f') 
            for date, time in zip(output['dates'], output['times'])
        ]
        for i in range(len(DateTimeSat)):
            # DateTimeSat = datetime.strptime(TransectInterGDF['dates'].iloc[Tr][i] + ' ' + TransectInterGDF['times'].iloc[Tr][i], '%Y-%m-%d %H:%M:%S.%f')
            # TrWaveDates.append(DateTimeSat[i]) # should total the same length of output['dates']
            # Interpolate wave data using number of minutes through the hour the satellite image was captured
            for WaveProp, WaveSat in zip([SigWaveHeight[:,IDLat,IDLong], MeanWaveDir[:,IDLat,IDLong], PeakWavePer[:,IDLat,IDLong]], 
                                         [TrWaveHs, TrWaveDir, TrWaveTp]):
                # if sat image date falls outside wave data window, assign nan
                if WaveTime[-1] < DateTimeSat[i]:
                    WaveSat.append(np.nan)
                else:
                    # find preceding and following hourly tide levels and times
                    Time_1 = WaveTime[Toolbox.find(min(item for item in WaveTime if item > DateTimeSat[i]-timedelta(hours=TimeStep)), WaveTime)]                        
                    Wave_1 = WaveProp[Toolbox.find(min(item for item in WaveTime if item > DateTimeSat[i]-timedelta(hours=TimeStep)), WaveTime)]
                    
                    Time_2 = WaveTime[Toolbox.find(min(item for item in WaveTime if item > DateTimeSat[i]), WaveTime)]
                    Wave_2 = WaveProp[Toolbox.find(min(item for item in WaveTime if item > DateTimeSat[i]), WaveTime)]
                    
                    # Find time difference of actual satellite timestamp (next wave timestamp minus sat timestamp)
                    TimeDiff = Time_2 - DateTimeSat[i]
                    # Get proportion of time back from the next 3-hour timestep
                    TimeProp = TimeDiff / timedelta(hours=TimeStep)
                    
                    # Get proportional difference between the two tidal stages
                    WaveDiff = (Wave_2 - Wave_1)
                    WaveSat.append(Wave_2 - (WaveDiff * TimeProp))

            for WaveProp, WaveSat, WaveType in zip([SigWaveHeight[:,IDLat,IDLong], MeanWaveDir[:,IDLat,IDLong], PeakWavePer[:,IDLat,IDLong]], 
                                               [TrNormWaveHs, TrNormWaveDir, TrNormWaveTp], ['Hs','Dir','Tp']):
                # if sat image date falls outside wave data window, assign nan
                if WaveTime[-1] < DateTimeSat[i]:
                    WaveSat.append(np.nan)
                else:
                    # Smooth over previous 3 month time period and get mean from this range
                    if Time_1-timedelta(days=90) in WaveTime:
                        Prev3Month = WaveTime.index(Time_1-timedelta(days=90))
                     # if timestep doesn't exist for exactly 3 months back, minus an hour
                    elif Time_1-timedelta(days=90,hours=1) in WaveTime:
                        Prev3Month = WaveTime.index(Time_1-timedelta(days=90,hours=1))
                     # if timestep doesn't exist for exactly 3 months back, add an hour
                    elif Time_1-timedelta(days=90,hours=-1) in WaveTime:
                        Prev3Month = WaveTime.index(Time_1-timedelta(days=90,hours=-1))
                    
                    if WaveType == 'Dir':
                        # if dealing with wave dir, use circular mean (to avoid problems with dirs around N i.e. 0deg)
                        SmoothWaveProp = Toolbox.CircMean(WaveProp[Prev3Month:WaveTime.index(Time_1)])
                    else:
                        SmoothWaveProp = np.mean(WaveProp[Prev3Month:WaveTime.index(Time_1)])
                    WaveSat.append(SmoothWaveProp)
                
            for WaveProp, WaveSat, WaveType in zip([SigWaveHeight[:,IDLat,IDLong], MeanWaveDir[:,IDLat,IDLong], PeakWavePer[:,IDLat,IDLong]], 
                                               [TrStDevWaveHs, TrStDevWaveDir, TrStDevWaveTp], ['Hs','Dir', 'Tp']):
                # if sat image date falls outside wave data window (only updated every 3 months or so), assign nan
                if WaveTime[-1] < DateTimeSat[i]:
                    WaveSat.append(np.nan)
                else:
                    # Smooth over previous 3 month time period and get stdev from this range
                    if WaveType == 'Dir':
                        # if dealing with wave dir, use circular std (to avoid problems with dirs around N i.e. 0deg)
                        StDevWaveProp = Toolbox.CircStd(WaveProp[Prev3Month:WaveTime.index(Time_1)])
                    else:
                        StDevWaveProp = np.std(WaveProp[Prev3Month:WaveTime.index(Time_1)])
                    WaveSat.append(StDevWaveProp)

        # append per-transect lists
        WaveDates.append(DateTimeSat)
        WaveHs.append(TrWaveHs)
        WaveDir.append(TrWaveDir)
        WaveTp.append(TrWaveTp)
        NormWaveHs.append(TrNormWaveHs)
        NormWaveDir.append(TrNormWaveDir)
        NormWaveTp.append(TrNormWaveTp)
        StDevWaveHs.append(TrStDevWaveHs)
        StDevWaveDir.append(TrStDevWaveDir)
        StDevWaveTp.append(TrStDevWaveTp)
        WaveDiffusivity.append(TrWaveDiffusivity)
        WaveStability.append(TrWaveStability)

    return WaveDates, WaveHs, WaveDir, WaveTp, NormWaveHs, NormWaveDir, NormWaveTp, StDevWaveHs, StDevWaveDir, StDevWaveTp, WaveDiffusivity, WaveStability, ShoreAngles


def SampleWavesSimple(settings, output, TransectInterGDF, WaveFilePath):
    """
    Function to extract wave information from Copernicus NWS data. Faster than
    SampleWaves() because once a grid cell of data from the Copernicus NetCDF is 
    processed (interpolated between timesteps to find exact value, mean and st dev calculated),
    the results are cached and just loaded in for the next transect that falls within
    that grid cell. 
    
    FM Nov 2024

    Parameters
    ----------
    settings : dict
        Veg edge extraction tool settings (including user inputs).
    output : dict
        Output veg edges (and waterlines) produced by VedgeSat.
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of transects with veg edge intersection info assigned.
    WaveFilePath : str
        Path to wave timeseries NetCDF file.

    Returns
    -------
    WaveDates : list of lists
        Datetimes of wave data observations (matched with unique satellite capture
        dates from output['dates']).
    WaveHs : list of lists
        Significant wave height at the time of each satellite image processed in run 
        (in metres).
    WaveDir : list of lists
        Mean wave direction (from) at the time of each satellite image processed in run 
        (in degrees clockwise from N).
    WaveTp : list of lists
        Peak wave period at the time of each satellite image processed in run 
        (in seconds).
    NormWaveHs : list of lists
        Mean/normalised significant wave height at the time of each satellite image processed in run
        (in metres).
    NormWaveDir : list of lists
        Mean/normalised wave direction (from) at the time of each satellite image processed in run
        (in degrees clockwise from N).
    NormWaveTp : list of lists
        Mean/normalised wave period at the time of each satellite image processed in run
        (in seconds).
    StDevWaveHs : list of lists
        Standard deviation of significant wave height at the time of each satellite image processed in run
        (in metres).
    StDevWaveDir : list of lists
        Standard deviation of mean wave direction (from) at the time of each satellite image processed in run
        (in degrees clockwise from N).
    StDevWaveTp : list of lists
        Standard deviation of peak wave period at the time of each satellite image processed in run
        (in seconds).
    WaveDiffusivity : list of floats
        Wave diffusivity calculated from entire wave timeseries available (in m/s^2). 
        From Ashton and Murray 2006: https://doi.org/10.1029/2005JF000423
    WaveStability : list of floats
        Wave stability calculated from entire wave timeseries available (dimensionless). 
        From Ashton and Murray 2006: https://doi.org/10.1029/2005JF000423
    ShoreAngles : list of floats
        Angle of shoreline at particular cross-shore transect 
        (in degrees clockwise from N).
    WaveDatesFD : list of lists
        Datetimes of wave data observations (defined daily usin start and end
        dates from output['dates']).
    WaveHsFD : list of lists
        Significant wave height for each transect calculated as daily mean 
        (in metres).
    WaveDir : list of lists
        Mean wave direction (from) for each transect calculated as daily circular mean  
        (in degrees clockwise from N).
    WaveTp : list of lists
        Peak wave period for each transect calculated as daily mean  
        (in seconds).

    """

    print('Loading wave data for extraction ...')
    
    # Read in wave data (and interpolate empty cells in rasters if need be)
    WaveX, WaveY, SigWaveHeight, MeanWaveDir, PeakWavePer, WaveTime, StormEvents = ReadWaveFile(WaveFilePath)

    # Extract unique satellite image dates list
    DateTimeSat = [datetime.strptime(f"{date} {time}", '%Y-%m-%d %H:%M:%S.%f')
                   for date, time in zip(output['dates'], output['times'])]
    # Generate daily timesteps between first and last output dates
    startDT = datetime.strptime(output['dates'][0]+' 00:00:00', '%Y-%m-%d %H:%M:%S')
    endDT = datetime.strptime(output['dates'][-1]+' 00:00:00', '%Y-%m-%d %H:%M:%S')+timedelta(days=1)
    # if wave timeseries is shorter than output date range, use smaller range
    WaveTimeDF = pd.Series(WaveTime)
    WaveTimeClip = WaveTimeDF[(WaveTimeDF >= startDT) & (WaveTimeDF < endDT)]
    DateTimeDaily = pd.date_range(start=WaveTimeClip.iloc[0],
                                  end=WaveTimeClip.iloc[-1],
                                  freq='D')

    # Get centroid locations of each transect (in lat long)
    Centroids = TransectInterGDF.to_crs('4326').centroid
    
    # Initialise caching and output dicts
    Cached = {}
    ResultsDict = {'WaveDates': [], 
                   'WaveHs': [],
                   'WaveDir': [],
                   'WaveTp': [],
                   'NormWaveHs': [],
                   'NormWaveDir': [],
                   'NormWaveTp': [],
                   'StDevWaveHs': [],
                   'StDevWaveDir': [],
                   'StDevWaveTp': [],
                   'WaveDiffusivity': [], # transect specific
                   'WaveStability': [], # transect specific
                   'ShoreAngles': [], # transect specific
                   'WaveDatesFD': [], # FD = full timeseries, daily median
                   'WaveHsFD': [],
                   'WaveDirFD': [],
                   'WaveTpFD': []}
    
    
    for Tr in range(len(TransectInterGDF)):
        print('\r %i / %i transects processed' % (Tr, len(TransectInterGDF)), end='')
        
        # Clculate the midpoint of each cross-shore transect geometry
        MidPnt = Centroids.iloc[Tr].coords[0]
        IDLat = (np.abs(WaveY - MidPnt[1])).argmin()
        IDLong = (np.abs(WaveX - MidPnt[0])).argmin()
        # Feed the matching wave grid lat and long IDs to be used as a key in the cached data dict
        GridCell = (IDLat, IDLong)

        # Calculate the angle of the shoreline at each transect (clockwise from north)
        ShoreAngle = CalcShoreAngle(TransectInterGDF, Tr)
        ResultsDict['ShoreAngles'].append(ShoreAngle)
        
        # Calculate WaveDiffusivity and WaveStability for each transect based on transect-specific ShoreAngle
        # Uses full wave timeseries rather than just sat obs
        TrWaveDiffusivity, TrWaveStability = WaveClimateSimple(ShoreAngle, 
                                                               SigWaveHeight[:, IDLat, IDLong], 
                                                               MeanWaveDir[:, IDLat, IDLong], 
                                                               PeakWavePer[:, IDLat, IDLong], 
                                                               WaveTime)
        # Append single values per-transect to results dictionary
        ResultsDict['WaveDiffusivity'].append(TrWaveDiffusivity)
        ResultsDict['WaveStability'].append(TrWaveStability)
        
        # If the grid cell is already present in the cached data file, 
        # don't re-calculate anything and just load in the wave values for that grid cell
        if GridCell in Cached:
            GridData = Cached[GridCell]
        else:
            # Otherwise, compute wave data for this grid cell
            # IDLat, IDLong = GridCell
            GridData = {
                'TrWaveDates': DateTimeSat,
                'TrWaveHs': SampleWavesSimple_interp(SigWaveHeight[:, IDLat, IDLong], WaveTime, DateTimeSat),
                'TrWaveDir': SampleWavesSimple_interp(MeanWaveDir[:, IDLat, IDLong], WaveTime, DateTimeSat),
                'TrWaveTp': SampleWavesSimple_interp(PeakWavePer[:, IDLat, IDLong], WaveTime, DateTimeSat),
                'TrNormWaveHs': SampleWavesSimple_norm(SigWaveHeight[:, IDLat, IDLong], WaveTime, DateTimeSat, 'Hs'),
                'TrNormWaveDir': SampleWavesSimple_norm(MeanWaveDir[:, IDLat, IDLong], WaveTime, DateTimeSat, 'Dir'),
                'TrNormWaveTp': SampleWavesSimple_norm(PeakWavePer[:, IDLat, IDLong], WaveTime, DateTimeSat, 'Tp'),
                'TrStDevWaveHs': SampleWavesSimple_stdev(SigWaveHeight[:, IDLat, IDLong], WaveTime, DateTimeSat, 'Hs'),
                'TrStDevWaveDir': SampleWavesSimple_stdev(MeanWaveDir[:, IDLat, IDLong], WaveTime, DateTimeSat, 'Dir'),
                'TrStDevWaveTp': SampleWavesSimple_stdev(PeakWavePer[:, IDLat, IDLong], WaveTime, DateTimeSat, 'Tp'),
                'TrWaveDatesFD': DateTimeDaily.to_list(),
                'TrWaveHsFD': SampleWavesSimple_daily(SigWaveHeight[:, IDLat, IDLong], 'Hs', WaveTime, DateTimeDaily),
                'TrWaveDirFD': SampleWavesSimple_daily(MeanWaveDir[:, IDLat, IDLong], 'Dir', WaveTime, DateTimeDaily),
                'TrWaveTpFD': SampleWavesSimple_daily(PeakWavePer[:, IDLat, IDLong], 'Tp', WaveTime, DateTimeDaily)
                }
            
            # Cache the results for this grid cell to avoid redundant calculations
            Cached[GridCell] = GridData
            
        # Append grid data results to output_results with the ShoreAngle-specific adjustments
        for key in ResultsDict:
            gridkey = 'Tr' + key
            if gridkey in GridData:
                ResultsDict[key].append(GridData[gridkey])
    
    # Return WaveDates, WaveHs, WaveDir, WaveTp, 
           # NormWaveHs, NormWaveDir, NormWaveTp, 
           # StDevWaveHs, StDevWaveDir, StDevWaveTp, 
           # WaveDiffusivity, WaveStability, ShoreAngles
    return tuple(ResultsDict[key] for key in ResultsDict)


def SampleWavesSimple_interp(WaveProp, WaveTime, DateTimeSat):
    interpolated_values = []

    for Dt in DateTimeSat:
        if WaveTime[-1] < Dt:
            interpolated_values.append(np.nan)
        else:
            # Find surrounding wave data points and times for interpolation
            Prev_t = max([t for t in WaveTime if t <= Dt], default=WaveTime[0])
            Next_t = min([t for t in WaveTime if t > Dt], default=WaveTime[-1])
            
            Prev_val = WaveProp[WaveTime.index(Prev_t)]
            Next_val = WaveProp[WaveTime.index(Next_t)]
            
            # Calculate interpolated wave value
            TimeFrac = (Dt - Prev_t) / (Next_t - Prev_t)
            interpolated_value = Prev_val + TimeFrac * (Next_val - Prev_val)
            interpolated_values.append(interpolated_value)
    
    return interpolated_values


def SampleWavesSimple_daily(WaveProp, PropName, WaveTime, DateTimeDaily):
    
    # Get dataframe of wave property and timestamps
    WavePropDF = pd.DataFrame({PropName:WaveProp}, index=WaveTime)
    # Clip down timeseries to just start and end of satellite output dates
    WavePropDFfilt = WavePropDF[(WavePropDF.index >= DateTimeDaily[0]) & (WavePropDF.index < DateTimeDaily[-1])]        
    # If wave property is direction, take daily circular mean instead
    if PropName == 'Dir':
        daily_values = WavePropDFfilt.resample('D')[PropName].apply(Toolbox.CircMean)
    # Otherwise calculate daily mean
    else:
        daily_values = WavePropDFfilt.resample('D')[PropName].mean()
    
    return daily_values.to_list()


def SampleWavesSimple_norm(WaveProp, WaveTime, DateTimeSat, WaveType):
    NormVals = []
    
    for Dt in DateTimeSat:
        # Find the 3-month period before sat_time
        start_t = Dt - timedelta(days=90)
        Indices = [i for i, t in enumerate(WaveTime) if start_t <= t <= Dt]
        
        if Indices:
            ValsIn = WaveProp[Indices]
            if WaveType == 'Dir':
                # Use circular mean for directions
                MeanVal = Toolbox.CircMean(ValsIn)
            else:
                MeanVal = np.mean(ValsIn)
            NormVals.append(MeanVal)
        else:
            NormVals.append(np.nan)
    
    return NormVals
    

def SampleWavesSimple_stdev(WaveProp, WaveTime, DateTimeSat, WaveType):
    StDVals = []

    for Dt in WaveTime:
        start_t = Dt - timedelta(days=90)
        Indices = [i for i, t in enumerate(WaveTime) if start_t <= t <= Dt]
        
        if Indices:
            ValsIn = WaveProp[Indices]
            if WaveType == 'Dir':
                # Use circular standard deviation for directions
                StDVal = Toolbox.CircStd(ValsIn)
            else:
                StDVal = np.std(ValsIn)
            StDVals.append(StDVal)
        else:
            StDVals.append(np.nan)
    
    return StDVals


def SampleWavesFuture(TransectInterGDF, WaveFilePath):
    """
    Function to extract wave prediction information from Copernicus NWS data. Faster than
    SampleWaves() because once a grid cell of data from the Copernicus NetCDF is 
    processed (interpolated between timesteps to find exact value, mean and st dev calculated),
    the results are cached and just loaded in for the next transect that falls within
    that grid cell. 
    
    FM Nov 2024

    Parameters
    ----------
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of transects with veg edge intersection info assigned.
    WaveFilePath : str
        Path to wave timeseries NetCDF file.

    Returns
    -------
    WaveDates : list of lists
        Datetimes of wave data observations
    WaveHs : list of lists
        Significant wave height at the time of each satellite image processed in run 
        (in metres).
    WaveDir : list of lists
        Mean wave direction (from) at the time of each satellite image processed in run 
        (in degrees clockwise from N).
    WaveTp : list of lists
        Peak wave period at the time of each satellite image processed in run 
        (in seconds).
    NormWaveHs : list of lists
        Mean/normalised significant wave height at the time of each satellite image processed in run
        (in metres).
    NormWaveDir : list of lists
        Mean/normalised wave direction (from) at the time of each satellite image processed in run
        (in degrees clockwise from N).
    NormWaveTp : list of lists
        Mean/normalised wave period at the time of each satellite image processed in run
        (in seconds).
    StDevWaveHs : list of lists
        Standard deviation of significant wave height at the time of each satellite image processed in run
        (in metres).
    StDevWaveDir : list of lists
        Standard deviation of mean wave direction (from) at the time of each satellite image processed in run
        (in degrees clockwise from N).
    StDevWaveTp : list of lists
        Standard deviation of peak wave period at the time of each satellite image processed in run
        (in seconds).
    WaveDiffusivity : list of floats
        Wave diffusivity calculated from entire wave timeseries available (in m/s^2). 
        From Ashton and Murray 2006: https://doi.org/10.1029/2005JF000423
    WaveStability : list of floats
        Wave stability calculated from entire wave timeseries available (dimensionless). 
        From Ashton and Murray 2006: https://doi.org/10.1029/2005JF000423
    ShoreAngles : list of floats
        Angle of shoreline at particular cross-shore transect 
        (in degrees clockwise from N).

    """

    print('Loading wave data for extraction ...')
    
    # Read in wave data (and interpolate empty cells in rasters if need be)
    WaveX, WaveY, SigWaveHeight, MeanWaveDir, PeakWavePer, WaveTime, StormEvents = ReadWaveFile(WaveFilePath)

    # Get centroid locations of each transect (in lat long)
    Centroids = TransectInterGDF.to_crs('4326').centroid
    
    # Initialise caching and output dicts
    Cached = {}
    ResultsDict = {'FutureWaveDates': [], 
                   'FutureWaveHs': [],
                   'FutureWaveDir': [],
                   'FutureWaveTp': [],
                   'FutureWaveDiffusivity': [], # transect specific
                   'FutureWaveStability': [], # transect specific
                   'ShoreAngles': []} # transect specific
    
    
    for Tr in range(len(TransectInterGDF)):
        print('\r %i / %i transects processed' % (Tr, len(TransectInterGDF)), end='')
        
        # Clculate the midpoint of each cross-shore transect geometry
        MidPnt = Centroids.iloc[Tr].coords[0]
        IDLat = (np.abs(WaveY - MidPnt[1])).argmin()
        IDLong = (np.abs(WaveX - MidPnt[0])).argmin()
        # Feed the matching wave grid lat and long IDs to be used as a key in the cached data dict
        GridCell = (IDLat, IDLong)

        # Calculate the angle of the shoreline at each transect (clockwise from north)
        ShoreAngle = CalcShoreAngle(TransectInterGDF, Tr)
        ResultsDict['ShoreAngles'].append(ShoreAngle)
        
        # Calculate WaveDiffusivity and WaveStability for each transect based on transect-specific ShoreAngle
        # Uses full wave timeseries rather than just sat obs
        TrWaveDiffusivity, TrWaveStability = WaveClimateSimple(ShoreAngle, 
                                                               SigWaveHeight[:, IDLat, IDLong], 
                                                               MeanWaveDir[:, IDLat, IDLong], 
                                                               PeakWavePer[:, IDLat, IDLong], 
                                                               WaveTime)
        # Append single values per-transect to results dictionary
        ResultsDict['WaveDiffusivity'].append(TrWaveDiffusivity)
        ResultsDict['WaveStability'].append(TrWaveStability)
        
        # If the grid cell is already present in the cached data file, 
        # don't re-calculate anything and just load in the wave values for that grid cell
        if GridCell in Cached:
            GridData = Cached[GridCell]
        else:
            # Otherwise, compute wave data for this grid cell
            # IDLat, IDLong = GridCell
            GridData = {
                'TrWaveDates': WaveTime,
                'TrWaveHs': SigWaveHeight[:, IDLat, IDLong],
                'TrWaveDir': MeanWaveDir[:, IDLat, IDLong],
                'TrWaveTp': PeakWavePer[:, IDLat, IDLong]}
            # Cache the results for this grid cell to avoid redundant calculations
            Cached[GridCell] = GridData
            
        # Append grid data results to output_results with the ShoreAngle-specific adjustments
        for key in ResultsDict:
            gridkey = 'Tr' + key
            if gridkey in GridData:
                ResultsDict[key].append(GridData[gridkey])
    
    # Return WaveDates, WaveHs, WaveDir, WaveTp, 
           # WaveDiffusivity, WaveStability, ShoreAngles
    return tuple(ResultsDict[key] for key in ResultsDict)


def WaveClimate(ShoreAngle, WaveHs, WaveDir, WaveTp, WaveTime):
    """
    Calculate indicators of wave climate per transect, following equations of
    Ashton & Murray (2006). 
    - Diffusivity (mu) varies with wave angle and represents the wave climate 
      that leads to either shoreline smoothing (+ve diffusivity, stability) or
      or growth of shoreline perturbations (-ve diffusivity, Stability)
    - Stability index (Gamma) represents wave angle with respect to shoreline
      orientation, with 1 = low-angle climate and -1 = high-angle climate
      
    FM March 2024

    Parameters
    ----------
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of transects with veg edge intersection info assigned.

    Returns
    -------
    WaveDiffusivity : float
        Wave climate indicating perturbation growth or smoothing.
    WaveStability : float
        Dimensionless measure of stability in offshore wave vs shore angles.

    """
    
    # for each transect in run
    # for Tr in range(len(TransectInterGDF)):
    # Set constant value for sig wave heights at 10m closure depth
    K2 = 0.15
    D = 10.
    
    TimeStep = np.mean(np.diff(WaveTime)).seconds    
    
    # Initialise per-wave diffusivity
    Mu = []
    # for each wave data point
    for i in range(len(WaveDir)):
        # Get smallest angle measured from clockwise and ignore shadowed waves
        Alpha = ((ShoreAngle - WaveDir[i]) + 180) % 360-180
        if Alpha > 0:
            # Wave shadowed = no wave energy = no diffusion effects
            # H0 = 0
            Mu.append(0.0)
        else:
            H0 = WaveHs[i]
            T = WaveTp[i]
            
            # Wave diffusivity (+ve = smoothing, -ve = growth)
            Term1 = (K2/D)# K2/D
            Term2 = T**(1./5.) # T^1.5
            Term3 = H0**(12./5.) # H0^12/5
            Term4 = abs(math.cos(Alpha))**(1./5.)*(math.cos(Alpha)/abs(math.cos(Alpha))) # cos^1/5(Alpha) [need to maintain sign]
            Term5 = (6./5.) * math.sin(Alpha)**2. # (6/5)sin^2(Alpha)
            Term6 = math.cos(Alpha)**2 # cos^2(Alpha)
            Mu.append( Term1 * Term2 * Term3 * (Term4 * (Term5 - Term6)) )

    # Net diffusivity (Mu_net) (m/s-2)
    WaveDiffusivity = np.sum([Mu_i * TimeStep for Mu_i in Mu]) / (len(Mu) * TimeStep)
    
    # Dimensionless Stability index (Gamma)
    WaveStability = np.sum([Mu_i * TimeStep for Mu_i in Mu]) / np.sum([abs(Mu_i) * TimeStep for Mu_i in Mu])
    
    return WaveDiffusivity, WaveStability


def WaveClimateSimple(ShoreAngle, WaveHs, WaveDir, WaveTp, WaveTime):
    """
    IN DEVELOPMENT
    Calculate indicators of wave climate per transect, following equations of
    Ashton & Murray (2006). 
    - Diffusivity (mu) varies with wave angle and represents the wave climate 
      that leads to either shoreline smoothing (+ve diffusivity, stability) or
      or growth of shoreline perturbations (-ve diffusivity, Stability)
    - Stability index (Gamma) represents wave angle with respect to shoreline
      orientation, with 1 = low-angle climate and -1 = high-angle climate
    Simplified version of WaveClimate() with angles in radians instead.
      
    FM Oct 2024

    Parameters
    ----------
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of transects with veg edge intersection info assigned.

    Returns
    -------
    WaveDiffusivity : float
        Wave climate indicating perturbation growth or smoothing.
    WaveStability : float
        Dimensionless measure of stability in offshore wave vs shore angles.

    """
    
    # for each transect in run
    # for Tr in range(len(TransectInterGDF)):
    K2 = 0.15 # Ashton & Murray (2006) value for significant wave heights
    D = 10. # average estimated depth of closure
    
    # Time interval between wave observations
    TimeStep = np.mean(np.diff(WaveTime)).seconds
    
    # Convert shore angle and wave directions to radians
    # theta_rad = np.radians(ShoreAngle)
    # Phi_0_rad = np.radians(WaveDir)
    # Calculate the angle difference (theta - Phi_0) in degrees
    Alpha = (ShoreAngle - WaveDir + 180) % 360 - 180  # Compute angle diff in degrees
      
    # Initialize an array to store mu values, applying shadowing condition
    mu_values = []
    for i in range(len(WaveDir)):        
        if Alpha[i] <= 0:  # Only include waves that are onshore (angle_diff <= 0)
            # Calculate the diffusivity (mu) using the formula for onshore waves
            # abs() value used to avoid NaNs from raising a negative number to a decimal power
            mu = (K2 / D) * (WaveTp[i]**(1/3)) * (WaveHs[i]**(12/5)) * \
                 (abs(np.cos(np.radians(Alpha[i])))**(1/5)) * \
                 ((6/5) * np.sin(np.radians(Alpha[i]))**2 - np.cos(np.radians(Alpha[i]))**2)

            mu_values.append(mu)
        else:
            # Set mu to zero for offshore waves (shadowed conditions)
            mu_values.append(0.0)
    mu_values = np.array(mu_values)
    
    # # Net diffusivity (Mu_net) [m/s-2]
    # Since each interval should be equal, delta_{t,i} cancels out in the division
    WaveDiffusivity = np.nanmean(mu_values)  # Equivalent to sum(mu * delta_t) / sum(delta_t) for equal intervals

    # Stability index (Gamma) [dimensionless]
    Stabil_num = np.nansum(mu_values * TimeStep)
    Stabil_denom = np.nansum(np.abs(mu_values) * TimeStep)
    # Check to make sure no division by zero; if that happens, return 0 
    # (low and high angle waves balanced out, no longshore wave effects)
    WaveStability = Stabil_num / Stabil_denom if Stabil_denom != 0 else 0
    
    return WaveDiffusivity, WaveStability


def CalcShoreAngle(TransectInterGDF, Tr):
    """
    Calculate shore angle using the perpendicular of each transect angle,
    as measured clockwise from N.
    FM Mar 2024

    Parameters
    ----------
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of transects with veg edge intersection info assigned.
    Tr : int
        Transect ID.

    Returns
    -------
    ShoreAngle : list
        List of shore angles in degrees for each transect.

    """
    x_on = list(TransectInterGDF.iloc[Tr]['geometry'].coords)[0][0]
    y_on = list(TransectInterGDF.iloc[Tr]['geometry'].coords)[0][1]
    x_off = list(TransectInterGDF.iloc[Tr]['geometry'].coords)[1][0]
    y_off = list(TransectInterGDF.iloc[Tr]['geometry'].coords)[1][1]
    
    # Translated to measure clockwise from N (same as waves)
    ShoreAngle = 360 - np.rad2deg(math.atan2(y_off - y_on, x_off - x_on))
    if ShoreAngle > 360:
        ShoreAngle = ShoreAngle - 360

    return ShoreAngle

def CalcAlpha(WaveDir, ShoreAngle, Tr):
    """
    Calculate angle between shore and each wave on each transect's timeseries.
    FM Sept 2024

    Parameters
    ----------
    WaveDir : list
        Timeseries of mean wave directions (in degrees from N).
    ShoreAngle : float
        Angle of shore in degrees from N.
    Tr : int
        Cross-shore transect ID.

    Returns
    -------
    WaveAlpha : list
        List of same length as WaveDir, representing angle between shore and each wave observation.

    """
    # If no wave directions were recorded (i.e. no VE or WL intersections on that transect), return NaN
    # No condition is needed if NaNs are present in the WaveDir timeseries (the sum just returns NaN);
    # this is why .all() is used
    if np.isnan(WaveDir[Tr]).all():
        WaveAlpha = np.nan
    else:
        # modulo ensures value returned is same sign as dividend
        WaveAlpha = [(Dir - ShoreAngle[Tr]) + 180 % 360-180 for Dir in WaveDir[Tr]]
    
    return WaveAlpha


def CalcRunup(WaveHs, WaveTp=None, beta=None, Model="Senechal"):
    """
    Calculate wave runup using Senechal et al., 2011 formula 
    (assuming runup scales directly with offshore wave height).
    Can alternatively use Stockdon  et al., 2006 if wave period is available.
    FM June 2024

    Parameters
    ----------
    WaveHs : list
        Offshore significant wave heights (from Copernicus hindcast) at each transect in a site.
    WaveTp : list, optional
        Offshore peak wave periods (from Copernicus hindcast) at each transect in a site.
        The default is None (period only needed for Stockdon equation).
    beta : float, optional
        Shoreface steepness (single value per transect).
    Model : string, optional
        Runup equation to be used, from either 'Senechal':
            Senechal, N., Coco, G., Bryan, K.R., Holman, R.A., 2011. Wave runup during extreme
            storm conditions. Journal of Geophysical Research 116.
            https://doi.org/10.1029/2010JC006819
        or 'Stockdon':
            Stockdon, H. F., Holman, R. A., Howd, P. A., & Sallenger, A. H. (2006).
            Empirical  parameterization of setup, swash, and runup. Coastal Engineering,
            53(7), 573–588. https://doi.org/10.1016/j.coastaleng.2005.12.005

    Returns
    -------
    Runups : list
        Calculated wave runups list (with same dimensions as input wave heights).

    """
    
    Runups = []
    # For each transect in list
    for Tr in range(len(WaveHs)):
        # If empty, add NaN
        if isinstance(WaveHs[Tr], list) == False:
            RunupTr = np.nan
        else:
            RunupTr = []
            # For each wave condition at time of each sat image (at single transect)
            for i in range(len(WaveHs[Tr])):
                
                if Model == 'Senechal':
                    # Senechal 2011, Castelle 2021 Runup (2% exceedance) calculation for macrotidal dissipative beach
                    R2 = 2.14 * np.tanh(0.4 * WaveHs[Tr][i])
                
                elif Model == 'Stockdon':
                    # Stockdon 2006 runup depending on Iribarren measure of dissipative vs reflective
                    L0 = (9.81 * WaveTp[Tr][i]**2) / (2 * np.pi)
                    zeta = beta / (WaveHs[Tr][i] * L0) # dynamic beach steepness
                    if zeta < 0.3: # dissipative
                        R2 = 0.043 * (WaveHs[Tr][i] * L0)**0.5
                    else: # intermediate to reflective
                        R2 = 1.1 * (0.35 * beta * (WaveHs[Tr][i] * L0) ** 0.5
                                    + ((WaveHs[Tr][i] * L0 * (0.563 * beta ** 2 + 0.004)) ** 0.5) / 2
                                    )
                        
                RunupTr.append(R2)
        
        # Add per-transect runup lists (or nan) to full list
        Runups.append(RunupTr)
        
    return Runups


def CalcStorms(WaveTime, SigWaveHeight):
    """
    Generate boolean array of same size as wave height timestack, marked 1 where
    wave heights exceed the 95th percentile for each individual cell's timeseries.
    FM Aug 2024

    Parameters
    ----------
    WaveTime : list
        List of timesteps matching length of wave height timestack.
    SigWaveHeight : array
        3D array of offshore significant wave height rasters (shape=(time, y, x)).

    Returns
    -------
    StormEvents : array
        3D boolean array of storm events, with shape=SigWaveHeight.shape

    """
 
    # Calculate 95th percentile of wave height for 'storm' limit (creates percentile array of shape y,x)
    pct = np.percentile(SigWaveHeight, 95, axis=0)
    # Create boolean mask for where wave height exceeds 95th percentile
    StormMask = SigWaveHeight > pct
    # Generate boolean array where 1 = storm (exceeded wave height) and 0 = normal conditions
    StormEvents = np.where(StormMask, 1, 0)
    
    return StormEvents


#%%
def TransformWaves(TransectInterGDF, Hs, Dir, Tp):
    """
    IN DEVELOPMENT
    Airy/linear wave theory transformations for offshore to nearshore wave conditions,
    based on shoaling, refraction, and breaking effects.
    FM Jan 2024

    Parameters
    ----------
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of transects with veg edge intersection info assigned.
    SigWaveHeight : list
        Timeseries of significant wave height (in metres).
    Dir : list
        Timeseries of mean wave direction (in degrees from).
    Tp : list
        Timeseries of peak wave period (in seconds).

    Returns
    -------
    None.

    """
    for Tr in range(len(TransectInterGDF)): # for each transect
        Hs_mask = Hs.copy()
        Dir_mask = Dir.copy()
        Tp_mask = Tp.copy()
        
        # Mask data for onshore waves only (waves less than shoreline orientation)
        # need to preserve the matrix size so that theta_0 can be calculated
        # previous method of taking a mean doesn't work on curved west-facing bay
        # since half is 270-360 and half is 0-180 giving a weird mean
        # new way: straight line from edges of hard headland to get mean = 45
        
        
        # create new wave direction with index of masked values
        # 90 - atan2(y2-y1, x2-x1)
        # shoreAngle = 90 - np.rad2deg(math.atan2(S.Y(end-1)-S.Y(1), S.X(end-1)-S.X(1)));
        dX = TransectInterGDF['geometry'].iloc[Tr].bounds[2] - TransectInterGDF['geometry'].iloc[Tr].bounds[0]
        dY = TransectInterGDF['geometry'].iloc[Tr].bounds[3] - TransectInterGDF['geometry'].iloc[Tr].bounds[1]
        shoreAngle = 90 - np.rad2deg(math.atan2(dY, dX))
        
        for W in range(len(Dir)): # for each wave condition recorded on each transect intersection
            if Dir[W] > shoreAngle and Dir[W] < shoreAngle+180:
                    Dir_mask[W] = np.nan
    
        #Dir_mask(Dir > shoreAngle && Dir < shoreAngle+180) = NaN; 
        #Dir_mask(Dir > shoreAngle+180) = NaN;
        mask = np.isnan(Dir_mask)
        Tp_mask[mask] = np.nan
        Hs_mask[mask] = 0   # using NaN mask caused issues with breaking condition loop; changed to Hs=0
            
        # Preallocation to save memory
        # waves = struct('ID', np.nan ,'t', np.nan ,'Dir', np.nan ,'Hs', np.nan ,'Tp', np.nan );
        waves = {'ID':np.nan ,'t':np.nan ,'Dir':np.nan ,'Hs':np.nan ,'Tp':np.nan}
        
        ## Shadow zones
        # From the intersection of offshore wave directions with two points 
        # along the shoreline.    
        
        g  = 9.81   # gravity m^s^2
        rho = 1025  # water density
    
        Nloop = 0    # breaking wave loop counter updates
        
        # Re-initialise shadow zone logic table for each transect
        for W in range(len(Hs)):    # for each wave condition recorded on each transect intersection

            Hs_maskSh = Hs_mask[W]
            Tp_maskSh = Tp_mask[W]
            Dir_maskSh = Dir_mask[W]
            
            # If transect sits in shadow of other transect/coast at a particular wave angle
            # TO DO: shadowing routine
            shadow = 0
            if shadow==1:
                Hs_maskSh = 0
                Tp_maskSh = np.nan
                Dir_maskSh = np.nan
            
            H_0 = Hs_maskSh
            C_0 = np.divide( (g*Tp_maskSh) , (2*np.pi) )   # Deepwater wave speed (m/s)
            L_0 = np.multiply( C_0 , Tp_maskSh )        # Deepwater wavelength (m) set by speed and period
            h = 3 * H_0   # water depth at wave base for later calcs of Hs
            
            # Define offshore wave condition based on shadow zone masking
            # Calculate wave energy
            En = (1/8) * rho * g * np.dot(H_0, 2)
            
            BREAK_WAV = 0  # flag for wave breaking  
            
            while BREAK_WAV == 0:
                
                # Calculate wave conditions in shallow water depth
                L = np.multiply( L_0 , (np.tanh( ( np.multiply((np.square(np.divide((2*np.pi),Tp_maskSh))) , (h/g)) )**(3/4) )) )**(2/3)    # wavelength; tanh(x)=1 when -2pi<x<2pi
                C = np.multiply( C_0 , np.tanh(np.multiply((2*np.pi*h) , L)) )  # shallow wave speed
                k = np.divide((2*np.pi),L)    # wave number (1/m)
                
                # Calculate shoaling coefficient
                n = ( np.divide( np.multiply((2*h),k) , (np.sinh(np.multiply((2*h),k))) ) + 1 ) / 2    # shoaling factor
                Ks = np.sqrt( np.divide(C_0 , np.multiply(np.multiply(n,C),2)) )   # shoaling coefficient
                
                # Calculate refraction coefficient
                if shoreAngle > 0 and shoreAngle < 90:
                    # theta_0 is wave dir wrt shore angle
                    Theta_0 = shoreAngle + 270 - Dir_maskSh 
                else:
                    Theta_0 = shoreAngle - 90 - Dir_maskSh
                
                Theta = np.rad2deg( math.asin( np.multiply(np.divide(C,C_0) , math.sin(np.deg2rad(Theta_0)) )) )   # update theta
                Kr = math.sqrt(abs(math.cos(np.deg2rad(Theta_0))/math.cos(np.deg2rad(Theta))))
                # update conditions using refracted shoaled waves
                Hs_near = H_0*Ks*Kr
                if shoreAngle > 0 and shoreAngle < 90:
                    Dir_near = shoreAngle+270-Theta    # recalculating direction using theta
                else:
                    Dir_near = shoreAngle-90-Theta
                    if Dir_near < 0:
                        Dir_near=360+Dir_near   # need to check this! was *-1, but this swings -ve values back W from N
                    
                
                Tp_near = Tp_maskSh[W,1] # offshore period
                
                # Test if the wave meets breaking conditions
                if Hs_near > h*0.78:
                    BREAK_WAV = 1
                    Hs_break = Hs_near # to record per timeseries AND transect
                    Dir_break = Dir_near #  offshore cond.
                    Tp_break = Tp_maskSh[W,1] 
                    Nloop = Nloop + 1    # breaking wave loop counter updates
                
                
                # Reduce water depth by -10cm each loop
                h = h-0.10
                
                # Catch negative water depths (assume 0 transport and set
                # wave height and transport angle to 0)
                if h<0:
                    Hs_break[W,Tr] = 0
                    if shoreAngle > 0 and shoreAngle < 90: # for shoreline angles <90 (perpendicular transformation of -90 leads to -ve values) 
                        # need conditionals for Dir orientations too
                        if Dir_near > shoreAngle+270:    # 0-90 + 270 = for waves 270-360
                            Dir_break[W,Tr] = shoreAngle # transport rate = 0 when alpha = +90
                        elif np.isnan(Dir_near):  # to catch offshore (NaN) wave directions
                            Dir_break[W,Tr] = np.nan
                        else:
                            Dir_break[W,Tr] = shoreAngle+180 # transport rate = 0 when alpha = -90
                        
                    else: # for shoreline angles 90-360
                        # need conditionals for Dir orientations too
                        if Dir_near > shoreAngle-90:     # 90-360 - 90 = for waves 0-270
                            Dir_break[W,Tr] = shoreAngle # transport rate = 0 when alpha = +90
                        elif np.isnan(Dir_near):  # to catch offshore (NaN) wave directions
                            Dir_break[W,Tr] = np.nan
                        else:    # for Dir_near less than alpha_shore-90                      
                            Dir_break[W,Tr] = shoreAngle-180 # transport rate = 0 when alpha = -90
                            if Dir_break[W,Tr] < 0: #added condition for when alpha_shore-90 becomes negative (alpha<135)
                                Dir_break[W,Tr] = 360 + Dir_break[W,Tr]
 
                    
                    Tp_break[W,Tr] = Tp_maskSh[W,1] # offshore cond.
                    BREAK_WAV = 1 # ignore refraction in this case, wave has already refracted around
                     
                # use loop vars to write transformed wave data to structure
                waves['ID'][Tr] = TransectInterGDF['TransectID'].iloc[Tr]
                waves['t'][Tr][W,1] = str(TransectInterGDF['dates'].iloc[Tr][W]+' '+TransectInterGDF['times'].iloc[Tr][W])
                waves['alpha_shore'] = shoreAngle
                waves['Dir_near'][Tr][W,1] = Dir_near    
                    
            # condition to store both types of waves (near/breaking)
            if BREAK_WAV == 1:
                waves['Dir'][Tr][W,1] = Dir_break[W,Tr]
                waves['Hs'][Tr][W,1] = Hs_break[W,Tr]
                waves['Tp'][Tr][W,1] = Tp_break[W,Tr]
            else:
                waves['Hs'][Tr][W] = Hs_near
                waves['Tp'][Tr][W,1] = Tp_near
                waves['Dir'][Tr][W,1] = Dir_near
                

        print('number of breaking wave conditions: '+str(Nloop))