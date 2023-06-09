"""
This module contains functions to analyze the 2D shorelines along shore-normal
transects
    
Martin Hurst, Freya Muir
"""

# load modules
import os
from osgeo import ogr
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pdb
from datetime import datetime
from pathlib import Path
import pyproj
from pyproj import Proj

# other modules
import skimage.transform as transform
from sklearn.linear_model import LinearRegression
from pylab import ginput

from Toolshed import Toolbox
from Toolshed.Coast import *


def ProduceTransectsAll(SmoothingWindowSize, NoSmooths, TransectSpacing, DistanceInland, DistanceOffshore, proj, BasePath):
    
    for subdir, dirs, files in os.walk(BasePath):
        for direc in dirs:
            FileSpec = '/' + str(os.path.join(direc)) + '/' + str(os.path.join(direc)) + '.shp'
            ReprojSpec = '/' + str(os.path.join(direc)) + '/Reproj.shp'
            TransectSpec = '/' + str(os.path.join(direc)) + '/Transect.shp'
            CoastSpec = '/' + str(os.path.join(direc)) + '/Coast.shp'
            Filename2SaveCoast = '/' + str(os.path.join(direc)) + '/' + "My_Baseline.shp"
        
            #Reprojects shape file from EPSG 4326 to 27700 (britain)
        
            shape = gpd.read_file(BasePath+FileSpec)
            #shape = shape.set_crs(4326)
            # change CRS to epsg 27700
            shape = shape.to_crs(crs=proj,epsg=4326)
            # write shp file
            shape.to_file(BasePath+ReprojSpec)
        
            #Creates coast objects
            CellCoast = Coast(BasePath+ReprojSpec, MinLength=5)

            if not CellCoast.BuiltTransects:
            
                # may need to think carefully about how much to smooth
                CellCoast.SmoothCoastLines(WindowSize=SmoothingWindowSize,NoSmooths=NoSmooths)
            
                # make sure each line is correctly orientated with sea on left as you look down the line
                # this is something we'll need to think about replacing
                # CellCoast.CheckOrientation(str(SoftPath),str(MLWSPath))
        
                # write smoothed coast/bathy to file
                CellCoast.WriteCoastShp(BasePath+CoastSpec)
    
                # create some initial dummy transects, check inland/offshore the right way around
                CellCoast.GenerateTransects(TransectSpacing, DistanceInland, DistanceOffshore, CheckTopology=False)
        
                CellCoast.BuiltTransects = True
            
                CellCoast.WriteTransectsShp(BasePath+TransectSpec)
            
                # SAVE ENTIRE COAST OBJECT
                with open(str(BasePath+Filename2SaveCoast), 'wb') as PFile:
                    pickle.dump(CellCoast, PFile)
    return

def ProduceTransects(SmoothingWindowSize, NoSmooths, TransectSpacing, DistanceInland, DistanceOffshore, proj, sitename, BasePath, RefShapePath):
    """
    Produce shore-normal transects using CoastalMappingTools
    FM Oct 2022

    Parameters
    ----------
    SmoothingWindowSize : TYPE
        DESCRIPTION.
    NoSmooths : TYPE
        DESCRIPTION.
    TransectSpacing : TYPE
        DESCRIPTION.
    DistanceInland : TYPE
        DESCRIPTION.
    DistanceOffshore : TYPE
        DESCRIPTION.
    proj : TYPE
        DESCRIPTION.
    sitename : TYPE
        DESCRIPTION.
    BasePath : TYPE
        DESCRIPTION.
    RefShapePath : TYPE
        DESCRIPTION.

    Returns
    -------
    TransectGDF : TYPE
        DESCRIPTION.

    """    
    FileSpec = RefShapePath
    ReprojSpec = BasePath + '/Baseline_Reproj.shp'
    TransectSpec = os.path.join(BasePath, sitename+'_Transects.shp')
    CoastSpec = BasePath + '/Coast.shp'
    Filename2SaveCoast = BasePath + '/Coast.pydata'
    
    if (SmoothingWindowSize % 2) == 0:
        SmoothingWindowSize = SmoothingWindowSize + 1
        print('Window size should be odd; changed to %s m' % SmoothingWindowSize)
    
    shape = gpd.read_file(FileSpec)
    # change CRS to epsg 27700
    shape = shape.to_crs(epsg=27700)
    # write shp file
    shape.to_file(ReprojSpec)
        
    #Creates coast objects
    CellCoast = Coast(ReprojSpec, MinLength=10)

    if not CellCoast.BuiltTransects:
            
        CellCoast.SmoothCoastLines(WindowSize=SmoothingWindowSize,NoSmooths=NoSmooths)
            
        CellCoast.WriteCoastShp(CoastSpec)
    
        CellCoast.GenerateTransects(TransectSpacing, DistanceInland, DistanceOffshore, CheckTopology=False)
        
        CellCoast.BuiltTransects = True
            
        CellCoast.WriteSimpleTransectsShp(TransectSpec)
            
        with open(str(Filename2SaveCoast), 'wb') as PFile:
            pickle.dump(CellCoast, PFile)
            
    TransectSpec =  os.path.join(BasePath, sitename+'_Transects.shp')
    TransectGDF = gpd.read_file(TransectSpec)
    
    return TransectGDF
    
def GetIntersections(BasePath, TransectGDF, ShorelineGDF):
    '''
    New intersection between transects and shorelines, based on geopandas GDFs/shapefiles 
    rather than shorelines represented as points.
    
    FM Sept 2022

    Parameters
    ----------
    BasePath : str
        Path to shapefiles of transects.
    TransectGDF : GeoDataFrame
        GDF of shore-normal transects created.
    ShorelineGDF : GeoDataFrame
        GDF of lines extracted from sat images.

    Returns
    -------
    TransectDict : dict
        Transects with newly added intersection info.

    '''
     
    print("performing intersections between transects")
    # initialise where each intersection between lines and transects will be saved
    ColumnData = []
    Geoms = []
    # for each row/feature in transect
    for _, _, ID, TrGeom in TransectGDF.itertuples():
        # for each row/feature shoreline
        for _,dates,times,filename,cloud,ids,vthresh,satn,wthresh,SGeom in ShorelineGDF.itertuples():
            # calculate intersections between each transect and shoreline
            Intersects = TrGeom.intersection(SGeom)
            ColumnData.append((ID,dates,times,filename,cloud,ids,vthresh,satn,wthresh))
            Geoms.append(Intersects)
            
    # create GDF from appended lists of intersections        
    AllIntersects = gpd.GeoDataFrame(ColumnData,geometry=Geoms,columns=['TransectID','dates','times','filename','cloud_cove','idx','vthreshold','wthreshold','satname'])
    # remove any rows with no intersections
    AllIntersects = AllIntersects[~AllIntersects.is_empty].reset_index().drop('index',axis=1)
    # duplicate geom column to save point intersections
    AllIntersects['interpnt'] = AllIntersects['geometry']
    # take only first point on any transects which intersected a single shoreline more than once
    for inter in range(len(AllIntersects)):
        if AllIntersects['interpnt'][inter].geom_type == 'MultiPoint':
            AllIntersects['interpnt'][inter] = list(AllIntersects['interpnt'][inter])[0] # list() accesses individual points in MultiPoint
    AllIntersects = AllIntersects.drop('geometry',axis=1)
    # attribute join on transect ID to get transect geometry back
    AllIntersects = AllIntersects.merge(TransectGDF[['TransectID','geometry']], on='TransectID')
    
    print("formatting back into dict...")
    # initialise distances of intersections 
    distances = []
    # for each intersection
    for i in range(len(AllIntersects)):
        # calculate distance of intersection along transect
        distances.append(np.sqrt( 
            (AllIntersects['interpnt'][i].x - AllIntersects['geometry'][i].coords[0][0])**2 + 
            (AllIntersects['interpnt'][i].y - AllIntersects['geometry'][i].coords[0][1])**2 ))
    AllIntersects['distances'] = distances
    
    TransectDict = TransectGDF.to_dict('list')
    for Key in AllIntersects.drop(['TransectID','geometry'],axis=1).keys():
        TransectDict[Key] = {}
    TransectDict['interpnt'] = AllIntersects['interpnt'].copy()
    TransectDict['distances'] = AllIntersects['distances'].copy()
    

    #initialise lists used for storing each transect's intersection values
    dates, times, filename, cloud_cove, idx, vthreshold, wthreshold, satname, distances, interpnt = ([] for i in range(10)) # per-transect lists of values

    Key = [dates,times,filename,cloud_cove,idx,vthreshold, wthreshold,satname, distances, interpnt]
    KeyName = ['dates','times','filename','cloud_cove','idx','vthreshold', 'wthreshold','satname', 'distances', 'interpnt']
    
    # for each column name
    for i in range(len(Key)):
        # for each transect
        for Tr in range(len(TransectGDF['TransectID'])):
            # refresh per-transect list
            TrKey = []
            # for each matching intersection on a single transect
            for j in range(len(AllIntersects.loc[AllIntersects['TransectID']==Tr])):
                # append each intersection value to a list for each transect
                # iloc used so index doesn't restart at 0 each loop
                TrKey.append(AllIntersects[KeyName[i]].loc[AllIntersects['TransectID']==Tr].iloc[j]) 
            Key[i].append(TrKey)
    
        TransectDict[KeyName[i]] = Key[i]
    
    print("TransectDict with intersections created.")
        
    return TransectDict


def GetTransitionDists(TransectDict,TransectInterGDF):
    '''
    

    Parameters
    ----------
    TransectDict : TYPE
        DESCRIPTION.
    TransectInterGDF : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    
    
    

def GetBeachWidth(BasePath, TransectGDF, TransectDict, WaterlineGDF, settings, output, AvBeachSlope):
    
    '''
    Intersection between veglines and shorelines, based on geopandas GDFs/shapefiles.
    Shorelines are tidally corrected using either a DEM of slopes or a single slope value for all transects.
    
    FM Sept 2022

    Parameters
    ----------
    BasePath : str
        Path to shapefiles of transects.
    TransectGDF : GeoDataFrame
        GDF of shore-normal transects created.
    ShorelineGDF : GeoDataFrame
        GDF of lines extracted from sat images.

    Returns
    -------
    TransectDict : dict
        Transects with newly added intersection info.

    '''
     
    print("performing intersections between transects and waterlines")
    # initialise where each intersection between lines and transects will be saved
    ColumnData = []
    Geoms = []
    # for each row/feature in transect
    for _, _, ID, TrGeom in TransectGDF.itertuples():
        # for each row/feature shoreline
        for _,dates,_,_,_,_,_,_,_,SGeom in WaterlineGDF.itertuples():
            # calculate intersections between each transect and shoreline
            Intersects = TrGeom.intersection(SGeom)
            ColumnData.append((ID,dates))
            Geoms.append(Intersects)
            
    # create GDF from appended lists of intersections        
    AllIntersects = gpd.GeoDataFrame(ColumnData,geometry=Geoms,columns=['TransectID', 'wldates'])
    # remove any rows with no intersections
    AllIntersects = AllIntersects[~AllIntersects.is_empty].reset_index().drop('index',axis=1)
    # duplicate geom column to save point intersections
    AllIntersects['wlinterpnt'] = AllIntersects['geometry']
    # take only first point on any transects which intersected a single shoreline more than once
    for inter in range(len(AllIntersects)):
        if AllIntersects['wlinterpnt'][inter].geom_type == 'MultiPoint':
            AllIntersects['wlinterpnt'][inter] = list(AllIntersects['wlinterpnt'][inter])[0] # list() accesses individual points in MultiPoint
    AllIntersects = AllIntersects.drop('geometry',axis=1)
    # attribute join on transect ID to get transect geometry back
    AllIntersects = AllIntersects.merge(TransectGDF[['TransectID','geometry']], on='TransectID')
    
    print("formatting back into dict...")
    # initialise distances of intersections 
    distances = []
    # for each intersection
    for i in range(len(AllIntersects)):
        # calculate distance of intersection along transect
        distances.append(np.sqrt( 
            (AllIntersects['wlinterpnt'][i].x - AllIntersects['geometry'][i].coords[0][0])**2 + 
            (AllIntersects['wlinterpnt'][i].y - AllIntersects['geometry'][i].coords[0][1])**2 ))
    AllIntersects['wldists'] = distances

    CorrectedDists = TidalCorrection(settings, output, AllIntersects, AvBeachSlope)
    AllIntersects['wlcorrdist'] = CorrectedDists
    
    #initialise lists used for storing each transect's intersection values
    dates, distances, corrdists, interpnt = ([] for i in range(4)) # per-transect lists of values

    Key = [dates, distances, corrdists, interpnt]
    KeyName = ['wldates','wldists', 'wlcorrdist','wlinterpnt']
       
    # for each column name
    for i in range(len(Key)):
        # for each transect
        for Tr in range(len(TransectGDF['TransectID'])):
            # refresh per-transect list
            TrKey = []
            # for each matching intersection on a single transect
            for j in range(len(AllIntersects.loc[AllIntersects['TransectID']==Tr])):
                # append each intersection value to a list for each transect
                # iloc used so index doesn't restart at 0 each loop
                TrKey.append(AllIntersects[KeyName[i]].loc[AllIntersects['TransectID']==Tr].iloc[j]) 
            Key[i].append(TrKey)
    
        TransectDict[KeyName[i]] = Key[i]
          
    # Create beach width attribute
    print('calculating distances between veg and water lines...')
    TransectDict['beachwidth'] = TransectDict['TransectID'].copy()
    # for each transect
    for Tr in range(len(TransectGDF['TransectID'])):
        # dates into transect-specific list
        WLDateList = [datetime.strptime(date, '%Y-%m-%d') for date in TransectDict['wldates'][Tr]]
        VLDateList = [datetime.strptime(date, '%Y-%m-%d') for date in TransectDict['dates'][Tr]]
        # find index of closest waterline date to each vegline date
        VLSLDists = []
        for D, WLDate in enumerate(WLDateList):
            # index of matching nearest date
            if VLDateList != []:
                DateLoc = Toolbox.NearDate(WLDate,VLDateList)
                if DateLoc == False:
                    VLSLDists.append(np.nan)
                    continue
                else:
                    DateIndex = VLDateList.index(DateLoc)
            else:
                continue
            # use date index to identify matching distance along transect
            # and calculate distance between two intersections (veg - water means +ve is veg measured seaward towards water)
            VLSLDists.append(TransectDict['wlcorrdist'][Tr][D] - TransectDict['distances'][Tr][DateIndex])
            
        TransectDict['beachwidth'][Tr] = VLSLDists
    
    print("TransectDict with beach width and waterline intersections created.")
        
    return TransectDict
    

def TidalCorrection(settings, output, IntersectDF, AvBeachSlope):

    # load tidal data
    tidefilepath = os.path.join(settings['inputs']['filepath'],'tides',settings['inputs']['sitename']+'_tides.csv')
    tide_data = pd.read_csv(tidefilepath, parse_dates=['date'])
    dates_ts = [_.to_pydatetime() for _ in tide_data['date']]
    tides_ts = np.array(tide_data['tide'])
    
    # get the tide level corresponding to the time of sat image acquisition
    dates_sat = []
    for i in range(len(output['dates'])):
        dates_sat_str = output['dates'][i] +' '+output['times'][i]
        dates_sat.append(datetime.strptime(dates_sat_str, '%Y-%m-%d %H:%M:%S.%f'))
    
    tide_sat = []
    def find(item, lst):
        start = 0
        start = lst.index(item, start)
        return start
    for i,date in enumerate(dates_sat):
        tide_sat.append(tides_ts[find(min(item for item in dates_ts if item > date), dates_ts)])
    tides_sat = np.array(tide_sat)
         
    # tidal correction along each transect
    # elevation at which you would like the shoreline time-series to be
    RefElev = 1.0
    
    # if a DEM exists, use it to extract cross-shore slope between MSL and MHWS
    # TO DO: figure out way of running this per transect
    DEMpath = os.path.join(settings['inputs']['filepath'],'tides',settings['inputs']['sitename']+'_DEM.tif')
    if os.path.exists(DEMpath):
        MSL = 1.0
        MHWS = 0.1
        BeachSlope = GetBeachSlopes(MSL, MHWS, DEMpath)
    else:
        # if no DEM exists, use same slope value for all transects
        BeachSlope = AvBeachSlope
    
    CorrIntDistances = []
    
    dates_sat_d = []
    for dt in dates_sat:
        dates_sat_d.append(dt.date())
    
    for D, Dist in enumerate(IntersectDF['wldists']):
        DateIndex = dates_sat_d.index(datetime.strptime(IntersectDF['wldates'][D], '%Y-%m-%d').date())
        # calculate and apply cross-shore correction 
        Correction = (tides_sat[DateIndex] - RefElev) / BeachSlope
        # correction is minus because transect dists are defined land to seaward
        CorrIntDistances.append(Dist - Correction)
    
    return CorrIntDistances

def GetBeachSlopes(MSL, MHWS, DEMpath):
    """
    Extract a list of cross-shore slopes from a DEM using provided water levels.
    In development!
    FM Nov 2022
    
    Parameters
    ----------
    MSL : TYPE
        DESCRIPTION.
    MHWS : TYPE
        DESCRIPTION.
    DEMpath : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    

def SaveIntersections(TransectDict, LinesGDF, BasePath, sitename, projection):
    '''
    Save transects with intersection info as shapefile. Rates of change also calculated.
    FM Sept 2022

    Parameters
    ----------
    TransectDict : dict
        Transects with newly added intersection info.
    BasePath : str
        Path to shapefiles of transects.
    sitename : str
        Name of site.
    projection : int
        Projection EPSG code for saving transect shapefile.

    Returns
    -------
    TransectInterGDF : GeoDataFrame
        GDF of transects with intersection info.
    '''
    
    
    print('saving new transect shapefile ...')
    
    TransectInterGDF = gpd.GeoDataFrame(TransectDict, crs="EPSG:"+str(projection))
    
    DateList = LinesGDF['dates'].unique()
    
    # # for each unique date in satellitee-derived shorelines list
    # for Date in DateList:
    #     # shortened date format to YYMMDD to fit in field heading
    #     dateshort = Date[2:].replace('-','')
    #     ColData = []
    #     # for each transect
    #     for Tr in range(len(TransectInterGDF)):
    #         if Date in TransectInterGDF['dates'].iloc[Tr]:
    #             # find matching distance along transect for  that date
    #             DateIndex = TransectInterGDF['dates'].iloc[Tr].index(Date)
    #             # append found distance to list which will make up a new attribute field of distances
    #             ColData.append(TransectInterGDF['distances'].iloc[Tr][DateIndex])
    #         else:
    #             ColData.append(np.nan)
    #     # attach back onto GDF as a new field per image date
    #     TransectInterGDF[dateshort + 'dist'] = ColData
    
    DateRange = []
    FullDateTime = []
    RecentDateTime = []
    for Tr in range(len(TransectInterGDF)):
        DateRange.append(TransectInterGDF['dates'].iloc[Tr][0]) # oldest date
        DateRange.append(TransectInterGDF['dates'].iloc[Tr][-2]) # second youngest date
        DateRange.append(TransectInterGDF['dates'].iloc[Tr][-1]) # youngest date
        
        # for each Tr, find difference between converted oldest and youngest dates and transform to decimal years
        DateTime = round((float((datetime.strptime(DateRange[2],'%Y-%m-%d')-datetime.strptime(DateRange[0],'%Y-%m-%d')).days)/365.2425),4)
        FullDateTime.append(DateTime)
        DateTime = round((float((datetime.strptime(DateRange[2],'%Y-%m-%d')-datetime.strptime(DateRange[1],'%Y-%m-%d')).days)/365.2425),4)
        RecentDateTime.append(DateTime)
        # convert dates to ordinals for linreg
        OrdDates = [datetime.strptime(i,'%Y-%m-%d').toordinal() for i in TransectInterGDF['dates'].iloc[0]]
        
        Slopes = []
        for idate in [0,-2]:
            X = np.array(OrdDates[idate:]).reshape((-1,1))
            y = np.array(TransectInterGDF['distances'][Tr][idate:])
            model = LinearRegression(fit_intercept=True).fit(X,y)
            Slope = round(model.coef_[0],2)*365.2425 # ordinal dates means slope is in m/day, converts to m/yr
            Slopes.append(Slope)

    TransectInterGDF['olddate'] = DateRange[0] # oldest date in timeseries
    TransectInterGDF['youngdate'] = DateRange[-1] # youngest date in timeseries
    TransectInterGDF['oldyoungT'] = FullDateTime # time difference in years between oldest and youngest date
    TransectInterGDF['oldyoungRT'] = Slopes[0] # rate of change from oldest to youngest veg edge in m/yr
    TransectInterGDF['recentT'] = RecentDateTime # time difference in years between second youngest and youngest date
    TransectInterGDF['recentRT'] = Slopes[1] # rate of change from second youngest to youngest veg edge in m/yr
               
    TransectInterShp = TransectInterGDF.copy()

    KeyName = ['dates','times','filename','cloud_cove','idx','vthreshold','wthreshold','satname', 'distances','interpnt']
    for Key in KeyName:
        TransectInterShp[Key] = TransectInterShp[Key].astype(str)
    
    TransectInterShp.to_file(os.path.join(BasePath,sitename+'_Transects_Intersected.shp'))
    
    print("Shapefile with sat intersections saved.")
    
    return TransectInterGDF
    
def SaveWaterIntersections(TransectDict, LinesGDF, TransectInterGDFwDates, BasePath, sitename, projection):
    '''
    Save transects with beach width intersection info as shapefile.
    FM Sept 2022

    Parameters
    ----------
    TransectDict : dict
        Transects with newly added intersection info.
    BasePath : str
        Path to shapefiles of transects.
    sitename : str
        Name of site.
    projection : int
        Projection EPSG code for saving transect shapefile.

    Returns
    -------
    TransectInterGDF : GeoDataFrame
        GDF of transects with intersection info.
    '''
    
    
    print('saving new transect shapefile ...')
    
    TransectInterGDF = gpd.GeoDataFrame(TransectDict, crs="EPSG:"+str(projection))
    for Key in TransectInterGDFwDates.columns[11:]:
        TransectInterGDF[Key] = TransectInterGDFwDates[Key]
    
    DateList = LinesGDF['dates'].unique()
    
    # # for each unique date in satellite-derived shorelines list
    # for Date in DateList:
    #     # shortened date format to YYMMDD to fit in field heading
    #     dateshort = Date[2:].replace('-','')
    #     DistColData = []
    #     BWColData = []
    #     # for each transect
    #     for Tr in range(len(TransectInterGDF)):
    #         if Date in TransectInterGDF['wldates'].iloc[Tr]:
    #             # find matching distance along transect for that date
    #             DateIndex = TransectInterGDF['wldates'].iloc[Tr].index(Date)
    #             # append found distance to list which will make up a new attribute field of distances
    #             DistColData.append(TransectInterGDF['wldists'].iloc[Tr][DateIndex])
    #             if TransectInterGDF['beachwidth'].iloc[Tr] != []: # for rare transects where shoreline exists but width doesn't
    #                 BWColData.append(TransectInterGDF['beachwidth'].iloc[Tr][DateIndex])
    #             else:
    #                 BWColData.append(np.nan)
    #         else:
    #             DistColData.append(np.nan)
    #             BWColData.append(np.nan)
    #     # attach back onto GDF as a new field per image date
    #     TransectInterGDF[dateshort + 'wld'] = DistColData
    #     TransectInterGDF[dateshort + 'bw'] = BWColData
    
    DateRange = []
    FullDateTime = []
    RecentDateTime = []
    for Tr in range(len(TransectInterGDF)):
        DateRange.append(TransectInterGDF['dates'].iloc[Tr][0]) # oldest date
        DateRange.append(TransectInterGDF['dates'].iloc[Tr][-2]) # second youngest date
        DateRange.append(TransectInterGDF['dates'].iloc[Tr][-1]) # youngest date
        
        # for each Tr, find difference between converted oldest and youngest dates and transform to decimal years
        DateTime = round((float((datetime.strptime(DateRange[2],'%Y-%m-%d')-datetime.strptime(DateRange[0],'%Y-%m-%d')).days)/365.2425),4)
        FullDateTime.append(DateTime)
        DateTime = round((float((datetime.strptime(DateRange[2],'%Y-%m-%d')-datetime.strptime(DateRange[1],'%Y-%m-%d')).days)/365.2425),4)
        RecentDateTime.append(DateTime)
        # convert dates to ordinals for linreg
        OrdDates = [datetime.strptime(i,'%Y-%m-%d').toordinal() for i in TransectInterGDF['dates'].iloc[0]]
        
        Slopes = []
        for idate in [0,-2]:
            X = np.array(OrdDates[idate:]).reshape((-1,1))
            y = np.array(TransectInterGDF['wlcorrdist'][Tr][idate:])
            model = LinearRegression(fit_intercept=True).fit(X,y)
            Slope = round(model.coef_[0],2)*365.2425
            Slopes.append(Slope)

    TransectInterGDF['olddateW'] = DateRange[0] # oldest date in timeseries
    TransectInterGDF['youngdateW'] = DateRange[-1] # youngest date in timeseries
    TransectInterGDF['oldyoungTW'] = FullDateTime # time difference in years between oldest and youngest date
    TransectInterGDF['oldyungRTW'] = Slopes[0] # rate of change from oldest to youngest veg edge in m/yr
    TransectInterGDF['recentTW'] = RecentDateTime # time difference in years between second youngest and youngest date
    TransectInterGDF['recentRTW'] = Slopes[1] # rate of change from second youngest to youngest veg edge in m/yr
               
    TransectInterShp = TransectInterGDF.copy()
    
    KeyName = ['dates','times','filename','cloud_cove','idx','vthreshold','wthreshold','satname', 'distances', 'normdists' ,'interpnt', 'wldates','wldists', 'wlcorrdist','wlinterpnt', 'beachwidth']
    for Key in KeyName:
        TransectInterShp[Key] = TransectInterShp[Key].astype(str)
    
    
    TransectInterShp.to_file(os.path.join(BasePath,sitename+'_Transects_Intersected.shp'))

    
    print("Shapefile with sat intersections saved.")
    
    return TransectInterGDF


def CalculateChanges(TransectDict,TransectInterGDF):
    
    
    TransectDict['normdists'] = TransectDict['distances'].copy()
    # for each transect
    for Tr in range(len(TransectDict['TransectID'])):
        Dists = []
        # for each intersection on each transect
        for i, Dist in enumerate(TransectInterGDF['distances'][Tr]):
            # intersection distance along transect minus midpoint distance gives +ve for seaward and -ve for landward
            Dists.append(Dist - TransectInterGDF.geometry[Tr].length/2)
        TransectDict['normdists'][Tr] = Dists
    
    print("TransectDict updated with distances between sat lines.")
            
    return TransectDict


def ValidateIntersects(ValidationShp, DatesCol, TransectGDF, TransectDict):
    """
    Intersects transects with validation lines from shapefile, matches date of
    each validation line to nearest sat line, and calculates distance along 
    transect between them.
    FM Oct 2022

    Parameters
    ----------
    ValidationShp : str
        File path to validation line shapefile.
    DatesCol : str
        Name of attribute field where dates are stored.
    TransectGDF : GeoDataFrame
        Transect GDF with no attributes, just geometries.
    TransectDict : dict
        Transect dictionary with attributes.

    Returns
    -------
    TYPE
        DESCRIPTION.
        
    """
    
    print('performing transect intersects on validation lines...')
    ValidGDF = gpd.read_file(ValidationShp)
    if DatesCol in ValidGDF.keys():
        ValidGDF = ValidGDF[[DatesCol,'geometry']]
    else:
        print('No date column found - check your spelling')
        return
    # initialise where each intersection between lines and transects will be saved
    ColumnData = []
    Geoms = []
     # for each row/feature in transect
    for _, _, ID, TrGeom in TransectGDF.itertuples():
         # for each row/feature shoreline
        for _,dates,SGeom in ValidGDF.itertuples():
             # calculate intersections between each transect and shoreline
            Intersects = TrGeom.intersection(SGeom)
            ColumnData.append((ID,dates))
            Geoms.append(Intersects)
            
    # create GDF from appended lists of intersections        
    AllIntersects = gpd.GeoDataFrame(ColumnData,geometry=Geoms,columns=['TransectID','Vdates'])
    # remove any rows with no intersections
    AllIntersects = AllIntersects[~AllIntersects.is_empty].reset_index().drop('index',axis=1)
    # duplicate geom column to save point intersections
    AllIntersects['Vinterpnt'] = AllIntersects['geometry']
    # take only first point on any transects which intersected a single shoreline more than once
    for inter in range(len(AllIntersects)):
        if AllIntersects['Vinterpnt'][inter].geom_type == 'MultiPoint':
            AllIntersects['Vinterpnt'][inter] = list(AllIntersects['Vinterpnt'][inter])[0] # list() accesses individual points in MultiPoint
    AllIntersects = AllIntersects.drop('geometry',axis=1)
    # attribute join on transect ID to get transect geometry back
    AllIntersects = AllIntersects.merge(TransectGDF[['TransectID','geometry']], on='TransectID')
    
    print("formatting back into dict...")
    # initialise distances of intersections 
    distances = []
    # for each intersection
    for i in range(len(AllIntersects)):
        # calculate distance of intersection along transect
        distances.append(Toolbox.CalcDistance(AllIntersects['Vinterpnt'][i], AllIntersects['geometry'][i]))
    AllIntersects['Vdists'] = distances
    
    #initialise lists used for storing each transect's intersection values
    Vdates,Vdists, Vinterpnt = ([] for i in range(3)) # per-transect lists of values

    Key = [Vdates, Vdists, Vinterpnt]
    KeyName = ['Vdates', 'Vdists', 'Vinterpnt']
    ValidDict = TransectDict.copy()
    
    # for each column name
    for i in range(len(Key)):
        # for each transect
        for Tr in range(len(TransectGDF['TransectID'])):
            # refresh per-transect list
            TrKey = []
            # for each matching intersection on a single transect
            for j in range(len(AllIntersects.loc[AllIntersects['TransectID']==Tr])):
                # append each intersection value to a list for each transect
                # iloc used so index doesn't restart at 0 each loop
                TrKey.append(AllIntersects[KeyName[i]].loc[AllIntersects['TransectID']==Tr].iloc[j]) 
            Key[i].append(TrKey)
    
        ValidDict[KeyName[i]] = Key[i]
    
    print('calculating distances between validation and sat lines...')
    ValidDict['valsatdist'] = ValidDict['TransectID'].copy()
    # for each transect
    for Tr in range(len(TransectGDF['TransectID'])):
        # dates into transect-specific list
        VDateList = [datetime.strptime(date, '%Y-%m-%d') for date in ValidDict['Vdates'][Tr]]
        DateList = [datetime.strptime(date, '%Y-%m-%d') for date in ValidDict['dates'][Tr]]
        # find index of closest sat date to each validation date
        ValSatDists = []
        for D, VDate in enumerate(VDateList):
            # index of matching nearest date
            if DateList != []:
                DateIndex = DateList.index(Toolbox.NearDate(VDate,DateList))
            else:
                continue
            # use date index to identify matching distance along transect
            # and calculate distance between two intersections (sat - validation means +ve is seaward/-ve is landward)
            ValSatDists.append(ValidDict['distances'][Tr][DateIndex] - ValidDict['Vdists'][Tr][D])
            
        ValidDict['valsatdist'][Tr] = ValSatDists
        
    print("TransectDict with intersections created.")
    
    return ValidDict

def ValidateSatIntersects(sitename, ValidationShp, DatesCol, TransectGDF, TransectDict):
    """
    Intersects transects with validation lines from shapefile, matches date of
    each sat line to nearest valid. line, and calculates distance along 
    transect between them.
    
    FM Oct 2022

    Parameters
    ----------
    ValidationShp : str
        File path to validation line shapefile.
    DatesCol : str
        Name of attribute field where dates are stored.
    TransectGDF : GeoDataFrame
        Transect GDF with no attributes, just geometries.
    TransectDict : dict
        Transect dictionary with attributes.

    Returns
    -------
    ValidDict : dict
        DESCRIPTION.
        
    """
    
    validpath = os.path.join(os.getcwd(), 'Data', sitename, 'validation')
    if os.path.isdir(validpath) is False:
        os.mkdir(validpath)
    
    print('performing transect intersects on validation lines...')
    ValidGDF = gpd.read_file(ValidationShp)
    if DatesCol in ValidGDF.keys():
        ValidGDF = ValidGDF[[DatesCol,'geometry']]
    else:
        print('No date column found - check your spelling')
        return
    # initialise where each intersection between lines and transects will be saved
    ColumnData = []
    Geoms = []
     # for each row/feature in transect
    for _, _, ID, TrGeom in TransectGDF.itertuples():
         # for each row/feature shoreline
        for _,dates,SGeom in ValidGDF.itertuples():
             # calculate intersections between each transect and shoreline
            Intersects = TrGeom.intersection(SGeom)
            ColumnData.append((ID,dates))
            Geoms.append(Intersects)
            
    # create GDF from appended lists of intersections        
    AllIntersects = gpd.GeoDataFrame(ColumnData,geometry=Geoms,columns=['TransectID','Vdates'])
    # remove any rows with no intersections
    AllIntersects = AllIntersects[~AllIntersects.is_empty].reset_index().drop('index',axis=1)
    # duplicate geom column to save point intersections
    AllIntersects['Vinterpnt'] = AllIntersects['geometry']
    # take only first point on any transects which intersected a single shoreline more than once
    for inter in range(len(AllIntersects)):
        if AllIntersects['Vinterpnt'][inter].geom_type == 'MultiPoint':
            AllIntersects['Vinterpnt'][inter] = list(AllIntersects['Vinterpnt'][inter])[0] # list() accesses individual points in MultiPoint
    AllIntersects = AllIntersects.drop('geometry',axis=1)
    # attribute join on transect ID to get transect geometry back
    AllIntersects = AllIntersects.merge(TransectGDF[['TransectID','geometry']], on='TransectID')
    
    print("formatting back into dict...")
    # initialise distances of intersections 
    distances = []
    # for each intersection
    for i in range(len(AllIntersects)):
        # calculate distance of intersection along transect
        distances.append(Toolbox.CalcDistance(AllIntersects['Vinterpnt'][i], AllIntersects['geometry'][i]))
    AllIntersects['Vdists'] = distances
    
    #initialise lists used for storing each transect's intersection values
    Vdates,Vdists, Vinterpnt = ([] for i in range(3)) # per-transect lists of values

    Key = [Vdates, Vdists, Vinterpnt]
    KeyName = ['Vdates', 'Vdists', 'Vinterpnt']
    ValidDict = TransectDict.copy()
    
    # for each column name
    for i in range(len(Key)):
        # for each transect
        for Tr in range(len(TransectGDF['TransectID'])):
            # refresh per-transect list
            TrKey = []
            # for each matching intersection on a single transect
            for j in range(len(AllIntersects.loc[AllIntersects['TransectID']==Tr])):
                # append each intersection value to a list for each transect
                # iloc used so index doesn't restart at 0 each loop
                TrKey.append(AllIntersects[KeyName[i]].loc[AllIntersects['TransectID']==Tr].iloc[j]) 
            Key[i].append(TrKey)
    
        ValidDict[KeyName[i]] = Key[i]
    
    print('calculating distances between validation and sat lines...')
    ValidDict['valsatdist'] = ValidDict['TransectID'].copy()
    # for each transect
    for Tr in range(len(TransectGDF['TransectID'])):
        # dates into transect-specific list
        VDateList = [datetime.strptime(date, '%Y-%m-%d') for date in ValidDict['Vdates'][Tr]]
        DateList = [datetime.strptime(date, '%Y-%m-%d') for date in ValidDict['dates'][Tr]]
        # find index of closest validation date to each sat date
        ValSatDists = []
        for D, Date in enumerate(DateList):
            # index of matching nearest date
            if VDateList != []:
                NearestDate = Toolbox.NearDate(Date,VDateList)
                if NearestDate == False: # if no matching validation date exists, add nan to the list
                    ValSatDists.append(np.nan)
                else:
                    # use date index to identify matching distance along transect
                    # and calculate distance between two intersections (sat - validation means +ve is seaward/-ve is landward)
                    VDateIndex = VDateList.index(NearestDate)
                    ValSatDists.append(ValidDict['distances'][Tr][D] - ValidDict['Vdists'][Tr][VDateIndex])
            else:
                continue


        ValidDict['valsatdist'][Tr] = ValSatDists
        
    print("ValidDict with intersections created.")
    
    return ValidDict


def compute_intersection(output, transects, settings, linetype):
    """
    Computes the intersection between the 2D shorelines and the shore-normal.
    transects. It returns time-series of cross-shore distance along each transect.
    
    KV WRL 2018       

    Arguments:
    -----------
    output: dict
        contains the extracted shorelines and corresponding metadata
    transects: dict
        contains the X and Y coordinates of each transect
    settings: dict with the following keys
        'along_dist': int
            alongshore distance considered caluclate the intersection
              
    Returns:    
    -----------
    cross_dist: dict
        time-series of cross-shore distance along each of the transects. 
        Not tidally corrected.        
    """  
    
    """
    if (linetype+'transect_time_series.csv') in os.listdir(settings['inputs']['filepath']):
        print('Cross-distance calculations already exist and were loaded')
        with open(os.path.join(settings['inputs']['filepath'], linetype+'transect_time_series.csv'), 'rb') as f:
            cross_dist = pickle.load(f)
        return cross_dist
    """
    
    # loop through shorelines and compute the median intersection    
    intersections = np.zeros((len(output['shorelines']),len(transects)))
    for i in range(len(output['shorelines'])):

        sl = output['shorelines'][i]
        
        print(" \r\tShoreline %4d / %4d" % (i+1, len(output['shorelines'])), end="")
        
        for j,key in enumerate(list(transects.keys())): 
            
            # compute rotation matrix
            X0 = transects[key][0,0]
            Y0 = transects[key][0,1]
            temp = np.array(transects[key][-1,:]) - np.array(transects[key][0,:])
            phi = np.arctan2(temp[1], temp[0])
            Mrot = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]])
    
            # calculate point to line distance between shoreline points and the transect
            p1 = np.array([X0,Y0])
            p2 = transects[key][-1,:]
            d_line = np.abs(np.cross(p2-p1,sl-p1)/np.linalg.norm(p2-p1))
            # calculate the distance between shoreline points and the origin of the transect
            d_origin = np.array([np.linalg.norm(sl[k,:] - p1) for k in range(len(sl))])
            # find the shoreline points that are close to the transects and to the origin
            # the distance to the origin is hard-coded here to 1 km 
            idx_dist = np.logical_and(d_line <= settings['along_dist'], d_origin <= 1000)
            # find the shoreline points that are in the direction of the transect (within 90 degrees)
            temp_sl = sl - np.array(transects[key][0,:])
            phi_sl = np.array([np.arctan2(temp_sl[k,1], temp_sl[k,0]) for k in range(len(temp_sl))])
            diff_angle = (phi - phi_sl)
            idx_angle = np.abs(diff_angle) < np.pi/2
            # combine the transects that are close in distance and close in orientation
            idx_close = np.where(np.logical_and(idx_dist,idx_angle))[0]     
            
            # in case there are no shoreline points close to the transect 
            if len(idx_close) == 0:
                intersections[i,j] = np.nan
            else:
                # change of base to shore-normal coordinate system
                xy_close = np.array([sl[idx_close,0],sl[idx_close,1]]) - np.tile(np.array([[X0],
                                   [Y0]]), (1,len(sl[idx_close])))
                xy_rot = np.matmul(Mrot, xy_close)
                # compute the median of the intersections along the transect
                intersections[i,j] = np.nanmedian(xy_rot[0,:])
    
    # fill the a dictionnary
    cross_dist = dict([])
    cross_dist['dates'] = output['dates']
    for j,key in enumerate(list(transects.keys())): 
        cross_dist[key] = intersections[:,j]  
    
    
    # save a .csv file for Excel users
    out_dict = dict([])
    out_dict['dates'] = output['dates']
    for key in transects.keys():
        out_dict['Transect '+ key] = cross_dist[key]
    df = pd.DataFrame(out_dict)
    fn = os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],linetype+
                      'transect_time_series.csv')
    df.to_csv(fn, sep=',')
    print('Time-series of the shoreline change along the transects saved as:\n%s'%fn)
    
    return cross_dist

def stuffIntoLibrary(geo, image_epsg, projection_epsg, filepath, sitename):
    
    print('Reading transects into library for further analysis...')
    transect = Path("Data/" + sitename + "/transect_proj.pkl")
    
    proj1 = Proj(init="epsg:"+str(projection_epsg)) 
    proj2 = Proj(init="epsg:"+str(4326))
    proj3 = Proj(init="epsg:"+str(image_epsg))
    
    if transect.is_file():
        with open(os.path.join(filepath, sitename + '_transect_proj' + '.pkl'), 'rb') as f:
            transects_proj = pickle.load(f)
        with open(os.path.join(filepath, sitename + '_transect_latlon' + '.pkl'), 'rb') as f:
            transects_latlon = pickle.load(f)
            
        return transects_latlon, transects_proj
    
    transects_latlon = dict([])
    transects_proj = dict([])

    for i in range (len(geo['geometry'])):
        
        lib = 'Transect_'+str(i+1)
    
        x,y = geo['geometry'][i].coords.xy
        
        # convert to lat lon
        xy0 = pyproj.transform(proj1,proj2,y[0],x[0])
        xy1 = pyproj.transform(proj1,proj2,y[1],x[1])
        coord0_latlon = [xy0[1],xy0[0]]
        coord1_latlon = [xy1[1],xy1[0]]

        transects_latlon[lib] = np.array([coord0_latlon, coord1_latlon])
        #x,y = pyproj.transform(proj2,proj3,transects_latlon[lib][0][1],transects_latlon[lib][0][0])
        #x1,y1 = pyproj.transform(proj2,proj3,transects_latlon[lib][1][1],transects_latlon[lib][1][0])
        transects_proj[lib] = np.array([[x[1],y[1]],[x[0],y[0]]])

        print(" \r\tCurrent Progress:",np.round(i/len(geo['geometry'])*100,2),"%",end='')
    
    with open(os.path.join(filepath, sitename + '_transect_proj.pkl'), 'wb') as f:
            pickle.dump(transects_proj, f)
            
    with open(os.path.join(filepath, sitename + '_transect_latlon.pkl'), 'wb') as f:
            pickle.dump(transects_latlon, f)
            
    return transects_latlon, transects_proj

def transect_compiler(Rows, transect_proj, transect_range, output):
    
    cross_distance_condensed = dict([])
    standard_err_condensed = dict([])
    transect_condensed = dict([])
    Dates = dict([])
    new_Transect = 1

    cross_arr = []
    trans_arr = []

    for i in range(len(transect_range)):
        cross_arr = []
        trans_arr = []
        for j in range(transect_range[i][0],transect_range[i][1]):
            try:
                arr = []
                for k in range(len(Rows)-1):
                    try:
                        arr.append(float(Rows[k][j]))
                    except:
                        arr.append(np.nan)
                cross_arr.append(arr)
                trans_arr.append(transect_proj[list(transect_proj.keys())[j]])
            except:
                continue
        std = np.nanstd(cross_arr,0)
        for j in range(len(std)):
            std[j] = std[j]/(abs(transect_range[i][0]-transect_range[i][1]))**0.5

        NaN_mask = np.isfinite(np.nanmean(cross_arr,0))
        cross_distance_condensed['Transect_'+str(transect_range[i][0])+'-'+str(transect_range[i][1])] = np.nanmean(cross_arr,0).astype(np.double)[NaN_mask]
        standard_err_condensed['Transect_'+str(transect_range[i][0])+'-'+str(transect_range[i][1])] = std.astype(np.double)[NaN_mask]
        Dates['Transect_'+str(transect_range[i][0])+'-'+str(transect_range[i][1])] = np.array(output['dates'])[NaN_mask]
        transect_condensed['Transect_'+str(transect_range[i][0])+'-'+str(transect_range[i][1])] = np.mean(trans_arr,0).astype(np.double)#[NaN_mask]
        
    return cross_distance_condensed, standard_err_condensed, transect_condensed, Dates