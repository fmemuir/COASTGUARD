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
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
pd.options.mode.chained_assignment = None # suppress pandas warning about setting a value on a copy of a slice
from scipy.interpolate import interp1d

# from Toolshed import Waves

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
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
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# Only use tensorflow in CPU mode
import tensorflow as tf
tf.config.set_visible_devices([],'GPU')



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
    CoastalDF = pd.merge(TransectInterGDF[['TransectID','dates','times','distances']], 
                         TransectInterGDFWater[['TransectID','wldates','wltimes','wlcorrdist', 'tideelev','beachwidth']],
                         how='inner', on='TransectID')
    # Merge combined dataframe with topographic info
    # TransectInterGDFTopo[['TransectID','TZwidth', 'TZwidthMn', 'SlopeMax', 'SlopeMean']]
    CoastalDF = pd.merge(CoastalDF, 
                         TransectInterGDFTopo[['TransectID','TZwidth']],
                         how='inner', on='TransectID')
    # Merge combined dataframe with wave info
    # TransectInterGDFWave[['TransectID','WaveHs', 'WaveDir', 'WaveTp', 'WaveDiffus']]
    CoastalDF = pd.merge(CoastalDF, 
                         TransectInterGDFWave[['TransectID','WaveDates','WaveHs', 'WaveDir', 'WaveTp', 'Runups','Iribarrens']],
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


def InterpVEWL(CoastalDF, Tr):
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

    # Interpolate over waterline and wave associated variables
    wl_numdates = pd.to_datetime(TransectDF['wlDTs'][Tr]).values.astype(np.int64)
    ve_numdates = pd.to_datetime(TransectDF['veDTs'][Tr]).values.astype(np.int64)
    wv_numdates = pd.to_datetime(TransectDF['WaveDates'][Tr]).values.astype(np.int64)
    # Match dates with veg edge dates and append back to TransectDF
    for wlcol in ['wlcorrdist', 'tideelev','beachwidth']:
        wl_interp_f = interp1d(wl_numdates, TransectDF[wlcol][Tr], kind='linear', fill_value='extrapolate')
        wl_interp = wl_interp_f(wv_numdates).tolist()
        TransectDF[wlcol] = [wl_interp]
    for vecol in ['distances','TZwidth']:
        ve_interp_f = interp1d(ve_numdates, TransectDF[vecol][Tr], kind='linear', fill_value='extrapolate')
        ve_interp = ve_interp_f(wv_numdates).tolist()
        TransectDF[vecol] = [ve_interp]
    
    # Recalculate beachwidth as values will now be mismatched
    beachwidth = [abs(wl_interp[i] - TransectDF['distances'][Tr][i]) for i in range(len(wl_interp))]
    TransectDF['beachwidth'] = [beachwidth]
    
    TransectDF.drop(columns=['veDTs','wldates','wltimes', 'wlDTs','dates','times'], inplace=True)
    
    # Transpose to get columns of variables and rows of timesteps
    TransectDF = pd.DataFrame({col: pd.Series(val.iloc[0]) for col,val in TransectDF.items()})
    
    # Reset index for timeseries
    TransectDF.index = TransectDF['WaveDates']
    TransectDF = TransectDF.drop(columns=['TransectID', 'WaveDates'])

    return TransectDF


def PreprocessTraining(CoastalDF):
    
    X = CoastalDF.drop(columns=['TransectID', 'labels'])
    y = CoastalDF['labels']
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    
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
    VarDF : DataFrame
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
    VarDF : DataFrame
        Dataframe of just coastal metrics/variables in timeseries, with cluster values attached to each timestep.

    """
    # Fill nans factoring in timesteps for interpolation
    TransectDF.replace([np.inf, -np.inf], np.nan, inplace=True)
    VarDF = TransectDF.interpolate(method='time', axis=0)
    
    # VarDF = VarDF[['distances', 'wlcorrdist','TZwidth','WaveHs']]
    
    VarDF_scaled = StandardScaler().fit_transform(VarDF)
    
    # Apply PCA to reduce the dimensions to 3D for visualization
    pca = PCA(n_components=3)
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
    pca_df = pd.DataFrame(data=pca_VarDF, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Cluster'] = ClusterMod.labels_

    # Visualization of clusters
    # Example clustered timeseries using one or two variables
    if ValPlots is True:
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
    
    scale_factor = 0.5
    # Plot the clusters in the PCA space
    fig, ax = plt.subplots(figsize=(5, 5))
    clusterDF = []
    for cluster in pca_df['Cluster'].unique():
        cluster_data = pca_df[pca_df['Cluster'] == cluster]
        plt.scatter(
            cluster_data['PC1']*scale_factor, 
            cluster_data['PC2']*scale_factor, 
            label=f'Cluster {cluster}', 
            s=40,
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
        
        
    # return VarDFClust



def PrepData(VarDF, l_mlabel, l_testS, l_hours, UseSMOTE=False):
    """
    Prepare features (X) and labels (y) for feeding into a NN for timeseries prediction.    
    FM Sept 2024
    
    Parameters
    ----------
    VarDF : DataFrame
        Dataframe of just coastal metrics/variables in timeseries, with cluster values attached to each timestep.
    UseSMOTE : bool, optional
        Flag for using SMOTE to oversample imbalanced data. The default is False.

    Returns
    -------
    PredDict : dict
        Dictionary to store all the NN model metadata.

    """
    
    PredDict = {'mlabel':l_mlabel,
                'model':[],
                'history':[],
                'loss':[],
                'accuracy':[],
                'train_time':[],
                'X_train':[],
                'y_train':[],
                'X_test':[],
                'y_test':[],
                'epochS':[],
                'batchS':[]}
    
    X = VarDF.drop(columns=['Cluster','Impact'])
    y = VarDF['Cluster']
    
    for mlabel, testS, hours in zip(PredDict['mlabel'], l_testS, l_hours):
        # Normalize the features
        scaler = StandardScaler()
        # scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        testS = testS # proportion of data to use for training vs. testing
        t_seq = hours # number of timesteps over which to generate sequence (hours)
        
        # Separate test and train data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=testS, random_state=0, stratify=y)
        # Create sequences
        X_train, y_train = CreateSequences(X_train, y_train, t_seq)
        # Create test sequences
        X_test, y_test = CreateSequences(X_test, y_test, t_seq)
        PredDict['X_test'].append(X_test)
        PredDict['y_test'].append(y_test)
        
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
            
        
    return PredDict


def GetFutureData(sitename, DateMin, DateMax, CoastalDF):
    
    with open(os.path.join(os.getcwd(),'Data', sitename, sitename + '_settings.pkl'), 'rb') as f:
        settings = pickle.load(f)
    

    # Download wave forecasts from Copernicus Marine
    WavePath, WaveOutFile = Waves.GetForecastWaveData(settings, DateMin, DateMax)
    WaveFilePath = os.path.join(WavePath, WaveOutFile)
    
    # Sample future waves using coastal DF transects
    WaveDates, WaveHs, WaveDir, WaveTp, WaveDiffusivity, WaveStability, ShoreAngles = Waves.SampleWavesFuture(CoastalDF, WaveFilePath)
    


def CompileRNN(PredDict, costsensitive=False):
    """
    Compile the NN using the settings and data stored in the NN dictionary.
    FM Sept 2024

    Parameters
    ----------
    PredDict : dict
        Dictionary to store all the NN model metadata.
    costsensitive : bool, optional
        Option for including a cost-sensitive loss function. The default is False.

    Returns
    -------
    PredDict : dict
        Dictionary to store all the NN model metadata, now with compiled models added

    """
    for mlabel in PredDict['mlabel']:
        # Index of model setup
        mID = PredDict['mlabel'].index(mlabel)
        
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
        N_hidden = round(inshape[0] / (2 * (inshape[1] + 3)))
        
        # LSTM (1 layer)
        # Input() takes input shape, used for sequential models
        # LSTM() has dimension of (batchsize, timesteps, units) and retains info at each timestep (return_sequences=True)
        # Dropout() randomly sets inputs to 0 during training to prevent overfitting
        # Dense() transforms output into normalised PDF across the 3 categories
        Model = Sequential([
                            Input(shape=inshape),
                            LSTM(units=N_hidden, return_sequences=True),
                            Dropout(0.2), 
                            Dense(3, activation='softmax') 
                            ])
        
        # Compile model and define loss function and metrics
        if costsensitive:
            # Define values for false +ve and -ve and create matrix
            falsepos_cost = 1   # Inconvenience of incorrect classification
            falseneg_cost = 100 # Risk to infrastructure by incorrect classification
            binary_thresh = 0.5
            LossFn = CostSensitiveLoss(falsepos_cost, falseneg_cost, binary_thresh)
        
            Model.compile(optimizer=Adam(learning_rate=0.001), 
                             loss=LossFn, 
                             metrics=['accuracy', 'loss'])
        else:
            # If not cost-sensitive, just use categorical loss fn
            Model.compile(optimizer=Adam(learning_rate=0.001), 
                             loss='sparse_categorical_crossentropy', 
                             metrics=['accuracy', 'loss'])
        
        # Save model infrastructure to dictionary of model sruns
        PredDict['model'].append(Model)
    
    return PredDict


def TrainRNN(PredDict,filepath,sitename):
    """
    Train the compiled NN based on the training data set aside for it.
    FM Sept 2024

    Parameters
    ----------
    PredDict : dict
        Dictionary to store all the NN model metadata.
    filepath : str
        Filepath to save the PredDict dictionary to (for reading the trained model back in).
    sitename : str
        Name of the site of interest.

    Returns
    -------
    PredDict : dict
        Dictionary to store all the NN model metadata, now with trained NN models.

    """
    for mlabel in PredDict['mlabel']:
        # Index of model setup
        mID = PredDict['mlabel'].index(mlabel)
        
        Model = PredDict['model'][mID]
        
        X_train = PredDict['X_train'][mID]
        y_train = PredDict['y_train'][mID]
        X_test = PredDict['X_test'][mID]
        y_test = PredDict['y_test'][mID]
        
        # Train the model on the training data, setting aside a small split of 
        # this data for validation 
        start=time.time() # start timer
        PredDict['history'].append(Model.fit(X_train, y_train, 
                                             epochs=PredDict['epochS'][mID], batch_size=PredDict['batchS'][mID], 
                                             validation_split=0.1, verbose=1))
        end=time.time() # end timer
        
        # Time taken to train model
        PredDict['train_time'].append(end-start)
        
        # Evaluate the model
        loss, accuracy = Model.evaluate(X_test, y_test)
        PredDict['loss'].append(loss)
        PredDict['accuracy'].append(accuracy)
        
        # Save trained models in dictionary for posterity
        with open(f"{os.path.join(filepath, sitename)}/predictions/{'_'.join(PredDict['mlabel'])}.pkl", 'wb') as f:
            pickle.dump(PredDict, f)
            
    return PredDict
    
    
def CreateSequences(X, y, time_steps=1):
    '''
    Function to create sequences
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
    if len(X) > time_steps:  # Check if there's enough data
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)]) # Slice feature set into sequences using moving window of size = number of timesteps
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)
    else:
        # Not enough data to create a sequence
        print(f"Not enough data to create sequences with time_steps={time_steps}")
        return np.array([]), np.array([])


def CostSensitiveLoss(falsepos_cost, falseneg_cost, binary_thresh):
    """
    Create a cost-sensitive loss function to implement within an NN model.compile() step.
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
