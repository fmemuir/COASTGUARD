#!/usr/bin/env python
# coding: utf-8

# # Train  a new classifier for VegSat
# 
# In this notebook the VegSat classifier is trained using satellite images from new sites. This can improve the accuracy of the veg edge detection if the users are experiencing issues with the default classifier.

# #### Initial settings

# In[8]:


# load modules
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import os, sys
import glob
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib import gridspec
import matplotlib.dates as mdates
plt.ion()
from datetime import datetime, timezone, timedelta
import timeit

from Toolshed import Classifier, Download, Image_Processing, Shoreline, Toolbox, Transects, VegetationLine

import mpl_toolkits as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import MaxNLocator

import seaborn as sns; sns.set()
import math
import geemap
import ee
import pprint
from shapely import geometry
from shapely.geometry import Point, LineString
import pandas as pd
import geopandas as gpd
import matplotlib.cm as cm
import pyproj
from IPython.display import clear_output
import scipy
from scipy import optimize
import csv
import math

# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import sklearn
if sklearn.__version__[:4] == '0.20':
    from sklearn.externals import joblib
else:
    import joblib

# coastsat modules
sys.path.insert(0, os.pardir)

ee.Initialize()

#%%
os.chdir('/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/ModelsFrameworks/CoastLearn-main/Classification')

# plotting params
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12

# filepaths 
filepath_images = os.path.join(os.getcwd(), 'data')
filepath_train = os.path.join(os.getcwd(), 'training_data')
filepath_models = os.path.join(os.getcwd(), 'models')

# settings
settings ={'filepath_train':filepath_train, # folder where the labelled images will be stored
           'cloud_thresh':0.9, # percentage of cloudy pixels accepted on the image
           'cloud_mask_issue':False, # set to True if problems with the default cloud mask 
           'inputs':{'filepath':filepath_images}, # folder where the images are stored
           'labels':{'veg':1,'non-veg':2}, # labels for the classifier
           'colors':{'veg':[0.3, 0.65, 0.3],'non-veg':[0.58,0.51,0.8]},
           'tolerance':0.01, # this is the pixel intensity tolerance, when using flood fill for sandy pixels
                             # set to 0 to select one pixel at a time
            }
        

#%% read kml files for the training sites
filepath_sites = os.path.join(os.getcwd(), 'training_sites')
#train_sites = [os.path.basename(x) for x in glob.glob(filepath_sites+'/*.kml')]
train_sites = [os.path.basename(x) for x in glob.glob(filepath_sites+'/MONTROSE.kml')]
print('Sites for training:\n%s\n'%train_sites)


#%% 1. Download images
# 
# For each site on which you want to train the classifier, save a .kml file with the region of interest (5 vertices clockwise, first and last points are the same, can be created from Google myMaps) in the folder *\training_sites*.
# 
# You only need a few images (~10) to train the classifier.


##ST ANDREWS EAST
# sitename = 'Montrose'
# lonmin, lonmax = -2.49, -2.42
# latmin, latmax = 56.70, 56.75

sitename = 'Dornoch'
lonmin, lonmax = -4.033, -3.996
latmin, latmax = 57.855, 57.885


train_sites = [sitename]
print('Sites for training:\n%s\n'%train_sites)

#point = ee.Geometry.Point([lonmin, latmin]) 
if latmin > latmax:
    print('Check your latitude min and max bounding box values!')
    oldlatmin = latmin
    oldlatmax = latmax
    latmin = oldlatmax
    latmax = oldlatmin
if lonmin > lonmax:
    print('Check your longitude min and max bounding box values!')
    oldlonmin = lonmin
    oldlonmax = lonmax
    lonmin = oldlonmax
    lonmax = oldlonmin
    
polygon = [[[lonmin, latmin],[lonmax, latmin],[lonmin, latmax],[lonmax, latmax]]]
point = ee.Geometry.Point(polygon[0][0]) 
polygon = Toolbox.smallest_rectangle(polygon)

filepath = os.path.join(os.getcwd(), 'Data')
direc = os.path.join(filepath, sitename)

if os.path.isdir(direc) is False:
    os.mkdir(direc)


# In[49]:


sat_list = ['S2']
dates = ['2019-06-01', '2019-08-31']
projection_epsg = 27700
image_epsg = 32630

# filepaths 
filepath_images = os.path.join(os.getcwd(), 'Data')
filepath_train = os.path.join(os.getcwd(), 'training_data')
filepath_models = os.path.join(os.getcwd(), 'models')

# put all the inputs into a dictionnary
inputs = {
    'polygon': polygon,
    'dates': dates, 
    'daterange':dates, 
    'sat_list': sat_list, 
    'sitename': sitename, 
    'filepath':filepath_images
}

Download.check_images_available(inputs)


#settings = {
#    'filepath_train':filepath_train, # folder where the labelled images will be stored
#    'labels':{'sand':1,'white-water':2,'water':3,'vegetation':4,'urban':5}, # labels for the classifier
#    'colors':{'sand':[1, 0.65, 0],'white-water':[1,0,1],'water':[0.1,0.1,0.7],'other land features':[0.8,0.8,0.1]},
settings = {
    'filepath_train':filepath_train, # folder where the labelled images will be stored
    'labels':{'veg':1,'nonveg':2}, # labels for the classifier
    'colors':{'veg':[0.2,0.8,0.2],'nonveg':[0.39,0.58,0.93]},
    'tolerance':0.01, # this is the pixel intensity tolerance, when using flood fill for sandy pixels
                             # set to 0 to select one pixel at a time
    'ref_epsg': 4326,
    'max_dist_ref': 500,
    # general parameters:
    'cloud_thresh': 0.9,        # threshold on maximum cloud cover
    'output_epsg': image_epsg,     # epsg code of spatial reference system desired for the output   
    # quality control:
    'check_detection': True,    # if True, shows each shoreline detection to the user for validation
    'adjust_detection': True,  # if True, allows user to adjust the postion of each shoreline by changing the threhold
    'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    'min_beach_area': 200,     # minimum area (in metres^2) for an object to be labelled as a beach
    'buffer_size': 250,         # radius (in metres) for buffer around sandy pixels considered in the shoreline detection
    'min_length_sl': 500,       # minimum length (in metres) of shoreline perimeter to be valid
    'cloud_mask_issue': True,  # switch this parameter to True if sand pixels are masked (in black) on many images  
    'sand_color': 'bright',    # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
    # add the inputs defined previously
    'inputs': inputs,
    'projection_epsg': projection_epsg,
    'hausdorff_threshold':3*(10**50)
}


# In[50]:


Sat = Toolbox.image_retrieval(inputs)
metadata = Toolbox.metadata_collection(sat_list, Sat, filepath, sitename)


#%% 2. Label Images


# label the images with an interactive annotator
get_ipython().run_line_magic('matplotlib', 'qt')
for site in train_sites:
    settings['inputs']['sitename'] = sitename
    settings['cloud_mask_issue'] = False
    # label images
    Classifier.label_vegimages(metadata, polygon, Sat, settings)


#%% 3. Train Classifier
# 
# A Multilayer Perceptron is trained with *scikit-learn*. To train the classifier, the training data needs to be loaded.
# 
# You can use the data that was labelled here and/or the original CoastSat training data.

# load labelled images
features = Classifier.load_labels(train_sites, settings)


#%% [OPTIONAL] original CoastSat data

# you can also load the original CoastSat training data (and optionally merge it with your labelled data)
with open(os.path.join(settings['filepath_train'], 'CoastSat_training_set_L8.pkl'), 'rb') as f:
    features_original = pickle.load(f)
for key in features_original.keys():
    print('%s : %d pixels'%(key,len(features_original[key])))


#%% Run this section to combine the original training data with your labelled data:


# add the white-water data from the original training data
features['white-water'] = np.append(features['white-water'], features_original['white-water'], axis=0)
# or merge all the classes
# for key in features.keys():
#     features[key] = np.append(features[key], features_original[key], axis=0)
#features = features_original 
for key in features.keys():
    print('%s : %d pixels'%(key,len(features[key])))


#%% [OPTIONAL] Subsample
#As the classes do not have the same number of pixels, it is good practice to subsample the very large classes 
# (in this case 'veg' and 'other land features')

# subsample randomly the land and water classes
# as the most important class is 'sand', the number of samples should be close to the number of sand pixels
n_samples = 10000
for key in ['veg', 'nonveg']:
    features[key] =  features[key][np.random.choice(features[key].shape[0], n_samples, replace=False),:]
# print classes again
for key in features.keys():
    print('%s : %d pixels'%(key,len(features[key])))




#%% X Y formatting
# When the labelled data is ready, format it into X, a matrix of features, and y, a vector of labels:

# format into X (features) and y (labels) 
classes = ['veg','nonveg']
labels = [1,2]
X,y = Classifier.format_training_data(features, classes, labels)


# 

#%% Divide the dataset into train and test
#train on 70% of the data and evaluate on the other 30%


# divide in train and test and evaluate the classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)
classifier = MLPClassifier(hidden_layer_sizes=(100,50), solver='adam')
classifier.fit(X_train,y_train)
print('Accuracy: %0.4f' % classifier.score(X_test,y_test))


#%% [OPTIONAL] 10-fold cross-validation (may take a few minutes to run)


# cross-validation
scores = cross_val_score(classifier, X, y, cv=10)
print('Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std() * 2))


# Plot a confusion matrix:

#%% plot confusion matrix


[1::2]# plot confusion matrix
get_ipython().run_line_magic('matplotlib', 'inline')
y_pred = classifier.predict(X_test)
SDS_classify.plot_confusion_matrix(y_test, y_pred,
                                   classes=['other land features','sand','white-water','water'],
                                   normalize=False);


# When satisfied with the accuracy and confusion matrix, train the model using ALL the training data and save it:

#%% Train with all the data and save the final classifier

start_time = timeit.default_timer()
classifier = MLPClassifier(hidden_layer_sizes=(100,50), solver='adam')
classifier.fit(X,y)
joblib.dump(classifier, os.path.join(filepath_models, sitename+'_MLPClassifier_Veg_S2.pkl'))
print(str(round(timeit.default_timer() - start_time), 5) + ' seconds elapsed')

#%%4. Evaluate the classifier
# Load a classifier that you have trained (specify the classifiers filename) and evaluate it on the satellite images.
# 
# This section will save the output of the classification for each site in a directory named \evaluation.

# load and evaluate a classifier
get_ipython().run_line_magic('matplotlib', 'qt')
classifier = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_test.pkl'))
settings['output_epsg'] = 3857
settings['min_beach_area'] = 4500
settings['buffer_size'] = 200
settings['min_length_sl'] = 200
settings['cloud_thresh'] = 0.5
# visualise the classified images
for site in train_sites:
    settings['inputs']['sitename'] = site[:site.find('.')] 
    # load metadata
    metadata = SDS_download.get_metadata(settings['inputs'])
    # plot the classified images
    SDS_classify.evaluate_classifier(classifier,metadata,settings)

