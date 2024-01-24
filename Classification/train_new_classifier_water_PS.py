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


os.chdir('/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/ModelsFrameworks/CoastSat.PlanetScope/coastsat_ps/classifier')


#%% read kml files for the training sites

# filepaths 
filepath_images = os.path.join(os.getcwd(), 'Data')
filepath_train = os.path.join(os.getcwd(), 'training_data')
filepath_models = os.path.join(os.getcwd(), 'models')

#%% 1. Download images
# 
# For each site on which you want to train the classifier, save a .kml file with the region of interest (5 vertices clockwise, first and last points are the same, can be created from Google myMaps) in the folder *\training_sites*.

sitename = 'PS_NARRA_new'
lonmin, lonmax = -4.036261,-3.995186
latmin, latmax = 57.853515,57.889722


sat_list = ['PSScene4Band']
# sat_list = ['L5']#, 'L8', 'S2']

# directory where the data will be stored
filepath = Toolbox.CreateFileStructure(sitename, sat_list)

projection_epsg = 27700
image_epsg = 32630
cloud_thresh = 0.5

train_sites = [sitename]
print('Sites for training:\n%s\n'%train_sites)

polygon, point = Toolbox.AOI(lonmin, lonmax, latmin, latmax,sitename, image_epsg)
# it's recommended to convert the polygon to the smallest rectangle (sides parallel to coordinate axes)       
polygon = Toolbox.smallest_rectangle(polygon)



# In[49]:


dates = ['2022-01-01', '2022-12-31'] # PSScene
# dates = ['2019-06-01', '2019-08-31']
# dates = ['2019-01-01', '2020-01-01']
# dates = ['2004-01-01', '2006-01-01']
# dates = ['2004-01-01', '2020-01-01']

# put all the inputs into a dictionnary
inputs = {
    'polygon': polygon,
    'dates': dates, 
    'daterange':dates, 
    'sat_list': sat_list, 
    'sitename': sitename, 
    'filepath':filepath_images,
    'cloud_thresh': cloud_thresh
}

settings = {
    'filepath_train':filepath_train, # folder where the labelled images will be stored
    'labels':{'sand':1,'white-water':2,'water':3,'other land features':4}, # labels for the classifier,
    'colors':{'sand':[1, 0.65, 0],'white-water':[1,0,1],'water':[0.1,0.1,0.7],'other land features':[0.8,0.8,0.1]},
    'tolerance':0.01, # this is the pixel intensity tolerance, when using flood fill for sandy pixels
                             # set to 0 to select one pixel at a time
    'ref_epsg': 4326,
    'max_dist_ref': 500,
    # general parameters:
    'cloud_thresh': cloud_thresh,        # threshold on maximum cloud cover
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

#%%
Download.check_images_available(inputs)

#%% Populate metadata (or skip if it already exists)


Sat = Toolbox.image_retrieval(inputs)
metadata = Toolbox.metadata_collection(inputs, Sat)


#%% 2. Label Images


# label the images with an interactive annotator
for site in train_sites:
    settings['inputs']['sitename'] = sitename
    settings['cloud_mask_issue'] = False
    # label images
    Classifier.label_vegimages(metadata, settings)


#%% 3. Train Classifier (Skip previous 2 and run this if already trained)
# 
# A Multilayer Perceptron is trained with *scikit-learn*. To train the classifier, the training data needs to be loaded.
# 
# You can use the data that was labelled here and/or the original CoastSat training data.

# load labelled images
features,labelmaps = Classifier.load_labels(train_sites, settings)

#%% 3.1 Combine Additional Classifier Data

features2,labelmaps2 = Classifier.load_labels(['DornochAberdeenFull'], settings) # rerun settings to redefine

newnonveg = np.append(features['nonveg'], features2['nonveg'], axis=0)
newveg = np.append(features['veg'], features2['veg'], axis=0)
newfeatures = dict({'veg':newveg,'nonveg':newnonveg})

features = newfeatures

for i in range(len(labelmaps2['filenames'])):
    labelmaps['filenames'].append(labelmaps2['filenames'][i])
    labelmaps['labelmaps'].append(labelmaps2['labelmaps'][i])
    
print('full features:\nveg:', str(len(features['veg'])), '\nnonveg:',str(len(features['nonveg'])) )

#%% 4. Subsample
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


#%% 5. Divide the dataset into train and test
#train on 70% of the data and evaluate on the other 30%

# divide in train and test and evaluate the classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)
runs = []
for i in range(5):
    start_time = timeit.default_timer()
    classifier = MLPClassifier(hidden_layer_sizes=(16,8,4), solver='adam')
    classifier.fit(X_train,y_train)
    runs.append(classifier.score(X_test,y_test))
    print('Accuracy: %0.4f' % classifier.score(X_test,y_test))
    print(str(round(timeit.default_timer() - start_time, 5)) + ' seconds elapsed')



#%% [OPTIONAL] Run heuristically through different values of nodes and layers
# divide in train and test and evaluate the classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)

# hidden_layer_sizes default is (100,) which represents one layer with 100 nodes
# goal is to cycle through different combinations to test accuracy
layerparams = [(1,), (5,), (10,), (50,), (100,), (500,), 
                (1,1), (5,5), (10,10), (50,50), (100,100), (500,500), 
                (1,1,1), (5,5,5), (10,10,10), (50,50,50), (100,100,100), (500,500,500)]
# layerparams = [(50,10,50),(100,50,10,50,100),(500,100,50,10,50,100,500)]

scores10k2 = pd.DataFrame(columns=['layers','accuracy','timeelapsed'])

for i, layerparam in enumerate(layerparams):
    start_time = timeit.default_timer()
    classifier = MLPClassifier(hidden_layer_sizes=layerparam, solver='adam')
    classifier.fit(X_train,y_train)
    scores10k2.loc[i] = [str(layerparam), classifier.score(X_test,y_test), round(timeit.default_timer() - start_time, 5)]
    print('Accuracy with %s: %0.6f' % (str(layerparam), classifier.score(X_test,y_test)))
    print(str(round(timeit.default_timer() - start_time, 5)) + ' seconds elapsed')

#%%
with open(os.path.join(filepath_train,'DornochScores.pkl'),'rb') as f:
    scores, scores10k = pickle.load(f)
#%% Plot bar graph of accuracies

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(18, 6))

# plot the same data on both axes
ax.bar(scores10k['layers'],scores10k['accuracy'],width=0.5)
ax2.bar(scores10k['layers'],scores10k['accuracy'],width=0.5)

# zoom-in / limit the view to different portions of the data
ax.set_ylim(0.992, 1.)  # outliers only
ax2.set_ylim(0, 0.55)  # most of the data

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
ax.set_ylabel('Accuracy')

d = .008  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# What's cool about this is that now if we vary the distance between
# ax and ax2 via fig.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'

# plt.xticks(scores10k['layers'], scores10k['layers'])
plt.tight_layout()
plt.savefig(os.path.join(filepath_train,'VegTraining_Accuracies.png'))
plt.show()

fig = plt.figure(figsize=(18, 6))

# plot the same data on both axes
plt.bar(scores10k['layers'],scores10k['timeelapsed'],width=0.5, color='darkgray')
plt.ylabel('Time elapsed to train (sec)')
plt.xlabel('Model config.')

plt.tight_layout()
plt.savefig(os.path.join(filepath_train,'VegTraining_TimeElapsed.png'))
plt.show()



#%% [OPTIONAL] 10-fold cross-validation (may take a few minutes to run)


# cross-validation
scores = cross_val_score(classifier, X, y, cv=10)
print('Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std() * 2))


# Plot a confusion matrix:

#%% plot confusion matrix


y_pred = classifier.predict(X_test)

Classifier.plot_confusion_matrix(y_test, y_pred,
                                    classes=['veg','nonveg'],
                                    normalize=False);


# When satisfied with the accuracy and confusion matrix, train the model using ALL the training data and save it:

#%% 6. Train with all the data and save the final classifier

start_time = timeit.default_timer()
classifier = MLPClassifier(hidden_layer_sizes=(16,8,4), solver='adam')
classifier.fit(X,y)
# joblib.dump(classifier, os.path.join(filepath_models, sitename+'_MLPClassifier_Veg_S2.pkl'))
joblib.dump(classifier, os.path.join(filepath_models, 'MLPClassifier_Veg_PSScene.pkl'))
# joblib.dump(classifier, os.path.join(filepath_models, 'MLPClassifier_Veg_Full.pkl'))

print(str(round(timeit.default_timer() - start_time, 5)) + ' seconds elapsed')

#%% Evaluate the classifier
# Load a classifier that you have trained (specify the classifiers filename) and evaluate it on the satellite images.
# 
# This section will save the output of the classification for each site in a directory named \evaluation.

# load and evaluate a classifier
# get_ipython().run_line_magic('matplotlib', 'qt')
classifier = joblib.load(os.path.join(filepath_models, sitename+'_MLPClassifier_Veg_S2.pkl'))
settings['output_epsg'] = 32630
settings['min_beach_area'] = 200
settings['buffer_size'] = 250
settings['min_length_sl'] = 500
settings['cloud_thresh'] = 0.5
# visualise the classified images
for site in train_sites:
    settings['inputs']['sitename'] = site[:site.find('.')] 
    # load metadata
    metadata = Toolbox.metadata_collection(sat_list, Sat, filepath, sitename)
    # plot the classified images
    Classifier.evaluate_classifier(classifier,metadata,settings)





#%% Seasonal specific tests
settings['inputs']['sitename'] = 'DornochSummer'
sumfeatures = Classifier.load_labels(train_sites, settings)
settings['inputs']['sitename'] = 'DornochWinter'
winfeatures = Classifier.load_labels(train_sites, settings)

#%% 
n_samples = 10000
    
for key in ['veg', 'nonveg']:
    sumfeatures[key] =  sumfeatures[key][np.random.choice(sumfeatures[key].shape[0], n_samples, replace=False),:]
# print classes again
for key in sumfeatures.keys():
    print('%s : %d pixels'%(key,len(sumfeatures[key])))
for key in ['veg', 'nonveg']:
    winfeatures[key] =  winfeatures[key][np.random.choice(winfeatures[key].shape[0], n_samples, replace=False),:]
# print classes again
for key in winfeatures.keys():
    print('%s : %d pixels'%(key,len(winfeatures[key])))
        
classes = ['veg','nonveg']
labels = [1,2]
sumX,sumy = Classifier.format_training_data(sumfeatures, classes, labels)
winX,winy = Classifier.format_training_data(winfeatures, classes, labels)

#%% divide in train and test and evaluate the classifier on opposite season models
sumX_train, sumX_test, sumy_train, sumy_test = train_test_split(sumX, sumy, test_size=0.3, shuffle=True, random_state=0)
winX_train, winX_test, winy_train, winy_test = train_test_split(winX, winy, test_size=0.3, shuffle=True, random_state=0)

#%%
classifier = MLPClassifier(hidden_layer_sizes=(100,50), solver='adam')
# train classifier on summer data
classifier.fit(sumX_train,sumy_train)
# test summer model on winter data
print('Summer model on winter data Accuracy: %0.4f' % classifier.score(winX_test,winy_test))

winy_pred = classifier.predict(winX_test)
Classifier.plot_confusion_matrix(winy_test, winy_pred,
                                    classes=['veg','nonveg'],
                                    normalize=False);

#%%
classifier = MLPClassifier(hidden_layer_sizes=(100,50), solver='adam')
# train classifier on winter data
classifier.fit(winX_train,winy_train)
# test winter model on summer data
print('Winter model on summer data Accuracy: %0.4f' % classifier.score(sumX_test,sumy_test))

sumy_pred = classifier.predict(sumX_test)
Classifier.plot_confusion_matrix(sumy_test, sumy_pred,
                                    classes=['veg','nonveg'],
                                    normalize=False);