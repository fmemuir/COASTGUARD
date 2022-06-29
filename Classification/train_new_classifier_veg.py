#!/usr/bin/env python
# coding: utf-8

# # Train  a new classifier for VegSat
# 
# In this notebook the VegSat classifier is trained using satellite images from new sites. This can improve the accuracy of the veg edge detection if the users are experiencing issues with the default classifier.

# #### Initial settings

# In[46]:


# load modules
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os, sys
import glob
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

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
#from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_classify
from Elves import Toolbox, Image_Processing, Download, Shoreline, Classifier

# plotting params
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12

# filepaths 
os.chdir('/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/ModelsFrameworks/CoastWatch-main/Classification')
filepath_images = os.path.join(os.getcwd(), 'data')
filepath_train = os.path.join(os.getcwd(), 'training_data')
filepath_models = os.path.join(os.getcwd(), 'models')

image_epsg = 32630
projection_epsg = 27700

# settings
settings ={'filepath_train':filepath_train, # folder where the labelled images will be stored
           'cloud_thresh':0.9, # percentage of cloudy pixels accepted on the image
           'cloud_mask_issue':False, # set to True if problems with the default cloud mask 
           'inputs':{'filepath':filepath_images}, # folder where the images are stored
           'labels':{'veg':1,'non-veg':2}, # labels for the classifier
           'colors':{'veg':[0.3, 0.65, 0.3],'non-veg':[0.58,0.51,0.8]},
           'tolerance':0.01, # this is the pixel intensity tolerance, when using flood fill for sandy pixels
                             # set to 0 to select one pixel at a time
            'projection_epsg':projection_epsg,
            'ref_epsg':image_epsg,
            'output_epsg':image_epsg
            }
        
# read kml files for the training sites
filepath_sites = os.path.join(os.getcwd(), 'training_sites')
#train_sites = [os.path.basename(x) for x in glob.glob(filepath_sites+'/*.kml')]
train_sites = [os.path.basename(x) for x in glob.glob(filepath_sites+'/STANDREWS.kml')]
print('Sites for training:\n%s\n'%train_sites)


# ### 1. Download images
# 
# For each site on which you want to train the classifier, save a .kml file with the region of interest (5 vertices clockwise, first and last points are the same, can be created from Google myMaps) in the folder *\training_sites*.
# 
# You only need a few images (~10) to train the classifier.

#%%


# dowload images at the sites
dates = ['2019-01-01', '2019-07-01']
sat_list = ['L8']
for site in train_sites:
    polygon = SDS_tools.polygon_from_kml(os.path.join(filepath_sites,site))
    polygon = SDS_tools.smallest_rectangle(polygon)
    sitename = site[:site.find('.')]  
    inputs = {'polygon':polygon, 'dates':dates, 'sat_list':sat_list,
             'sitename':sitename, 'filepath':filepath_images}
    print(sitename)
    metadata = SDS_download.retrieve_images(inputs)


#%%


lonmin, lonmax = -2.89087, -2.79878
latmin, latmax = 56.34700, 56.36987
polygon = [[[lonmin, latmin],[lonmax, latmin],[lonmin, latmax],[lonmax, latmax]]]
#point = ee.Geometry.Point(polygon[0][0]) 

# for already downloaded images
for site in train_sites:
    polygon = Toolbox.smallest_rectangle(polygon)
    sitename = 'StAndrewsPlanet'
    sat_list = ['PSScene4Band']
    filepath = os.path.join(os.getcwd(),'..','Data')
    inputs = {'polygon': polygon, 
              'sat_list': sat_list, 
              'sitename': sitename, 
              'filepath':filepath}

Sat = Toolbox.LocalImageRetrieval(inputs)
metadata = Toolbox.LocalImageMetadata(inputs, Sat)


#%%


metadata['PSScene4Band']['dates']=metadata['PSScene4Band']['dates'][1]
metadata['PSScene4Band']['acc_georef']=metadata['PSScene4Band']['acc_georef'][1]
metadata['PSScene4Band']['epsg']=metadata['PSScene4Band']['epsg'][1]
metadata['PSScene4Band']['filenames']=metadata['PSScene4Band']['filenames'][1]



#%%  2. Label images
# 
# Label the images into 2 classes: veg, non-veg.
# 
# The labelled images are saved in the *filepath_train* and can be visualised afterwards for quality control. If yo make a mistake, don't worry, this can be fixed later by deleting the labelled image.


# label the images with an interactive annotator
get_ipython().run_line_magic('matplotlib', 'qt')
for site in train_sites:
    settings['inputs']['sitename'] = sitename
    settings['cloud_mask_issue'] = False
    # label images
    Classifier.label_images(metadata, polygon, Sat, settings)


#%% 3. Train Classifier
# 
# A Multilayer Perceptron is trained with *scikit-learn*. To train the classifier, the training data needs to be loaded.
# 
# You can use the data that was labelled here and/or the original CoastSat training data.

# In[ ]:


# load labelled images
features = Classifier.load_labels(train_sites, settings)


# In[ ]:


# you can also load the original CoastSat training data (and optionally merge it with your labelled data)
with open(os.path.join(settings['filepath_train'], 'CoastSat_training_set_L8.pkl'), 'rb') as f:
    features_original = pickle.load(f)
for key in features_original.keys():
    print('%s : %d pixels'%(key,len(features_original[key])))


# Run this section to combine the original training data with your labelled data:

# In[ ]:


# add the white-water data from the original training data
features['white-water'] = np.append(features['white-water'], features_original['white-water'], axis=0)
# or merge all the classes
# for key in features.keys():
#     features[key] = np.append(features[key], features_original[key], axis=0)
#features = features_original 
for key in features.keys():
    print('%s : %d pixels'%(key,len(features[key])))


# [OPTIONAL] As the classes do not have the same number of pixels, it is good practice to subsample the very large classes (in this case 'water' and 'other land features'):

# In[ ]:


# subsample randomly the land and water classes
# as the most important class is 'sand', the number of samples should be close to the number of sand pixels
n_samples = 5000
for key in ['water', 'other land features']:
    features[key] =  features[key][np.random.choice(features[key].shape[0], n_samples, replace=False),:]
# print classes again
for key in features.keys():
    print('%s : %d pixels'%(key,len(features[key])))


# When the labelled data is ready, format it into X, a matrix of features, and y, a vector of labels:

# In[ ]:


# format into X (features) and y (labels) 
classes = ['veg','non-veg']
labels = [1,2]
X,y = SDS_classify.format_training_data(features, classes, labels)


# Divide the dataset into train and test: train on 70% of the data and evaluate on the other 30%:

# In[ ]:


# divide in train and test and evaluate the classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)
classifier = MLPClassifier(hidden_layer_sizes=(100,50), solver='adam')
classifier.fit(X_train,y_train)
print('Accuracy: %0.4f' % classifier.score(X_test,y_test))


# [OPTIONAL] A more robust evaluation is 10-fold cross-validation (may take a few minutes to run):

# In[ ]:


# cross-validation
scores = cross_val_score(classifier, X, y, cv=10)
print('Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std() * 2))


# Plot a confusion matrix:

# In[ ]:


[1::2]# plot confusion matrix
get_ipython().run_line_magic('matplotlib', 'inline')
y_pred = classifier.predict(X_test)
SDS_classify.plot_confusion_matrix(y_test, y_pred,
                                   classes=['other land features','sand','white-water','water'],
                                   normalize=False);


# When satisfied with the accuracy and confusion matrix, train the model using ALL the training data and save it:

# In[ ]:


# train with all the data and save the final classifier
classifier = MLPClassifier(hidden_layer_sizes=(100,50), solver='adam')
classifier.fit(X,y)
joblib.dump(classifier, os.path.join(filepath_models, 'NN_4classes_Landsat_test.pkl'))


# ### 4. Evaluate the classifier
# 
# Load a classifier that you have trained (specify the classifiers filename) and evaluate it on the satellite images.
# 
# This section will save the output of the classification for each site in a directory named \evaluation.

# In[ ]:


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

