# CoastLearn

CoastLearn is a Python toolkit for coastal monitoring and modelling using machine learning approaches. 

## Description and Scope
The goal of this toolkit is to have a fully operational framework for predicting coastal change, using machine learning techniques that are trained with satellite observations. We have a plethora of satellite imagery being generated every day to be used freely in a number of automated, API-based ways. These datasets are therefore well-suited to machine learning approaches which require a lot of data to train sufficiently. With just one satellite image, multiple indicators of coastal change can be automatically extracted such as wave breaking zones, wet-dry boundaries, high water marks and vegetation edges. These automatically extracted indicators can then be fed into a machine learning network which makes future predictions based on the past changes and relationships between these indicators. The result is an automated, early warning system for coastal erosion at a potentially global scale.

Currently, the main toolset is for extracting <b><span style="color:#2EA043">coastal vegetation edges</span></b> from satellite imagery, built from the CoastSat toolbox (https://github.com/kvos/CoastSat).

https://github.com/fmemuir/CoastLearn-main/assets/22475417/cb27e704-f361-4f34-b999-dcd5c990816c


## Enhancements
Various improvements have been made to the toolkit to address more accurate approaches recently reported on, and to incorporate new Python packages and tools for more seamlessness. These are detailed further in the methods paper (), but include:

* The use of geemap to download and process satellite imagery from Google Earth Engine entirely from within the cloud server;
* Improved transect creation based on the Dynamic Coast project's Coastal Mapping Tools;
* The use of geopandas to handle geospatial data (both loading in and exporting out) and for transect+shoreline intersections;
* Beach width (the distance between vegetation edge and wet-dry line) extracted for each transect (based on calling of some of the original CoastSat functions to classify the water line);
* Validation functions to quantify the error between satellite-derived vegetation edges and ground-truthed validation edges (from ground surveys or manual digitisation of aerial imagery);
* Various plotting functions, such as violin plots for distances between satellite lines and validation lines, and GIFs of extracted edges and their respective satellite images.


## Installation

Jump to the short version [down below](#install-quick-version)

### 1.1 Download the code
The Python tool relies on packages downloaded through Anaconda and the Google Earth Engine API to run. The preliminary step is downloading this repository. You can do this either by clicking the <span style="color:white;background-color:#2EA043;">Code</span> button at the top and downloading + extracting the zipped folder, or by navigating to where you want to download it on your local machine and running 
```
git clone https://github.com/fmemuir/CoastLearn-main.git
```
from a command line (if you have git command line tools installed).

### 1.2 Create a conda enviroment

To run the toolbox you first need to install the required Python packages in an environment. If you don't already have it, **Anaconda** can be downloaded freely [here](https://www.anaconda.com/download/).

Once you have Anaconda installed on your PC:
- Windows: open the Anaconda Prompt (not Powershell)
- Mac and Linux: open a terminal window

and navigate to the folder with the repository files. If you downloaded the code zip file manually, it's recommended you extract the files to a new local folder rather than keeping it in your Downloads!.

Navigate to the repository folder and then create a new `conda` environment named `coastlearn` with all the required packages by entering this command (make sure you're in the repo folder!):

```
cd CoastLearn

conda env create -f coastlearn_environment.yml
```
Please note that solving and building the environment can take some time (minutes to hours depending on the the nature of your base environment). Once this step is complete, all the required packages will have been installed in an environment called `coastlearn`. Always make sure that the environment is activated with:

```
conda activate coastlearn
```
### 1.3 Activate Google Earth Engine API

This tool uses Google Earth Engine (GEE) API to access satellite image metadata. You need to request access to GEE API by signing up at https://signup.earthengine.google.com/ with a Google account and filling in a few questions about your intended usage (the safest bet is 'research'). It can take up to 24 hours to approve a request, but it's usually fairly quick. 

In the meantime, you will also need to install a program called Google Cloud Command Line Interface (gcloud CLI). It shouldn't matter where you download this to. Find installation instructions here: https://cloud.google.com/sdk/docs/install. 

Once your GEE request has been approved, you should get a confirmation email. Open a prompt/terminal window and `activate coastlearn` environment. Run this command to link your `conda` environment to the GEE server:

```
earthengine authenticate
```

A web browser will open; log in with the GMail account you used to sign up to GEE. The authenticator should then redirect back to your terminal window. If it doesn't, copy+paste the authorization code into the terminal.


### **INSTALL QUICK VERSION**
1. Download repo: `$ git clone https://github.com/fmemuir/CoastLearn-main.git`
2. Create conda environment: `conda env create -f coastlearn_environment.yml`
3. Activate env: `conda activate coastlearn`
4. Authenticate GEE: `earthengine authenticate`

**Remember!**: Always run `conda activate coastlearn` each time you want to use the toolbox. You *should not* need to authenticate `earthengine` each time, just the once when installing. 


## Run-through Example

https://github.com/fmemuir/CoastLearn-main/assets/22475417/1bd4722b-ece9-4ed9-a9ac-104f71c241d7

There are 7 main steps to setting up the vegetation extraction tool. You can see [this paper]() for a flowchart and more info on the methodology. These steps are run from a driver file which takes care of all the user-driven params when setting up a new site. The main steps found in a driver file are:

1. Import relevant packages (including initialising the `earthengine` tools);
2. Define an area of interest. For the time being, this must be **smaller than 262144 (512 x 512) pixels**, equivalent to 5.12 x 5.12 km for Sentinel and 7.68 x 7.68 km for Landsat;
3. Define image parameters (start and end date, satellites, CRS/projections, sitename);
4. Retrieve and save image collection metadata*;
5. Set coastal boundary parameters (cloud cover threshold, plotting flags, minimum area for contouring);
6. Define a reference shore along which to create a buffer (boundaries will only be extracted along here);
7. Run the main edge extraction function.

You can use the [`VegEdge_DriverTemplate.ipynb`](https://github.com/fmemuir/CoastLearn-main/blob/master/VegEdge_DriverTemplate.ipynb) or [`VegEdge_DriverTemplate.py`](https://github.com/fmemuir/CoastLearn-main/blob/master/VegEdge_DriverTemplate.py) files to create a driver file for your own site.

*<sub>This is an update from the original CoastSat toolkit! Raw satellite images will **not** be downloaded, but merely the filenames will be passed to `geemap` and converted from the cloud server straight to `numpy` arrays. This is to save time and bandwidth. TIFs of true colour images and their classified and NDVI counterparts will however be exported throughout the process to be explored in a GIS environment.</sub>

The tool takes all the input settings the user has defined, and performs these steps:

1. Preprocess each image in the metadata collection (downsample or pansharpen, mask clouds, clean nodata);
2. Create buffer around reference shoreline (or most recent shore extracted, useful for dynamic shores and image collections over a long period);
3. Classify image using the pre-trained neural network; 
4. Show/adjust detected boundary between image classes (depending on if user has requested to be shown the interactive plot window);
5. Export boundaries and relevant metadata to a `.pkl` file and a shapefile of lines.

## Roadmap
This code is live and the master branch is being updated often (daily to weekly). If you clone this repo, please update it regularly with `git pull`!

## Contributions
We are in testing phase and not currently taking contributions, but [reach out to Freya](mailto:f.muir.1@research.gla.ac.uk) with suggestions.

## Authors and acknowledgements
This tool is based on work by Kilian Vos ([github: kvos](https://github.com/kvos)) at University of New South Wales. The veg adaptation for the tool was originally conceived by Freya Muir, Luke Richardson-Foulger and Martin Hurst, and was executed, tested and refined by Freya Muir and Luke Richardson-Foulger.
