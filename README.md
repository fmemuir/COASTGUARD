# CoastLearn

CoastLearn is a Python toolkit for coastal monitoring and modelling using machine learning approaches. Currently, the main toolset is for extracting coastal vegetation edges from satellite imagery, built from the CoastSat toolbox (https://github.com/kvos/CoastSat).

## Description and Scope
The goal of this toolkit is to have a fully operational framework for predicting coastal change, using machine learning techniques that are trained with satellite observations. We have a plethora of satellite imagery being generated every day to be used freely in a number of automated, API-based ways. These datasets are therefore well-suited to machine learning approaches which require a lot of data to train sufficiently. With just one satellite image, multiple indicators of coastal change can be automatically extracted such as wave breaking zones, wet-dry boundaries, high water marks and vegetation edges. These automatically extracted indicators can then be fed into a machine learning network which makes future predictions based on the past changes and relationships between these indicators. The result is an automated, early warning system for coastal erosion at a potentially global scale.

## Installation

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

Create a new `conda` environment named `coastlearn` with all the required packages by entering this command (make sure you're in the repo folder!):

```
conda env create -f coastlearn_environment.yml
```

All the required packages have now been installed in an environment called `coastlearn`. Always make sure that the environment is activated with:

```
conda activate coastlearn
```
### 1.3 Activate Google Earth Engine API

This tool uses Google Earth Engine (GEE) API to access satellite image metadata. You need to request access to GEE API by signing up at https://signup.earthengine.google.com/ with a Google account and filling in a few questions about your intended usage (the safest bet is 'research'). It can take up to 24 hours to approve a request, but it's usually fairly quick. 

In the meantime, you will also need to install a program called Google Cloud for the authenticator to work. It shouldn't matter where you download this to. Find installation instructions here: https://cloud.google.com/sdk/docs/install. 

Once your GEE request has been approved, you should get a confirmation email. Open a prompt/terminal window and activate the `coastlearn` environment. Run this command to link your `conda` environment to the GEE server:

```
earthengine authenticate
```

A web browser will open; log in with the GMail account you used to sign up to GEE. The authenticator should then redirect back to your terminal window. If it doesn't, copy+paste the authorization code into the terminal.

**INSTALL TL;DR**
1. Download repo: `$ git clone https://github.com/fmemuir/CoastLearn-main.git`
2. Create conda environment: `conda env create -f coastlearn_environment.yml`
3. Activate env: `conda activate coastlearn`
4. Authenticate GEE: `earthengine authenticate`

**Remember!**: Always run `conda activate coastlearn` each time you want to use the toolbox. You *should not* need to authenticate `earthengine` each time, just the once when installing. 

## Run-through Example
There are 7 main steps to setting up the vegetation extraction tool. You can see [this paper]() for a flowchart and more info on the methodology. These steps are run from a driver file which takes care of all the user-driven params when setting up a new site. The main steps found in a driver file are:

1. Import relevant packages (including initialising the `earthengine` tools);
2. Define an area of interest. For the time being, this must be **smaller than 262144 (512 x 512) pixels**, equivalent to 5.12 x 5.12 km for Sentinel and 7.68 x 7.68 km for Landsat;
3. Define image parameters (start and end date, satellites, CRS/projections, sitename);
4. Retrieve and save image collection metadata*;
5. Set coastal boundary parameters (cloud cover threshold, plotting flags, minimum area for contouring);
6. Define a reference shore along which to create a buffer (boundaries will only be extracted along here);
7. Run the main edge extraction function.

You can use the `CL_DriverTemplate.ipynb` or `CL_DriverTemplate.py` files to create a driver file for your own site.

*<sub>This is an update from the original CoastSat toolkit! Raw satellite images will **not** be downloaded, but merely the filenames will be passed to `geemap` and converted from the cloud server straight to `numpy` arrays. This is to save time and bandwidth. TIFs of true colour images and their classified and NDVI counterparts will however be exported throughout the process to be explored in a GIS environment.</sub>

The tool takes all the input settings the user has defined, and performs these steps:

1. Preprocess each image in the metadata collection (downsample or pansharpen, mask clouds, clean nodata);
2. Create buffer around reference shoreline (or most recent shore extracted, useful for dynamic shores and image collections over a long period);
3. Classify image using the pre-trained neural network; 
4. Show/adjust detected boundary between image classes (depending on if user has requested to be shown the interactive plot window);
5. Export boundaries and relevant metadata to a `.pkl` file and a shapefile of lines.

## Roadmap
This code is live and the master branch is being updated often (weekly). If you clone this repo, please update it regularly with `git pull`!

## Contributions
We are in testing phase and not currently taking contributions, but please [reach out to Freya](mailto:f.muir.1@research.gla.ac.uk) with suggestions!

## Authors and acknowledgements
This tool is heavily based on work by Kilian Vos ([github: kvos](https://github.com/kvos)) at University of New South Wales. The veg adaptation for the tool was originally conceived by Freya Muir and Martin Hurst, and was executed, tested and refined by Freya Muir and Luke Richardson-Foulger. Please see these papers for more information: