# COASTGUARD

COASTGUARD ($\textcolor{#00B0B7}{\textsf{C}}$ oastal $\textcolor{#00B0B7}{\textsf{O}}$ bservation + $\textcolor{#00B0B7}{\textsf{A}}$ nalysis using $\textcolor{#00B0B7}{\textsf{S}}$ atellite-derived $\textcolor{#00B0B7}{\textsf{T}}$ imeseries, $\textcolor{#00B0B7}{\textsf{G}}$ enerated $\textcolor{#00B0B7}{\textsf{U}}$ sing $\textcolor{#00B0B7}{\textsf{A}}$ I + $\textcolor{#00B0B7}{\textsf{R}}$ eal-time $\textcolor{#00B0B7}{\textsf{D}}$ ata) is a Python toolkit for coastal monitoring and modelling using machine learning approaches. 


## Description and Scope
The goal of this toolkit is to have a fully operational framework for predicting coastal change, using machine learning techniques that are trained with satellite observations. We have a plethora of satellite imagery being generated every day to be used freely in a number of automated, API-based ways. These datasets are therefore well-suited to machine learning approaches which require a lot of data to train sufficiently. With just one satellite image, multiple indicators of coastal change can be automatically extracted such as wave breaking zones, wet-dry boundaries, high water marks and vegetation edges. These automatically extracted indicators can then be fed into a machine learning network which makes future predictions based on the past changes and relationships between these indicators. The result is an automated, early warning system for coastal erosion at a potentially global scale.

Currently, the main toolset <b>VedgeSat</b> is for extracting $\textcolor{#2EA043}{\textsf{coastal vegetation edges}}$ from satellite imagery, built from the CoastSat toolbox (https://github.com/kvos/CoastSat).

https://github.com/fmemuir/COASTGUARD/assets/22475417/cb27e704-f361-4f34-b999-dcd5c990816c


## Enhancements
Various improvements have been made to the toolkit to address more accurate approaches recently reported on, and to incorporate new Python packages and tools for more seamlessness. These are detailed further in the methods paper (), but include:

* The use of geemap to download and process satellite imagery from Google Earth Engine entirely from within the cloud server;
* Improved transect creation based on the Dynamic Coast project's Coastal Mapping Tools;
* The use of geopandas to handle geospatial data (both loading in and exporting out) and for transect+shoreline intersections;
* Beach width (the distance between vegetation edge and wet-dry line) extracted for each transect (based on calling of some of the original CoastSat functions to classify the water line);
* Validation functions to quantify the error between satellite-derived vegetation edges and ground-truthed validation edges (from ground surveys or manual digitisation of aerial imagery);
* Various plotting functions, such as violin plots for distances between satellite lines and validation lines, and GIFs of extracted edges and their respective satellite images.


## Installation

### **INSTALL QUICK VERSION**
1. Download repo: `$ git clone https://github.com/fmemuir/COASTGUARD.git`
2. Create conda environment: `conda env create -f coastguard_environment.yml`
3. Activate env: `conda activate coastguard`
4. Authenticate GEE: `earthengine authenticate`

**Remember!**: Always run `conda activate coastguard` each time you want to use the toolbox. You *should not* need to authenticate `earthengine` each time, just the once when installing.


### 1.1 Download the code
The Python tool relies on packages downloaded through Anaconda and the Google Earth Engine API to run. The preliminary step is downloading this repository. You can do this either by clicking the <span style="color:white;background-color:#2EA043;">Code</span> button at the top and downloading + extracting the zipped folder, or by navigating to where you want to download it on your local machine and running 
```
git clone https://github.com/fmemuir/COASTGUARD.git
```
from a command line (if you have git command line tools installed).

### 1.2 Create a conda enviroment

To run the toolbox you first need to install the required Python packages in an environment. If you don't already have it, **Anaconda** can be downloaded freely [here](https://www.anaconda.com/download/).

Once you have Anaconda installed on your PC:
- Windows: open the Anaconda Prompt (not Powershell)
- Mac and Linux: open a terminal window

and navigate to the folder with the repository files. If you downloaded the code zip file manually, it's recommended you extract the files to a new local folder rather than keeping it in your Downloads!

Navigate to the COASTGUARD repository folder and then create a new `conda` environment named `coastguard` with all the required packages by entering this command (make sure you're in the repo folder!):
```
cd COASTGUARD

conda update -n base conda

conda create --name coastguard python=3.10
```
Note: the Python version is currently dependent on `pyfes`, see these issues [here](https://github.com/CNES/aviso-fes/issues/19) for details.

#### OPTIONAL: Install pyFES for FES2014 tidal corrections
If you would like to call the original CoastSat functions to get cross-shore shoreline timeseries, there are some tidal correction shortcuts you can now run using the [FES2014] tide model. The [tidal corrections](https://www.sciencedirect.com/science/article/pii/S1364815219300490#sectitle0065) are performed using a calculation based on shoreface slope, to correct cross-shore waterline positions (that will be biased by the tidal height at the time an image was captured) to a standard elevation. You don't need this for the VedgeSat vegetation routines, ONLY the waterline extraction routines. But if you do want them, you must run these steps **FIRST** before installing the other packages:
```
conda activate coastguard

conda update --all -c conda-forge

conda install pyfes -c fbriol
```
You can then continue with installing the other packages (a list of which can be found in [coastguard_environment.yml](https://github.com/fmemuir/COASTGUARD/blob/master/coastguard_environment.yml) in the repo):
```
conda install -c conda-forge earthengine-api geopandas spyder geemap scikit-image matplotlib rasterio seaborn astropy geopy notebook motuclient netcdf4
```


Please note that solving and building the environment can take some time (minutes to hours depending on the the nature of your base environment). If you want to make things go faster, it's recommended you solve the conda environment installation with [Mamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community). You can set Mamba as the default conda solver with these steps:
```
conda update -n base conda

conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Once the installation steps are complete, all the required packages will have been installed in an environment called `coastguard`. Always make sure that the environment is activated with:
```
conda activate coastguard
```
before you start working with the tools each time.


### 1.3 Activate Google Earth Engine API

This tool uses Google Earth Engine (GEE) API to access satellite image metadata. You need to request access to GEE API by signing up at [https://signup.earthengine.google.com/](https://signup.earthengine.google.com/) with a Google account and filling in a few questions about your intended usage (the safest bet is 'research'). It can take up to 24 hours to approve a request, but it's usually fairly quick. 

In the meantime, you will also need to install a program called Google Cloud Command Line Interface (gcloud CLI). It shouldn't matter where you download this to. Find installation instructions here: [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install). 

Once your GEE request has been approved, you should get a confirmation email. Open a prompt/terminal window and `activate coastguard` environment. Run this command to link your `conda` environment to the GEE server:

```
earthengine authenticate
```

A web browser will open; log in with the GMail account you used to sign up to GEE. The authenticator should then redirect back to your terminal window. If it doesn't, copy+paste the authorization code into the terminal.
 


## Getting Started

The process of extracting coastal vegetation edges from satellite data is run through a driver file. Driver files can be customised for your own site of interest. There are a couple of template examples in the repository to help you get started. 
The interactive python notebook [`VedgeSat_DriverTemplate.ipynb`](https://github.com/fmemuir/COASTGUARD/blob/master/VedgeSat_DriverTemplate.ipynb) can be viewed and executed in an interactive notebook environment such as jupyter-notebook which can be launched at the command line:
```
(coastguard) $ jupyter-notebook VedgeSat_DriverTemplate.ipynb
```
Alternatively, you can customise and run the standard python script [`VedgeSat_DriverTemplate.py`](https://github.com/fmemuir/COASTGUARD/blob/master/VedgeSat_DriverTemplate.py) using a python IDE such as spyder:
```
(coastguard) $ spyder VedgeSat_DriverTemplate.py
```
https://github.com/fmemuir/COASTGUARD/assets/22475417/1bd4722b-ece9-4ed9-a9ac-104f71c241d7

There are 7 main steps to setting up the vegetation extraction tool. You can see [this paper]() for a flowchart and more info on the methodology. These steps are run from a driver file which takes care of all the user-driven params when setting up a new site. The main steps found in a driver file are:

1. Import relevant packages (including initialising the `earthengine` tools);
2. Define an area of interest. For the time being, this must be **smaller than 262144 (512 x 512) pixels**, equivalent to 5.12 x 5.12 km for Sentinel and 7.68 x 7.68 km for Landsat;
3. Define image parameters (start and end date, satellites, CRS/projections, sitename);
4. Retrieve and save image collection metadata*;
5. Set coastal boundary parameters (cloud cover threshold, plotting flags, minimum area for contouring);
6. Define a reference shore along which to create a buffer (boundaries will only be extracted along here);
7. Run the main edge extraction function.

*<sub>This is an update from the original CoastSat toolkit! Raw satellite images will **not** be downloaded, but merely the filenames will be passed to `geemap` and converted from the cloud server straight to `numpy` arrays. This is to save time and bandwidth. TIFs of true colour images and their classified and NDVI counterparts will however be exported throughout the process to be explored in a GIS environment.</sub>

The tool takes all the input settings the user has defined, and performs these steps:

1. Preprocess each image in the metadata collection (downsample or pansharpen, mask clouds, clean nodata);
2. Create buffer around reference shoreline (or most recent shore extracted, useful for dynamic shores and image collections over a long period);
3. Classify image using the pre-trained neural network; 
4. Show/adjust detected boundary between image classes (depending on if user has requested to be shown the interactive plot window);
5. Export boundaries and relevant metadata to a `.pkl` file and a shapefile of lines.

### Extracting Waterlines Alongside Vegetation Edges
As this tool is built from the original CoastSat toolkit, it is possible to extract instantaneous waterlines as well as vegetation edges from each satellite image. To do this, change the `wetdry` flag in the user requirements to `True`. Any tidal correction on the extracted waterlines is performed using the FES2014 tidal model. You will need to use pyFES and the Aviso FES2014 repo for this, which **you should clone** from [their github repo](https://github.com/CNES/aviso-fes). When loading in the tidal data, you should **change the tidal files path to wherever you have cloned the FES2014 repo to on your machine.**

## Roadmap
This code is live and the master branch is being updated often (daily to weekly). If you clone this repo, please update it regularly with `git pull`!

## Contributions
We are in testing phase and not currently taking contributions, but [reach out to Freya](mailto:f.muir.1@research.gla.ac.uk) with suggestions.

## Authors and acknowledgements
This tool is based on work by Kilian Vos ([github: kvos](https://github.com/kvos)) at University of New South Wales. The veg adaptation for the tool was originally conceived by Freya Muir, Luke Richardson-Foulger and Martin Hurst, and was executed, tested and refined by Freya Muir and Luke Richardson-Foulger.


