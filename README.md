# CoastLearn

CoastLearn is a Python toolkit for coastal monitoring and modelling using machine learning approaches. Currently, the main toolset is for extracting coastal vegetation edges from satellite imagery, built from the CoastSat toolbox (https://github.com/kvos/CoastSat).

## Description

## Usage

## Examples

## Installation
The Python tool relies on packages downloaded through Anaconda and the Google Earth Engine API to run. The preliminary step is downloading this repository. You can do this either by clicking the <span style="color:white">**Code**</span> button at the top and downloading then extracting the zipped folder, or by navigating to where you want to download it on your local machine and running 
```
$ git clone https://github.com/fmemuir/CoastLearn-main.git
```
from a command line.

### 1.1 Create an Anaconda enviroment

To run the toolbox you first need to install the required Python packages in an environment. If you don't already have it, **Anaconda** can be downloaded freely [here](https://www.anaconda.com/download/).

Once you have Anaconda installed on your PC:
- Windows: open the Anaconda Prompt (not Powershell)
- Mac and Linux: open a terminal window)
Navigate to the folder where you downloaded this repository.

Create a new environment named `coastsat` with all the required packages by entering these commands in succession:

```
conda create -n coastsat python=3.8
conda activate coastsat
conda install -c conda-forge geopandas earthengine-api scikit-image matplotlib astropy notebook -y
pip install pyqt5
```

All the required packages have now been installed in an environment called `coastsat`. Always make sure that the environment is activated with:

```
conda activate coastsat
```


## Support

## Roadmap

## Contributions
We are in testing phase and not currently taking contributions, but please [reach out to Freya](mailto:f.muir.1@research.gla.ac.uk) with suggestions!

## Authors and acknowledgements
This tool is heavily based on work by Kilian Vos at University of New South Wales. The veg adaptation for the tool was originally conceived by Freya Muir and Martin Hurst, and was executed, tested and refined by Freya Muir and Luke Richardson-Foulger. Please see these papers for more details: