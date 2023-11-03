#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:40:16 2023

@author: mhurst
"""

# import module
import pyfes, datetime
import numpy as np


# optional parameters
StartDate = datetime.datetime(1983,1,1)
Latitude = 55.946752316996154
Longitude = -3.067919473233273

# Path to the configuration file that contains the definition of grids to use to compute the ocean tide
OceanPath = "./aviso-fes/data/fes2014/ocean_tide_extrapolated.ini"
# Path to the configuration file that contains the definition of grids to use to compute the radial tide
RadialPath = "./aviso-fes/data/fes2014/load_tide.ini" 

# Create handler
short_tide = pyfes.Handler('ocean', 'io', OceanPath)
radial_tide = pyfes.Handler('radial', 'io', RadialPath)

# Creating the time series
dates = np.array([StartDate + datetime.timedelta(seconds=item * 3600) for item in range(24)])
print(dates)