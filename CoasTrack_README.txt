This CoasTrack_StAndrews_FullS2Run_README.txt file was generated on 2025-04-28 by FREYA M. E. MUIR


GENERAL INFORMATION

1. Title of Dataset: 

2. Author Information
	A. Principal Investigator Contact Information
		Name: Freya Muir
		Institution: University of Glasgow
		Address: School of Geographical and Earth Sciences
		Email: f.muir.1@research.gla.ac.uk

	B. Associate or Co-investigator Contact Information
		Name: Martin Hurst
		Institution: University of Glasgow
		Address: School of Geographical and Earth Sciences
		Email: f.muir.1@research.gla.ac.uk


3a. Temporal period of data collection:
2025-01-24 to 2025-03-19

3b. Temporal period the data itself covers:
2015-06-28 to 2025-01-08

4. Geographic location of data collection (xmin,ymin, xmax,ymax in EPSG:32630):
510452,6244012, 512442,6252182

5. Information about funding sources that supported the collection of the data: 
This work was supported by the Natural Environment Research Council via an IAPETUS2 PhD studentship held by Freya M. E. Muir (grant reference NE/S007431/1). Contributions were provided by CASE partner JBA Trust and in-kind support was provided by JBA Consulting.


SHARING/ACCESS INFORMATION

1. Licenses/restrictions placed on the data: 
None (CC-BY)

2. Links to publications that cite or use the data: 
Muir, F. M. E, Hurst, M. D., Naylor, L. A., Rennie, A. F., 2025. Towards a digital twin for coastal geomorphology: coupling satellite-derived vegetation edges and other remotely sensed metrics. ISPRS Journal of Photogrammetry and Remote Sensing [in review]

3. Links to other publicly accessible locations of the data: 
N/A

4. Links/relationships to ancillary data sets: 
All additional functions called within the Python file are held in the COASTGUARD Python toolbox: https://github.com/fmemuir/COASTGUARD

5. Was data derived from another source? 
yes
	A. Source(s): 
	- Copernicus Sentinel-2 data 2015-2025. Retrieved from Google Earth Engine, processed by ESA (https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2);
	- North West Atlantic Shelf Wave Physics Reanalysis provided by the EU Copernicus Marine Service (2025). Available from: \href{https://doi.org/10.48670/moi-00060}{https://doi.org/10.48670/moi-00060}
	- Scottish Public Sector Lidar (Phase 5) provided with Crown copyright by Scottish Government, SEPA and Fugro (2021). Available from: \href{https://remotesensingdata.gov.scot/}{https://remotesensingdata.gov.scot/}.
	- The FES2022 Tide product was funded by CNES, produced by LEGOS, NOVELTIS and CLS and made freely available by AVISO (2024). Available from: \href{https://doi.org/10.24400/527896/A01-2024.004}{https://doi.org/10.24400/527896/A01-2024.004}. 
	

DATA & FILE OVERVIEW

1. File List: 

- CoasTrack_StAndrews_FullS2Run_README.txt: in-depth description of data;

- CoasTrack_StAndrews_FullS2Run.py: VedgeSat driver file for obtaining the attached data and running the analyses associated with the publication;

- img_files:
	- 1,638 x geoTIFF image files (“YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_T30VWH_*.tif”) representing Copernicus Sentinel-2 multispectral images at different stages of VedgeSat/CoastSat processing, clipped to the outer Eden Estuary in Scotland.
	- “_RGB” is the red-green-blue composite image (3 bands of range 0–1);
	- “_NDVI” is the Normalised Difference Vegetation Index calculated from the red and near-infrared spectral bands (range -1–1);
	- “_CLASS” is the classified image with values representing different land cover types (1=vegetation, 2=non-vegetation);
	- “_TZ” is the vegetation transition zone as a binary raster (1=transitional vegetation).

- shapefiles:
	- StAndrewsEastS2Full2024_2015-06-28_2025-01-08_veglines_clean_clip.shp: coastal vegetation edges extracted automatically from Copernicus Sentinel-2 public satellite imagery using VedgeSat (housed in the COASTGUARD Python toolbox: https://github.com/fmemuir/COASTGUARD);
	- StAndrewsEastS2Full2024_2015-06-28_2025-01-08_waterlines_clean_clip.shp: coastal waterlines extracted automatically from Copernicus Sentinel-2 public satellite imagery using CoastSat (housed in the COASTGUARD Python toolbox: https://github.com/fmemuir/COASTGUARD).
	- StAndrewsEastS2Full2024_Transects_Intersected.shp: cross-shore transects defined using the CoasTrack functions within the COASTGUARD toolbox. The base transect file holds statistics and data related to the cross-shore intersection of each transect and each satellite-derived vegetation edge; 
	- StAndrewsEastS2Full2024_Transects_Intersected_Water.shp: transects intersected with satellite-derived waterlines;
	- StAndrewsEastS2Full2024_Transects_Intersected_Slope.shp: transects intersected with Scottish Government Phase 5 lidar for the site (https://www.data.gov.uk/dataset/78daa005-99d9-4f9f-ac0a-29dd52c59d66/lidar-for-scotland-phase-5-dtm); 
	- StAndrewsEastS2Full2024_Transects_Intersected_TZ.shp: transects intersected with vegetation transition zone rasters derived from Copernicus Sentinel-2 imagery; 
	- StAndrewsEastS2Full2024_Transects_Intersected_Waves.shp: transects intersected with a gridded timeseries of offshore wave conditions modelled using MetOffice data and hosted by Copernicus Marine Service (“NWSHELF_REANALYSIS_WAV_004_015”, https://doi.org/10.48670/moi-00060).

- transect_intersections:
	- StAndrewsEastS2Full2024_transect_intersects.pkl: serialised Python object file, holing a variable of GeoDataFrame type, representing cross-shore transects across the outer Eden Estuary and vegetation edge statistics from intersecting each transect with StAndrewsEastS2Full2024_2015-06-28_2025-01-08_veglines_clean_clip.shp. 
	- StAndrewsEastS2Full2024_transect_water_intersects.pkl: the same transect GeoDataFrame but holding statistics from intersecting each transect with StAndrewsEastS2Full2024_2015-06-28_2025-01-08_waterlines_clean_clip.shp; 
	- StAndrewsEastS2Full2024_transect_wave_intersects.pkl: the transect GeoDataFrame intersected with statistics from the gridded timeseries of offshore wave conditions from Copernicus Marine Service;
	- StAndrewsEastS2Full2024_transect_topo_intersects.pkl: the transect GeoDataFrame holding statistics from intersection with Scottish Government Phase 5 lidar.


2. Relationship between files, if important: 
.shp files must be kept in the same folder as their auxiliary files at all times (.shx, .prj, .dbf, .cpg, .qmd) as they are of type ESRI Shapefile.

3. Additional related data collected that was not included in the current data package:
Original source datasets (see SHARING/ACCESS INFORMATION 5); not included for efficiency, as data is publicly available.

4. Are there multiple versions of the dataset? 
No


METHODOLOGICAL INFORMATION

1. Description of methods used for collection/generation of data: 
All data curation and processing information is within the COASTGUARD documentation (https://github.com/fmemuir/COASTGUARD) and driver file (CoasTrack_StAndrews_FullS2Run.py), as well as the paper associated with this data deposit.

2. Methods for processing the data: 
All methods can be found in the paper associated with this data deposit.

3. Instrument- or software-specific information needed to interpret the data: 
Packages and versions can be found in the .yml file (below) in the COASTGUARD repository (https://github.com/fmemuir/COASTGUARD). To run any of the Python files (in an IDE or from the command line), Anaconda should be used to create an environment: "conda env create -f coastguard.yml":
channels:
  - conda-forge
  - fbriol
dependencies:
  - pip
  - pip:
    - copernicusmarine>=1.0,<=2.0
  - python=3.10
  - pyfes
Shapefiles and geoTIFFs can be viewed in any GIS software.

4. Standards and calibration information, if appropriate: 
N/A

5. Environmental/experimental conditions: 
N/A

6. Describe any quality-assurance procedures performed on the data:
QA information is available from the individual original data sources (see SHARING/ACCESS INFORMATION 5).

7. People involved with sample collection, processing, analysis and/or submission: 
Freya M. E. Muir


-----------------------------------------------------------------------------------------

DATA-SPECIFIC INFORMATION FOR: img_files/YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_T30VWH_RGB.tif

- Copernicus Sentinel-2 red-green-blue composite image, clipped to the outer Eden Estuary in Scotland. Filename format is: (date and time of sensing start)_(date and time of product discriminator which is usually sensing end)_(tile identification number)_RGB.tif. Reflectance values have been normalised to between 0 and 1.

1. Number of variables/bands: 
3

2. Number of cases/rows/pixels:
X: 199 Y: 817 (pixel size: 10. -10)

3. Variable List: 
- Band 1: red spectral band reflectance (0 to 1);
- Band 2: green spectral band reflectance (0 to 1);
- Band 3: blue spectral band reflectance (0 to 1)

4. Missing data codes: 
nan

5. Specialized formats or other abbreviations used: 
None


DATA-SPECIFIC INFORMATION FOR: img_files/YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_T30VWH_NDVI.tif

- Copernicus Sentinel-2 Normalised Difference Vegetation Index image, clipped to the outer Eden Estuary in Scotland. Filename format is: (date and time of sensing start)_(date and time of product discriminator which is usually sensing end)_(tile identification number)_NDVI.tif. NDVI (range -1 to 1) is calculated with the band index formula: (near-infrared - red) / (near-infrared + red).

1. Number of variables/bands: 
1

2. Number of cases/rows/pixels: 
X: 199 Y: 817 (pixel size: 10. -10)

3. Variable List: 
- Band 1: NDVI value, dimensionless index representing strength of vegetation presence (-1 to 1)

4. Missing data codes: 
nan

5. Specialized formats or other abbreviations used: 
None


DATA-SPECIFIC INFORMATION FOR: img_files/YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_T30VWH_CLASS.tif

- Copernicus Sentinel-2 image classified into different land cover classes automatically using the VedgeSat tool, clipped to the outer Eden Estuary in Scotland. Filename format is: (date and time of sensing start)_(date and time of product discriminator which is usually sensing end)_(tile identification number)_CLASS.tif. The multilayer perceptron classifier can be found in the COASTGUARD repository (https://github.com/fmemuir/COASTGUARD).

1. Number of variables/bands: 
1

2. Number of cases/rows/pixels:
X: 199 Y: 817 (pixel size: 10. -10)

3. Variable List: 
- Band 1: binary classes (1=vegetation, 2=non-vegetation)

4. Missing data codes: 
nan

5. Specialized formats or other abbreviations used: 
None


DATA-SPECIFIC INFORMATION FOR: img_files/YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_T30VWH_TZ.tif

- Copernicus Sentinel-2 image of pixels classed as falling within the vegetation transition zone, a zone of overlap between _CLASS image classes representing NDVI values that are sometimes classed as vegetation and sometimes as non-vegetation, clipped to the outer Eden Estuary in Scotland. Filename format is: (date and time of sensing start)_(date and time of product discriminator which is usually sensing end)_(tile identification number)_TZ.tif. The multilayer perceptron classifier can be found in the COASTGUARD repository (https://github.com/fmemuir/COASTGUARD).

1. Number of variables/bands: 
1

2. Number of cases/rows/pixels: 
X: 199 Y: 817 (pixel size: 10. -10)

3. Variable List: 
- Band 1: binary class (1=transitional vegetation)

4. Missing data codes: 
nan

5. Specialized formats or other abbreviations used: 
None


DATA-SPECIFIC INFORMATION FOR: shapefiles/StAndrewsEastS2Full2024_2015-06-28_2025-01-08_veglines_clean_clip.shp

- Coastal vegetation edges extracted automatically from Copernicus Sentinel-2 public satellite imagery using VedgeSat (housed in the COASTGUARD Python toolbox: https://github.com/fmemuir/COASTGUARD)

1. Number of variables/bands: 
9

2. Number of cases/rows/pixels:
2,465 

3. Variable List: 
- dates: date of satellite image capture (string, length 80, precision 0);
- times: time of satellite image capture (string, length 80, precision 0);
- filename: Google Earth Engine server filename of matching satellite image (string, length 80, precision 0);
- cloud_cove: percentage of cloud cover over image (proportion of pixels classed as cloud) (float, length 23, precision 15);
- idx: satellite image ID (integer, length 18, precision 0);
- vthreshold: threshold Normalised Difference Vegetation Index used to extract veg edge contour along (float, length 23, precision 15);
- wthreshold: threshold Modified Normalised Difference Water Index used to extract waterline contour along (float, length 23, precision 15);
- tideelev: tidal elevation at date and time of satellite image capture, derived from FES2022 global tide model (float, length 23, precision 15);
- satname: abbreviated name of satellite platform sourcing the imagery (chosen from L5/L7/L8/L9=Landsat 5/7/8/9, S2=Sentinel-2, PS=PlanetScope)

4. Missing data codes: 
nan

5. Specialized formats or other abbreviations used: 
None


DATA-SPECIFIC INFORMATION FOR: shapefiles/StAndrewsEastS2Full2024_2015-06-28_2025-01-08_waterlines_clean_clip.shp

- Coastal waterlines extracted automatically from Copernicus Sentinel-2 public satellite imagery using CoastSat (housed in the COASTGUARD Python toolbox: https://github.com/fmemuir/COASTGUARD).

1. Number of variables/bands: 
9

2. Number of cases/rows/pixels: 
2,116

3. Variable List: 
- dates: date of satellite image capture (string, length 80, precision 0);
- times: time of satellite image capture (string, length 80, precision 0);
- filename: Google Earth Engine server filename of matching satellite image (string, length 80, precision 0);
- cloud_cove: percentage of cloud cover over image (proportion of pixels classed as cloud) (float, length 23, precision 15);
- idx: satellite image ID (integer, length 18, precision 0);
- vthreshold: threshold Normalised Difference Vegetation Index used to extract veg edge contour along (float, length 23, precision 15);
- wthreshold: threshold Modified Normalised Difference Water Index used to extract waterline contour along (float, length 23, precision 15);
- tideelev: tidal elevation at date and time of satellite image capture, derived from FES2022 global tide model (float, length 23, precision 15);
- satname: abbreviated name of satellite platform sourcing the imagery (chosen from L5/L7/L8/L9=Landsat 5/7/8/9, S2=Sentinel-2, PS=PlanetScope)

4. Missing data codes: 
nan

5. Specialized formats or other abbreviations used: 
None


DATA-SPECIFIC INFORMATION FOR: shapefiles/StAndrewsEastS2Full2024_Transects_Intersected.shp

- Cross-shore transects defined using the CoasTrack functions within the COASTGUARD toolbox. The base transect file holds statistics and data related to the cross-shore intersection of each transect with each satellite-derived vegetation edge.

1. Number of variables/bands: 
20

2. Number of cases/rows/pixels: 
1,412

3. Variable List: 
- LineID: reference shoreline ID number (integer, length 18, precision 0);
- TransectID: transect ID number (integer, length 18, precision 0);
- reflinepnt: shapely POINT object of intersection between transect and reference shoreline, formatted as a Python list of WKTs (string, length 254, precision 0); 
- dates: dates of satellite image capture for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings '%y-%m-%d' (string, length 254, precision 0); 
- times: times of satellite image capture for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings '%H:%M:%S.%f' (string, length 254, precision 0); 
- filename: Google Earth Engine server filename of matching satellite image for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings (string, length 254, precision 0); 
- cloud_cove: percentage of cloud cover over image (proportion of pixels classed as cloud) for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- idx: satellite image ID for timeseries of vegetation edges intersected with transect, formatted as a Python list of integers (string, length 254, precision 0); 
- vthreshold: threshold Normalised Difference Vegetation Index used to extract veg edge contour along for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- wthreshold: threshold Modified Normalised Difference Water Index used to extract waterline contour along for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- tideelev: tidal elevation at date and time of satellite image capture, derived from FES2022 global tide model for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- satname:  abbreviated name of satellite platform sourcing the imagery for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings (string, length 254, precision 0); 
- interpnt:  shapely POINT object of intersection between transect and each vegetation edge in timeseries, formatted as a Python list of WKTs (string, length 254, precision 0); 
- distances: distance along transect in metres of intersection point between transect and each vegetation edge in timeseries, formatted as a Python list of floats (string, length 254, precision 0); 
- olddate: oldest vegetation edge capture date in %y-%m-%d (string, length 80, precision 0); 
- youngdate: youngest/most recent vegetation edge capture date, in %y-%m-%d (string, length 80, precision 0); 
- oldyoungT: number of (decimal) years between the oldest and youngest vegetation edge capture dates (float, length 23, precision 15); 
- oldyoungRt: rate of cross-shore change between the oldest and youngest vegetation edge, calculated using linear regression, in metres per year (float, length 23, precision 15); 
- recentT: number of (decimal) years between the second youngest and youngest vegetation edge capture dates (float, length 23, precision 15); 
- recentRt: rate of cross-shore change between the second youngest and youngest vegetation edge, calculated using linear regression, in metres per year (float, length 23, precision 15)

4. Missing data codes: 
nan

5. Specialized formats or other abbreviations used: 
None


DATA-SPECIFIC INFORMATION FOR: shapefiles/StAndrewsEastS2Full2024_Transects_Intersected_Water.shp

- Cross-shore transects defined using the CoasTrack functions within the COASTGUARD toolbox, holding statistics and data related to the cross-shore intersection of each transect with each satellite-derived vegetation edge and satellite-derived waterline.

1. Number of variables/bands: 
37

2. Number of cases/rows/pixels: 
1,412

3. Variable List: 
- LineID: reference shoreline ID number (integer, length 18, precision 0);
- TransectID: transect ID number (integer, length 18, precision 0);
- reflinepnt: shapely POINT object of intersection between transect and reference shoreline, formatted as a Python list of WKTs (string, length 254, precision 0); 
- dates: dates of satellite image capture for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings '%y-%m-%d' (string, length 254, precision 0); 
- times: times of satellite image capture for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings '%H:%M:%S.%f' (string, length 254, precision 0); 
- filename: Google Earth Engine server filename of matching satellite image for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings (string, length 254, precision 0); 
- cloud_cove: percentage of cloud cover over image (proportion of pixels classed as cloud) for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- idx: satellite image ID for timeseries of vegetation edges intersected with transect, formatted as a Python list of integers (string, length 254, precision 0); 
- vthreshold: threshold Normalised Difference Vegetation Index used to extract veg edge contour along for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- wthreshold: threshold Modified Normalised Difference Water Index used to extract waterline contour along for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- tideelev: tidal elevation at date and time of satellite image capture, derived from FES2022 global tide model for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- satname:  abbreviated name of satellite platform sourcing the imagery for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings (string, length 254, precision 0); 
- interpnt: shapely POINT object of intersection between transect and each vegetation edge in timeseries, formatted as a Python list of WKTs (string, length 254, precision 0); 
- distances: distance along transect in metres of intersection point between transect and each vegetation edge in timeseries, formatted as a Python list of floats (string, length 254, precision 0); 
- olddate: oldest vegetation edge capture date in %y-%m-%d (string, length 80, precision 0); 
- youngdate: youngest/most recent vegetation edge capture date, in %y-%m-%d (string, length 80, precision 0); 
- oldyoungT: number of (decimal) years between the oldest and youngest vegetation edge capture dates (float, length 23, precision 15); 
- oldyoungRt: rate of cross-shore change between the oldest and youngest vegetation edge, calculated using linear regression, in metres per year (float, length 23, precision 15); 
- recentT: number of (decimal) years between the second youngest and youngest vegetation edge capture dates (float, length 23, precision 15); 
- recentRt: rate of cross-shore change between the second youngest and youngest vegetation edge, calculated using linear regression, in metres per year (float, length 23, precision 15);
- normdists: distance along transect in metres of intersection point between transect and each vegetation edge in timeseries, normalised to distance of first intersection, formatted as a Python list of floats (string, length 254, precision 0); 
- wldates: dates of satellite image capture for timeseries of waterlines intersected with transect, formatted as a Python list of strings '%y-%m-%d' (string, length 254, precision 0); 
- wltimes: times of satellite image capture for timeseries of waterlines intersected with transect, formatted as a Python list of strings '%H:%M:%S.%f' (string, length 254, precision 0); 
- wldists: distance along transect in metres of intersection point between transect and each waterline in timeseries, formatted as a Python list of floats (string, length 254, precision 0); 
- wlinterpnt: shapely POINT object of intersection between transect and each waterline in timeseries, formatted as a Python list of WKTs (string, length 254, precision 0); 
- wlcorrdist: distance along transect in metres of intersection point between transect and each waterline in timeseries, corrected to remove the effects of tides and wave runup, formatted as a Python list of floats (string, length 254, precision 0); 
- beachslope: intertidal beach slope in degrees, calculated using the frequency domain analysis on satellite-derived waterlines from Vos et al. (2020) (https://doi.org/10.1029/2020GL088365) (float, length 23, precision 15); 
- beachwidth: distance along transect in metres each vegetation edge and corrected waterline in timeseries, formatted as a Python list of floats (string, length 254, precision 0); 
- tidezone: tidal zone (lower = 0 to 33%, middle = 33% to 66%, upper = 66% to 100%) at date and time of satellite image capture, derived from FES2022 global tide model for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0);
- olddateW: oldest waterline capture date in %y-%m-%d (string, length 80, precision 0); 
- youngdateW: youngest/most recent waterline capture date in %y-%m-%d (string, length 80, precision 0); 
- oldyoungTW: number of (decimal) years between the oldest and youngest waterline capture dates (float, length 23, precision 15); 
- oldyungRtW: rate of cross-shore change between the oldest and youngest waterline, calculated using linear regression, in metres per year (float, length 23, precision 15); 
- oldyungMEW: margin of error (plus or minus) on the rate of cross-shore change between the oldest and youngest waterline, in metres per year (float, length 23, precision 15); 
- recentTW: number of (decimal) years between the second youngest and youngest waterline capture dates (float, length 23, precision 15); 
- recentRtW: rate of cross-shore change between the second youngest and youngest waterline, calculated using linear regression, in metres per year (float, length 23, precision 15); 
- recentMEW: margin of error (plus or minus) on the rate of cross-shore change between the second youngest and youngest waterline, in metres per year (float, length 23, precision 15); 

4. Missing data codes: 
nan

5. Specialized formats or other abbreviations used: 
None


DATA-SPECIFIC INFORMATION FOR: shapefiles/StAndrewsEastS2Full2024_Transects_Intersected_Slope.shp

- Cross-shore transects defined using the CoasTrack functions within the COASTGUARD toolbox, holding statistics and data related to the cross-shore intersection of each transect with each satellite-derived vegetation edge and lidar-derived dune face slope.

1. Number of variables/bands: 
25

2. Number of cases/rows/pixels: 
1,412

3. Variable List: 
- LineID: reference shoreline ID number (integer, length 18, precision 0);
- TransectID: transect ID number (integer, length 18, precision 0);
- reflinepnt: shapely POINT object of intersection between transect and reference shoreline, formatted as a Python list of WKTs (string, length 254, precision 0); 
- dates: dates of satellite image capture for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings '%y-%m-%d' (string, length 254, precision 0); 
- times: times of satellite image capture for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings '%H:%M:%S.%f' (string, length 254, precision 0); 
- filename: Google Earth Engine server filename of matching satellite image for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings (string, length 254, precision 0); 
- cloud_cove: percentage of cloud cover over image (proportion of pixels classed as cloud) for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- idx: satellite image ID for timeseries of vegetation edges intersected with transect, formatted as a Python list of integers (string, length 254, precision 0); 
- vthreshold: threshold Normalised Difference Vegetation Index used to extract veg edge contour along for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- wthreshold: threshold Modified Normalised Difference Water Index used to extract waterline contour along for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- tideelev: tidal elevation at date and time of satellite image capture, derived from FES2022 global tide model for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- satname:  abbreviated name of satellite platform sourcing the imagery for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings (string, length 254, precision 0); 
- interpnt:  shapely POINT object of intersection between transect and each vegetation edge in timeseries, formatted as a Python list of WKTs (string, length 254, precision 0); 
- distances: distance along transect in metres of intersection point between transect and each vegetation edge in timeseries, formatted as a Python list of floats (string, length 254, precision 0); 
- olddate: oldest vegetation edge capture date in %y-%m-%d (string, length 80, precision 0); 
- youngdate: youngest/most recent vegetation edge capture date, in %y-%m-%d (string, length 80, precision 0); 
- oldyoungT: number of (decimal) years between the oldest and youngest vegetation edge capture dates (float, length 23, precision 15); 
- oldyoungRt: rate of cross-shore change between the oldest and youngest vegetation edge, calculated using linear regression, in metres per year (float, length 23, precision 15); 
- recentT: number of (decimal) years between the second youngest and youngest vegetation edge capture dates (float, length 23, precision 15); 
- recentRt: rate of cross-shore change between the second youngest and youngest vegetation edge, calculated using linear regression, in metres per year (float, length 23, precision 15);
- normdists: distance along transect in metres of intersection point between transect and each vegetation edge in timeseries, normalised to distance of first intersection, formatted as a Python list of floats (string, length 254, precision 0); 
- TZwidth: timeseries of cross-shore width in metres of each satellite image vegetation transition zone, found from measuring the distance between the point of transect intersection with the seaward and landward edges of the transition zone raster pixels, formatted as a Python list of floats (string, length 254, precision 0); 
- TZwidthMn: timeseries mean of cross-shore width in metres of vegetation transition zone (float, length 23, precision 15); 
- SlopeMax: maximum of all slopes extracted at the vegetation edge intersection point ('interpnt'), from Scottish Government Phase 5 lidar, in degrees (float, length 23, precision 15); 
- SlopeMean: mean of all slopes extracted at the vegetation edge intersection point ('interpnt'), from Scottish Government Phase 5 lidar, in degrees (float, length 23, precision 15)

4. Missing data codes: 
nan

5. Specialized formats or other abbreviations used: 
None


DATA-SPECIFIC INFORMATION FOR: shapefiles/StAndrewsEastS2Full2024_Transects_Intersected_TZ.shp

- Cross-shore transects defined using the CoasTrack functions within the COASTGUARD toolbox, holding statistics and data related to the cross-shore intersection of each transect with each satellite-derived vegetation edge and satellite-derived vegetation transition zone raster.

1. Number of variables/bands: 
23

2. Number of cases/rows/pixels: 
1,412

3. Variable List: 
- LineID: reference shoreline ID number (integer, length 18, precision 0);
- TransectID: transect ID number (integer, length 18, precision 0);
- reflinepnt: shapely POINT object of intersection between transect and reference shoreline, formatted as a Python list of WKTs (string, length 254, precision 0); 
- dates: dates of satellite image capture for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings '%y-%m-%d' (string, length 254, precision 0); 
- times: times of satellite image capture for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings '%H:%M:%S.%f' (string, length 254, precision 0); 
- filename: Google Earth Engine server filename of matching satellite image for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings (string, length 254, precision 0); 
- cloud_cove: percentage of cloud cover over image (proportion of pixels classed as cloud) for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- idx: satellite image ID for timeseries of vegetation edges intersected with transect, formatted as a Python list of integers (string, length 254, precision 0); 
- vthreshold: threshold Normalised Difference Vegetation Index used to extract veg edge contour along for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- wthreshold: threshold Modified Normalised Difference Water Index used to extract waterline contour along for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- tideelev: tidal elevation at date and time of satellite image capture, derived from FES2022 global tide model for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- satname:  abbreviated name of satellite platform sourcing the imagery for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings (string, length 254, precision 0); 
- interpnt:  shapely POINT object of intersection between transect and each vegetation edge in timeseries, formatted as a Python list of WKTs (string, length 254, precision 0); 
- distances: distance along transect in metres of intersection point between transect and each vegetation edge in timeseries, formatted as a Python list of floats (string, length 254, precision 0); 
- olddate: oldest vegetation edge capture date in %y-%m-%d (string, length 80, precision 0); 
- youngdate: youngest/most recent vegetation edge capture date, in %y-%m-%d (string, length 80, precision 0); 
- oldyoungT: number of (decimal) years between the oldest and youngest vegetation edge capture dates (float, length 23, precision 15); 
- oldyoungRt: rate of cross-shore change between the oldest and youngest vegetation edge, calculated using linear regression, in metres per year (float, length 23, precision 15); 
- recentT: number of (decimal) years between the second youngest and youngest vegetation edge capture dates (float, length 23, precision 15); 
- recentRt: rate of cross-shore change between the second youngest and youngest vegetation edge, calculated using linear regression, in metres per year (float, length 23, precision 15);
- normdists: distance along transect in metres of intersection point between transect and each vegetation edge in timeseries, normalised to distance of first intersection, formatted as a Python list of floats (string, length 254, precision 0); 
- TZwidth: timeseries of cross-shore width in metres of each satellite image vegetation transition zone, found from measuring the distance between the point of transect intersection with the seaward and landward edges of the transition zone raster pixels, formatted as a Python list of floats (string, length 254, precision 0); 
- TZwidthMn: timeseries mean of cross-shore width in metres of vegetation transition zone (float, length 23, precision 15)

4. Missing data codes: 
nan

5. Specialized formats or other abbreviations used: 
None


DATA-SPECIFIC INFORMATION FOR: shapefiles/StAndrewsEastS2Full2024_Transects_Intersected_Waves.shp

- Cross-shore transects defined using the CoasTrack functions within the COASTGUARD toolbox, holding statistics and data related to the cross-shore intersection of each transect with each satellite-derived vegetation edge and satellite-assimilated NW Atlantic wave hindcast netCDF slices from Copernicus Marine Service.

1. Number of variables/bands: 
37

2. Number of cases/rows/pixels: 
1,412

3. Variable List: 
- LineID: reference shoreline ID number (integer, length 18, precision 0);
- TransectID: transect ID number (integer, length 18, precision 0);
- reflinepnt: shapely POINT object of intersection between transect and reference shoreline, formatted as a Python list of WKTs (string, length 254, precision 0); 
- dates: dates of satellite image capture for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings '%y-%m-%d' (string, length 254, precision 0); 
- times: times of satellite image capture for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings '%H:%M:%S.%f' (string, length 254, precision 0); 
- filename: Google Earth Engine server filename of matching satellite image for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings (string, length 254, precision 0); 
- cloud_cove: percentage of cloud cover over image (proportion of pixels classed as cloud) for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- idx: satellite image ID for timeseries of vegetation edges intersected with transect, formatted as a Python list of integers (string, length 254, precision 0); 
- vthreshold: threshold Normalised Difference Vegetation Index used to extract veg edge contour along for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- wthreshold: threshold Modified Normalised Difference Water Index used to extract waterline contour along for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- tideelev: tidal elevation at date and time of satellite image capture, derived from FES2022 global tide model for timeseries of vegetation edges intersected with transect, formatted as a Python list of floats (string, length 254, precision 0); 
- satname:  abbreviated name of satellite platform sourcing the imagery for timeseries of vegetation edges intersected with transect, formatted as a Python list of strings (string, length 254, precision 0); 
- interpnt:  shapely POINT object of intersection between transect and each vegetation edge in timeseries, formatted as a Python list of WKTs (string, length 254, precision 0); 
- distances: distance along transect in metres of intersection point between transect and each vegetation edge in timeseries, formatted as a Python list of floats (string, length 254, precision 0); 
- olddate: oldest vegetation edge capture date in %y-%m-%d (string, length 80, precision 0); 
- youngdate: youngest/most recent vegetation edge capture date, in %y-%m-%d (string, length 80, precision 0); 
- oldyoungT: number of (decimal) years between the oldest and youngest vegetation edge capture dates (float, length 23, precision 15); 
- oldyoungRt: rate of cross-shore change between the oldest and youngest vegetation edge, calculated using linear regression, in metres per year (float, length 23, precision 15); 
- recentT: number of (decimal) years between the second youngest and youngest vegetation edge capture dates (float, length 23, precision 15); 
- recentRt: rate of cross-shore change between the second youngest and youngest vegetation edge, calculated using linear regression, in metres per year (float, length 23, precision 15);
- normdists: distance along transect in metres of intersection point between transect and each vegetation edge in timeseries, normalised to distance of first intersection, formatted as a Python list of floats (string, length 254, precision 0); 
- WaveDates: timeseries of dates of wave hindcasts at each satellite image capture date and time, extracted onto each transect from the nearest wave data grid cell, formatted as a Python list of datetime objects (%y,%m,%d,%H,%M,%S,%f) (string, length 254, precision 0); 
- WaveDatesF: daily dates of full timeseries of wave hindcasts (where the start and end date match the first and last Sentinel-2 satellite image collected), extracted onto each transect from the nearest wave raster grid cell, formatted as a Python list of pandas Timestamps ('%y-%m-%d %H:%M:%S') (string, length 254, precision 0); 
- WaveHs: timeseries of significant wave height in metres at each satellite image capture date and time, extracted onto each transect from the nearest wave data grid cell, formatted as a Python list of floats (string, length 254, precision 0); 
- WaveHsFD: full timeseries of daily significant wave heights in metres from wave hindcasts (where the start and end date match the first and last Sentinel-2 satellite image collected), extracted onto each transect from the nearest wave raster grid cell, formatted as a Python list of floats (string, length 254, precision 0); 
- WaveDir: timeseries of mean wave direction (from) in degrees at each satellite image capture date and time, extracted onto each transect from the nearest wave data grid cell, formatted as a Python list of floats (string, length 254, precision 0); 
- WaveDirFD: full timeseries of daily mean wave directions (from) in degrees from wave hindcasts (where the start and end date match the first and last Sentinel-2 satellite image collected), extracted onto each transect from the nearest wave raster grid cell, formatted as a Python list of floats (string, length 254, precision 0); 
- WaveAlpha: timeseries of difference between mean wave directions (from) and shoreline angle in degrees at each satellite image capture date and time, extracted onto each transect from the nearest wave data grid cell, formatted as a Python list of floats (string, length 254, precision 0); 
- WaveAlphaF: full timeseries of daily difference between mean wave directions (from) and shoreline angle in degrees from wave hindcasts (where the start and end date match the first and last Sentinel-2 satellite image collected), extracted onto each transect from the nearest wave raster grid cell, formatted as a Python list of floats (string, length 254, precision 0); 
- WaveTp: timeseries of peak wave period in seconds at each satellite image capture date and time, extracted onto each transect from the nearest wave data grid cell, formatted as a Python list of floats (string, length 254, precision 0); 
- WaveTpFD: full timeseries of daily peak wave periods in seconds from wave hindcasts (where the start and end date match the first and last Sentinel-2 satellite image collected), extracted onto each transect from the nearest wave raster grid cell, formatted as a Python list of floats (string, length 254, precision 0); 
- WaveQs: full timeseries of daily relative longshore sediment transport flux defined in Ashton & Murray (2006b) (https://doi.org/10.1029/2005JF000423) in metres cubed per second, calculated from wave hindcasts (where the start and end date match the first and last Sentinel-2 satellite image collected), extracted onto each transect from the nearest wave raster grid cell, formatted as a Python list of floats (string, length 254, precision 0); 
- WaveQsNet: net relative longshore sediment transport flux defined in Ashton & Murray (2006b) (https://doi.org/10.1029/2005JF000423) in metres cubed per second (float, length 23, precision 15); 
- WaveDiffus: net wave-driven shoreline diffusivity defined in Ashton & Murray (2006b) (https://doi.org/10.1029/2005JF000423) in metres per second squared (float, length 23, precision 15); 
- WaveStabil: net wave-driven shoreline instability index defined in Ashton & Murray (2006b) (https://doi.org/10.1029/2005JF000423) dimensionless (-1 to 1) (float, length 23, precision 15); 
- Shore Angle: shoreline angle i.e. perpendicular anticlockwise to each transect, in degrees with sea on right (float, length 23, precision 15); 
- Runups: full timeseries of daily wave runup elevations in metres, calculated using the formula from Senechal et al. (2011) (https://doi.org/10.1029/2010JC006819) from wave hindcasts (where the start and end date match the first and last Sentinel-2 satellite image collected), extracted onto each transect from the nearest wave raster grid cell, formatted as a Python list of floats (string, length 254, precision 0)

4. Missing data codes: 
nan

5. Specialized formats or other abbreviations used: 
None


DATA-SPECIFIC INFORMATION FOR: transect_intersections/StAndrewsEastS2Full2024_transect_intersects.pkl

- Python variable of type GeoDataFrame, representing cross-shore transects defined using the CoasTrack functions within the COASTGUARD toolbox. The base transect variable holds statistics and data related to the cross-shore intersection of each transect with each satellite-derived vegetation edge.

1. Number of variables/bands: 
22

2. Number of cases/rows/pixels: 
1,412

3. Variable List (see variable names in corresponding shapefiles above for descriptions): 
- LineID (type: pandas.core.series.Series);
- TransectID (type: pandas.core.series.Series);
- geometry: cross-shore transect geometry (type: geopandas.array.GeometryDtype);
- reflinepnt (type: list of shapely.geometry.point.Point);
- dates (type: list of str);
- times (type: list of str);
- filename (type: list of str);
- cloud_cove (type: list of numpy.float64);
- idx (type: list of numpy.int64);
- vthreshold (type: list of numpy.float64);
- wthreshold (type: list of numpy.float64);
- tideelev (type: list of numpy.float64);
- satname (type: list of str);
- interpnt (type: list of shapely.geometry.point.Point);
- distances (type: list of numpy.float64);
- olddate (type: str);
- youngdate (type: str);
- oldyoungT (type: pandas.core.series.Series);
- oldyoungRt (type: pandas.core.series.Series);
- recentT (type: pandas.core.series.Series);
- recentRt (type: pandas.core.series.Series);
- normdists (type: list of numpy.float64)

4. Missing data codes: 
nan

5. Specialized formats or other abbreviations used: 
None


DATA-SPECIFIC INFORMATION FOR: transect_intersections/StAndrewsEastS2Full2024_transect_water_intersects.pkl

- Python variable of type GeoDataFrame, representing cross-shore transects defined using the CoasTrack functions within the COASTGUARD toolbox, holding statistics and data related to the cross-shore intersection of each transect with each satellite-derived vegetation edge and satellite-derived waterline.

1. Number of variables/bands: 
41

2. Number of cases/rows/pixels: 
1,412

3. Variable List (see duplicate variable names in corresponding shapefiles above for descriptions): 
- LineID (type: pandas.core.series.Series);
- TransectID (type: pandas.core.series.Series);
- geometry: cross-shore transect geometry  (type: geopandas.array.GeometryDtype);
- reflinepnt (type: list of shapely.geometry.point.Point);
- dates (type: list of str);
- times (type: list of str);
- filename (type: list of str);
- cloud_cove (type: list of numpy.float64);
- idx (type: list of numpy.int64);
- vthreshold (type: list of numpy.float64);
- wthreshold (type: list of numpy.float64);
- tideelev (type: list of numpy.float64);
- satname (type: list of str);
- interpnt (type: list of shapely.geometry.point.Point);
- distances (type: list of numpy.float64);
- olddate (type: str);
- youngdate (type: str);
- oldyoungT (type: pandas.core.series.Series);
- oldyoungRt (type: pandas.core.series.Series);
- recentT (type: pandas.core.series.Series);
- recentRt (type: pandas.core.series.Series);
- normdists (type: list of numpy.float64);
- wldates (type: list of str);
- wltimes (type: list of str);
- wldists (type: list of float);
- wlinterpnt (type: list of shapely.geometry.point.Point);
- wlcorrdist (type: list of numpy.float64);
- beachslope (type: pandas.core.series.Series);
- beachwidth (type: list of numpy.float64);
- tidezone (type: str);
- olddateW (type: str);
- youngdateW (type: str);
- oldyoungTW (type: pandas.core.series.Series);
- oldyungRtW (type: pandas.core.series.Series);
- oldyungMEW (type: pandas.core.series.Series);
- recentTW (type: pandas.core.series.Series);
- recentRtW (type: pandas.core.series.Series);
- recentMEW (type: pandas.core.series.Series);
- tideelevFD: full timeseries of daily mean tidal elevation in metres, from the FES2022 tide model (where the start and end date match the first and last Sentinel-2 satellite image collected), extracted onto each transect from the nearest tidal grid cell (type: list of float);
- tideelevMx: full timeseries of daily maximum tidal elevation in metres, from the FES2022 tide model (where the start and end date match the first and last Sentinel-2 satellite image collected), extracted onto each transect from the nearest tidal grid cell (type: list of float);
- tidedatesFD: full timeseries of daily dates from the FES2022 tide model (where the start and end date match the first and last Sentinel-2 satellite image collected), extracted onto each transect from the nearest tidal grid cell (type: list of pandas._libs.tslibs.timestamps.Timestamp)

4. Missing data codes: 
nan

5. Specialized formats or other abbreviations used: 
None


DATA-SPECIFIC INFORMATION FOR: transect_intersections/StAndrewsEastS2Full2024_transect_wave_intersects.pkl

- Python variable of type GeoDataFrame, representing cross-shore transects defined using the CoasTrack functions within the COASTGUARD toolbox, holding statistics and data related to the cross-shore intersection of each transect with each satellite-derived vegetation edge and satellite-assimilated NW Atlantic wave hindcast netCDF slices from Copernicus Marine Service.

1. Number of variables/bands: 
39

2. Number of cases/rows/pixels: 
1,412

3. Variable List (see variable names in corresponding shapefiles above for descriptions): 
- LineID (type: pandas.core.series.Series);
- TransectID (type: pandas.core.series.Series);
- geometry: cross-shore transect geometry (type: geopandas.array.GeometryDtype);
- reflinepnt (type: list of shapely.geometry.point.Point);
- dates (type: list of str);
- times (type: list of str);
- filename (type: list of str);
- cloud_cove (type: list of numpy.float64);
- idx (type: list of numpy.int64);
- vthreshold (type: list of numpy.float64);
- wthreshold (type: list of numpy.float64);
- tideelev (type: list of numpy.float64);
- satname (type: list of str);
- interpnt (type: list of shapely.geometry.point.Point);
- distances (type: list of numpy.float64);
- olddate (type: str);
- youngdate (type: str);
- oldyoungT (type: pandas.core.series.Series);
- oldyoungRt (type: pandas.core.series.Series);
- recentT (type: pandas.core.series.Series);
- recentRt (type: pandas.core.series.Series);
- normdists (type: list of numpy.float64);
- WaveDates (type: list of datetime.datetime);
- WaveDatesFD (type: list of pandas._libs.tslibs.timestamps.Timestamp);
- WaveHs (type: list of numpy.float64);
- WaveHsFD (type: list of float);
- WaveDir (type: list of numpy.float64);
- WaveDirFD (type: list of float);
- WaveAlpha (type: list of numpy.float64);
- WaveAlphaFD (type: list of numpy.float64);
- WaveTp (type: list of numpy.float64);
- WaveTpFD (type: list of float);
- WaveQs (type: list of float);
- WaveQsNet (type: pandas.core.series.Series);
- WaveDiffus (type: pandas.core.series.Series);
- WaveStabil (type: pandas.core.series.Series);
- ShoreAngle (type: pandas.core.series.Series);
- Runups (type: list of numpy.float64);
- Iribarren: timeseries of dimensionless Iribarren numbers at each satellite image capture date and time, calculated using the following formula: beachslope / WaveHs * ((9.81 * WaveTp**2) / 2pi), using wave hindcasts extracted onto each transect from the nearest wave data grid cell (type: list of numpy.float64)

4. Missing data codes: 
nan

5. Specialized formats or other abbreviations used: 
None


DATA-SPECIFIC INFORMATION FOR: transect_intersections/StAndrewsEastS2Full2024_transect_topo_intersects.pkl

- Python variable of type GeoDataFrame, representing cross-shore transects defined using the CoasTrack functions within the COASTGUARD toolbox, holding statistics and data related to the cross-shore intersection of each transect with each satellite-derived vegetation edge, lidar-derived dune face slope, and satellite-derived vegetation transition zone raster.

1. Number of variables/bands: 
26

2. Number of cases/rows/pixels: 
1,412

3. Variable List (see variable names in corresponding shapefiles above for descriptions): 
- LineID (type: pandas.core.series.Series);
- TransectID (type: pandas.core.series.Series);
- geometry: cross-shore transect geometry (type: geopandas.array.GeometryDtype);
- reflinepnt (type: list of shapely.geometry.point.Point);
- dates (type: list of str);
- times (type: list of str);
- filename (type: list of str);
- cloud_cove (type: list of numpy.float64);
- idx (type: list of numpy.int64);
- vthreshold (type: list of numpy.float64);
- wthreshold (type: list of numpy.float64);
- tideelev (type: list of numpy.float64);
- satname (type: list of str);
- interpnt (type: list of shapely.geometry.point.Point);
- distances (type: list of numpy.float64);
- olddate (type: str);
- youngdate (type: str);
- oldyoungT (type: pandas.core.series.Series);
- oldyoungRt (type: pandas.core.series.Series);
- recentT (type: pandas.core.series.Series);
- recentRt (type: pandas.core.series.Series);
- normdists (type: list of numpy.float64);
- TZwidth (type: list of float);
- TZwidthMn (type: pandas.core.series.Series);
- SlopeMax (type: pandas.core.series.Series);
- SlopeMean (type: pandas.core.series.Series)

4. Missing data codes: 
nan

5. Specialized formats or other abbreviations used: 
None


