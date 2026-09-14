"""
Standalone verification script for the s2cloudless cloud-masking patch in
Toolshed/Image_Processing.py (S2 branch of preprocess_single()).

Does NOT run the VedgeSat/CoastSat pipeline. It reads three Sentinel-2 images
directly from Earth Engine, over the same AOI the driver notebook builds, and
prints the numbers needed to sanity-check the patch:

  - CLOUDY_PIXEL_PERCENTAGE (scene-level, from the image properties)
  - the QA60-derived AOI cloud fraction (using the real, unmodified
    create_cloud_mask() from Image_Processing.py)
  - the s2cloudless AOI cloud fraction at probability cutoffs of 40, 50 and 60

for three images:
  1. one acquired before 2024-02-28 (inside the QA60 production gap)
  2. the acquisition nearest KNOWN_GOOD_DATE (2024-04-02 by default - processed
     successfully in the previous run)
  3. the image with the single highest scene-level CLOUDY_PIXEL_PERCENTAGE in
     the date range (stands in for "one rejected in the previous run at a high
     scene-level cloud percentage")

It then reports two explicit checks called out in the task brief:
  1. Empty-QA60 diagnosis - the pre-gap image's QA60 AOI fraction should be
     0.00 while s2cloudless returns a real value.
  2. Threshold direction - the known-good image should score LOWER on
     s2cloudless than the high-scene-cloud image.

Run this with the same environment as the driver notebook (ee, geemap,
geopandas importable, and ee.Authenticate() already completed), from the
repo root:

    python test_s2cloudless.py

Edit the CONFIG block below to match your site if it differs from the
Karekare_2024_Test_20cThresh example in VedgeSat_DriverTemplate_1.ipynb.
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np
import ee
import geemap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Toolshed import Toolbox, Image_Processing

# -------------------------------------------------------------------------- #
# CONFIG - edit to match your site/run
# -------------------------------------------------------------------------- #
EE_PROJECT = 'civic-ripsaw-454707-c4'  # matches VedgeSat_DriverTemplate_1.ipynb

SITENAME = 'Karekare_2024_Test_20cThresh'
SAT_LIST = ['S2']
REFERENCE_LINE_SHP = os.path.join('Data', 'referenceLines', 'Karekare_refLine.shp')
MAX_DIST_REF = 250  # metres, matches the driver template

DATES = ['2024-01-01', '2024-05-02']  # matches the driver template date range

QA60_GAP_END = '2024-02-28'    # QA60 legacy reconstruction start date (task brief, section 2)
KNOWN_GOOD_DATE = '2024-04-02'  # acquisition that processed cleanly in the previous run

PROB_CUTOFFS = [40, 50, 60]
# -------------------------------------------------------------------------- #


def iso_date(feature):
    ts = feature['properties']['system:time_start'] / 1000
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')


def days_from_target(isodate, target):
    return abs((datetime.strptime(isodate, '%Y-%m-%d') - datetime.strptime(target, '%Y-%m-%d')).days)


def qa60_aoi_fraction(img, aoi):
    """AOI cloud fraction from the existing QA60 path (unmodified create_cloud_mask)."""
    try:
        im60 = geemap.ee_to_numpy(img, bands=['QA60'], region=aoi, scale=60)
    except Exception:
        im60 = geemap.ee_to_numpy(img, bands=['MSK_CLASSI_OPAQUE'], region=aoi, scale=60)
    if im60 is None:
        return None
    im_QA = im60[:, :, 0]
    cloud_mask = Image_Processing.create_cloud_mask(im_QA, 'S2', False)
    return float(cloud_mask.sum() / cloud_mask.size)


def s2cloudless_aoi_fractions(img_id, aoi, cutoffs):
    """AOI cloud fraction from s2cloudless at each of `cutoffs`, or None if unavailable."""
    s2_index = img_id.split('/')[-1]
    coll = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filter(
        ee.Filter.eq('system:index', s2_index))
    if coll.size().getInfo() == 0:
        return None
    im_prob = geemap.ee_to_numpy(coll.first(), bands=['probability'], region=aoi, scale=60)
    if im_prob is None:
        return None
    prob_arr = im_prob[:, :, 0]
    return {c: float(np.sum(prob_arr > c) / prob_arr.size) for c in cutoffs}


def main():
    ee.Initialize(project=EE_PROJECT)

    # Make sure the site folder exists (idempotent - matches the driver's own first step)
    Toolbox.CreateFileStructure(SITENAME, SAT_LIST)

    if not os.path.isfile(REFERENCE_LINE_SHP):
        raise FileNotFoundError(
            f"Reference line shapefile not found at '{REFERENCE_LINE_SHP}'. "
            "Edit REFERENCE_LINE_SHP at the top of this script to point at your site's "
            "reference line (same file the driver notebook's Reference Shore Option 1 uses).")

    # Build the same AOI polygon the driver notebook builds (Option 1 cell)
    polygon, point, lonmin, lonmax, latmin, latmax = Toolbox.AOIfromLine(
        REFERENCE_LINE_SHP, MAX_DIST_REF, SITENAME)
    polygon = Toolbox.smallest_rectangle(polygon)
    aoi = ee.Geometry.Polygon(polygon)

    # Pull the S2 collection over the AOI/date range, as Download.py does
    s2_coll = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
               .filterBounds(aoi)
               .filterDate(DATES[0], DATES[1]))
    features = s2_coll.getInfo()['features']
    if not features:
        raise RuntimeError('No Sentinel-2 images found for this AOI/date range - '
                            'check REFERENCE_LINE_SHP, DATES and EE_PROJECT.')

    for feat in features:
        feat['_isodate'] = iso_date(feat)

    pre_gap_candidates = [f for f in features if f['_isodate'] < QA60_GAP_END]
    if not pre_gap_candidates:
        raise RuntimeError(f"No S2 images found before {QA60_GAP_END} in this date range; "
                            "widen DATES to include a pre-gap image.")
    img_pre_gap = sorted(pre_gap_candidates, key=lambda f: f['_isodate'])[-1]

    img_known_good = min(features, key=lambda f: days_from_target(f['_isodate'], KNOWN_GOOD_DATE))
    img_high_cloud = max(features, key=lambda f: f['properties']['CLOUDY_PIXEL_PERCENTAGE'])

    targets = [
        ('pre-QA60-gap', img_pre_gap),
        ('known-good (nearest %s)' % KNOWN_GOOD_DATE, img_known_good),
        ('high scene-cloud reject', img_high_cloud),
    ]

    print('=' * 78)
    print(f'AOI: {SITENAME}  |  date range: {DATES[0]} to {DATES[1]}')
    print('=' * 78)

    results = []
    for label, feat in targets:
        img_id = feat['id']
        img = ee.Image(img_id)
        scene_pct = feat['properties']['CLOUDY_PIXEL_PERCENTAGE'] / 100
        isodate = feat['_isodate']

        qa60_frac = qa60_aoi_fraction(img, aoi)
        s2c_fracs = s2cloudless_aoi_fractions(img_id, aoi, PROB_CUTOFFS)

        print(f"\n[{label}]  {isodate}")
        print(f"  id: {img_id}")
        print(f"  CLOUDY_PIXEL_PERCENTAGE (scene):        {scene_pct * 100:6.2f} %")
        print(f"  QA60-derived AOI cloud fraction:        "
              f"{'N/A (no QA60/MSK_CLASSI_OPAQUE data)' if qa60_frac is None else f'{qa60_frac * 100:6.2f} %'}")
        if s2c_fracs is None:
            print("  s2cloudless AOI cloud fraction:         N/A (no matching s2cloudless image)")
        else:
            for c in PROB_CUTOFFS:
                print(f"  s2cloudless AOI cloud fraction (p{c}):   {s2c_fracs[c] * 100:6.2f} %")

        results.append({'label': label, 'date': isodate, 'id': img_id, 'scene_pct': scene_pct,
                         'qa60_frac': qa60_frac, 's2c_fracs': s2c_fracs})

    pre_gap_result, good_result, bad_result = results

    # ---------------------------------------------------------------------- #
    # Check 1: empty-QA60 diagnosis
    # ---------------------------------------------------------------------- #
    print('\n' + '=' * 78)
    print(f"Check 1: Empty-QA60 diagnosis  ({pre_gap_result['date']}, before {QA60_GAP_END})")
    print('=' * 78)
    if pre_gap_result['qa60_frac'] is None:
        print('  INCONCLUSIVE: QA60/MSK_CLASSI_OPAQUE could not be read at all for this image.')
    elif pre_gap_result['qa60_frac'] == 0.0:
        print('  CONFIRMED: QA60 AOI fraction is 0.00, as expected inside the QA60 production gap.')
        if pre_gap_result['s2c_fracs'] is not None and pre_gap_result['s2c_fracs'][60] > 0:
            print(f"  s2cloudless returns a real, non-zero value "
                  f"(p60={pre_gap_result['s2c_fracs'][60]:.3f}), consistent with s2cloudless "
                  "covering the gap that QA60 cannot.")
        else:
            print('  WARNING: s2cloudless ALSO returned zero/unavailable for this image - cannot '
                  'confirm it covers the gap here. Try a different pre-gap date with visible cloud.')
    else:
        print(f"  *** UNEXPECTED: QA60 AOI fraction is {pre_gap_result['qa60_frac']:.3f}, NOT 0.00. ***")
        print('  The empty-QA60 diagnosis in the task brief does not hold for this image/site - do '
              'not trust the patch until this is investigated further.')

    # ---------------------------------------------------------------------- #
    # Check 2: threshold direction
    # ---------------------------------------------------------------------- #
    print('\n' + '=' * 78)
    print(f"Check 2: Threshold direction  ({good_result['date']} vs {bad_result['date']})")
    print('=' * 78)
    if good_result['s2c_fracs'] is None or bad_result['s2c_fracs'] is None:
        print('  INCONCLUSIVE: s2cloudless data unavailable for one or both comparison images.')
    else:
        g60, b60 = good_result['s2c_fracs'][60], bad_result['s2c_fracs'][60]
        print(f"  known-good image  s2cloudless AOI fraction (p60): {g60:.3f}  "
              f"(scene-level: {good_result['scene_pct']:.3f})")
        print(f"  high-scene-cloud  s2cloudless AOI fraction (p60): {b60:.3f}  "
              f"(scene-level: {bad_result['scene_pct']:.3f})")
        if b60 > g60:
            print('  CONFIRMED: the high-scene-cloud image scores higher than the known-good image - '
                  'threshold direction (cloud_mask = probability > threshold) is correct.')
        elif b60 == g60:
            print('  INCONCLUSIVE: both images score identically - pick a more clearly cloudy '
                  'comparison image (lower QA60_GAP_END / different KNOWN_GOOD_DATE).')
        else:
            print('  *** UNEXPECTED: the known-good image scored HIGHER than the high-scene-cloud '
                  'image. ***')
            print('  Either the comparison operator is inverted (should be prob_arr > threshold, '
                  "not <), or this 'high scene-cloud' image's cloud genuinely sits away from the "
                  'AOI - compare the scene-level percentages printed above to judge which.')

    print()


if __name__ == '__main__':
    main()
