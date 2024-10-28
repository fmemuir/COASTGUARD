#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is adapted from CoastSat.slope (SDS_slope.py) for finding beach slopes
from satellite-derived shorelines which avoid effects of dynamic shorefaces.

COASTGUARD edits and updates: Freya Muir, University of Glasgow

"""

import numpy as np
import pytz
import datetime
from datetime import datetime, timedelta

from scipy import integrate as sintegrate
from scipy import signal as ssignal
from scipy import interpolate as sinterpolate
from astropy.timeseries import LombScargle

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colorbar
from matplotlib import lines


#%%

def CoastSatSlope(dates_sat_tr, tides_sat_tr, cross_distances):
    """
    Run the main CoastSat.slope routines to get beach slope which minimises high
    frequency tidal fluctuations compared to lower-frequency erosion/accretion signals.
    FM Oct 2024

    Parameters
    ----------
    dates_sat_tr : list
        Datetime of satellite image capture times (per-transect timeseries).
    tides_sat_tr : list
        List of tidal elevations at each satellite image capture time (per-transect timeseries).
    cross_distances : list
        List of distances along a cross-shore transect where each sat-derived 
        waterline intersects (per-transect timeseries).

    Returns
    -------
    slope_est : float
        Estimated tan(Beta) beach slope value.

    """
    # Slope calculation happens per-transect, so single value returned if only
    # one timeseries list is provided
    settings_slope = DefineSlopeSettings(cross_distances)
    
    # find tidal peak frequency
    settings_slope['freqs_max'] = find_tide_peak(dates_sat_tr, tides_sat_tr, settings_slope)
    
    # remove NaNs (but won't be any in VedgeSat GDF per-transect list)
    # idx_nan = np.isnan(cross_distances)
    # tide = tides_sat_tr[~idx_nan]
    # composite = cross_distances[~idx_nan]
    
    tcorr = tide_correct(cross_distances, tides_sat_tr, settings_slope['beach_slopes'])
    slope_est, conf_ints = integrate_power_spectrum(dates_sat_tr, tcorr, settings_slope)
    
    return slope_est
    
#%%

def DefineSlopeSettings(cross_distances):
    """
    Function to define the slope routine settings as a dictionary. Includes
    the minimum number of days for peak frequency interval, a date range (which
    isn't relevant if you have less than the 1999-2020 range), and a list of 
    beach slopes to check the energy levels of.
    FM Oct 2024

    Parameters
    ----------
    cross_distances : list
        List of distances along a cross-shore transect where each sat-derived 
        waterline intersects (per-transect timeseries).

    Returns
    -------
    settings_slope : dict
        Dictionary of settings required for slope-finding routine.

    """
    seconds_in_day = 24*3600
    settings_slope = {'slope_min':        0.001,                  # minimum slope to trial (global 10th%)
                      'slope_max':        0.5,                    # maximum slope to trial (global 90th%)
                      'delta_slope':      0.003,                  # resolution of slopes to trial
                      'date_range':       [1999,2020],            # range of dates over which to perform the analysis
                      'n0':               50,                     # for Nyquist criterium
                      'freqs_cutoff':     1./(seconds_in_day*30), # 1 month frequency
                      'delta_f':          1e-8,                   # deltaf for buffer around max peak
                      'prc_conf':         0.05,                   # percentage above minimum to define confidence bands in energy curve
                      'n_days':           8}                      # minimum number of days for peak freq interval
    settings_slope['date_range'] = [pytz.utc.localize(datetime(settings_slope['date_range'][0],5,1)),
                                    pytz.utc.localize(datetime(settings_slope['date_range'][1],1,1))]
    settings_slope['beach_slopes'] = range_slopes(settings_slope['slope_min'], settings_slope['slope_max'], settings_slope['delta_slope'])
        
    # # clip the dates between 1999 and 2020 as we need at least 2 Landsat satellites in orbit simultaneously 
    # idx_dates = [np.logical_and(_>settings_slope['date_range'][0],_<settings_slope['date_range'][1]) for _ in output['dates']]
    # for key in cross_distance.keys():
    #     cross_distance[key] = cross_distance[key][idx_dates]

    return settings_slope


def FreqParams(dates_sat_tr, settings_slope):
    """
    Define common parameters required in various functions throughout the Slope routine.
    FM Oct 2024

    Parameters
    ----------
    dates_sat_tr : list
        Datetime of satellite image capture times (per-transect timeseries).
    settings_slope : dict
        Dictionary of settings required for slope-finding routine.

    Returns
    -------
    t : array
        Array of timesteps as floats.
    days_in_year : float
        Number of decimal days in a year.
    seconds_in_day : int
        Number of seconds in a day.
    time_step : int
        Time interval (7 or 8 days) converted to seconds.
    freqs : TYPE
        Frequency grid created with Slope.frequency_grid().

    """
    # set common params
    t = np.array([_.timestamp() for _ in dates_sat_tr]).astype('float64')
    days_in_year = 365.2425
    seconds_in_day = 24*3600
    time_step = settings_slope['n_days']*seconds_in_day
    # create frequency grid
    freqs = frequency_grid(t,time_step,settings_slope['n0'])
    
    return t, days_in_year, seconds_in_day, time_step, freqs


#%%
def find_tide_peak(dates, tide_level, settings_slope, Plot=False):
    'find the high frequency peak in the tidal time-series'
    # set common params
    t, days_in_year, seconds_in_day, time_step, freqs = FreqParams(dates, settings_slope)
    
    # check if tidal frequency peak is on Day 7 or 8
    delta_t = np.diff(t)
    bin_edges = np.arange(np.min(delta_t)/seconds_in_day, np.max(delta_t)/seconds_in_day+1,1)-0.5
    counts, bin_edges = np.histogram(delta_t / seconds_in_day, bins=bin_edges)
    day7 = np.digitize(7,bin_edges)-1
    day8 = np.digitize(8,bin_edges)-1
    
    if counts[day7] > counts[day8]: # if Day 7 peak is larger than Day 8
        # change day interval to 7
        settings_slope['n_days'] = 7
    
    # compute power spectrum
    ps_tide,_,_ = power_spectrum(t,tide_level,freqs,[])
    # find peaks in spectrum
    idx_peaks,_ = ssignal.find_peaks(ps_tide, height=0)
    y_peaks = _['peak_heights']
    idx_peaks = idx_peaks[np.flipud(np.argsort(y_peaks))]
    # find the strongest peak at the high frequency (defined by freqs_cutoff[1])
    idx_max = idx_peaks[freqs[idx_peaks] > settings_slope['freqs_cutoff']][0]
    # compute the frequencies around the max peak with some buffer (defined by buffer_coeff)
    freqs_max = [freqs[idx_max] - settings_slope['delta_f'], freqs[idx_max] + settings_slope['delta_f']]
    
    # make a plot of the spectrum
    if Plot is True:
        fig = plt.figure()
        fig.set_size_inches([12,4])
        fig.set_tight_layout(True)
        ax = fig.add_subplot(111)
        ax.grid(linestyle=':', color='0.5')
        ax.plot(freqs,ps_tide)
        ax.set_title('$\Delta t$ = %d days'%settings_slope['n_days'], x=0, ha='left')
        ax.set(xticks=[(days_in_year*seconds_in_day)**-1, (30*seconds_in_day)**-1, (16*seconds_in_day)**-1, (8*seconds_in_day)**-1],
                       xticklabels=['1y','1m','16d','8d']);
        # show top 3 peaks
        for k in range(2):
            ax.plot(freqs[idx_peaks[k]], ps_tide[idx_peaks[k]], 'ro', ms=4)
            ax.text(freqs[idx_peaks[k]], ps_tide[idx_peaks[k]]+1, '%.1f d'%((freqs[idx_peaks[k]]**-1)/(3600*24)),
                    ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='square', ec='k',fc='w', alpha=0.5))
        ax.axvline(x=freqs_max[1], ls='--', c='0.5')
        ax.axvline(x=freqs_max[0], ls='--', c='0.5')
        ax.axvline(x=(2*settings_slope['n_days']*seconds_in_day)**-1, ls='--', c='k')
    
    return freqs_max


def frequency_grid(time, time_step, n0):
    'define frequency grid for Lomb-Scargle transform'
    T = np.max(time) - np.min(time)
    fmin = 1/T
    fmax = 1/(2*time_step) # Niquist criterium
    df = 1/(n0*T)
    N = np.ceil((fmax - fmin)/df).astype(int)
    freqs = fmin + df * np.arange(N)
    return freqs


def power_spectrum(t, y, freqs, idx_cut):
    'compute power spectrum and integrate'
    model = LombScargle(t, y, dy=None, fit_mean=True, center_data=True, nterms=1, normalization='psd')
    ps = model.power(freqs)
    # integrate the entire power spectrum
    E = sintegrate.simps(ps, x=freqs, even='avg')
    if len(idx_cut) == 0:
        idx_cut = np.ones(freqs.size).astype(bool)
    # integrate only frequencies above cut-off
    Ec = sintegrate.simps(ps[idx_cut], x=freqs[idx_cut], even='avg')
    return ps, E, Ec


def range_slopes(min_slope, max_slope, delta_slope):
    'create list of beach slopes to test'
    beach_slopes = []
    slope = min_slope
    while slope < max_slope:
        beach_slopes.append(slope)
        slope = slope + delta_slope
    beach_slopes.append(slope)
    beach_slopes = np.round(beach_slopes,len(str(delta_slope).split('.')[1]))
    return beach_slopes


def integrate_power_spectrum(dates_rand, tsall, settings_slope, Plot=False):
    'integrate power spectrum at the frequency band of peak tidal signal'
    # set common params
    t, days_in_year, seconds_in_day, time_step, freqs = FreqParams(dates_rand, settings_slope)

    beach_slopes = settings_slope['beach_slopes']
    # integrate power spectrum
    idx_interval = np.logical_and(freqs >= settings_slope['freqs_max'][0], freqs <= settings_slope['freqs_max'][1]) 
    E = np.zeros(beach_slopes.size)
    for i in range(len(tsall)):
        ps, _, _ = power_spectrum(t,tsall[i],freqs,[])
        E[i] = sintegrate.simps(ps[idx_interval], x=freqs[idx_interval], even='avg')
    # calculate confidence interval
    delta = 0.0001
    prc = settings_slope['prc_conf']
    f = sinterpolate.interp1d(beach_slopes, E, kind='linear')
    beach_slopes_interp = range_slopes(settings_slope['slope_min'],settings_slope['slope_max']-delta,delta)
    E_interp = f(beach_slopes_interp)
    # find values below minimum + 5%
    slopes_min = beach_slopes_interp[np.where(E_interp <= np.min(E)*(1+prc))[0]]
    if len(slopes_min) > 1:
        ci = [slopes_min[0],slopes_min[-1]]
    else:
        ci = [beach_slopes[np.argmin(E)],beach_slopes[np.argmin(E)]]
    
    if Plot is True:
        # plot energy vs slope curve
        fig = plt.figure()
        fig.set_size_inches([12,4])
        fig.set_tight_layout(True)
        ax = fig.add_subplot(111)
        ax.grid(linestyle=':', color='0.5')
        ax.set(title='Energy in tidal frequency band', xlabel='slope values',ylabel='energy')
        ax.plot(beach_slopes_interp,E_interp,'-k',lw=1.5)
        cmap = cm.get_cmap('RdYlGn')
        color_list = cmap(np.linspace(0,1,len(beach_slopes)))
        for i in range(len(beach_slopes)): ax.plot(beach_slopes[i], E[i],'o',ms=8,mec='k',mfc=color_list[i,:])
        ax.plot(beach_slopes[np.argmin(E)],np.min(E),'bo',ms=14,mfc='None',mew=2)
        ax.text(0.65,0.85,
                'slope estimate = %.3f\nconf. band = [%.3f , %.3f]'%(beach_slopes[np.argmin(E)],ci[0],ci[1]),
                transform=ax.transAxes,va='center',ha='left',
                bbox=dict(boxstyle='round', ec='k',fc='w', alpha=0.5),fontsize=12)
        ax.axhspan(ymin=np.min(E),ymax=np.min(E)*(1+prc),fc='0.7',alpha=0.5)
        ybottom = ax.get_ylim()[0]
        ax.plot([ci[0],ci[0]],[ybottom,f(ci[0])],'k--',lw=1,zorder=0)
        ax.plot([ci[1],ci[1]],[ybottom,f(ci[1])],'k--',lw=1,zorder=0)
        ax.plot([ci[0],ci[1]],[ybottom,ybottom],'k--',lw=1,zorder=0)
    
        
    return beach_slopes[np.argmin(E)], ci


def tide_correct(chain, tide_level, beach_slopes):
    'apply tidal correction with a range of slopes'
    tcorr = []
    for i,slope in enumerate(beach_slopes):
        # apply tidal correction
        tide_correction = (tide_level)/slope
        ts = chain + tide_correction
        tcorr.append(ts)
    return tcorr