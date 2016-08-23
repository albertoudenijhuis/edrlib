#!/usr/bin/env python

__readme = \
'''
edrlib.py
===============
A Python Module with functions that are necessary to calculate the Eddy Dissipation Rate (EDR) with different methods,
based on a sequence of velocities.

Author
======
Albert Oude Nijhuis <albertoudenijhuis@gmail.com>

Institute
=========
Delft University of Technology

Date
====
March 7th, 2016

Version
=======
1.0

Project
=======
EU FP7 program, the UFO project

Acknowledgement and citation
============================
Whenever this Python module is used for publication,
the code writer should be informed, acknowledged and referenced.
If you have any suggestions for improvements or amendments, please inform the author of this class.

Oude Nijhuis, A. C. P., Unal, C. M. H., Krasnov, O. A., Russchenberg, H. W. J., & Yarovoy, A. (2016). Drop size distribution independent radar based EDR retrieval techniques applied during rain: I. Assessment by two case studies (in preperation). Journal of Atmospheric and Oceanic Technology.

Typical usage
=============
import edrlib
velocity_series                 = {}        #dictionary containing everything
velocity_series['domain']       = 'space'   #either 'space' or 'time'
velocity_series['dx']           = 0.1       #units: m
velocity_series['y']            = v         #place here the velocity series, units: m/s

edrlib.kolmogorov_constants(velocity_series, 'full')           #set Kolmogorov constants 
edrlib.do_edr_retrievals(velocity_series)           #do edr retrievals with different methods
edrlib.printstats(velocity_series)                  #print retrieved edr values
edrlib.makeplots(velocity_series)                   #make plots of it all

#for the time domain, update two lines to:
#velocity_series['domain']      = 'time'
#velocity_series['dt']          = 0.1       #units: s

Testing
=======
For testing the class can be executed from the command line:
./edrlib.py
Three test will be run. See the function test() at the bottom of this file for the details.

Revision History
================
-
'''
print __readme


import numpy as np
import sys
from copy import deepcopy
from pprint import pprint

from scipy.fftpack import fft, ifft, fftfreq
from scipy.special import gamma

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc; rc('text',usetex=True)

import os

#do analysis of velocity_series
def do_analysis(velocity_series):
    #input velocity_series is a dictionary either for the space or time domain
    #input: velocity_series['domain'] = 'space', velocity_series['dx'], velocity_series['y']
    #input: velocity_series['domain'] = 'time',  velocity_series['dt'], velocity_series['y']

    velocity_series['n']                        = len(velocity_series['y'])
    velocity_series['i']                        = np.arange(velocity_series['n'])

    if (np.sum(np.isnan(velocity_series['y']) | np.isinf(velocity_series['y'])) > 0):
        print "\n\nWARNING: your velocity_series contains nans and/or infs!\n\n"

    if velocity_series['domain'] == 'space':
        velocity_series['x']                    = velocity_series['i'] * velocity_series['dx']
        velocity_series['tau']                  = fftfreq(velocity_series['n']) * 1. * velocity_series['n'] * velocity_series['dx']
        velocity_series['d']                    = velocity_series['dx']
        
    if velocity_series['domain'] == 'time':
        velocity_series['t']                    = velocity_series['i'] * velocity_series['dt']
        velocity_series['tau']                  = fftfreq(velocity_series['n']) * 1. * velocity_series['n'] * velocity_series['dt']
        velocity_series['d']                    = velocity_series['dt']
        
    velocity_series['freq']                     = 2. * np.pi * fftfreq(velocity_series['n'], velocity_series['d'])
    velocity_series['freqsort']                 = np.argsort(velocity_series['freq'])
    velocity_series['freqsort+']                = np.compress(velocity_series['freq'][velocity_series['freqsort']] > 0., velocity_series['freqsort'])

    velocity_series['freqmin']                  = np.min(velocity_series['freq'][velocity_series['freqsort+']])
    velocity_series['freqmax']                  = np.max(velocity_series['freq'][velocity_series['freqsort+']])

    #analysis
    velocity_series['mu']                       = np.average(velocity_series['y'])
    velocity_series['std']                      = np.std(velocity_series['y'])
    velocity_series['var']                      = np.var(velocity_series['y'])

    velocity_series['nonperiodic_d2']           = struct_function(velocity_series['y'], 2, False)
    velocity_series['nonperiodic_d3']           = struct_function(velocity_series['y'], 3, False)
    velocity_series['nonperiodic_autocov']      = autocovariance(velocity_series['y'], False)
    velocity_series['nonperiodic_pow']          = power(velocity_series['y'], False)

    velocity_series['periodic_d2']              = struct_function(velocity_series['y'], 2, True)
    velocity_series['periodic_d3']              = struct_function(velocity_series['y'], 3, True)
    velocity_series['periodic_autocov']         = autocovariance(velocity_series['y'], True)
    velocity_series['periodic_fft'],           \
    velocity_series['periodic_phase'],         \
    velocity_series['periodic_pow']            = power(velocity_series['y'], True)

    return True


#add Kolmogorov constants to a dictionary
#kolmogorov_constant_power  : Kolmogorov constant for the power spectrum
#kolmogorov_constant_struc2 : Kolmogorov constant for the second order structure function
#kolmogorov_constant_struc3 : Kolmogorov constant for the third order structure function
def kolmogorov_constants(dct, choice):
    C = 1.5

    q           = 2./3.                                                 #~0.66
    C1_div_C2   = (1. / np.pi) * gamma(1.+q) * np.sin(np.pi * q / 2.)   #~0.25
    C2_div_C1   = 1. / C1_div_C2                                        #~4.

    if choice=='longitudinal':        
        dct['kolmogorov_constant_power']     = (18./55.) * C                             #0.49
        dct['kolmogorov_constant_struc2']    = C2_div_C1 * (18./55.) * C                 
        dct['kolmogorov_constant_struc3']    = -4./5.
    elif choice=='transverse':
        dct['kolmogorov_constant_power']     = (4./3.) * (18./55.) * C                   #0.65
        dct['kolmogorov_constant_struc2']    = (4./3.) * C2_div_C1 *  (18./55.) * C
        dct['kolmogorov_constant_struc3']    = -4./5.            
    elif choice=='full':
        dct['kolmogorov_constant_power']     = C
        dct['kolmogorov_constant_struc2']    = C2_div_C1 * dct['kolmogorov_constant_power']
        dct['kolmogorov_constant_struc3']    = -4./5.                                   
    return True

#add Kolmogorov constants for a (radar / lidar) line of sight to a dictionary
def kolmogorov_constants_los(dct, azimuthrad, azimuth0rad, elevationrad):
    delta_azimuthrad = azimuthrad - azimuth0rad
    
    kc_trans = {}; kc_longi = {}
    kolmogorov_constants(kc_trans, 'transverse')
    kolmogorov_constants(kc_longi, 'longitudinal')

    dct['kolmogorov_constant_power']     = \
                            ((np.cos(elevationrad) * np.cos(delta_azimuthrad))**2.) * kc_longi['kolmogorov_constant_power'] \
                            + ((np.cos(elevationrad) * np.sin(delta_azimuthrad))**2.) * kc_trans['kolmogorov_constant_power'] \
                            + (np.sin(elevationrad)**2.) * kc_trans['kolmogorov_constant_power']
    dct['kolmogorov_constant_struc2']    = \
                            ((np.cos(elevationrad) * np.cos(delta_azimuthrad))**2.) * kc_longi['kolmogorov_constant_struc2'] \
                            + ((np.cos(elevationrad) * np.sin(delta_azimuthrad))**2.) * kc_trans['kolmogorov_constant_struc2'] \
                            + (np.sin(elevationrad)**2.) * kc_trans['kolmogorov_constant_struc2']
    dct['kolmogorov_constant_struc3']    = -4./5.
    return True


#One EDR is retrieved for the sequence of velocities
def do_edr_retrievals(dct):
    do_analysis(dct)       #do analysis first
    
    for key_str in ['periodic', 'nonperiodic']:          
        #edr via variance method
        res   = retr_edr_via_variance(dct, key_str)
        dct[key_str+'_variancemethod_edr']          = res['edr']
        dct[key_str+'_variancemethod_edrerr']       = res['edr.err']
        dct[key_str+'_variancemethod_edr+']         = (res['edr'] ** (1./3.) + res['edr.err'])**3.
        dct[key_str+'_variancemethod_edr-']         = (res['edr'] ** (1./3.) - res['edr.err'])**3.        
        dct[key_str+'_variancemethod_edr++']        = (res['edr'] ** (1./3.) + 2. * res['edr.err'])**3.
        dct[key_str+'_variancemethod_edr--']        = (res['edr'] ** (1./3.) - 2. * res['edr.err'])**3.        

        #edr via power spectrum method
        res = retr_edr_via_power_spectrum(dct,key_str)
        dct[key_str+'_powerspectrum_edr']           = res['edr']
        dct[key_str+'_powerspectrum_edrerr']        = res['edr.err']
        dct[key_str+'_powerspectrum_edr+']          = (res['edr'] ** (1./3.) + res['edr.err'])**3.
        dct[key_str+'_powerspectrum_edr-']          = (res['edr'] ** (1./3.) - res['edr.err'])**3.        
        dct[key_str+'_powerspectrum_edr++']         = (res['edr'] ** (1./3.) + 2. * res['edr.err'])**3.
        dct[key_str+'_powerspectrum_edr--']         = (res['edr'] ** (1./3.) - 2. * res['edr.err'])**3.        
        dct[key_str+'_powerspectrum_lstedr']        = res['lst_edr']
        dct[key_str+'_powerspectrum_lstfreq']       = res['lst_freq']
        dct[key_str+'_powerspectrum_lstfreqmin']    = res['lst_freqmin']
        dct[key_str+'_powerspectrum_lstfreqmax']    = res['lst_freqmax']

        #edr via second order structure function
        res = retr_edr_via_2nd_order(dct,key_str)
        dct[key_str+'_2ndorder_edr']                = res['edr']
        dct[key_str+'_2ndorder_edrerr']             = res['edr.err']
        dct[key_str+'_2ndorder_edr+']               = (res['edr'] ** (1./3.) + res['edr.err'])**3.
        dct[key_str+'_2ndorder_edr-']               = (res['edr'] ** (1./3.) - res['edr.err'])**3.
        dct[key_str+'_2ndorder_edr++']              = (res['edr'] ** (1./3.) + 2. * res['edr.err'])**3.
        dct[key_str+'_2ndorder_edr--']              = (res['edr'] ** (1./3.) - 2. * res['edr.err'])**3.
        dct[key_str+'_2ndorder_lstedr']             = res['lst_edr']

        #edr via third order structure function        
        res = retr_edr_via_3rd_order(dct,key_str)
        dct[key_str+'_3rdorder_edr']                = res['edr']
        dct[key_str+'_3rdorder_edrerr']             = res['edr.err']
        dct[key_str+'_3rdorder_edr+']               = (np.sign(res['edr']) * (np.abs(res['edr']) ** (1./3.)) + res['edr.err'])**3.
        dct[key_str+'_3rdorder_edr-']               = (np.sign(res['edr']) * (np.abs(res['edr']) ** (1./3.)) - res['edr.err'])**3.
        dct[key_str+'_3rdorder_edr++']              = (np.sign(res['edr']) * (np.abs(res['edr']) ** (1./3.)) + 2. * res['edr.err'])**3.
        dct[key_str+'_3rdorder_edr--']              = (np.sign(res['edr']) * (np.abs(res['edr']) ** (1./3.)) - 2. * res['edr.err'])**3.
        dct[key_str+'_3rdorder_lstedr']             = res['lst_edr']

        #Calculation according to Siebert et al. (2006)
        #Siebert, H., Lehmann, K., & Wendisch, M. (2006). Observations of Small-Scale Turbulence and Energy Dissipation Rates in the Cloudy Boundary Layer. Journal of the Atmospheric Sciences, 63(5), 1451-1466. http://doi.org/10.1175/JAS3687.1
        nu = 1.5e-5 #kinematic viscosity
        dct[key_str+'_powerspectrum_taylorreynolds'] = (dct['std'] ** 2. ) * np.sqrt(15. / (nu * dct[key_str+'_powerspectrum_edr']))      
    return True







f_even = lambda x: x % 2 == 0
#help function
def taulist(n):
    if f_even(n):
        lst1 = 1+np.arange(n/2-1)
        lst2 = np.hstack((0,lst1, -n/2,-lst1[::-1]))
    else:
        lst1 = 1+np.arange((n-1)/2)
        lst2 = np.hstack((0,lst1,-lst1[::-1]))
    return lst2

#calculate autocovariance of x
def autocovariance(x, periodic=False):
    n = len(x)
    x_auto = np.zeros(n) + np.nan
    if periodic:
        #assume that sequence x is periodic
        for i in taulist(n):   #0, 1, 2, 3, ... -3, -2, -1
            a = x[:]
            b = np.hstack(( x[i:], x[0:i] ))
            x_auto[i] = np.cov(a,b, bias=1)[0,1]
    else:
        #assume that sequence x is non-periodic
        for i in taulist(n):   #0, 1, 2, 3, ... -3, -2, -1
            if i > 0:
                a = x[:-i]
                b = x[i:]
            if i <= 0:
                a = x[-i:]
                b = x[0:n+i]
            x_auto[i] = np.cov(a,b, bias=1)[0,1]        
    return x_auto

#calculate power density spectrum
def power(x, periodic=False):
    if periodic:
        o_fft   = fft(x)
        o_phase = np.angle(o_fft)
        o_pow   = np.abs(o_fft /len(x))**2.
        return o_fft, o_phase, o_pow
    else:
        ac      = autocovariance(x, False)
        mu      = np.average(x)
        o_acfft = fft((ac + mu**2.) / len(x))
        o_pow   = np.real(o_acfft)
        o_pow   = np.abs(o_pow)
        return o_pow

#calculate structure function of x
def struct_function(x, order=2, periodic=False):
    n = len(x)
    x_struc = np.zeros(n) + np.nan

    if periodic:
        #assume that sequence x is periodic
        for i in taulist(n):   #0, 1, 2, 3, ... -3, -2, -1
            a = x[:]
            b = np.hstack(( x[i:], x[0:i] ))
            lst1 = (np.sign(a-b)**order) *(np.abs(a-b)**(order))
            x_struc[i] = np.average(lst1)
    else:
        #assume that sequence x is non-periodic
        for i in taulist(n):   #0, 1, 2, 3, ... -3, -2, -1
            if i > 0:
                a = x[:-i]
                b = x[i:]
            if i <= 0:
                a = x[-i:]    
                b = x[:n+i]     
            lst1 = (np.sign(a-b)**order) *(np.abs(a-b)**(order))
            x_struc[i] = np.average(lst1)
    return x_struc

#make fft coefficients real
def make_fft_coef_real(fft_coef_in):
    fft_coef = deepcopy(fft_coef_in)
    n = len(fft_coef)
    if f_even(n):
        fft_coef[-1:-n/2:-1] = np.conj(fft_coef[1:n/2]) #make signal real for even number
        fft_coef[n/2] = np.abs(fft_coef[n/2])           #for even number            
    else:
        fft_coef[-1:-n/2:-1] = np.conj(fft_coef[1:n/2+1])   #make signal real for uneven number         
    return fft_coef 




#EDR via variance
#k      : wavenumber
#dk     : smallest wavenumber
#eps    : eddy dissipation rate
#C      : Kolmogorov universal constant
def retr_edr_via_variance(dct,key_str):
    thisedr     = (((3.0/2.0) * dct['kolmogorov_constant_power'] * \
                        (  ((dct['freqmin'] - 0.5 * dct['freqmin']) ** (-2.0/3.0))  - ((dct['freqmax'] + 0.5 * dct['freqmin']) ** (-2.0/3.0)) ) \
                        ) ** (-3.0/2.0) ) \
                        * (dct['var'] ** (3.0/2.0))

    if dct['domain'] == 'time':
        thisedr *= dct['u0'] ** (-1.0)

    res = {}
    res['edr']      = thisedr
    
    if dct['domain'] == 'time':
        res['edr.err'] = (1./3.) * \
                            (res['edr'] ** (1. / 3.)) \
                            * np.sqrt(((dct['freqmin']/dct['freqmax'])**(4./3.)) + (dct['var'] / (dct['u0'] ** 2.)) + (9. / (2. * (dct['n'] - 1.))))
    else:
        res['edr.err'] = (1./3.) * \
                            (res['edr'] ** (1. / 3.)) \
                            * np.sqrt(((dct['freqmin']/dct['freqmax'])**(4./3.)) +  (9. / (2. * (dct['n'] - 1.))))
    
    return res






#EDR via power spectrum.
#k      : wavenumber
#dk     : smallest wavenumber
#eps    : eddy dissipation rate
#C      : Kolmogorov universal constant
model_edr_via_power_spectrum    = lambda k,eps, dk, C: (3./2.) * C * ( eps ** (2./3.)) * (((k - (0.5 * dk)) ** (-2./3.))    - ((k + (0.5*dk)) ** (-2./3.))  )
model_edr_via_power_spectrum2   = lambda k,eps, dk, C: C * ( eps ** (2./3.)) * k ** (-5./3.)
#model_edr_via_power_spectrum3   = lambda k,eps, dk, C: dk * C * ( eps ** (2./3.)) * k ** (-2./3.)

def retr_edr_via_power_spectrum(dct,key_str):
    fft_pfreq   = dct['freq'][dct['freqsort+']]
    fft_ppow    = dct[key_str+'_pow'][dct['freqsort+']]
    
    nintervals  = 3
    nfreq       = max(1,int(np.ceil((1. * len(fft_ppow)) / nintervals)))        #number of frequencies per interval
    
    j = -1
    res = {}
    res['lst_pow']      = []
    res['lst_freq']     = []
    res['lst_freqmin']  = []
    res['lst_freqmax']  = []
    res['lst_edr']      = []

    for i1 in range(0,len(fft_ppow), nfreq):
        i2 = i1 + nfreq
        
        j+=1
        thispow     = 2.0 * np.sum(fft_ppow[i1:i2]) #2.0 because of only positive frequencies
        thisfreq    = np.average(fft_pfreq[i1:i2])
        thisfreqmin = np.min(fft_pfreq[i1:i2])
        thisfreqmax = np.max(fft_pfreq[i1:i2])
        thisedr     = (((3.0/2.0) * dct['kolmogorov_constant_power'] * \
                        (  ((thisfreqmin - 0.5 * dct['freqmin']) ** (-2.0/3.0))  - ((thisfreqmax + 0.5 * dct['freqmin']) ** (-2.0/3.0)) ) \
                        ) ** (-3.0/2.0) ) \
                        * (thispow ** (3.0/2.0))

        if dct['domain'] == 'time':
            thisedr *= dct['u0'] ** (-1.0)
            
        res['lst_pow'].append(thispow)
        res['lst_freq'].append(thisfreq)
        res['lst_freqmin'].append(thisfreqmin)
        res['lst_freqmax'].append(thisfreqmax)
        res['lst_edr'].append(thisedr)      



    res['edr']      = np.average(np.array(res['lst_edr'])**(1./3.))**3.
    res['edr.err']  = np.std(np.array(res['lst_edr'])**(1./3.))
    
    return res

#2nd order structure function
model_edr_via_2nd_order = lambda s,eps, C: C * ( (eps * s  ) ** (2./3.)) 
f_retr_edr_via_2nd_order = lambda s, d2, C: (1./ s) * ((d2 / C) ** (3./2.))
def retr_edr_via_2nd_order(dct,key_str):
    res = {}

    if dct['domain'] == 'time':
        res['lst_edr']  = f_retr_edr_via_2nd_order(dct['u0'] * dct['tau'][dct['freqsort+']],dct[key_str+'_d2'][dct['freqsort+']], dct['kolmogorov_constant_struc2'])
    else:
        res['lst_edr']  = f_retr_edr_via_2nd_order(dct['tau'][dct['freqsort+']],dct[key_str+'_d2'][dct['freqsort+']], dct['kolmogorov_constant_struc2'])
                
    res['edr']      = np.average(res['lst_edr'][1:]**(1./3.))**3.
    res['edr.err']  = np.std(res['lst_edr'][1:]**(1./3.))
    return res
        
#3rd order structure function
model_edr_via_3rd_order = lambda s,eps, C: C *  eps * s 
f_retr_edr_via_3rd_order = lambda s, d3, C: (1./C) * (d3/ s)
def retr_edr_via_3rd_order(dct,key_str):
    res = {}
    if dct['domain'] == 'time':
        res['lst_edr']  = f_retr_edr_via_3rd_order(dct['u0'] * dct['tau'][dct['freqsort+']],dct[key_str+'_d3'][dct['freqsort+']], dct['kolmogorov_constant_struc3'])
    else:
        res['lst_edr']  = f_retr_edr_via_3rd_order(dct['tau'][dct['freqsort+']],dct[key_str+'_d3'][dct['freqsort+']], dct['kolmogorov_constant_struc3'])
                
    res['lst_edr1/3'] = np.sign(res['lst_edr']) * (np.abs(res['lst_edr']) ** (1./3.))
    res['edr']      = np.average(res['lst_edr1/3'][1:])**3.
    res['edr.err']  = np.std(res['lst_edr1/3'][1:])
    return res



def makeplots(dct, name='edrlib', seperate=False, plot_periodic = False, plot_nonperiodic = True, plot_legend = True, plot_errors = False, units_in = {}):
    fontsize0 = 18
    fontsize1 = 14
    matplotlib.rc('xtick', labelsize=fontsize0) 
    matplotlib.rc('ytick', labelsize=fontsize0) 

    #sorting
    sorting = dct['freqsort+']     

    st_1 = {'color':'red' , 'alpha':0.7, 'linewidth':3}                                         #non-periodic
    st_1p= {'color':'black' , 'alpha':0.7, 'marker':'x', 'linestyle':'None', 'markersize':3}    #non-periodic
    st_1s= {'color':'red' , 'alpha':0.7, 'linewidth':3, 'linestyle':'--'}                       #non-periodic
    st_1f= {'color':'red' , 'alpha':0.1}                                                        #non-periodic
    
    st_1i= {'color':'red' , 'alpha':0.7, 'linewidth':3, 'linestyle':'-', 'zorder':10}           #non-periodic

    st_2 = {'color':'green'   , 'alpha':0.7, 'linewidth':3}                                     #periodic
    st_2p= {'color':'black'   , 'alpha':0.7, 'marker':'x', 'linestyle':'None'}                  #periodic
    st_2s= {'color':'green'   , 'alpha':0.7, 'linewidth':3, 'linestyle':'--'}                   #periodic
    st_2f= {'color':'green'   , 'alpha':0.1}                                                    #periodic
    st_2i= {'color':'green'   , 'alpha':0.7, 'linewidth':3, 'linestyle':'-', 'zorder':10}       #periodic

    if not seperate:      
        if plot_periodic:
            plot_lst = ['velocity_series', 'autocovariance', 'd2', 'd3', 'pow', 'phase']
        if plot_nonperiodic:
            plot_lst = ['velocity_series', 'autocovariance', 'd2', 'd3', 'pow']
        nrows = len(plot_lst)
        fig = plt.figure(figsize=(5,5*nrows))

    if seperate:
        plot_lst = ['velocity_series', 'autocovariance', 'd2', 'd3', 'pow', 'd22', 'd32', 'pow2', 'phase']
    
    for plot in plot_lst:
        if seperate:
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1,1,1)

        if plot == 'velocity_series':
            #velocity_series
            if not seperate:
                ax = fig.add_subplot(nrows,1,1)
                ax.set_title('velocity series')
            if dct['domain'] == 'time':
                ax.plot(dct['t'], dct['y'])
                ax.set_xlabel('$t$ [s]', fontsize=fontsize0)
            else:
                ax.plot(dct['x'], dct['y'])
                ax.set_xlabel('$x$ [m]', fontsize=fontsize0)
            ax.set_ylabel('$v$ [m/s]', fontsize=fontsize0)
           
        if plot == 'autocovariance':
            #autocovariance
            if not seperate:                
                ax = fig.add_subplot(nrows,1,2)
                ax.set_title('autocovariance')

            if plot_nonperiodic:
                ln1 = ax.plot(dct['tau'][sorting], dct['nonperiodic_autocov'][sorting], label='non-periodic',**st_1)
            if plot_periodic:
                ln2 = ax.plot(dct['tau'][sorting], dct['periodic_autocov'][sorting], label='periodic',**st_2)  

            if dct['domain'] == 'time':
                ax.set_xlabel(r'$t$ [s]', fontsize=fontsize0)
            else:
                ax.set_xlabel(r'$s$ [m]', fontsize=fontsize0)
            ax.set_ylabel(r'$R$ [m$^2$s$^{-2}$]', fontsize=fontsize0)
            if plot_legend:
                ax.legend(frameon=False, ncol=2, loc='lower left', fontsize=fontsize1)

        if plot == 'd2':
            if not seperate:
                ax = fig.add_subplot(nrows,1,3)
                ax.set_title('$D_{2}$')

            if plot_nonperiodic:
                ax.plot(dct['tau'][sorting], dct['nonperiodic_d2'][sorting], label='non-periodic', **st_1p)
            if plot_periodic:
                ax.plot(dct['tau'][sorting], dct['periodic_d2'][sorting], label='periodic', **st_2p)
            
            if dct['domain'] == 'time':
                if plot_nonperiodic:
                    ax.plot(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting], dct['u0'] * dct['nonperiodic_2ndorder_edr'], dct['kolmogorov_constant_struc2']), label='fit', **st_1)
                    if plot_errors:
                        ax.plot(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting], dct['u0'] * dct['nonperiodic_2ndorder_edr++'], dct['kolmogorov_constant_struc2']), **st_1s)
                        ax.plot(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting], dct['u0'] * dct['nonperiodic_2ndorder_edr--'], dct['kolmogorov_constant_struc2']), **st_1s)
                        ax.fill_between(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting], dct['u0'] * dct['nonperiodic_2ndorder_edr--'], dct['kolmogorov_constant_struc2']),
                            model_edr_via_2nd_order(dct['tau'][sorting], dct['u0'] * dct['nonperiodic_2ndorder_edr++'], dct['kolmogorov_constant_struc2']), **st_1f)
                
                if plot_periodic:
                    ax.plot(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting], dct['u0'] * dct['periodic_2ndorder_edr'], dct['kolmogorov_constant_struc2']), label='fit', **st_2)
                    if plot_errors:
                        ax.plot(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting], dct['u0'] * dct['periodic_2ndorder_edr++'], dct['kolmogorov_constant_struc2']), **st_2s)
                        ax.plot(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting], dct['u0'] * dct['periodic_2ndorder_edr--'], dct['kolmogorov_constant_struc2']), **st_2s)
                        ax.fill_between(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting], dct['u0'] * dct['periodic_2ndorder_edr--'], dct['kolmogorov_constant_struc2']),
                            model_edr_via_2nd_order(dct['tau'][sorting], dct['u0'] * dct['periodic_2ndorder_edr++'], dct['kolmogorov_constant_struc2']), **st_2f)
            else:
                if plot_nonperiodic:
                    ax.plot(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting] ,dct['nonperiodic_2ndorder_edr'], dct['kolmogorov_constant_struc2']), label='fit', **st_1)
                    if plot_errors:
                        ax.plot(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting] ,dct['nonperiodic_2ndorder_edr++'], dct['kolmogorov_constant_struc2']), **st_1s)
                        ax.plot(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting] ,dct['nonperiodic_2ndorder_edr--'], dct['kolmogorov_constant_struc2']), **st_1s)
                        ax.fill_between(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting], dct['nonperiodic_2ndorder_edr--'], dct['kolmogorov_constant_struc2']),
                            model_edr_via_2nd_order(dct['tau'][sorting], dct['nonperiodic_2ndorder_edr++'], dct['kolmogorov_constant_struc2']), **st_1f)                
                if plot_periodic:
                    ax.plot(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting] ,dct['periodic_2ndorder_edr'], dct['kolmogorov_constant_struc2']), label='fit', **st_2)
                    if plot_errors:
                        ax.plot(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting] ,dct['periodic_2ndorder_edr++'], dct['kolmogorov_constant_struc2']), **st_2s)
                        ax.plot(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting] ,dct['periodic_2ndorder_edr--'], dct['kolmogorov_constant_struc2']), **st_2s)
                        ax.fill_between(dct['tau'][sorting], model_edr_via_2nd_order(dct['tau'][sorting], dct['periodic_2ndorder_edr--'], dct['kolmogorov_constant_struc2']),
                            model_edr_via_2nd_order(dct['tau'][sorting], dct['periodic_2ndorder_edr++'], dct['kolmogorov_constant_struc2']), **st_2f)

            if dct['domain'] == 'time':
                ax.set_xlabel(r'$t$ [s]', fontsize=fontsize0)
                ax.set_ylabel(r'$D_{2}$ [m$^2$s$^{-2}$]', fontsize=fontsize0)
            else:
                ax.set_xlabel(r'$s$ [m]', fontsize=fontsize0)
                ax.set_ylabel(r'$D_{2}$ [m$^2$s$^{-2}$]', fontsize=fontsize0)
            if plot_legend:
                ax.legend(frameon=False, loc='upper left', ncol=2, fontsize=fontsize1)
            
        if plot == 'd22':
            if not seperate:
                ax = fig.add_subplot(nrows,1,3)
                ax.set_title('$D_{2}$')

            if plot_nonperiodic:
                ax.plot(dct['tau'][dct['freqsort+']], dct['nonperiodic_2ndorder_lstedr'], label='non-periodic', **st_1p)            
            if plot_periodic:
                ax.plot(dct['tau'][dct['freqsort+']], dct['periodic_2ndorder_lstedr'],label='periodic', **st_2p)

            if plot_nonperiodic:
                mylst = np.zeros(len(dct['nonperiodic_3rdorder_lstedr'])) 
                ax.plot(dct['tau'][dct['freqsort+']]  , mylst + dct['nonperiodic_2ndorder_edr'], label='fit', **st_1)
                if plot_errors:
                    ax.plot(dct['tau'][dct['freqsort+']]  , mylst + dct['nonperiodic_2ndorder_edr++'], **st_1s)
                    ax.plot(dct['tau'][dct['freqsort+']]  , mylst + dct['nonperiodic_2ndorder_edr--'], **st_1s)
                    ax.fill_between(dct['tau'][dct['freqsort+']]  , mylst + dct['nonperiodic_2ndorder_edr++'],
                                    mylst + dct['nonperiodic_2ndorder_edr--'], **st_1f)
            
            if plot_periodic:
                mylst = np.zeros(len(dct['periodic_3rdorder_lstedr'])) 
                ax.plot(dct['tau'][dct['freqsort+']]  , mylst + dct['periodic_2ndorder_edr'], label='fit', **st_2)
                if plot_errors:
                    ax.plot(dct['tau'][dct['freqsort+']]  , mylst + dct['periodic_2ndorder_edr++'], **st_2s)
                    ax.plot(dct['tau'][dct['freqsort+']]  , mylst + dct['periodic_2ndorder_edr--'], **st_2s)
                    ax.fill_between(dct['tau'][dct['freqsort+']]  , mylst + dct['periodic_2ndorder_edr++'],
                                    mylst + dct['periodic_2ndorder_edr--'], **st_2f)

            if dct['domain'] == 'time':
                ax.set_xlabel(r'$t$ [s]', fontsize=fontsize0)
            else:
                ax.set_xlabel(r'$s$ [m]', fontsize=fontsize0)
            ax.set_ylabel(r'$\epsilon$ [m$^2$s$^{-3}$]', fontsize=fontsize0)
                                
            if plot_legend:
                ax.legend(frameon=False, ncol=2, loc='lower left', fontsize=fontsize1)

            try:
                ax.set_yscale('log')
            except:
                pass


        if plot == 'd3':
            #d3              
            if not seperate:
                ax = fig.add_subplot(nrows,1,4)
                ax.set_title('$D_{3}$')

            if plot_nonperiodic:
                ax.plot(dct['tau'][sorting], dct['nonperiodic_d3'][sorting], label='non-periodic', **st_1p)
            if plot_periodic:
                ax.plot(dct['tau'][sorting], dct['periodic_d3'][sorting], label='periodic', **st_2p)
            
            if dct['domain'] == 'time':
                if plot_nonperiodic:
                    ax.plot(dct['tau'][sorting], model_edr_via_3rd_order(dct['tau'][sorting], dct['u0'] * dct['nonperiodic_3rdorder_edr'], dct['kolmogorov_constant_struc3']), label='fit', **st_1)
                    if plot_errors:
                        ax.plot(dct['tau'][sorting], model_edr_via_3rd_order(dct['tau'][sorting], dct['u0'] * dct['nonperiodic_3rdorder_edr++'], dct['kolmogorov_constant_struc3']), **st_1s)
                        ax.plot(dct['tau'][sorting], model_edr_via_3rd_order(dct['tau'][sorting], dct['u0'] * dct['nonperiodic_3rdorder_edr--'], dct['kolmogorov_constant_struc3']), **st_1s)
                        ax.fill_between(dct['tau'][sorting],   model_edr_via_3rd_order(dct['tau'][sorting], dct['u0'] * dct['nonperiodic_3rdorder_edr--'], dct['kolmogorov_constant_struc3']),
                                                                model_edr_via_3rd_order(dct['tau'][sorting], dct['u0'] * dct['nonperiodic_3rdorder_edr++'], dct['kolmogorov_constant_struc3']), **st_1f)
                
                if plot_periodic:
                    ax.plot(dct['tau'][sorting], model_edr_via_3rd_order(dct['tau'][sorting], dct['u0'] * dct['periodic_3rdorder_edr'], dct['kolmogorov_constant_struc3']), label='fit', **st_2)
                    if plot_errors:
                        ax.plot(dct['tau'][sorting], model_edr_via_3rd_order(dct['tau'][sorting], dct['u0'] * dct['periodic_3rdorder_edr++'], dct['kolmogorov_constant_struc3']), **st_2s)
                        ax.plot(dct['tau'][sorting], model_edr_via_3rd_order(dct['tau'][sorting], dct['u0'] * dct['periodic_3rdorder_edr--'], dct['kolmogorov_constant_struc3']), **st_2s)
                        ax.fill_between(dct['tau'][sorting],   model_edr_via_3rd_order(dct['tau'][sorting], dct['u0'] * dct['periodic_3rdorder_edr--'], dct['kolmogorov_constant_struc3']),
                                                                model_edr_via_3rd_order(dct['tau'][sorting], dct['u0'] * dct['periodic_3rdorder_edr++'], dct['kolmogorov_constant_struc3']), **st_2f)

            else:
                if plot_nonperiodic:
                    ax.plot(dct['tau'][sorting], model_edr_via_3rd_order(dct['tau'][sorting] ,dct['nonperiodic_3rdorder_edr'], dct['kolmogorov_constant_struc3']), label='fit', **st_1)
                    if plot_errors:
                        ax.plot(dct['tau'][sorting], model_edr_via_3rd_order(dct['tau'][sorting] ,dct['nonperiodic_3rdorder_edr++'], dct['kolmogorov_constant_struc3']), **st_1s)
                        ax.plot(dct['tau'][sorting], model_edr_via_3rd_order(dct['tau'][sorting] ,dct['nonperiodic_3rdorder_edr--'], dct['kolmogorov_constant_struc3']), **st_1s)
                        ax.fill_between(dct['tau'][sorting],   model_edr_via_3rd_order(dct['tau'][sorting], dct['nonperiodic_3rdorder_edr--'], dct['kolmogorov_constant_struc3']),
                                                                model_edr_via_3rd_order(dct['tau'][sorting], dct['nonperiodic_3rdorder_edr++'], dct['kolmogorov_constant_struc3']), **st_1f)
                
                if plot_periodic:
                    ax.plot(dct['tau'][sorting], model_edr_via_3rd_order(dct['tau'][sorting] ,dct['periodic_3rdorder_edr'], dct['kolmogorov_constant_struc3']), label='fit', **st_2)
                    if plot_errors:
                        ax.plot(dct['tau'][sorting], model_edr_via_3rd_order(dct['tau'][sorting] ,dct['periodic_3rdorder_edr++'], dct['kolmogorov_constant_struc3']), **st_2s)
                        ax.plot(dct['tau'][sorting], model_edr_via_3rd_order(dct['tau'][sorting] ,dct['periodic_3rdorder_edr--'], dct['kolmogorov_constant_struc3']), **st_2s)
                        ax.fill_between(dct['tau'][sorting],   model_edr_via_3rd_order(dct['tau'][sorting], dct['periodic_3rdorder_edr--'], dct['kolmogorov_constant_struc3']),
                                                                model_edr_via_3rd_order(dct['tau'][sorting], dct['periodic_3rdorder_edr++'], dct['kolmogorov_constant_struc3']), **st_2f)

            if dct['domain'] == 'time':
                ax.set_xlabel(r'$t$ [s]', fontsize=fontsize0)
            else:
                ax.set_xlabel(r'$s$ [m]', fontsize=fontsize0)
            ax.set_ylabel(r'$D_{3}$ [m$^3$s$^{-3}$]', fontsize=fontsize0)
            if plot_legend:
                ax.legend(frameon=False, ncol=2, loc='upper left', fontsize=fontsize1)
            
        if plot == 'd32':
            if not seperate:
                ax = fig.add_subplot(nrows,1,4)
                ax.set_title('$D_{3}$')

            if plot_nonperiodic:
                ax.plot(dct['tau'][dct['freqsort+']], np.abs(dct['nonperiodic_3rdorder_lstedr'])
                                        , label='non-periodic', **st_1p)
            if plot_periodic:
                ax.plot(dct['tau'][dct['freqsort+']], np.abs(dct['periodic_3rdorder_lstedr'])
                                            , label='periodic', **st_2p)

            if plot_nonperiodic:
                mylst = np.zeros(len(dct['nonperiodic_3rdorder_lstedr'])) 
                ax.plot(dct['tau'][dct['freqsort+']]  , mylst + np.abs(dct['nonperiodic_3rdorder_edr']), label='fit', **st_1)
                if plot_errors:
                    ax.plot(dct['tau'][dct['freqsort+']]  , mylst + np.abs(dct['nonperiodic_3rdorder_edr++']), **st_1s)
                    ax.plot(dct['tau'][dct['freqsort+']]  , mylst + np.abs(dct['nonperiodic_3rdorder_edr--']), **st_1s)
                    ax.fill_between(dct['tau'][dct['freqsort+']], mylst   + np.abs(dct['nonperiodic_3rdorder_edr--']),
                                                                    mylst   + np.abs(dct['nonperiodic_3rdorder_edr++']), **st_1f)


            if plot_periodic:
                mylst = np.zeros(len(dct['periodic_3rdorder_lstedr'])) 
                ax.plot(dct['tau'][dct['freqsort+']]      , mylst + np.abs(dct['periodic_3rdorder_edr']), label='fit', **st_2)
                if plot_errors:
                    ax.plot(dct['tau'][dct['freqsort+']]      , mylst + np.abs(dct['periodic_3rdorder_edr++']), **st_2s)
                    ax.plot(dct['tau'][dct['freqsort+']]      , mylst + np.abs(dct['periodic_3rdorder_edr--']), **st_2s)
                    ax.fill_between(dct['tau'][dct['freqsort+']], mylst   + np.abs(dct['periodic_3rdorder_edr--']),
                                                                    mylst   + np.abs(dct['periodic_3rdorder_edr++']), **st_2f)

            if dct['domain'] == 'time':
                ax.set_xlabel(r'$t$ [s]', fontsize=fontsize0)
            else:
                ax.set_xlabel(r'$s$ [m]', fontsize=fontsize0)

            ax.set_ylabel(r'$\epsilon$\ [m$^2$s$^{-3}$]', fontsize=fontsize0)
                
            if plot_legend:
                ax.legend(frameon=False, ncol=2, loc='lower left', fontsize=fontsize1)

            try:
                ax.set_yscale('log')
            except:
                pass
                
        if plot == 'pow':
            #pow              
            if not seperate:
                ax = fig.add_subplot(nrows,1,5)
                ax.set_title('pow')
                
            if plot_nonperiodic:
                dat = deepcopy(dct['nonperiodic_pow'][sorting])
                dat = np.where(dat < (1.e-10 * np.max(dat)), np.nan , dat)
                ax.plot(dct['freq'][sorting], dat, label='non-periodic', **st_1p) 
            if plot_periodic:
                dat = deepcopy(dct['periodic_pow'][sorting])
                dat = np.where(dat < (1.e-10 * np.max(dat)), np.nan , dat)
                ax.plot(dct['freq'][sorting], dat, label='periodic', **st_2p)
                                        
            if dct['domain'] == 'time':
                if plot_nonperiodic:
                    #ax.plot(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['u0'] * dct['nonperiodic_powerspectrum_edr'], dct['freqmin'],dct['kolmogorov_constant_power']), label='fit', **st_1)
                    for int_i in range(len(dct['nonperiodic_powerspectrum_lstedr'])):
                        tmpx = np.array( [dct['nonperiodic_powerspectrum_lstfreqmin'][int_i], dct['nonperiodic_powerspectrum_lstfreqmax'][int_i]] )
                        ax.plot(tmpx   , 0.5 * model_edr_via_power_spectrum (tmpx ,dct['u0'] * dct['nonperiodic_powerspectrum_lstedr'][int_i] , dct['freqmin'],dct['kolmogorov_constant_power']), label='interval', **st_1i)                

                    if plot_errors:                    
                        ax.plot(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['u0'] * dct['nonperiodic_powerspectrum_edr++'], dct['freqmin'],dct['kolmogorov_constant_power']), **st_1s)
                        ax.plot(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['u0'] * dct['nonperiodic_powerspectrum_edr--'], dct['freqmin'],dct['kolmogorov_constant_power']), **st_1s)
                        ax.fill_between(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['u0'] * dct['nonperiodic_powerspectrum_edr--'], dct['freqmin'],dct['kolmogorov_constant_power']),
                                                                  0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['u0'] * dct['nonperiodic_powerspectrum_edr++'], dct['freqmin'],dct['kolmogorov_constant_power']), **st_1f)
                if plot_periodic:                                
                    #ax.plot(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['u0'] * dct['periodic_powerspectrum_edr'], dct['freqmin'],dct['kolmogorov_constant_power']), label='fit', **st_2)
                    for int_i in range(len(dct['periodic_powerspectrum_lstedr'])):
                        tmpx = np.array( [dct['periodic_powerspectrum_lstfreqmin'][int_i], dct['periodic_powerspectrum_lstfreqmax'][int_i]] )
                        ax.plot(tmpx   , 0.5 * model_edr_via_power_spectrum (tmpx ,dct['u0'] * dct['periodic_powerspectrum_lstedr'][int_i] , dct['freqmin'],dct['kolmogorov_constant_power']), label='interval', **st_2i)

                    if plot_errors:
                        ax.plot(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['u0'] * dct['periodic_powerspectrum_edr++'], dct['freqmin'],dct['kolmogorov_constant_power']), **st_2s)
                        ax.plot(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['u0'] * dct['periodic_powerspectrum_edr--'], dct['freqmin'],dct['kolmogorov_constant_power']), **st_2s)
                        ax.fill_between(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['u0'] * dct['periodic_powerspectrum_edr--'], dct['freqmin'],dct['kolmogorov_constant_power']),
                                                              0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['u0'] * dct['periodic_powerspectrum_edr++'], dct['freqmin'],dct['kolmogorov_constant_power']), **st_2f)

            else:
                if plot_nonperiodic:
                    #ax.plot(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['nonperiodic_powerspectrum_edr'], dct['freqmin'],dct['kolmogorov_constant_power']), label='fit', **st_1)
                    for int_i in range(len(dct['nonperiodic_powerspectrum_lstedr'])):
                        tmpx = np.array( [dct['nonperiodic_powerspectrum_lstfreqmin'][int_i], dct['nonperiodic_powerspectrum_lstfreqmax'][int_i]] )
                        ax.plot(tmpx   , 0.5 * model_edr_via_power_spectrum (tmpx ,dct['nonperiodic_powerspectrum_lstedr'][int_i] , dct['freqmin'],dct['kolmogorov_constant_power']), label='interval', **st_1i)

                    if plot_errors:
                        ax.plot(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['nonperiodic_powerspectrum_edr++'], dct['freqmin'],dct['kolmogorov_constant_power']), **st_1s)
                        ax.plot(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['nonperiodic_powerspectrum_edr--'], dct['freqmin'],dct['kolmogorov_constant_power']), **st_1s)
                        ax.fill_between(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['nonperiodic_powerspectrum_edr--'], dct['freqmin'],dct['kolmogorov_constant_power']),
                                                                  0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['nonperiodic_powerspectrum_edr++'], dct['freqmin'],dct['kolmogorov_constant_power']), **st_1f)

                if plot_periodic:
                    #ax.plot(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['periodic_powerspectrum_edr'], dct['freqmin'],dct['kolmogorov_constant_power']), label='fit', **st_2)
                    for int_i in range(len(dct['periodic_powerspectrum_lstedr'])):
                        tmpx = np.array( [dct['periodic_powerspectrum_lstfreqmin'][int_i], dct['periodic_powerspectrum_lstfreqmax'][int_i]] )
                        ax.plot(tmpx   , 0.5 * model_edr_via_power_spectrum (tmpx , dct['periodic_powerspectrum_lstedr'][int_i] , dct['freqmin'],dct['kolmogorov_constant_power']), label='interval', **st_2i)

                    if plot_errors:
                        ax.plot(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['periodic_powerspectrum_edr++'], dct['freqmin'],dct['kolmogorov_constant_power']), **st_2s)
                        ax.plot(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['periodic_powerspectrum_edr--'], dct['freqmin'],dct['kolmogorov_constant_power']), **st_2s)
                        ax.fill_between(dct['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['periodic_powerspectrum_edr--'], dct['freqmin'],dct['kolmogorov_constant_power']),
                                                                  0.5 * model_edr_via_power_spectrum (dct['freq'][sorting] ,dct['periodic_powerspectrum_edr++'], dct['freqmin'],dct['kolmogorov_constant_power']), **st_2f)

            if dct['domain'] == 'time':
                ax.set_xlabel('$\chi$ [s$^{-1}$]', fontsize=fontsize0)
            else:
                ax.set_xlabel('$\kappa$ [m$^{-1}$]', fontsize=fontsize0)
            ax.set_ylabel('$P_k$ [m$^2$s$^{-2}$]', fontsize=fontsize0)
            
            if plot_legend:
                ax.legend(frameon=False, loc='lower left', ncol=2, fontsize=fontsize1)
            try:
                ax.set_xscale('log')
                ax.set_yscale('log')
            except:
                pass
        
            xmin = 10.**min(np.floor(np.log10(dct['freq'][sorting])))
            xmax = 10.**max(np.ceil (np.log10(dct['freq'][sorting])))
            ax.set_xlim(xmin,xmax)

        if plot == 'pow2':
            #pow              
            #~ if not seperate:
                #~ ax = fig.add_subplot(nrows,1,5)
                #~ ax.set_title('pow')

            if plot_nonperiodic:
                dat = dct['nonperiodic_powerspectrum_lstedr']
                dat = np.where(dat < (1.e-10 * np.max(dat)), np.nan , dat)
                ax.plot(dct['nonperiodic_powerspectrum_lstfreq'], dat
                                            , label='non-periodic', **st_1p) 
            if plot_periodic:
                dat = dct['periodic_powerspectrum_lstedr']
                dat = np.where(dat < (1.e-10 * np.max(dat)), np.nan , dat)
                ax.plot(dct['periodic_powerspectrum_lstfreq'], dat
                                            , label='periodic', **st_2p)
            
            if plot_nonperiodic:
                mylst = np.zeros(len(dct['freq'][sorting])) 
                ax.plot(dct['freq'][sorting]   , mylst + dct['nonperiodic_powerspectrum_edr']
                                                , label='fit', **st_1)
                ax.plot(dct['freq'][sorting]   , mylst + dct['nonperiodic_powerspectrum_edr++']
                                                , **st_1s)
                ax.plot(dct['freq'][sorting]   , mylst + dct['nonperiodic_powerspectrum_edr--']
                                                , **st_1s)
                ax.fill_between(dct['freq'][sorting],  mylst   + dct['nonperiodic_powerspectrum_edr--'],
                                                        mylst   + dct['nonperiodic_powerspectrum_edr++'], **st_1f)

            if plot_periodic:
                mylst = np.zeros(len(dct['freq'][sorting])) 
                ax.plot(dct['freq'][sorting]       , mylst + dct['periodic_powerspectrum_edr']
                                                , label='fit', **st_2)
                ax.plot(dct['freq'][sorting]       , mylst + dct['periodic_powerspectrum_edr++']
                                               , **st_2s)
                ax.plot(dct['freq'][sorting]       , mylst + dct['periodic_powerspectrum_edr--']
                                                , **st_2s)
                ax.fill_between(dct['freq'][sorting],  mylst   + dct['periodic_powerspectrum_edr--'],
                                                        mylst   + dct['periodic_powerspectrum_edr++'], **st_2f)
                                                        
            if dct['domain'] == 'time':
                ax.set_xlabel('$\chi$ [s$^{-1}$]', fontsize=fontsize0)
            else:
                ax.set_xlabel('$\kappa$ [m$^{-1}$]', fontsize=fontsize0)


            ax.set_ylabel(r'$\epsilon$ [m$^2$s$^{-3}$]', fontsize=fontsize0)

            if plot_legend:
                ax.legend(frameon=False, ncol=2, loc='lower left', fontsize=fontsize1)
            try:
                ax.set_xscale('log')
                ax.set_yscale('log')
            except:
                pass
        
            xmin = 10.**min(np.floor(np.log10(dct['freq'][sorting])))
            xmax = 10.**max(np.ceil (np.log10(dct['freq'][sorting])))
            ax.set_xlim(xmin,xmax)
            
        if plot == 'phase':
            #phase
            if not seperate:
                ax = fig.add_subplot(nrows,1,6)                
                ax.set_title('phase')

            if plot_periodic:
                ax.plot(dct['freq'][sorting], dct['periodic_phase'][sorting] % (2. * np.pi),  **st_2)
            else:
                ax.text(0, 0, "only for periodic analysis")

            ax.set_xlabel('$\kappa$ [m$^{-1}$]', fontsize=fontsize0)
            ax.set_ylabel('phase', fontsize=fontsize0)

            ax.set_xscale('log')

            if plot_legend:
                ax.legend(frameon=False, loc='lower left', fontsize=fontsize1)

        if seperate:
            try:
                plt.tight_layout()
                myname = name[:-4]+'_'+plot+name[-4:]
                plt.savefig(myname)
                plt.close(fig)
            except Exception as e:
                print "an error occured on line number ", sys.exc_traceback.tb_lineno
                print str(e)
                print "script will continue!"
                pass
            
    if not seperate:            
        #fig.subplots_adjust(hspace=.8)
        plt.tight_layout()


        plt.savefig(name)
        plt.close(fig)


def printstats(dct):
    print
    print "mu:      {:.4e}".format(dct['mu'])
    print "sigma:   {:.4e}".format(dct['std'])

    print
    print 'results for variance analysis'
    print "variance method  , edr: {:.4e}".format(dct['nonperiodic_variancemethod_edr'])
    print
    print 
    
    print 'results for non-periodic analysis'
    print
    print "power spectrum  , edr: {:.4e}".format(dct['nonperiodic_powerspectrum_edr'])
    print "2nd order struct, edr: {:.4e}".format(dct['nonperiodic_2ndorder_edr'])
    print "3rd order struct, edr: {:.4e}".format(dct['nonperiodic_3rdorder_edr'])
    print 
    print
    
    print 'results for periodic analysis'
    print
    print "power spectrum  , edr: {:.4e}".format(dct['periodic_powerspectrum_edr'])
    print "2nd order struct, edr: {:.4e}".format(dct['periodic_2ndorder_edr'])
    print "3rd order struct, edr: {:.4e}".format(dct['periodic_3rdorder_edr'])
    print 
    print
    

def test():
    test1()
    test2()
    test3()

def test1():
    print 'TEST I'
    print 'start from power spectrum'
    print 'eddy dissipation rate is set to 1.0'
    print
    
    dct = {}
    dct['n'] = 201
    dct['mu'] = 0.
    kolmogorov_constants(dct, 'longitudinal')  
    dct['domain'] = 'space'

    #edr = 1.0
    dct['f_pow']           = lambda x: model_edr_via_power_spectrum(x, 1.0, 2. * np.pi / dct['n'], dct['kolmogorov_constant_power'])

    dct['dx']              = 1.
    dct['tau']             = fftfreq(dct['n']) * 1. * dct['n']
    dct['i']               = np.arange(dct['n'])
    dct['freq']            = 2. * np.pi * fftfreq(dct['n'])
    dct['freqsort']        = np.argsort(dct['freq'])
    dct['freqsort+']       = np.compress(dct['freq'][dct['freqsort']] > 0., dct['freqsort'])

    #power spectrum
    dct['in_pow']          = dct['f_pow'](np.abs(dct['freq'])) / 2.
    dct['in_pow'][0]       = 0.

    #calculate fft coefficients
    dct['y_fft_coef']      = dct['n'] * np.sqrt(np.abs(dct['in_pow']))

    dct['y_fft']           = dct['y_fft_coef'] * np.exp(2.j * np.pi * np.random.random(dct['n']))
    dct['y_fft']           = make_fft_coef_real(dct['y_fft'])
    dct['y']               = np.real(ifft(dct['y_fft']))

    #set sign of mu
    dct['y']               = dct['y'] + dct['mu'] - np.average(dct['y'])

    do_edr_retrievals(dct)
    printstats(dct)
    makeplots(dct, 'edrlib_testI_periodic', plot_periodic = True, plot_nonperiodic = False)
    makeplots(dct, 'edrlib_testI_nonperiodic', plot_periodic = False, plot_nonperiodic = True)

    return dct
    
def test2():
    print 'TEST II'
    print 'start from second order structure function'
    print 'eddy dissipation rate is set to 1.0'
    print 'please note: bias can occur as d2 is periodic in simulation, and d2 is calculated non-periodic in retrieval'
    print
    
    dct = {}
    dct['n'] = 201
    dct['mu'] = 0.
    kolmogorov_constants(dct, 'longitudinal')
    dct['domain'] = 'space'

    #edr = 1.0
    dct['f_d2']           = lambda x: model_edr_via_2nd_order(x, 1.0, dct['kolmogorov_constant_struc2'])

    dct['dx']              = 1.
    dct['tau']             = fftfreq(dct['n']) * 1. * dct['n']
    dct['i']               = np.arange(dct['n'])
    dct['freq']            = 2. * np.pi * fftfreq(dct['n'])
    dct['freqsort']        = np.argsort(dct['freq'])
    dct['freqsort+']       = np.compress(dct['freq'][dct['freqsort']] > 0., dct['freqsort'])

    dct['in_d2']          = dct['f_d2'](np.abs(dct['tau']))

    dct['var']             = np.sum(dct['in_d2']) / (2. * dct['n'])
    dct['f_autocov']       = lambda x: dct['var'] - 0.5* dct['f_d2'](x)

    #autocovariance
    dct['in_autocov']      = dct['f_autocov'](np.abs(dct['tau']))

    #power spectrum
    dct['in_pow']          = np.abs(np.real(fft((dct['in_autocov'] + dct['mu'] ** 2.)/dct['n'])))

    #calculate fft coefficients
    dct['y_fft_coef']      = dct['n'] * np.sqrt(np.abs(dct['in_pow']))

    dct['y_fft']           = dct['y_fft_coef'] * np.exp(2.j * np.pi * np.random.random(dct['n']))
    dct['y_fft']           = make_fft_coef_real(dct['y_fft'])
    dct['y']               = np.real(ifft(dct['y_fft']))

    #set sign of mu
    dct['y']               = dct['y'] + dct['mu'] - np.average(dct['y'])

    do_edr_retrievals(dct)
    printstats(dct)
    makeplots(dct, 'edrlib_testII_periodic', plot_periodic = True, plot_nonperiodic = False)  
    makeplots(dct, 'edrlib_testII_nonperiodic', plot_periodic = False, plot_nonperiodic = True)  

    return dct

def test3():
    print 'TEST III'
    print 'start from power spectrum'
    print 'test the time domain'
    print 'eddy dissipation rate is set to 1.0'
    print
    
    dct = {}
    dct['n'] = 201
    dct['mu'] = 0.
    kolmogorov_constants(dct, 'longitudinal')  
    dct['domain'] = 'time'
    dct['u0']  = 5.

    #edr = 1.0
    dct['f_pow']           = lambda k: model_edr_via_power_spectrum(k, dct['u0'] * 1.0, 2. * np.pi / dct['n'], dct['kolmogorov_constant_power'])

    dct['dt']              = 1.
    dct['tau']             = fftfreq(dct['n']) * 1. * dct['n']
    dct['i']               = np.arange(dct['n'])
    dct['freq']            = 2. * np.pi * fftfreq(dct['n'])
    dct['freqsort']        = np.argsort(dct['freq'])
    dct['freqsort+']       = np.compress(dct['freq'][dct['freqsort']] > 0., dct['freqsort'])

    #power spectrum
    dct['in_pow']          = dct['f_pow'](np.abs(dct['freq'])) / 2.
    dct['in_pow'][0]       = 0.

    #calculate fft coefficients
    dct['y_fft_coef']      = dct['n'] * np.sqrt(np.abs(dct['in_pow']))

    dct['y_fft']           = dct['y_fft_coef'] * np.exp(2.j * np.pi * np.random.random(dct['n']))
    dct['y_fft']           = make_fft_coef_real(dct['y_fft'])
    dct['y']               = np.real(ifft(dct['y_fft']))

    #set sign of mu
    dct['y']               = dct['y'] + dct['mu'] - np.average(dct['y'])

    do_edr_retrievals(dct)
    printstats(dct)
    makeplots(dct, 'edrlib_testIII_periodic', plot_periodic = True, plot_nonperiodic = False)
    makeplots(dct, 'edrlib_testIII_nonperiodic', plot_periodic = False, plot_nonperiodic = True)

    return dct

if __name__ == '__main__':
    test()
