#!/usr/bin/env python

__readme = \
'''
edr_analysis.py
===============
A Python class to calculate the eddy dissipation rate with different methods, based on a sequence of velocities

Author
======
Albert Oude Nijhuis <albertoudenijhuis@gmail.com>

Institute
=========
Delft University of Technology

Date
====
June 23th, 2015

Version
=======
0.2

Project
=======
EU FP7 program, the UFO project

Acknowledgement and citation
============================
Whenever this Python class is used for publication of scientific results,
the code writer should be informed, acknowledged and referenced.

If you have any suggestions for improvements or amendments, please inform the author of this class.

typical usage
=============
    import edr_analysis
    signal = {}
    signal['domain']    = 'space'
    signal['dx']        = 0.1       #units: m
    signal['y']         = v         #place here the velocity signal, units: m/s

    edr_analysis.do_analysis(signal)                #do analysis
    edr_analysis.kolmogorov_constants(zeta, 'full') #set Kolmogorov constants 
    edr_analysis.do_edr_retrievals(signal)          #do edr retrievals with different methods
    edr_analysis.printstats(signal)                 #print retrieved edr values
    edr_analysis.makeplots(signal)                  #make plots of it all

for the time domain, update two lines to:
    signal['domain']    = 'time'
    signal['dt']        = 0.1       #units: s

Testing
=======
For testing the class can be executed from the command line:
    ./edr_analysis.py
Three test will be run. See the function test() at the bottom of this file for the details.

Revision History
================
    June 23, 2015:
    - Small bugs in plotting have been updated.
'''


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

f_even = lambda x: x % 2 == 0

def autocov(x, periodic=False):
    n = len(x)
    x_auto = np.zeros(n) + np.nan
    if periodic:
        #assume that signal is periodic
        for i in taulist(n):   #0, 1, 2, 3, ... -3, -2, -1
            a = x[:]
            b = np.hstack(( x[i:], x[0:i] ))
            x_auto[i] = np.cov(a,b, bias=1)[0,1]
    else:
        #assume that signal is non-periodic
        for i in taulist(n):   #0, 1, 2, 3, ... -3, -2, -1
            if i > 0:
                a = x[:-i]
                b = x[i:]
            if i <= 0:
                a = x[-i:]
                b = x[0:n+i]
            x_auto[i] = np.cov(a,b, bias=1)[0,1]        
    return x_auto

def power(x, periodic=False):
    if periodic:
        o_fft   = fft(x)
        o_phase = np.angle(o_fft)
        o_pow   = np.abs(o_fft /len(x))**2.
        return o_fft, o_phase, o_pow
    else:
        ac      = autocov(x, False)
        mu      = np.average(x)
        o_acfft = fft((ac + mu**2.) / len(x))
        o_pow   = np.real(o_acfft)
        o_pow   = np.abs(o_pow)
        return o_pow

def struct_function(xin, order=2, periodic=False):
    x = xin
    n = len(x)
    x_struc = np.zeros(n) + np.nan

    if periodic:
        #assume that signal is periodic
        for i in taulist(n):   #0, 1, 2, 3, ... -3, -2, -1
            a = x[:]
            b = np.hstack(( x[i:], x[0:i] ))
            lst1 = (np.sign(a-b)**order) *(np.abs(a-b)**(order))
            x_struc[i] = np.average(lst1)

    else:
        #assume that signal is non-periodic
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

def make_fft_coef_real(fft_coef_in):
    fft_coef = deepcopy(fft_coef_in)
    n = len(fft_coef)
    if f_even(n):
        fft_coef[-1:-n/2:-1] = np.conj(fft_coef[1:n/2]) #make signal real for even number
        fft_coef[n/2] = np.abs(fft_coef[n/2])           #for even number            
    else:
        fft_coef[-1:-n/2:-1] = np.conj(fft_coef[1:n/2+1])   #make signal real for uneven number         
    return fft_coef 

def make_3Dfft_coef_real(fft_coef_in):
    fft_coef = deepcopy(fft_coef_in)
    #swipe trough all axes

    n = len(fft_coef)    
    for k0 in range(n):
        for k1 in range(n):
            for k2 in range(n):
                if (k0 == ((-k0 -1) % n)) & (k1 == ((-k1 -1) % n)) & (k2 == ((-k2-1) % n)):
                    fft_coef[k0, k1, k2] = np.abs(fft_coef[n-k0-1, n-k1-1, n-k2-1])
                else:
                    fft_coef[k0, k1, k2] = np.conj(fft_coef[n-k0-1, n-k1-1, n-k2-1])

    return fft_coef 

def do_analysis(zeta):
    #input: zeta['domain'] = 'space', zeta['dx'], zeta['y']
    #input: zeta['domain'] = 'time',  zeta['dt'], zeta['y']

    zeta['n']                       = len(zeta['y'])
    zeta['i']                       = np.arange(zeta['n'])

    if zeta['domain'] == 'space':
        zeta['x']                   = zeta['i'] * zeta['dx']
        zeta['tau']                 = fftfreq(zeta['n']) * 1. * zeta['n'] * zeta['dx']
        zeta['d']                   = zeta['dx']
        
    if zeta['domain'] == 'time':
        zeta['t']                   = zeta['i'] * zeta['dt']
        zeta['tau']                 = fftfreq(zeta['n']) * 1. * zeta['n'] * zeta['dt']
        zeta['d']                   = zeta['dt']
        
    zeta['freq']                    = 2. * np.pi * fftfreq(zeta['n'], zeta['d'])
    zeta['freqsort']                = np.argsort(zeta['freq'])
    zeta['freqsort+']               = np.compress(zeta['freq'][zeta['freqsort']] > 0., zeta['freqsort'])

    zeta['freqmin']                 = np.min(zeta['freq'][zeta['freqsort+']])
    zeta['freqmax']                 = np.max(zeta['freq'][zeta['freqsort+']])

    #analysis
    zeta['mu']                      = np.average(zeta['y'])
    zeta['std']                     = np.std(zeta['y'])
    zeta['var']                     = np.var(zeta['y'])

    zeta['nonperiodic_dll']         = struct_function(zeta['y'], 2, False)
    zeta['nonperiodic_dlll']        = struct_function(zeta['y'], 3, False)
    zeta['nonperiodic_autocov']     = autocov(zeta['y'], False)
    zeta['nonperiodic_pow']         = power(zeta['y'], False)

    zeta['periodic_dll']            = struct_function(zeta['y'], 2, True)
    zeta['periodic_dlll']           = struct_function(zeta['y'], 3, True)
    zeta['periodic_autocov']        = autocov(zeta['y'], True)
    zeta['periodic_fft'],           \
    zeta['periodic_phase'],         \
    zeta['periodic_pow']            = power(zeta['y'], True)

    return True

def taulist(n):
    if f_even(n):
        lst1 = 1+np.arange(n/2-1)
        lst2 = np.hstack((0,lst1, -n/2,-lst1[::-1]))
    else:
        lst1 = 1+np.arange((n-1)/2)
        lst2 = np.hstack((0,lst1,-lst1[::-1]))
    return lst2


#EDR via variance
#k      : wavenumber
#dk     : smallest wavenumber
#eps    : eddy dissipation rate
#C      : Kolmogorov universal constant
def retr_edr_via_variance(zeta,key_str):
    thisedr     = (((3.0/2.0) * zeta['power_C'] * \
                        (  ((zeta['freqmin'] - 0.5 * zeta['freqmin']) ** (-2.0/3.0))  - ((zeta['freqmax'] + 0.5 * zeta['freqmin']) ** (-2.0/3.0)) ) \
                        ) ** (-3.0/2.0) ) \
                        * (zeta['var'] ** (3.0/2.0))

    if zeta['domain'] == 'time':
        thisedr *= zeta['u0'] ** (-1.0)

    res = {}
    res['edr']      = thisedr
    
    if zeta['domain'] == 'time':
        res['edr.err'] = (1./3.) * \
                            (res['edr'] ** (1. / 3.)) \
                            * np.sqrt(((zeta['freqmin']/zeta['freqmax'])**(4./3.)) + (zeta['var'] / (zeta['u0'] ** 2.)) + (9. / (2. * (zeta['n'] - 1.))))
    else:
        res['edr.err'] = (1./3.) * \
                            (res['edr'] ** (1. / 3.)) \
                            * np.sqrt(((zeta['freqmin']/zeta['freqmax'])**(4./3.)) +  (9. / (2. * (zeta['n'] - 1.))))
    
    return res






#EDR via power spectrum.
#k      : wavenumber
#dk     : smallest wavenumber
#eps    : eddy dissipation rate
#C      : Kolmogorov universal constant
model_edr_via_power_spectrum    = lambda k,eps, dk, C: (3./2.) * C * ( eps ** (2./3.)) * (((k - (0.5 * dk)) ** (-2./3.))    - ((k + (0.5*dk)) ** (-2./3.))  )
model_edr_via_power_spectrum2   = lambda k,eps, dk, C: C * ( eps ** (2./3.)) * k ** (-5./3.)
#model_edr_via_power_spectrum3   = lambda k,eps, dk, C: dk * C * ( eps ** (2./3.)) * k ** (-2./3.)

def retr_edr_via_power_spectrum(zeta,key_str):
    fft_pfreq   = zeta['freq'][zeta['freqsort+']]
    fft_ppow    = zeta[key_str+'_pow'][zeta['freqsort+']]
    
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
        thisedr     = (((3.0/2.0) * zeta['power_C'] * \
                        (  ((thisfreqmin - 0.5 * zeta['freqmin']) ** (-2.0/3.0))  - ((thisfreqmax + 0.5 * zeta['freqmin']) ** (-2.0/3.0)) ) \
                        ) ** (-3.0/2.0) ) \
                        * (thispow ** (3.0/2.0))

        if zeta['domain'] == 'time':
            thisedr *= zeta['u0'] ** (-1.0)
            
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
f_retr_edr_via_2nd_order = lambda s, dll, C: (1./ s) * ((dll / C) ** (3./2.))
def retr_edr_via_2nd_order(zeta,key_str):
    res = {}
    #zeta['dll']
    #zeta['tau']

    if zeta['domain'] == 'time':
        res['lst_edr']  = f_retr_edr_via_2nd_order(zeta['u0'] * zeta['tau'][zeta['freqsort+']],zeta[key_str+'_dll'][zeta['freqsort+']], zeta['struc2_C'])
    else:
        res['lst_edr']  = f_retr_edr_via_2nd_order(zeta['tau'][zeta['freqsort+']],zeta[key_str+'_dll'][zeta['freqsort+']], zeta['struc2_C'])
                
    res['edr']      = np.average(res['lst_edr'][1:]**(1./3.))**3.
    res['edr.err']  = np.std(res['lst_edr'][1:]**(1./3.))
    return res
        
#3rd order structure function
model_edr_via_3rd_order = lambda s,eps, C: C *  eps * s 
f_retr_edr_via_3rd_order = lambda s, dlll, C: (1./C) * (dlll/ s)
def retr_edr_via_3rd_order(zeta,key_str):
    res = {}
    #zeta['dlll']
    #zeta['tau']
    if zeta['domain'] == 'time':
        res['lst_edr']  = f_retr_edr_via_3rd_order(zeta['u0'] * zeta['tau'][zeta['freqsort+']],zeta[key_str+'_dlll'][zeta['freqsort+']], zeta['struc3_C'])
    else:
        res['lst_edr']  = f_retr_edr_via_3rd_order(zeta['tau'][zeta['freqsort+']],zeta[key_str+'_dlll'][zeta['freqsort+']], zeta['struc3_C'])
                
    res['lst_edr1/3'] = np.sign(res['lst_edr']) * (np.abs(res['lst_edr']) ** (1./3.))
    res['edr']      = np.average(res['lst_edr1/3'][1:])**3.
    res['edr.err']  = np.std(res['lst_edr1/3'][1:])
    return res

def do_edr_retrievals(zeta):
    for key_str in ['periodic', 'nonperiodic']:          
        #variance method
        res   = retr_edr_via_variance(zeta, key_str)
        zeta[key_str+'_variancemethod_edr']      = res['edr']
        zeta[key_str+'_variancemethod_edrerr']   = res['edr.err']
        zeta[key_str+'_variancemethod_edr+']     = (res['edr'] ** (1./3.) + res['edr.err'])**3.
        zeta[key_str+'_variancemethod_edr-']     = (res['edr'] ** (1./3.) - res['edr.err'])**3.        
        zeta[key_str+'_variancemethod_edr++']    = (res['edr'] ** (1./3.) + 2. * res['edr.err'])**3.
        zeta[key_str+'_variancemethod_edr--']    = (res['edr'] ** (1./3.) - 2. * res['edr.err'])**3.        


        res = retr_edr_via_power_spectrum(zeta,key_str)
        zeta[key_str+'_powerspectrum_edr']      = res['edr']
        zeta[key_str+'_powerspectrum_edrerr']   = res['edr.err']
        zeta[key_str+'_powerspectrum_edr+']     = (res['edr'] ** (1./3.) + res['edr.err'])**3.
        zeta[key_str+'_powerspectrum_edr-']     = (res['edr'] ** (1./3.) - res['edr.err'])**3.        
        zeta[key_str+'_powerspectrum_edr++']    = (res['edr'] ** (1./3.) + 2. * res['edr.err'])**3.
        zeta[key_str+'_powerspectrum_edr--']    = (res['edr'] ** (1./3.) - 2. * res['edr.err'])**3.        
        zeta[key_str+'_powerspectrum_lstedr']   = res['lst_edr']
        zeta[key_str+'_powerspectrum_lstfreq']  = res['lst_freq']
        zeta[key_str+'_powerspectrum_lstfreqmin']  = res['lst_freqmin']
        zeta[key_str+'_powerspectrum_lstfreqmax']  = res['lst_freqmax']


        res = retr_edr_via_2nd_order(zeta,key_str)
        zeta[key_str+'_2ndorder_edr']           = res['edr']
        zeta[key_str+'_2ndorder_edrerr']        = res['edr.err']
        zeta[key_str+'_2ndorder_edr+']       = (res['edr'] ** (1./3.) + res['edr.err'])**3.
        zeta[key_str+'_2ndorder_edr-']       = (res['edr'] ** (1./3.) - res['edr.err'])**3.
        zeta[key_str+'_2ndorder_edr++']      = (res['edr'] ** (1./3.) + 2. * res['edr.err'])**3.
        zeta[key_str+'_2ndorder_edr--']      = (res['edr'] ** (1./3.) - 2. * res['edr.err'])**3.
        zeta[key_str+'_2ndorder_lstedr']        = res['lst_edr']
        
        #essential problem, retrieved edr can be negative!!
        res = retr_edr_via_3rd_order(zeta,key_str)
        zeta[key_str+'_3rdorder_edr']           = res['edr']
        zeta[key_str+'_3rdorder_edrerr']        = res['edr.err']
        zeta[key_str+'_3rdorder_edr+']          = (np.sign(res['edr']) * (np.abs(res['edr']) ** (1./3.)) + res['edr.err'])**3.
        zeta[key_str+'_3rdorder_edr-']          = (np.sign(res['edr']) * (np.abs(res['edr']) ** (1./3.)) - res['edr.err'])**3.
        zeta[key_str+'_3rdorder_edr++']         = (np.sign(res['edr']) * (np.abs(res['edr']) ** (1./3.)) + 2. * res['edr.err'])**3.
        zeta[key_str+'_3rdorder_edr--']         = (np.sign(res['edr']) * (np.abs(res['edr']) ** (1./3.)) - 2. * res['edr.err'])**3.
        zeta[key_str+'_3rdorder_lstedr']        = res['lst_edr']

        results = {}
        results['nu'] = 1.5e-5 #kinematic viscosity
        zeta[key_str+'_powerspectrum_taylorreynolds'] = (zeta['std'] ** 2. ) * np.sqrt(15. / (results['nu'] * zeta[key_str+'_powerspectrum_edr']))      

    return True


def kolmogorov_constants(zeta, choice):
    C = 1.5

    q           = 2./3.                                                 #~0.66
    C1_div_C2   = (1. / np.pi) * gamma(1.+q) * np.sin(np.pi * q / 2.)   #~0.25
    C2_div_C1   = 1. / C1_div_C2                                        #~4.

    if choice=='longitudinal':        
        zeta['power_C']     = (18./55.) * C                             #0.49
        zeta['struc2_C']    = C2_div_C1 * (18./55.) * C                 
        zeta['struc3_C']    = -4./5.
    elif choice=='transverse':
        zeta['power_C']     = (4./3.) * (18./55.) * C                   #0.65
        zeta['struc2_C']    = (4./3.) * C2_div_C1 *  (18./55.) * C
        zeta['struc3_C']    = -4./5.            #not sure
    elif choice=='full':
        zeta['power_C']     = C
        zeta['struc2_C']    = C2_div_C1 * zeta['power_C']
        zeta['struc3_C']    = -4./5.            #not sure
    return True

def radial_kolmogorov_constants(zeta, azimuthrad, azimuth0rad, elevationrad):
    delta_azimuthrad = azimuthrad - azimuth0rad
    
    kc_trans = {}; kc_longi = {}
    kolmogorov_constants(kc_trans, 'transverse')
    kolmogorov_constants(kc_longi, 'longitudinal')

    zeta['power_C']     =     ((np.cos(elevationrad) * np.cos(delta_azimuthrad))**2.) * kc_longi['power_C'] \
                            + ((np.cos(elevationrad) * np.sin(delta_azimuthrad))**2.) * kc_trans['power_C'] \
                            + (np.sin(elevationrad)**2.) * kc_trans['power_C']
    zeta['struc2_C']    =     ((np.cos(elevationrad) * np.cos(delta_azimuthrad))**2.) * kc_longi['struc2_C'] \
                            + ((np.cos(elevationrad) * np.sin(delta_azimuthrad))**2.) * kc_trans['struc2_C'] \
                            + (np.sin(elevationrad)**2.) * kc_trans['struc2_C']
    zeta['struc3_C']    = -4./5.
    return True

def makeplots(zeta, name='edr_analysis', seperate=False, plot_periodic = False, plot_nonperiodic = True, plot_legend = True, units_in = {}):
    fontsize0 = 16
    fontsize1 = 14
    matplotlib.rc('xtick', labelsize=fontsize0) 
    matplotlib.rc('ytick', labelsize=fontsize0) 

    #sorting
    sorting = zeta['freqsort+']     
    #sorting = signal['freqsort']       

    st_1 = {'color':'green' , 'alpha':0.7, 'linewidth':2}                       #non-periodic
    st_1p= {'color':'black' , 'alpha':0.7, 'marker':'x', 'linestyle':'None'}    #non-periodic
    st_1s= {'color':'green' , 'alpha':0.7, 'linewidth':2, 'linestyle':'--'}     #non-periodic
    st_1f= {'color':'green' , 'alpha':0.1}                                      #non-periodic
    
    st_1i= {'color':'green' , 'alpha':0.7, 'linewidth':2, 'linestyle':'-', 'zorder':10}      #non-periodic

    st_2 = {'color':'red'   , 'alpha':0.7, 'linewidth':2}                       #periodic
    st_2p= {'color':'black'   , 'alpha':0.7, 'marker':'x', 'linestyle':'None'}    #periodic
    st_2s= {'color':'red'   , 'alpha':0.7, 'linewidth':2, 'linestyle':'--'}     #periodic
    st_2f= {'color':'red'   , 'alpha':0.1}                                      #periodic
    st_2i= {'color':'red'   , 'alpha':0.7, 'linewidth':2, 'linestyle':'-', 'zorder':10}      #periodic

    #~ units = {}
    #~ units.update(units_in)
    #~ if not 'eps' in units.keys():
        #~ units['eps'] = 

    if not seperate:      
        if plot_periodic:
            plot_lst = ['signal', 'autocovariance', 'dll', 'dlll', 'pow', 'phase']
        if plot_nonperiodic:
            plot_lst = ['signal', 'autocovariance', 'dll', 'dlll', 'pow']
        nrows = len(plot_lst)
        fig = plt.figure(figsize=(5,5*nrows))

    if seperate:
        plot_lst = ['signal', 'autocovariance', 'dll', 'dlll', 'pow', 'dll2', 'dlll2', 'pow2', 'phase']
    
    for plot in plot_lst:
        if seperate:
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(1,1,1)

        if plot == 'signal':
            #signal
            if not seperate:
                ax = fig.add_subplot(nrows,1,1)
                ax.set_title('signal')
            if zeta['domain'] == 'time':
                ax.plot(zeta['t'], zeta['y'])
                ax.set_xlabel('$t$ [s]', fontsize=fontsize0)
            else:
                ax.plot(zeta['x'], zeta['y'])
                ax.set_xlabel('$x$ [m]', fontsize=fontsize0)
            ax.set_ylabel('$v$ [m/s]', fontsize=fontsize0)
           
        if plot == 'autocovariance':
            #autocovariance
            if not seperate:                
                ax = fig.add_subplot(nrows,1,2)
                ax.set_title('autocovariance')

            if plot_nonperiodic:
                ln1 = ax.plot(zeta['tau'][sorting], zeta['nonperiodic_autocov'][sorting], label='non-periodic',**st_1)
            if plot_periodic:
                ln2 = ax.plot(zeta['tau'][sorting], zeta['periodic_autocov'][sorting], label='periodic',**st_2)  

            if zeta['domain'] == 'time':
                ax.set_xlabel(r'$t$ [s]', fontsize=fontsize0)
            else:
                ax.set_xlabel(r'$s$ [m]', fontsize=fontsize0)
            ax.set_ylabel(r'$R$ [m$^2$s$^{-2}$]', fontsize=fontsize0)
            if plot_legend:
                ax.legend(frameon=False, ncol=2, loc='lower left', fontsize=fontsize1)

        if plot == 'dll':
            #dll
            if not seperate:
                ax = fig.add_subplot(nrows,1,3)
                ax.set_title('$D_{LL}$')

            if plot_nonperiodic:
                ax.plot(zeta['tau'][sorting], zeta['nonperiodic_dll'][sorting], label='non-periodic', **st_1p)
            if plot_periodic:
                ax.plot(zeta['tau'][sorting], zeta['periodic_dll'][sorting], label='periodic', **st_2p)
            
            if zeta['domain'] == 'time':
                if plot_nonperiodic:
                    ax.plot(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting], zeta['u0'] * zeta['nonperiodic_2ndorder_edr'], zeta['struc2_C']), label='fit', **st_1)
                    ax.plot(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting], zeta['u0'] * zeta['nonperiodic_2ndorder_edr++'], zeta['struc2_C']), **st_1s)
                    ax.plot(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting], zeta['u0'] * zeta['nonperiodic_2ndorder_edr--'], zeta['struc2_C']), **st_1s)
                    ax.fill_between(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting], zeta['u0'] * zeta['nonperiodic_2ndorder_edr--'], zeta['struc2_C']),
                        model_edr_via_2nd_order(zeta['tau'][sorting], zeta['u0'] * zeta['nonperiodic_2ndorder_edr++'], zeta['struc2_C']), **st_1f)
                
                if plot_periodic:
                    ax.plot(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting], zeta['u0'] * zeta['periodic_2ndorder_edr'], zeta['struc2_C']), label='fit', **st_2)
                    ax.plot(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting], zeta['u0'] * zeta['periodic_2ndorder_edr++'], zeta['struc2_C']), **st_2s)
                    ax.plot(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting], zeta['u0'] * zeta['periodic_2ndorder_edr--'], zeta['struc2_C']), **st_2s)
                    ax.fill_between(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting], zeta['u0'] * zeta['periodic_2ndorder_edr--'], zeta['struc2_C']),
                        model_edr_via_2nd_order(zeta['tau'][sorting], zeta['u0'] * zeta['periodic_2ndorder_edr++'], zeta['struc2_C']), **st_2f)
            else:
                if plot_nonperiodic:
                    ax.plot(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting] ,zeta['nonperiodic_2ndorder_edr'], zeta['struc2_C']), label='fit', **st_1)
                    ax.plot(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting] ,zeta['nonperiodic_2ndorder_edr++'], zeta['struc2_C']), **st_1s)
                    ax.plot(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting] ,zeta['nonperiodic_2ndorder_edr--'], zeta['struc2_C']), **st_1s)
                    ax.fill_between(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting], zeta['nonperiodic_2ndorder_edr--'], zeta['struc2_C']),
                        model_edr_via_2nd_order(zeta['tau'][sorting], zeta['nonperiodic_2ndorder_edr++'], zeta['struc2_C']), **st_1f)                
                if plot_periodic:
                    ax.plot(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting] ,zeta['periodic_2ndorder_edr'], zeta['struc2_C']), label='fit', **st_2)
                    ax.plot(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting] ,zeta['periodic_2ndorder_edr++'], zeta['struc2_C']), **st_2s)
                    ax.plot(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting] ,zeta['periodic_2ndorder_edr--'], zeta['struc2_C']), **st_2s)
                    ax.fill_between(zeta['tau'][sorting], model_edr_via_2nd_order(zeta['tau'][sorting], zeta['periodic_2ndorder_edr--'], zeta['struc2_C']),
                        model_edr_via_2nd_order(zeta['tau'][sorting], zeta['periodic_2ndorder_edr++'], zeta['struc2_C']), **st_2f)

            if zeta['domain'] == 'time':
                ax.set_xlabel(r'$t$ [s]', fontsize=fontsize0)
                ax.set_ylabel(r'$D_{LL}$ [m$^2$s$^{-2}$]', fontsize=fontsize0)
            else:
                ax.set_xlabel(r'$s$ [m]', fontsize=fontsize0)
                ax.set_ylabel(r'$D_{LL}$ [m$^2$s$^{-2}$]', fontsize=fontsize0)
            if plot_legend:
                ax.legend(frameon=False, loc='upper left', ncol=2, fontsize=fontsize1)
            
        if plot == 'dll2':
            #dll
            if not seperate:
                ax = fig.add_subplot(nrows,1,3)
                ax.set_title('$D_{LL}$')

            if plot_nonperiodic:
                ax.plot(zeta['tau'][zeta['freqsort+']], zeta['nonperiodic_2ndorder_lstedr'], label='non-periodic', **st_1p)            
            if plot_periodic:
                ax.plot(zeta['tau'][zeta['freqsort+']], zeta['periodic_2ndorder_lstedr'],label='periodic', **st_2p)

            if plot_nonperiodic:
                mylst = np.zeros(len(zeta['nonperiodic_3rdorder_lstedr'])) 
                ax.plot(zeta['tau'][zeta['freqsort+']]  , mylst + zeta['nonperiodic_2ndorder_edr'], label='fit', **st_1)
                ax.plot(zeta['tau'][zeta['freqsort+']]  , mylst + zeta['nonperiodic_2ndorder_edr++'], **st_1s)
                ax.plot(zeta['tau'][zeta['freqsort+']]  , mylst + zeta['nonperiodic_2ndorder_edr--'], **st_1s)
                ax.fill_between(zeta['tau'][zeta['freqsort+']]  , mylst + zeta['nonperiodic_2ndorder_edr++'],
                                mylst + zeta['nonperiodic_2ndorder_edr--'], **st_1f)
            
            if plot_periodic:
                mylst = np.zeros(len(zeta['periodic_3rdorder_lstedr'])) 
                ax.plot(zeta['tau'][zeta['freqsort+']]  , mylst + zeta['periodic_2ndorder_edr'], label='fit', **st_2)
                ax.plot(zeta['tau'][zeta['freqsort+']]  , mylst + zeta['periodic_2ndorder_edr++'], **st_2s)
                ax.plot(zeta['tau'][zeta['freqsort+']]  , mylst + zeta['periodic_2ndorder_edr--'], **st_2s)

                ax.fill_between(zeta['tau'][zeta['freqsort+']]  , mylst + zeta['periodic_2ndorder_edr++'],
                                mylst + zeta['periodic_2ndorder_edr--'], **st_2f)

            if zeta['domain'] == 'time':
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


        if plot == 'dlll':
            #dlll              
            if not seperate:
                ax = fig.add_subplot(nrows,1,4)
                ax.set_title('$D_{LLL}$')

            if plot_nonperiodic:
                ax.plot(zeta['tau'][sorting], zeta['nonperiodic_dlll'][sorting], label='non-periodic', **st_1p)
            if plot_periodic:
                ax.plot(zeta['tau'][sorting], zeta['periodic_dlll'][sorting], label='periodic', **st_2p)
            
            if zeta['domain'] == 'time':
                if plot_nonperiodic:
                    ax.plot(zeta['tau'][sorting], model_edr_via_3rd_order(zeta['tau'][sorting], zeta['u0'] * zeta['nonperiodic_3rdorder_edr'], zeta['struc3_C']), label='fit', **st_1)
                    ax.plot(zeta['tau'][sorting], model_edr_via_3rd_order(zeta['tau'][sorting], zeta['u0'] * zeta['nonperiodic_3rdorder_edr++'], zeta['struc3_C']), **st_1s)
                    ax.plot(zeta['tau'][sorting], model_edr_via_3rd_order(zeta['tau'][sorting], zeta['u0'] * zeta['nonperiodic_3rdorder_edr--'], zeta['struc3_C']), **st_1s)
                    ax.fill_between(zeta['tau'][sorting],   model_edr_via_3rd_order(zeta['tau'][sorting], zeta['u0'] * zeta['nonperiodic_3rdorder_edr--'], zeta['struc3_C']),
                                                            model_edr_via_3rd_order(zeta['tau'][sorting], zeta['u0'] * zeta['nonperiodic_3rdorder_edr++'], zeta['struc3_C']), **st_1f)
                
                if plot_periodic:
                    ax.plot(zeta['tau'][sorting], model_edr_via_3rd_order(zeta['tau'][sorting], zeta['u0'] * zeta['periodic_3rdorder_edr'], zeta['struc3_C']), label='fit', **st_2)
                    ax.plot(zeta['tau'][sorting], model_edr_via_3rd_order(zeta['tau'][sorting], zeta['u0'] * zeta['periodic_3rdorder_edr++'], zeta['struc3_C']), **st_2s)
                    ax.plot(zeta['tau'][sorting], model_edr_via_3rd_order(zeta['tau'][sorting], zeta['u0'] * zeta['periodic_3rdorder_edr--'], zeta['struc3_C']), **st_2s)
                    ax.fill_between(zeta['tau'][sorting],   model_edr_via_3rd_order(zeta['tau'][sorting], zeta['u0'] * zeta['periodic_3rdorder_edr--'], zeta['struc3_C']),
                                                            model_edr_via_3rd_order(zeta['tau'][sorting], zeta['u0'] * zeta['periodic_3rdorder_edr++'], zeta['struc3_C']), **st_2f)

            else:
                if plot_nonperiodic:
                    ax.plot(zeta['tau'][sorting], model_edr_via_3rd_order(zeta['tau'][sorting] ,zeta['nonperiodic_3rdorder_edr'], zeta['struc3_C']), label='fit', **st_1)
                    ax.plot(zeta['tau'][sorting], model_edr_via_3rd_order(zeta['tau'][sorting] ,zeta['nonperiodic_3rdorder_edr++'], zeta['struc3_C']), **st_1s)
                    ax.plot(zeta['tau'][sorting], model_edr_via_3rd_order(zeta['tau'][sorting] ,zeta['nonperiodic_3rdorder_edr--'], zeta['struc3_C']), **st_1s)
                    ax.fill_between(zeta['tau'][sorting],   model_edr_via_3rd_order(zeta['tau'][sorting], zeta['nonperiodic_3rdorder_edr--'], zeta['struc3_C']),
                                                            model_edr_via_3rd_order(zeta['tau'][sorting], zeta['nonperiodic_3rdorder_edr++'], zeta['struc3_C']), **st_1f)
                
                if plot_periodic:
                    ax.plot(zeta['tau'][sorting], model_edr_via_3rd_order(zeta['tau'][sorting] ,zeta['periodic_3rdorder_edr'], zeta['struc3_C']), label='fit', **st_2)
                    ax.plot(zeta['tau'][sorting], model_edr_via_3rd_order(zeta['tau'][sorting] ,zeta['periodic_3rdorder_edr++'], zeta['struc3_C']), **st_2s)
                    ax.plot(zeta['tau'][sorting], model_edr_via_3rd_order(zeta['tau'][sorting] ,zeta['periodic_3rdorder_edr--'], zeta['struc3_C']), **st_2s)
                    ax.fill_between(zeta['tau'][sorting],   model_edr_via_3rd_order(zeta['tau'][sorting], zeta['periodic_3rdorder_edr--'], zeta['struc3_C']),
                                                            model_edr_via_3rd_order(zeta['tau'][sorting], zeta['periodic_3rdorder_edr++'], zeta['struc3_C']), **st_2f)

            if zeta['domain'] == 'time':
                ax.set_xlabel(r'$t$ [s]', fontsize=fontsize0)
            else:
                ax.set_xlabel(r'$s$ [m]', fontsize=fontsize0)
            ax.set_ylabel(r'$D_{LLL}$ [m$^3$s$^{-3}$]', fontsize=fontsize0)
            if plot_legend:
                ax.legend(frameon=False, ncol=2, loc='upper left', fontsize=fontsize1)
            
        if plot == 'dlll2':
            #dlll              
            if not seperate:
                ax = fig.add_subplot(nrows,1,4)
                ax.set_title('$D_{LLL}$')

            if plot_nonperiodic:
                ax.plot(zeta['tau'][zeta['freqsort+']], np.abs(zeta['nonperiodic_3rdorder_lstedr'])
                                        , label='non-periodic', **st_1p)
            if plot_periodic:
                ax.plot(zeta['tau'][zeta['freqsort+']], np.abs(zeta['periodic_3rdorder_lstedr'])
                                            , label='periodic', **st_2p)

            if plot_nonperiodic:
                mylst = np.zeros(len(zeta['nonperiodic_3rdorder_lstedr'])) 
                ax.plot(zeta['tau'][zeta['freqsort+']]  , mylst + np.abs(zeta['nonperiodic_3rdorder_edr']), label='fit', **st_1)
                ax.plot(zeta['tau'][zeta['freqsort+']]  , mylst + np.abs(zeta['nonperiodic_3rdorder_edr++']), **st_1s)
                ax.plot(zeta['tau'][zeta['freqsort+']]  , mylst + np.abs(zeta['nonperiodic_3rdorder_edr--']), **st_1s)
                ax.fill_between(zeta['tau'][zeta['freqsort+']], mylst   + np.abs(zeta['nonperiodic_3rdorder_edr--']),
                                                                mylst   + np.abs(zeta['nonperiodic_3rdorder_edr++']), **st_1f)


            if plot_periodic:
                mylst = np.zeros(len(zeta['periodic_3rdorder_lstedr'])) 
                ax.plot(zeta['tau'][zeta['freqsort+']]      , mylst + np.abs(zeta['periodic_3rdorder_edr']), label='fit', **st_2)
                ax.plot(zeta['tau'][zeta['freqsort+']]      , mylst + np.abs(zeta['periodic_3rdorder_edr++']), **st_2s)
                ax.plot(zeta['tau'][zeta['freqsort+']]      , mylst + np.abs(zeta['periodic_3rdorder_edr--']), **st_2s)
                ax.fill_between(zeta['tau'][zeta['freqsort+']], mylst   + np.abs(zeta['periodic_3rdorder_edr--']),
                                                                mylst   + np.abs(zeta['periodic_3rdorder_edr++']), **st_2f)

            if zeta['domain'] == 'time':
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
                dat = deepcopy(zeta['nonperiodic_pow'][sorting])
                dat = np.where(dat < (1.e-10 * np.max(dat)), np.nan , dat)
                ax.plot(zeta['freq'][sorting], dat, label='non-periodic', **st_1p) 
            if plot_periodic:
                dat = deepcopy(zeta['periodic_pow'][sorting])
                dat = np.where(dat < (1.e-10 * np.max(dat)), np.nan , dat)
                ax.plot(zeta['freq'][sorting], dat, label='periodic', **st_2p)
                                        
            if zeta['domain'] == 'time':
                if plot_nonperiodic:
                    #ax.plot(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['u0'] * zeta['nonperiodic_powerspectrum_edr'], zeta['freqmin'],zeta['power_C']), label='fit', **st_1)
                    ax.plot(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['u0'] * zeta['nonperiodic_powerspectrum_edr++'], zeta['freqmin'],zeta['power_C']), **st_1s)
                    ax.plot(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['u0'] * zeta['nonperiodic_powerspectrum_edr--'], zeta['freqmin'],zeta['power_C']), **st_1s)
                    ax.fill_between(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['u0'] * zeta['nonperiodic_powerspectrum_edr--'], zeta['freqmin'],zeta['power_C']),
                                                              0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['u0'] * zeta['nonperiodic_powerspectrum_edr++'], zeta['freqmin'],zeta['power_C']), **st_1f)
                    for int_i in range(len(zeta['nonperiodic_powerspectrum_lstedr'])):
                        tmpx = np.array( [zeta['nonperiodic_powerspectrum_lstfreqmin'][int_i], zeta['nonperiodic_powerspectrum_lstfreqmax'][int_i]] )
                        ax.plot(tmpx   , 0.5 * model_edr_via_power_spectrum (tmpx ,zeta['u0'] * zeta['nonperiodic_powerspectrum_lstedr'][int_i] , zeta['freqmin'],zeta['power_C']), label='interval', **st_1i)                
                if plot_periodic:                                
                    #ax.plot(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['u0'] * zeta['periodic_powerspectrum_edr'], zeta['freqmin'],zeta['power_C']), label='fit', **st_2)
                    ax.plot(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['u0'] * zeta['periodic_powerspectrum_edr++'], zeta['freqmin'],zeta['power_C']), **st_2s)
                    ax.plot(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['u0'] * zeta['periodic_powerspectrum_edr--'], zeta['freqmin'],zeta['power_C']), **st_2s)
                    ax.fill_between(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['u0'] * zeta['periodic_powerspectrum_edr--'], zeta['freqmin'],zeta['power_C']),
                                                              0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['u0'] * zeta['periodic_powerspectrum_edr++'], zeta['freqmin'],zeta['power_C']), **st_2f)
                    for int_i in range(len(zeta['periodic_powerspectrum_lstedr'])):
                        tmpx = np.array( [zeta['periodic_powerspectrum_lstfreqmin'][int_i], zeta['periodic_powerspectrum_lstfreqmax'][int_i]] )
                        ax.plot(tmpx   , 0.5 * model_edr_via_power_spectrum (tmpx ,zeta['u0'] * zeta['periodic_powerspectrum_lstedr'][int_i] , zeta['freqmin'],zeta['power_C']), label='interval', **st_2i)

            else:
                if plot_nonperiodic:
                    #ax.plot(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['nonperiodic_powerspectrum_edr'], zeta['freqmin'],zeta['power_C']), label='fit', **st_1)
                    ax.plot(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['nonperiodic_powerspectrum_edr++'], zeta['freqmin'],zeta['power_C']), **st_1s)
                    ax.plot(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['nonperiodic_powerspectrum_edr--'], zeta['freqmin'],zeta['power_C']), **st_1s)
                    ax.fill_between(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['nonperiodic_powerspectrum_edr--'], zeta['freqmin'],zeta['power_C']),
                                                              0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['nonperiodic_powerspectrum_edr++'], zeta['freqmin'],zeta['power_C']), **st_1f)
                    for int_i in range(len(zeta['nonperiodic_powerspectrum_lstedr'])):
                        tmpx = np.array( [zeta['nonperiodic_powerspectrum_lstfreqmin'][int_i], zeta['nonperiodic_powerspectrum_lstfreqmax'][int_i]] )
                        ax.plot(tmpx   , 0.5 * model_edr_via_power_spectrum (tmpx ,zeta['nonperiodic_powerspectrum_lstedr'][int_i] , zeta['freqmin'],zeta['power_C']), label='interval', **st_1i)

                if plot_periodic:
                    #ax.plot(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['periodic_powerspectrum_edr'], zeta['freqmin'],zeta['power_C']), label='fit', **st_2)
                    ax.plot(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['periodic_powerspectrum_edr++'], zeta['freqmin'],zeta['power_C']), **st_2s)
                    ax.plot(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['periodic_powerspectrum_edr--'], zeta['freqmin'],zeta['power_C']), **st_2s)
                    ax.fill_between(zeta['freq'][sorting]   , 0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['periodic_powerspectrum_edr--'], zeta['freqmin'],zeta['power_C']),
                                                              0.5 * model_edr_via_power_spectrum (zeta['freq'][sorting] ,zeta['periodic_powerspectrum_edr++'], zeta['freqmin'],zeta['power_C']), **st_2f)
                    for int_i in range(len(zeta['periodic_powerspectrum_lstedr'])):
                        tmpx = np.array( [zeta['periodic_powerspectrum_lstfreqmin'][int_i], zeta['periodic_powerspectrum_lstfreqmax'][int_i]] )
                        ax.plot(tmpx   , 0.5 * model_edr_via_power_spectrum (tmpx , zeta['periodic_powerspectrum_lstedr'][int_i] , zeta['freqmin'],zeta['power_C']), label='interval', **st_2i)

            if zeta['domain'] == 'time':
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
        
            xmin = 10.**min(np.floor(np.log10(zeta['freq'][sorting])))
            xmax = 10.**max(np.ceil (np.log10(zeta['freq'][sorting])))
            ax.set_xlim(xmin,xmax)

        if plot == 'pow2':
            #pow              
            #~ if not seperate:
                #~ ax = fig.add_subplot(nrows,1,5)
                #~ ax.set_title('pow')

            if plot_nonperiodic:
                dat = zeta['nonperiodic_powerspectrum_lstedr']
                dat = np.where(dat < (1.e-10 * np.max(dat)), np.nan , dat)
                ax.plot(zeta['nonperiodic_powerspectrum_lstfreq'], dat
                                            , label='non-periodic', **st_1p) 
            if plot_periodic:
                dat = zeta['periodic_powerspectrum_lstedr']
                dat = np.where(dat < (1.e-10 * np.max(dat)), np.nan , dat)
                ax.plot(zeta['periodic_powerspectrum_lstfreq'], dat
                                            , label='periodic', **st_2p)
            
            if plot_nonperiodic:
                mylst = np.zeros(len(zeta['freq'][sorting])) 
                ax.plot(zeta['freq'][sorting]   , mylst + zeta['nonperiodic_powerspectrum_edr']
                                                , label='fit', **st_1)
                ax.plot(zeta['freq'][sorting]   , mylst + zeta['nonperiodic_powerspectrum_edr++']
                                                , **st_1s)
                ax.plot(zeta['freq'][sorting]   , mylst + zeta['nonperiodic_powerspectrum_edr--']
                                                , **st_1s)
                ax.fill_between(zeta['freq'][sorting],  mylst   + zeta['nonperiodic_powerspectrum_edr--'],
                                                        mylst   + zeta['nonperiodic_powerspectrum_edr++'], **st_1f)

            if plot_periodic:
                mylst = np.zeros(len(zeta['freq'][sorting])) 
                ax.plot(zeta['freq'][sorting]       , mylst + zeta['periodic_powerspectrum_edr']
                                                , label='fit', **st_2)
                ax.plot(zeta['freq'][sorting]       , mylst + zeta['periodic_powerspectrum_edr++']
                                               , **st_2s)
                ax.plot(zeta['freq'][sorting]       , mylst + zeta['periodic_powerspectrum_edr--']
                                                , **st_2s)
                ax.fill_between(zeta['freq'][sorting],  mylst   + zeta['periodic_powerspectrum_edr--'],
                                                        mylst   + zeta['periodic_powerspectrum_edr++'], **st_2f)
                                                        
            if zeta['domain'] == 'time':
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
        
            xmin = 10.**min(np.floor(np.log10(zeta['freq'][sorting])))
            xmax = 10.**max(np.ceil (np.log10(zeta['freq'][sorting])))
            ax.set_xlim(xmin,xmax)
            
        if plot == 'phase':
            #phase
            if not seperate:
                ax = fig.add_subplot(nrows,1,6)                
                ax.set_title('phase')

            if plot_periodic:
                ax.plot(zeta['freq'][sorting], zeta['periodic_phase'][sorting] % (2. * np.pi),  **st_2)
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


def printstats(zeta):
    print
    print "mu:      {:.4e}".format(zeta['mu'])
    print "sigma:   {:.4e}".format(zeta['std'])

    print
    print 'results for variance analysis'
    print "variance method  , edr: {:.4e}".format(zeta['nonperiodic_variancemethod_edr'])
    print
    print 
    
    print 'results for non-periodic analysis'
    print
    print "power spectrum  , edr: {:.4e}".format(zeta['nonperiodic_powerspectrum_edr'])
    print "2nd order struct, edr: {:.4e}".format(zeta['nonperiodic_2ndorder_edr'])
    print "3rd order struct, edr: {:.4e}".format(zeta['nonperiodic_3rdorder_edr'])
    print 
    print
    
    print 'results for periodic analysis'
    print
    print "power spectrum  , edr: {:.4e}".format(zeta['periodic_powerspectrum_edr'])
    print "2nd order struct, edr: {:.4e}".format(zeta['periodic_2ndorder_edr'])
    print "3rd order struct, edr: {:.4e}".format(zeta['periodic_3rdorder_edr'])
    print 
    print
    
    #estimate for taylor reynolds number
    print
    #print "taylor reynolds number: {:.4e}".format(zeta['powerspectrum_taylorreynolds'])


def test():
    test1()
    test2()
    test3()

def test1():
    print 'TEST I'
    print 'start from power spectrum'
    print 'eddy dissipation rate is set to 1.0'

    zeta = {}
    zeta['n'] = 201
    zeta['mu'] = 0.
    kolmogorov_constants(zeta, 'longitudinal')  
    zeta['domain'] = 'space'

    #edr = 1.0
    zeta['f_pow']           = lambda x: model_edr_via_power_spectrum(x, 1.0, 2. * np.pi / zeta['n'], zeta['power_C'])

    zeta['dx']              = 1.
    zeta['tau']             = fftfreq(zeta['n']) * 1. * zeta['n']
    zeta['i']               = np.arange(zeta['n'])
    zeta['freq']            = 2. * np.pi * fftfreq(zeta['n'])
    zeta['freqsort']        = np.argsort(zeta['freq'])
    zeta['freqsort+']       = np.compress(zeta['freq'][zeta['freqsort']] > 0., zeta['freqsort'])

    #power spectrum
    zeta['in_pow']          = zeta['f_pow'](np.abs(zeta['freq'])) / 2.
    zeta['in_pow'][0]       = 0.

    #calculate fft coefficients
    zeta['y_fft_coef']      = zeta['n'] * np.sqrt(np.abs(zeta['in_pow']))

    zeta['y_fft']           = zeta['y_fft_coef'] * np.exp(2.j * np.pi * np.random.random(zeta['n']))
    zeta['y_fft']           = make_fft_coef_real(zeta['y_fft'])
    zeta['y']               = np.real(ifft(zeta['y_fft']))

    #set sign of mu
    zeta['y']               = zeta['y'] + zeta['mu'] - np.average(zeta['y'])

    do_analysis(zeta)
    do_edr_retrievals(zeta)
    printstats(zeta)
    makeplots(zeta, 'edr_analysis_testI_periodic', plot_periodic = True, plot_nonperiodic = False)
    makeplots(zeta, 'edr_analysis_testI_nonperiodic', plot_periodic = False, plot_nonperiodic = True)

    return zeta
    
def test2():
    print 'TEST II'
    print 'start from second order structure function'
    print 'eddy dissipation rate is set to 1.0'
    print 'please note: bias can occur as dll is periodic in simulation, and dll is calculated non-periodic in retrieval'

    zeta = {}
    zeta['n'] = 201
    zeta['mu'] = 0.
    kolmogorov_constants(zeta, 'longitudinal')
    zeta['domain'] = 'space'

    #edr = 1.0
    zeta['f_dll']           = lambda x: model_edr_via_2nd_order(x, 1.0, zeta['struc2_C'])

    zeta['dx']              = 1.
    zeta['tau']             = fftfreq(zeta['n']) * 1. * zeta['n']
    zeta['i']               = np.arange(zeta['n'])
    zeta['freq']            = 2. * np.pi * fftfreq(zeta['n'])
    zeta['freqsort']        = np.argsort(zeta['freq'])
    zeta['freqsort+']       = np.compress(zeta['freq'][zeta['freqsort']] > 0., zeta['freqsort'])

    zeta['in_dll']          = zeta['f_dll'](np.abs(zeta['tau']))

    zeta['var']             = np.sum(zeta['in_dll']) / (2. * zeta['n'])
    zeta['f_autocov']       = lambda x: zeta['var'] - 0.5* zeta['f_dll'](x)

    #autocovariance
    zeta['in_autocov']      = zeta['f_autocov'](np.abs(zeta['tau']))

    #power spectrum
    zeta['in_pow']          = np.abs(np.real(fft((zeta['in_autocov'] + zeta['mu'] ** 2.)/zeta['n'])))

    #calculate fft coefficients
    zeta['y_fft_coef']      = zeta['n'] * np.sqrt(np.abs(zeta['in_pow']))

    zeta['y_fft']           = zeta['y_fft_coef'] * np.exp(2.j * np.pi * np.random.random(zeta['n']))
    zeta['y_fft']           = make_fft_coef_real(zeta['y_fft'])
    zeta['y']               = np.real(ifft(zeta['y_fft']))

    #set sign of mu
    zeta['y']               = zeta['y'] + zeta['mu'] - np.average(zeta['y'])

    do_analysis(zeta)
    do_edr_retrievals(zeta)
    printstats(zeta)
    makeplots(zeta, 'edr_analysis_testII_periodic', plot_periodic = True, plot_nonperiodic = False)  
    makeplots(zeta, 'edr_analysis_testII_nonperiodic', plot_periodic = False, plot_nonperiodic = True)  

    return zeta

def test3():
    print 'TEST III'
    print 'start from power spectrum'
    print 'test the time domain'
    print 'eddy dissipation rate is set to 1.0'

    zeta = {}
    zeta['n'] = 201
    zeta['mu'] = 0.
    kolmogorov_constants(zeta, 'longitudinal')  
    zeta['domain'] = 'time'
    zeta['u0']  = 5.

    #edr = 1.0
    zeta['f_pow']           = lambda k: model_edr_via_power_spectrum(k, zeta['u0'] * 1.0, 2. * np.pi / zeta['n'], zeta['power_C'])

    zeta['dt']              = 1.
    zeta['tau']             = fftfreq(zeta['n']) * 1. * zeta['n']
    zeta['i']               = np.arange(zeta['n'])
    zeta['freq']            = 2. * np.pi * fftfreq(zeta['n'])
    zeta['freqsort']        = np.argsort(zeta['freq'])
    zeta['freqsort+']       = np.compress(zeta['freq'][zeta['freqsort']] > 0., zeta['freqsort'])

    #power spectrum
    zeta['in_pow']          = zeta['f_pow'](np.abs(zeta['freq'])) / 2.
    zeta['in_pow'][0]       = 0.

    #calculate fft coefficients
    zeta['y_fft_coef']      = zeta['n'] * np.sqrt(np.abs(zeta['in_pow']))

    zeta['y_fft']           = zeta['y_fft_coef'] * np.exp(2.j * np.pi * np.random.random(zeta['n']))
    zeta['y_fft']           = make_fft_coef_real(zeta['y_fft'])
    zeta['y']               = np.real(ifft(zeta['y_fft']))

    #set sign of mu
    zeta['y']               = zeta['y'] + zeta['mu'] - np.average(zeta['y'])

    do_analysis(zeta)
    do_edr_retrievals(zeta)
    printstats(zeta)
    makeplots(zeta, 'edr_analysis_testIII_periodic', plot_periodic = True, plot_nonperiodic = False)
    makeplots(zeta, 'edr_analysis_testIII_nonperiodic', plot_periodic = False, plot_nonperiodic = True)

    return zeta

print __readme
if __name__ == '__main__':
    test()
