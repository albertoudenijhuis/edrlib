#!/usr/bin/env python

import numpy as np
import edrlib

def worked_example_1():
    print("Worked example 1")
    print("")

    #obtain raw data from ASCII file
    rawdata = np.array([line.split( ) for line in open("example1_radardata_ascii.txt")])
    rawdata = rawdata[20:368]
    rawdata = np.array([[float(item) for item in lst] for lst in rawdata], dtype='float')
    
    #make dictrionary of the raw data
    data = {
        'slant_range': rawdata[:,0],
        'power': rawdata[:,1],
        'rad_vel': rawdata[:,2],
        }
    
    #parameters necessary for the Kolmogorov constants
    azimuthrad = 3.130000114440918
    azimuth0rad = azimuthrad + 45. #Best guess for azimuth direction of the wind, when the wind direction is unknown
    elevationrad = 0.11299999803304672

    #for the analysis a slice is used from the data.
    #50 data points, spanning ~5 km in space.
    
    j0 = 0
    j1 = 50
    myslice = slice(j0,j1)

    dct = {}
    dct['n']    = len(data['rad_vel'][myslice])
    dct['dx']   = (data['slant_range'][myslice][-1] - data['slant_range'][myslice][0]) / (dct['n'] - 1.)
    dct['y']    = data['rad_vel'][myslice]
    dct['domain'] = 'space'
    edrlib.kolmogorov_constants_los(dct, azimuthrad, azimuth0rad, elevationrad)

    edrlib.do_edr_retrievals(dct)
    edrlib.printstats(dct)
    edrlib.makeplots(dct, 'example1_radardata', plot_periodic = False, plot_nonperiodic = True)

    
    
if __name__ == '__main__':
    worked_example_1()
