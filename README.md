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
