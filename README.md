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
November 17th, 2018

Version
=======
1.1

Project
=======
EU FP7 program, the UFO project

Acknowledgement and citation
============================
Whenever this Python module is used for publication,
the code writer should be informed, acknowledged and referenced.
If you have any suggestions for improvements or amendments, please inform the author of this class.

Reference:
Oude Nijhuis, A. C. P., Unal, C. M. H., Krasnov, O. A., Russchenberg, H. W. J., & Yarovoy, A. (2018). Velocity-based EDR retrieval techniques applied to Doppler radar measurements from rain: Two case studies (under review). Journal of Atmospheric and Oceanic Technology.

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
- 17th November 2018: tested with Python3.6 and resolved some minor issues.
