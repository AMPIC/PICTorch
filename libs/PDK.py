#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 01:28:03 2022

@author: vsaxena
"""

# Generic SiP PDK

# Constants
C = 3e8
WL0 = 1550e-9

###############################################################################
# SOI Waveguide 
# Version 1 - standard waveguide
# height = 220nm
# width  = 500nm
###############################################################################

class Si_WG1:
    NG     = 4.24      # Group index
    NEFF   = 2.34      # Effective index
    LOSS   = 2         # dB/cm


###############################################################################
# Grating Coupler 
# Version 1 
###############################################################################

class GC1:
    CENTER_WAVELENGTH   = 1550e-9 # m
    REFLECTION          = 0.2**0.5
    IN_REFLECTION       = 0
    PEAK_TRANSMISSION   = 0.60**0.5
    BANDWIDTH           = 0.6e-6  # m
    
