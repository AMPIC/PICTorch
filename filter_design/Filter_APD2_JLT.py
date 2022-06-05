#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 21:17:26 2021

@author: vsaxena
"""
#%matplotlib inline

import numpy as np
import sys

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm # Progress bar
import torch 
import photontorch as pt 

# setting path
sys.path.append('../libs/')

# Import SiP and PDK libraries
import sip_library as sip
import constants, PDK 


###############################################################################
## Ring Resonator with Four Ports
###############################################################################
class RingResonator4(pt.Network):
    r""" This dual bus RingResonator has 4 ports. They are numbered 0 to 3 as shown in the
        docstring below: 
        Ports:
     (drop) 3_________________2 (add)
                ---------
                |        |
                |        |
                |        |
                |________|
       (in) 0-----------------1 (thru)   
        """    
    def __init__(self, 
            ring_length = 500e-6,
            loss = 3, # in dB/cm
            neff = 2.34, # effective index
            ng = 4.24,  # group index
            wl0 = 1.55e-6,  # reference wavelength
            kappa_thru = 0.1, # through port coupling coefficient
            kappa_drop = 0.1, # drop/monitor power coupling coefficient
            phase_input = 0 # excess phase in the ring
    ):
        super(RingResonator4, self).__init__() # always initialize parent first
        # define waveguides and directional couplers:        
        self.wg1 = sip.Waveguide(ring_length/2, loss, neff, ng, wl0, phase_input, trainable=True)
        self.wg2 = sip.Waveguide(ring_length/2, loss, neff, ng, wl0, phase=0, trainable=False)
        self.cp1 = sip.DirectionalCoupler(kappa_thru)
        self.cp2 = sip.DirectionalCoupler(kappa_drop)
        self.link('cp1:2', '0:wg2:1', '3:cp2:2', '0:wg1:1', '3:cp1')
    
# see if the network is terminated
#print(torch.where(RingResonator4(ring_length, loss, neff, ng, wl0, kappa_thru, kappa_drop, phase_input).free_ports_at)[0])


###############################################################################
## 2nd Order APF-type Filter 
###############################################################################
class APF2(pt.Network):
    r""" This filter has 6 ports and 4 phase settings and 2 coupler settings. They are numbered 0 to 3 as shown in the
        docstring below: 
        Ports:
            
        """  
        
    def __init__(self, 
            ring_length = 500e-6,
            loss = 3, # in dB/cm
            neff = 2.34, # effective index
            ng = 4.24,  # group index
            wl0 = 1.55e-6,  # reference wavelength
            kappa1 = 0.0308, # through port coupling coefficient
            kappa1_drop = 0.05, # drop/monitor power coupling coefficient
            MZ_length = 100e-6,
            phi1 = 0.0158, # excess phase in the ring
            beta = -1.5809 # quadrature bias
    ):
            super(APF2, self).__init__() # always initialize parent first
            # define waveguides, directional couplers and ring resonators
            self.cp1 = sip.DirectionalCoupler(0.5)
            self.cp2 = sip.DirectionalCoupler(0.5)
            self.wg_top = sip.Waveguide(MZ_length, loss, neff, ng, wl0, phase= beta % (2*np.pi) , trainable=True)
            self.wg_bot = sip.Waveguide(MZ_length, loss, neff, ng, wl0, phase= -beta % (2*np.pi) , trainable=True)
            self.RR_top = RingResonator4(ring_length, loss, neff, ng, wl0, kappa1, kappa1_drop, phi1)    
            self.RR_bot = RingResonator4(ring_length, loss, neff, ng, wl0, kappa1, kappa1_drop, -phi1)    
            
            # define source and detectors:
            self.source = pt.Source()
            self.in2 =  pt.Detector()
            self.bar =   pt.Detector()
            self.cross =  pt.Detector()
            # Monitors
            self.mon_top =   pt.Detector()
            self.mon_bot =  pt.Detector()
    
            # Terminate the unused ring add ports
            self.add_top = pt.Detector()
            self.add_bot = pt.Detector()
            
            # Make the connections
            self.link('source:0', '0:cp1:1', '0:wg_top:1', '0:RR_top:1', '0:cp2:1', '0:bar')
            self.link('in2:0','3:cp1:2', '0:wg_bot:1', '0:RR_bot:1', '3:cp2:2', '0:cross')
            self.link('mon_top:0', "3:RR_top:2", "0:add_top")    
            self.link('mon_bot:0', "3:RR_bot:2", "0:add_bot")    
            
            
# print(torch.where(APF2().free_ports_at)[0])            

###############################################################################
## Simulation Testbench
###############################################################################

# Define constants
c       =   constants.C # speed of light
ng      =   PDK.WG1_NG # group index
neff    =   PDK.WG1_NEFF # effective index
loss    =   PDK.WG1_LOSS # dB/cm
wl0     =   1550e-9


# Define ring dimensions
ring_length = 2200e-6 #[m]
MZ_length = 100e-6
kappa1_drop = 1/100 # 1% monitor tap

# Butterworth filter coefficients from APD synthesis
kappa1= 0.4385 # through port coupling coefficient
phi1 = 0.2969
beta = -1.6137

# Chebyshev filter coefficients from APD synthesis
# kappa1= 0.0791 # through port coupling coefficient
# phi1 = 0.0416
# beta = -1.5817

# define the simulation environment:  
GHz = 1e9
size = 10001
fmin = 0 # GHz
fmax = 70 # GHz
freq = GHz*np.linspace(fmin, fmax, size) #[m]
fc = (c/wl0) 
f0 = fc * np.linspace(1, 1, size)
f = f0 + freq   

env = pt.Environment(
    #wavelength = 1e-6*np.linspace(1.50, 1.501, 10001), #[m]
    f = f,
    freqdomain=True, # we will be doing frequency domain simulations
)

# Set the global simulation environment:
pt.set_environment(env)

# Instantiate the circuit
circuit1 = APF2(ring_length, loss, neff, ng, wl0, kappa1, kappa1_drop, MZ_length, phi1, beta)

# Run simulation with a source
det = circuit1(source=1)  # Detects power

# Plot all detector outputs
#fig1 = plt.figure()
#circuit1.plot(det)
#plt.show()

# Use db10 as all outputs are powers
bar     = sip.dB10(det[0,:,2,0])
#in2    = sip.dB10(det[0,:,1,0])
cross  = sip.dB10(det[0,:,0,0])
mon_top = sip.dB10(det[0,:,3,0])
mon_bot = sip.dB10(det[0,:,5,0])

fig1 = plt.figure()
plt.plot((f-fc)/GHz, bar, label='Bar', color='black')
plt.plot((f-fc)/GHz, cross, label='Cross', color='gray')
#plt.plot((f-fc)/GHz, in2)

plt.plot((f-fc)/GHz, mon_top, label='UR Drop')
plt.plot((f-fc)/GHz, mon_bot, label='LR Drop')
plt.grid(color='0.95')


ymin = -40
ymax = 0
plt.axis([fmin, fmax, ymin, ymax])
plt.ylabel('Transmission, dB')
plt.xlabel('frequency offset, GHz')

plt.legend(loc=1)
plt.savefig('filter_sim.png')
plt.show()



###############################################################################