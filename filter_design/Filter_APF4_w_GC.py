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

# Relative
from photontorch.components.terms import Detector



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
            phase_input = 0, # excess phase in the ring
            name = None,
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
## 4th-Order APD-based Filter 
###############################################################################
class APF(pt.Network):
    r""" This filter uses APF type topology with n pair of rings: 
        Ports:
            
        """  
        
    def __init__(self, 
            num_rings   = 2, # Number of rings in each arm
            ring_length = 500e-6,
            loss        = 2.5, # in dB/cm
            neff        = 2.34, # effective index
            ng          = 4.24,  # group index
            wl0         = 1.55e-6,  # reference wavelength            
            kappa       = np.array([0.3461, 0.1223]), # Coupling coefficients            
            phi         = np.array([-0.1732, 0.3503]), # Phase shifts in the rings
            kappa_drop  = 1/100, # Drop port kappa, 1% monitor tap
            MZ_length   = 100e-6,
            beta        = -1.5809 # quadrature bias
    ):
            super(APF, self).__init__() # always initialize parent first
            # Handle variables
            self.num_rings = int(num_rings)                        
            # define global source and detectors
            self.source     = pt.Source()
            self.in2        = pt.Detector()
            self.bar        = pt.Detector()            
            self.cross      = pt.Detector()                                    
            
            # Input and output 50% directional couplers            
            self.cp1 = sip.DirectionalCoupler(0.5)
            self.cp2 = sip.DirectionalCoupler(0.5)                        
            # Waveguides in the MZ arm
            self.wg_top = sip.Waveguide(MZ_length, loss, neff, ng, wl0, phase= beta % (2*np.pi) , trainable=True)
            self.wg_bot = sip.Waveguide(MZ_length, loss, neff, ng, wl0, phase= -beta % (2*np.pi) , trainable=True)        
            
            # Define rings and their add and drop port monitor
            for i in range(self.num_rings):                
                self.add_component(
                        name = "RR_top%i"%i,
                        comp = RingResonator4(ring_length, loss, neff, ng, wl0, kappa[i], kappa_drop, phi[i])
                    ) # Top ring
                self.add_component(
                        name = "RR_bot%i"%i,
                        comp = RingResonator4(ring_length, loss, neff, ng, wl0, kappa[i], kappa_drop, -phi[i])
                    ) # Bottom ring              
                self.add_component(
                        name = "term_top%i"%i,
                        comp = pt.Term()                        
                    ) # Top add port terminated                 
                self.add_component(
                        name = "term_bot%i"%i,
                        comp = pt.Term()                        
                    ) # Bottom add port terminated                              
                self.add_component(
                        name = "mon_top%i"%i,
                        comp = pt.Detector()                        
                    ) # Top monitor port                
                self.add_component(
                        name = "mon_bot%i"%i,
                        comp = pt.Detector()                        
                    ) # Bottom monitor port                
            # end for i 
            

            self.gc_in = self.gc_in2 = self.gc_bar = self.gc_cross = pt.GratingCoupler(
                    R       = PDK.GC1.REFLECTION,
                    R_in    = PDK.GC1.IN_REFLECTION,
                    Tmax    = PDK.GC1.PEAK_TRANSMISSION,
                    bandwidth = PDK.GC1.BANDWIDTH,
                    wl0       = PDK.GC1.CENTER_WAVELENGTH,
                ) 
            
            
            # Define the links      
            link1   = ["source:0", '0:gc_in:1', '0:cp1:1', '0:wg_top:1']
            link2   = ["in2:0", '0:gc_in2:1', '3:cp1:2', '0:wg_bot:1']          
            
            # Create the top and bottom arm rings
            for i in range(self.num_rings):                
                link1   += ["0:RR_top%i:1"%i]
                link2   += ["0:RR_bot%i:1"%i]
            # end for i     
            
            link1   +=  ['0:cp2:1', '1:gc_bar:0', '0:cross']
            link2   +=  ['3:cp2:2', '1:gc_cross:0', '0:bar']
            self.link(*link1)
            self.link(*link2)                
            
            # Create the monitors on the top and bottom arm rings
            for i in range(self.num_rings):                
                self.link("mon_top%i:0"%i, "3:RR_top%i:2"%i, "0:term_top%i"%i)
                self.link("mon_bot%i:0"%i, "3:RR_bot%i:2"%i, "0:term_bot%i"%i)
            # end for i     
                                                
print(torch.where(APF().free_ports_at)[0])            

###############################################################################
## Simulation Testbench
###############################################################################
c           = constants.C # speed of light
ng      =   PDK.Si_WG1.NG # group index
neff    =   PDK.Si_WG1.NEFF # effective index
loss    =   PDK.Si_WG1.LOSS # dB/cm

wl0         = 1.55e-6
num_rings   = 2 # Two pair of rings
ring_length = 2200e-6 #[m]
kappa       = np.array([0.3461, 0.1223]) # Coupling coefficients
phi         = np.array([-0.1732, 0.3503]) # Phase shifts in the rings
kappa_drop  = 1/100 # 1% monitor tap
MZ_length   = 100e-6
beta        = 1.58

# Simualation Parameters
GHz = 1e9
size = 10001
fmin = 0 # GHz
fmax = 70 # GHz

# detector list for plotting
det_list = []

# Define the simulation environment:
freq = GHz*np.linspace(fmin, fmax, size) # frequency offset points
fc = (c/wl0)                             # reference frequency
f = fc * np.linspace(1, 1, size) + freq  # actual frequency points

env = pt.Environment(
    #wavelength = 1e-6*np.linspace(1.50, 1.501, 10001), #[m]
    f = f,
    freqdomain=True, # we will be doing frequency domain simulations
)

# set the global simulation environment:
pt.set_environment(env)

circuit1 = APF(num_rings, ring_length, loss, neff, ng, wl0, kappa, phi, kappa_drop, MZ_length, beta)

# get detector outputs
det = circuit1(source=1)
 
# Use this block to plot all detector signals
# Cant do dB plots here
# fig0 = plt.figure() 
# circuit1.plot(det)
# plt.show()

# extract all the detectors from the circuit
det_list = [
        name for name, comp in circuit1.components.items() if isinstance(comp, Detector)
    ]

bar = sip.dB10(det[0,:,det_list.index('bar'),0])
#in2 = dB10(det[0,:,1,0])
cross = sip.dB10(det[0,:,det_list.index('cross'),0])

mon_top0  = sip.dB10(det[0,:,det_list.index('mon_top0'),0])
mon_bot0  = sip.dB10(det[0,:,det_list.index('mon_bot0'),0])
mon_top1  = sip.dB10(det[0,:,det_list.index('mon_top1'),0])
mon_bot1  = sip.dB10(det[0,:,det_list.index('mon_bot1'),0])


fig1 = plt.figure()


plt.plot((f-fc)/GHz, bar, label='bar', color='black')
plt.plot((f-fc)/GHz, cross, label='cross', color='gray')


plt.plot((f-fc)/GHz, mon_top0, label='mon_top0')
plt.plot((f-fc)/GHz, mon_bot0, label='mon_bot0')
plt.plot((f-fc)/GHz, mon_top1, label='mon_top1')
plt.plot((f-fc)/GHz, mon_bot1, label='mon_bot1')

ymin = -80
ymax = 0
plt.grid(color='0.95')
#plt.axis([fmin, fmax, ymin, ymax])
plt.ylabel('Transmission, dB')
plt.xlabel('frequency offset, GHz')
plt.legend(loc='lower right')

plt.legend(loc=1)
plt.savefig('filter_APD4_w_GC_sim.png')
plt.show()

###############################################################################