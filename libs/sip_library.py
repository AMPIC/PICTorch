#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main SiP library file that contains the optical circuit primitives
such as Waveguide, Couplers, Ring Resonator and a 2x2 MZI Switch


The code is copyright of Vishal Saxena, 2022 and permission and license is 
required to reuse this code.

Created on Fri Jul 30 21:17:26 2021

@author: vsaxena
"""
import numpy as np
import torch 
import photontorch as pt 

# Relative
from photontorch.components.terms import Detector
from photontorch.components  import Component
from photontorch.nn.nn import Parameter, Buffer

##############################################################################
## WAVEGUIDE CLASS
##############################################################################
class Waveguide(pt.Component):
    """ Waveguide with configurable phase shift 
    Each waveguides has two ports. They are numbered 0 and 1:
    Ports:
        0 ---- 1
    """
    # photontorch requires you to explicitly define the number of
    # ports in the component as a class variable:
    num_ports = 2

    def __init__(
        self,
        length=1e-5,
        loss=0, # in dB/cm
        neff=2.34, # effective index of the waveguide
        ng=3.40, # group index of the waveguide
        wl0=1.55e-6, # center wavelength for which the waveguide is defined
        phase=0, # additional phase PARAMETER added to the waveguide
        trainable=True, # a flag to make the the component trainable or not
        name=None, # name of the waveguide
    ):
        """ creation of a new waveguide """
        super(Waveguide, self).__init__(name=name)# always initialize parent first
        # Handle inputs
        self.length = float(length)
        self.loss = float(loss)
        self.neff = float(neff)
        self.wl0 = float(wl0)
        self.ng = float(ng)

        # handle phase input
        phase = float(phase) % (2*np.pi)
        if not trainable: # if the network is not trainable, just store it as a normal float:
            self.phase = phase
        else: # else, make an optimizable parameter out of it:
            # create a torch tensor from the phase
            phase = torch.tensor(phase, dtype=torch.float64)
            # store the phase as a optimizable parameter
            self.phase = torch.nn.Parameter(data=phase)

    def set_delays(self, delays):
        """ set the delays for time-domain simulations """
        delays[:] = self.ng * self.length / self.env.c

    def set_S(self, S):
        """ set the S-matrix

        NOTE: because PyTorch does not support complex tensors, the real
        ane imaginary part of the S-matrix are stored in an extra dimension

        NOTE2: the S-matrix needs to be defined for all wavelengths, therefore
        one needs an extra dimension to store each different S-matrix for each
        wavelength

        ----------------

        Taking the above two notes into account, the S-matrix is thus a
        4-D tensor with shape

        (2=(real|imag), #wavelengths, #ports, #ports)

        """
        # during a photontorch simulation, the simulation environment
        # containing all the global simulation parameters will be
        # available to you as `self.env`:
        current_simulation_environment = self.env

        # you can use this environment to get information about the
        # wavelengths used in the simulation:
        wavelength = current_simulation_environment.wavelength

        # however, this wavelength is stored as a numpy array, while
        # photontorch expect torch tensors. We need to make a torch
        # tensor out of this:
        wavelength = torch.tensor(
            wavelength, # make this numpy array into a torch tensor
            dtype=torch.float64, # keep float64 dtype
            device=self.device, # put it on the current device ('cpu' or 'gpu')
        )

        # next we implement the dispersion, which will depend on the
        # wavelength tensor
        neff = self.neff - (wavelength - self.wl0) * (self.ng - self.neff) / self.wl0

        # we have now calculated an neff for each different wavelength.
        # let's calculate the phase depending on this neff:
        phase = (2 * np.pi * neff * self.length / wavelength) % (2 * np.pi)

        # next, we add the phase correction parameter.
        phase = phase + self.phase
        # note that in pytorch, inplace operations, such as
        # phase += self.phase
        # are not allowed, as they obscure the computation graph necessary to
        # perform the backpropagation algorithm later on...

        # because pytorch does not allow complex numbers, we split up exp(1j*phase) into
        # its real and imaginary part and revert back to the default dtype (usually float32).
        cos_phase = torch.cos(phase).to(torch.get_default_dtype())
        sin_phase = torch.sin(phase).to(torch.get_default_dtype())

        # finally, we can calculate the loss and add it to the phase, which
        # gives us the S-matrix parameters
        loss_dB =  (self.length * 100) * self.loss # Convert loss/dB into loss/m
        loss = 10 ** (- loss_dB/ 20)  # 20 because loss works on power
        #                                                     
        re = loss * cos_phase
        ie = loss * sin_phase

        # the last thing to do is to add the S-matrix parameters to the S-matrix:
        S[0, :, 0, 1] = S[0, :, 1, 0] = re
        S[1, :, 0, 1] = S[1, :, 1, 0] = ie
        
        
##############################################################################
## DIRECTIONAL COUPLER CLASS
##############################################################################
class DirectionalCoupler(pt.Component):
    r""" A directional coupler is a component with 4 ports that introduces no delays

    Each directional coupler has four ports. They are numbered 0 to 3:

    Ports:
       3        2
        \______/
        /------\
       0        1

    """

    # photontorch requires you to explicitly define the number of
    # ports in the component as a class variable:
    num_ports = 4

    def __init__(self, coupling=0.5, name=None):
        """ creation of a new waveguide """
        super(DirectionalCoupler, self).__init__(name=name)# always initialize parent first

        # to save the coupling as an optimizable parameter, we could just do the
        # same as we did for the waveguide: create a torch tensor and store it as a parameter:
        # coupling = torch.tensor(float(coupling))
        # self.phase = torch.nn.Parameter(data=coupling)

        # however, this could lead to problems, as this parameter would be unbounded
        # and we know for a fact the coupling should be bounded between 0 and 1.
        # an easy solution is to define the coupling as the cosine of a hidden parameter
        # which we call (with little imagination) `parameter`:

        # create a parameter. The coupling will be derived from the parameter as cos(self.parameter):
        parameter = torch.tensor(np.arccos(float(coupling)), dtype=torch.get_default_dtype())
        self.parameter = torch.nn.Parameter(data=parameter)

    @property
    def coupling(self):
        return torch.cos(self.parameter)

    def set_S(self, S):
        """ Fill the S-matrix with elements. Rememeber that the S-matrix has a shape

        (2=(real|imag), #wavelengths, #ports, #ports)

        """

        t = (1 - self.coupling) ** 0.5 # Take square root for field 
        k = self.coupling ** 0.5

        # real part scattering matrix (transmission):
        S[0, :, 0, 1] = S[0, :, 1, 0] = t # same for all wavelengths
        S[0, :, 2, 3] = S[0, :, 3, 2] = t # same for all wavelengths

        # imag part scattering matrix (coupling):
        S[1, :, 0, 2] = S[1, :, 2, 0] = k # same for all wavelengths
        S[1, :, 1, 3] = S[1, :, 3, 1] = k # same for all wavelengths
        
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
   
################################################
## Basic Transmission Unit : A 2X2 MZI Switch ##
################################################


class BTU(Component):
    r""" A BTU is a component with 4 ports.
    An BTU has two tunable/trainable parameters: the phase shifts in the upper and lower arms (phiU and phiL).
    Terms::
                    ___phiU____
        3________  /           \  ___2
                 \/             \/
        0________/\____phiL_____/\___1
    Note:
        This 2x2 switch implementation assumes the arms are idential and considers the propagation delay due to the finite length
    """

    num_ports = 4

    def __init__(
        self,
        phiU=0,
        phiL=0,
        neff=2.34,
        ng=3.40,
        wl0=1.55e-6,
        length=500e-6,
        loss=0,
        trainable=True,
        name=None,
    ):
        """
        Args:
            phi (float): input phase
            theta (float): phase difference between the arms
            neff (float): effective index of the waveguide
            ng (float): group index of the waveguide
            wl0 (float): the center wavelength for which neff is defined.
            length (float): length of the waveguide in meter.
            loss (float): loss in the waveguide [dB/m]
            trainable (bool): whether phi and theta are trainable
            name (optional, str): name of this specific MZI
        """
        super(BTU, self).__init__(name=name)

        parameter = Parameter if trainable else Buffer

        self.ng = float(ng)
        self.neff = float(neff)
        self.length = float(length)
        self.loss = float(loss)
        self.wl0 = float(wl0)
        self.phiU = parameter(torch.tensor(phiU, dtype=torch.float64, device=self.device))
        self.phiL = parameter(torch.tensor(phiL, dtype=torch.float64, device=self.device))
        self.phi00 = 0
        
    # def set_port_order(self, port_order):
    #     port_order = torch.tensor([1, 3, 2, 0])
     
    def set_delays(self, delays):
        delays[:] = self.ng * self.length / self.env.c

    def set_S(self, S):
        wls = torch.tensor(self.env.wl, dtype=torch.float64, device=self.device)
        
        ## Common-mode and differential hases
        # Always use the self.variable for variables, otherwise the model will
        # not update with network.initialize()
        phiA = (self.phiU + self.phiL)/2
        phiD = (self.phiU - self.phiL)/2
        
        # neff depends on the wavelength:
        neff = self.neff - (wls - self.wl0) * (self.ng - self.neff) / self.wl0
        self.phi00 = (2 * np.pi * self.neff * self.length / self.wl0) % (2 * np.pi) # Global Phi0
        
        # Wavelength dependent phi0
        phi0 = (2 * np.pi * neff * self.length / wls) % (2 * np.pi)
        phiC = phi0 + phiA  # total common-mode phi
        # cos / sin of phases
        cos_phiC = torch.cos(phiC).to(torch.get_default_dtype())
        sin_phiC = torch.sin(phiC).to(torch.get_default_dtype())
        cos_phiD = torch.cos(phiD).to(torch.get_default_dtype())
        sin_phiD = torch.sin(phiD).to(torch.get_default_dtype())
        # scattering matrix
        S[0, :, 0, 1] = S[0, :, 1, 0] = -sin_phiC * sin_phiD
        S[1, :, 0, 1] = S[1, :, 1, 0] = cos_phiC * sin_phiD
        S[0, :, 0, 2] = S[0, :, 2, 0] = -sin_phiC * cos_phiD
        S[1, :, 0, 2] = S[1, :, 2, 0] = cos_phiC * cos_phiD
        S[0, :, 3, 1] = S[0, :, 1, 3] = -sin_phiC * cos_phiD
        S[1, :, 3, 1] = S[1, :, 1, 3] = cos_phiC * cos_phiD
        S[0, :, 3, 2] = S[0, :, 2, 3] = sin_phiC * sin_phiD
        S[1, :, 3, 2] = S[1, :, 2, 3] = -cos_phiC * sin_phiD
        # return scattering matrix

        # add loss
        #loss = self.loss * self.length * 100 # loss defined per unit length
        loss = self.loss # Absolute loss in dB for the whole BTU
        return S * 10 ** (-loss / 20)  # 20 bc loss is defined on power.
###############################################################################
     
###############################################################################
## dB10 function
###############################################################################

def dB10(input,zero_nan=True):
    '''
    converts linear magnitude to db
 
     db is given by
            20*log10(|z|)
    where z is a complex number
    '''
    out = 10 * np.log10(input)
    # if zero_nan:
    #     try:
    #         out[np.isnan(out)] = LOG_OF_NEG
    #     except (TypeError):
    #         # input is a number not array-like
    #         if npy.isnan(out):
    #             return LOG_OF_NEG
    return out
###############################################################################        