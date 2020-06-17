""" PhotonTorch: a photonic simulation framework based on the deep learning framework PyTorch.

PhotonTorch features CUDA enabled optimization of photonic circuits. It leverages the
deep learning framework PyTorch to view the photonic circuit as essentially a recurrent
neural network. This enables the use of native PyTorch optimizers to optimize the
(physical) parameters of your circuit.

"""

__version__ = "0.0.4"

# Test pytorch version
import torch

torch_version = tuple(int(v) for v in torch.__version__.split("."))
if torch_version[0] < 1 or (torch_version[0] < 2 and torch_version[1] < 3):
    raise ImportError(
        "Photontorch requires PyTorch>=1.3.0. Your version: %s" % torch.__version__
    )

import warnings

warnings.filterwarnings(
    "ignore",
    message="Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable.",
)

## Submodules
from . import components
from . import environment
from . import networks
from . import torch_ext


## Components

# Component
from .components.component import Component

# Terms
from .components.terms import Term
from .components.terms import Source
from .components.terms import Detector

# Mirrors
from .components.mirrors import Mirror

# SOAs
from .components.soas import Soa
from .components.soas import BaseSoa
from .components.soas import LinearSoa
from .components.soas import AgrawalSoa

# MMIs
from .components.mmis import Mmi

# MZIs
from .components.mzis import Mzi

# Waveguides
from .components.waveguides import Waveguide
from .components.waveguides import Connection

# Grating Couplers
from .components.gratingcouplers import GratingCoupler

# Directional couplers
from .components.directionalcouplers import DirectionalCoupler
from .components.directionalcouplers import RealisticDirectionalCoupler
from .components.directionalcouplers import DirectionalCouplerWithLength


## Networks

# Base Network
from .networks.network import Network
from .networks.network import current_network
from .networks.network import link

# Ring Networks
from .networks.rings import RingNetwork

# Reck Network
from .networks.reck import ReckNxN

# Clements Network
from .networks.clements import ClementsNxN


## Environment

from .environment.environment import Environment
from .environment.environment import set_environment
from .environment.environment import current_environment


## Detectors

from .detectors.photodetector import Photodetector
from .detectors.lowpassdetector import LowpassDetector


## PyTorch extensions

# autograd
from .torch_ext.autograd import block_diag

# neural networks
from .torch_ext import nn
from .torch_ext.nn import Buffer
from .torch_ext.nn import BoundedParameter
from .torch_ext.nn import Module

# tensor
from .torch_ext.tensor import where

# custom functional additions
from .torch_ext.functional import BERLoss
from .torch_ext.functional import MSELoss
from .torch_ext.functional import BitStreamGenerator
