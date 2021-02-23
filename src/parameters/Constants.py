#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPH simulation code by Leonard Romano
"""
from numpy import pi
from os import environ
from psutil import cpu_count
from sys import exit

from src.parameters.Parameters import NDIM, Scheduler, InitialTime, FinalTime, \
    NTimesteps, Kernel
from src.data.utility import factorial


"""
This file stores some constant global parameters.
Specify the parameters of your simulation in this file 
before running the simulation with 'run'.
"""

#Parallelization constants
if Scheduler == "SGE":
    NCPU = int(environ['NSLOTS'])
elif Scheduler == "SLURM":
    NCPU = int(environ['SLURM_CPUS_PER_TASK'])
elif Scheduler == "PBS":
    NCPU = int(environ['PBS_NP'])
else:
    NCPU = cpu_count(logical=False)
MIN_LOAD_PER_CORE = 20

#small and big numbers
SMALL_NUM = 1e-15
LARGE_NUM = 1e15

#Integer coordinate utility
BITS_FOR_POSITIONS = 32
MAX_INT = (1 << BITS_FOR_POSITIONS) - 1

#tree utility
TREE_NUM_BEFORE_NODESPLIT = 3

#Time parameters
Dt = (FinalTime - InitialTime)/NTimesteps #Maximum Timestep

#######################################
#Geometrical constants and normalization constants
#Volume of DIM-dimensional unit ball
if NDIM % 2 == 0:
    NORM_COEFF =pi**(NDIM//2)/factorial(NDIM//2)
else:
    NORM_COEFF = 2 * factorial((NDIM - 1)//2) * (4 *pi)**((NDIM-1)//2) / factorial(NDIM)
    
#Kernel normalisation constant
if Kernel == "cubic":
    NORM_FAC = factorial(NDIM + 3, NDIM + 1) / \
        (12 * NORM_COEFF * (1 - 2**(-(NDIM + 1))))
elif Kernel == "Wendland_C2":
    if NDIM == 1:
        NORM_FAC = 5/4
    else:
        NORM_FAC = factorial(NDIM + 5, NDIM + 2) / (120 * NORM_COEFF)
elif Kernel == "Wendland_C4":
    if NDIM == 1:
        NORM_FAC = 3/2
    else:
        NORM_FAC = 3 * (NDIM + 2) * factorial(NDIM + 8, NDIM + 4) / \
            (NORM_COEFF * 40320)
else:
    print("Kernel function not defined!")
    exit()