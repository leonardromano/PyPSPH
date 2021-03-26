#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 19:47:22 2020

@author: leonard
"""
from sys import exit

from src.parameters.Constants import NORM_FAC
from src.parameters.Parameters import NDIM, Kernel, DESNNGBS

"""
This file contains the implementation for the choice and evaluation of 
smoothing kernels. The implementation is general enough so that different 
implementations of the kernels can be used by the code simply by specifying
the name of the kernel function by 'Kernel'.
"""
if Kernel == "cubic":     
    def kernel(x, h, derivative = False):
        if derivative:
            if x < 0.5:
                return -6 * NORM_FAC * (2 * x - 3 * x**2) * h**(-(NDIM+1))
            elif x < 1:
                return -6 * NORM_FAC * (1-x)**2 * h**(-(NDIM+1))
            else:
                return 0
        else:
            if x < 0.5:
                return NORM_FAC * (1-6*x**2 + 6*x**3) * h**(-NDIM)
            elif x < 1:
                return 2 * NORM_FAC * (1-x)**3 * h**(-NDIM)
            else:
                return 0
elif Kernel == "Wendland_C2":
    if NDIM == 1:
        def kernel(x, h, derivative = False):
            if derivative:
                if x < 1:
                    return -12 * NORM_FAC * (1-x)**2 * x / h**2
                else:
                    return 0
            else:
                if x < 1:
                    return NORM_FAC * (1-x)**3 * (1 + 3 * x) /h
                else:
                    return 0
    else:
        def kernel(x, h, derivative = False):
            if derivative:
                if x < 1:
                    return -20 * NORM_FAC * (1-x)**3 * x * h**(-(NDIM+1))
                else:
                    return 0
            else:
                if x < 1:
                    return NORM_FAC * (1-x)**4 * (1 + 4 * x) * h**(-NDIM)
                else:
                    return 0 
elif Kernel == "Wendland_C4":
    if NDIM == 1:
        def kernel(x, h, derivative = False):
            if derivative:
                if x < 1:
                    return -14 * NORM_FAC * x * (1-x)**4 * (1 + 4 * x) / h**2
                else:
                    return 0
            else:
                if x < 1:
                    return NORM_FAC * (1-x)**5 * (1 + 5 * x + 8 * x**2) /h
                else:
                    return 0
    else:
        def kernel(x, h, derivative = False):
            if derivative:
                if x < 1:
                    return -56/3 * NORM_FAC * x * (1-x)**5 * (1 + 5 * x) * h**(-(NDIM+1))
                else:
                    return 0
            else:
                if x < 1:
                    return NORM_FAC * (1-x)**6 * (1 + 6 * x + 35/3 * x**2) * h**(-NDIM)
                else:
                    return 0 
else:
    print("Kernel function not defined!")
    exit()

def bias_correction(particle):
    "correct for density bias according to Dehn & Aly (2012)"
    epsilon = 0
    if Kernel == "Wendland_C2":
        if NDIM == 3:
            epsilon = 0.0294 * (DESNNGBS * 0.01)**(-0.977)
        elif NDIM == 2:
            epsilon = 0.0932 * (DESNNGBS/16)**(-1.48)
    elif Kernel == "Wendland_C4":
        if NDIM == 2:
            epsilon = 0.1075 * (DESNNGBS/16)**(-1.78)
    elif Kernel == "cubic":
        if NDIM == 2:
            epsilon = 0.0340 * (DESNNGBS/16)**(-1.91)
    if epsilon != 0:
        particle.Rho -= epsilon * NORM_FAC * particle.Hsml**(-NDIM)
                     
