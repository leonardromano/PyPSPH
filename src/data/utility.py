#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:21:28 2020

@author: leonard
"""
from numpy import sqrt

from src.parameters.Constants import NCPU, MIN_LOAD_PER_CORE
from src.parameters.Parameters import NDIM

def factorial(n):
    result = 1
    for i in range(2, n+1):
        result *= i
    return result

def dot(v1, v2):
    r = 0
    for i in range(NDIM):
        r += v1[i] * v2[i]
    return r

def norm(vector):
    r = 0
    for component in vector:
        r += component * component
    return sqrt(r)

def get_optimal_load(total_load):
    "Divide the load in as many reasonably big chunks as possible"
    if total_load <= MIN_LOAD_PER_CORE:
        return total_load, 1
    
    ncpu = NCPU
    load = total_load//ncpu
    while load < MIN_LOAD_PER_CORE and ncpu > 1:
        ncpu -= 1
        load = total_load//ncpu
    
    return load, ncpu