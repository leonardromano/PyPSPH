#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 16:25:48 2021

@author: leonard
"""
from src.parameters.Constants import NCPU, MIN_LOAD_PER_CORE

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