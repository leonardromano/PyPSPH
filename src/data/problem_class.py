#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 17:08:32 2021

@author: leonard
"""
from numpy import ones

from src.parameters.Constants import BITS_FOR_POSITIONS
from src.parameters.Parameters import NDIM

class problem():
    def __init__(self):
        self.Mpart         = 0.
        self.Boxsize       = ones(NDIM)
        self.FacIntToCoord = self.Boxsize/(1 << BITS_FOR_POSITIONS)
        self.Periodic      = ones(NDIM, dtype = int)
        self.Timer         = {"INIT" : 0,
                              "TREE": 0,
                              "DENSITY": 0,
                              "TIMEBINS": 0,
                              "FORCE": 0,
                              "DRIFTS": 0,
                              "OUTPUT": 0}
        
    def update_int_conversion(self):
        self.FacIntToCoord = self.Boxsize/(1 << BITS_FOR_POSITIONS)