#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:15:30 2020

@author: leonard
"""
from numpy import sqrt, zeros

from src.data.utility import norm
from src.parameters.Constants import Dt
from src.parameters.Parameters import AdiabaticIndex, CourantParameter, \
    TimestepLimiter, NDIM

class particle:
    "A class to store all the properties of a single particle"
    def __init__(self, position, velocity, entropy, index):
        #dynamic quantities
        self.position = position
        self.velocity = velocity
        self.velocity_ahead = zeros(NDIM, dtype = float)
        self.acceleration = zeros(NDIM, dtype = float)
        self.entropy = entropy
        self.entropy_ahead = 0
        self.entropyChange = 0
        #constants
        self.index = index
        #SPH quantities
        self.Hsml = 0
        self.VarHsmlFac = 0
        self.Rho = 0
        self.pressure = 0
        self.csound = 0
        #access variables
        self.timestepCriterion = Dt
        self.neighbors = list()
        self.timeBin = 0
        self.CloseToWall = 0
        
    def update_pressure(self, ahead):
        if ahead:
            self.pressure = self.entropy_ahead * self.Rho**(AdiabaticIndex)
        else:
            self.pressure = self.entropy * self.Rho**(AdiabaticIndex)            
        
    def update_soundspeed(self):
        self.csound = sqrt(AdiabaticIndex * self.pressure/self.Rho)
    
    def update_timestep_criterion(self, NgbTree):
        cmax = 0
        for no in self.neighbors:
            ngb = NgbTree.Tp[no]
            if ngb.csound > cmax:
                cmax = ngb.csound
        courantTimestep = CourantParameter * 2 * self.Hsml/(self.csound + cmax)
        if norm(self.acceleration) > 0:
            kinematicTimestep = sqrt(2 * TimestepLimiter * self.Hsml / \
                                     norm(self.acceleration))
        else:
            kinematicTimestep = courantTimestep
        self.timestepCriterion = min(kinematicTimestep, courantTimestep)
    