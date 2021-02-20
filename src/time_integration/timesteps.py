#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:22:49 2020

@author: leonard
"""
from math import ceil
from numpy import log2
import ray
from sys import exit
from time import time

from src.data.parallel_utility import get_optimal_load
from src.parameters.Constants import Dt
from src.parameters.Parameters import NTimebins

###############################################################################
#parallel worker job

@ray.remote(num_cpus=1)
def process(Particles, activeTimeBin):
    "this function does the heavy lifting for the parallelization"
    lowestPopulatedBin = activeTimeBin
    for particle in Particles:
        k = assign_timebin(particle, activeTimeBin)
        lowestPopulatedBin = max(lowestPopulatedBin, k)
    return Particles, lowestPopulatedBin

###############################################################################
#per particle processing

def assign_timebin(particle, activeTimeBin):
    "assign the particle to its timebin"
    if particle.timestepCriterion < Dt/(1 << activeTimeBin): 
        #need to reduce timestep
        #determine the new timebin
        k = ceil(log2(Dt/particle.timestepCriterion))
        particle.timeBin = k
        if k > NTimebins:
            print("Too small timestep detected! Increase Ntimesteps!")
            ray.shutdown()
            exit()
    return particle.timeBin

###############################################################################
    
def assign_timestep_classes(Particles, activeTimeBin, Problem):
    """
    For all active particles check the timestep criterion and populate 
    bins with shorter timesteps if necessary
    """
    t0 = time()
    lowestPopulatedBin = activeTimeBin
    
    active = list()
    done = list()
    for particle in Particles:
        if particle.timeBin >= activeTimeBin:
            active.append(particle)
        else:
            done.append(particle)
            
    load, ncpu = get_optimal_load(len(active))
    
    if ncpu > 1:
        #split work evenly among processes
        result = [process.remote(active[i * load:(i+1) * load], activeTimeBin) \
                  for i in range(ncpu-1)]
        result.append(process.remote(active[(ncpu-1) * load:], activeTimeBin))
        while len(result):
            done_id, result = ray.wait(result)
            particles, timebin = ray.get(done_id[0])
            done += particles
            lowestPopulatedBin = max(lowestPopulatedBin, timebin)
    else:
        #do the remaining work locally
        for particle in active:
            k = assign_timebin(particle, activeTimeBin)
            lowestPopulatedBin = max(lowestPopulatedBin, k)
        done += active
    
    t1 = time()
    Problem.Timer["TIMEBINS"] += t1 - t0
    return done, lowestPopulatedBin
        
def get_active_time_bin(localTimeStep):
    "determine the current active time bin"
    for level in range(NTimebins):
        if (localTimeStep*2**level)%2 == 1:
            return level