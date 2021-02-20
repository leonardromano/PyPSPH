#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:06:22 2020

@author: leonard
"""
import ray
from time import time

from src.data.int_conversion import convert_to_int_position
from src.data.parallel_utility import get_optimal_load
from src.Parameters.Constants import Dt, BITS_FOR_POSITIONS, MAX_INT, NCPU
from src.parameters.Parameters import NDIM

###############################################################################
#parallel worker job

@ray.remote(num_cpus=1)
def process_full(Particles, activeTimeBin, dt, Problem):
    "this function does the heavy lifting for the parallelization"
    for particle in Particles:
        kick_and_drift(particle, activeTimeBin, dt, Problem)
    return Particles

@ray.remote(num_cpus=1)
def process_half(Particles):
    "this function does the heavy lifting for the parallelization"
    for particle in Particles:
        kick(particle)
    return Particles

###############################################################################
#per particle processing

def kick_and_drift(particle, activeTimeBin, dt, Problem):
    "compute the SPH acceleration for a single particle"
    #if particle is active: kick
    if particle.timeBin >= activeTimeBin:
        kick(particle)
    
    #now drift
    particle.position = particle.position \
        + convert_to_int_position(particle.velocity * dt, Problem.FacIntToCoord)
    keep_inside_box(particle, Problem.Periodic)
    
def kick(particle):
    tb = particle.timeBin
    v = particle.velocity
    e = particle.entropy
    #update velocity
    particle.velocity_ahead = v + particle.acceleration * Dt/(1 << tb)
    particle.velocity = v + particle.acceleration*Dt/(1 << (1+tb))
    #update entropy
    particle.entropy_ahead = e + particle.entropyChange * Dt/(1 << tb)
    particle.entropy = e + particle.entropyChange * Dt/(1 << (1+tb))

###############################################################################

def Kick_and_Drift(Particles, Problem, activeTimeBin, dt):
    "drift all active velocities and entropies all positions"
    load = len(Particles)//NCPU
    t0 = time()
    #split work evenly among processes
    result = [process_full.remote(Particles[i * load:(i+1) * load], activeTimeBin, \
                             dt, Problem) for i in range(NCPU-1)]
    result.append(process_full.remote(Particles[(NCPU-1) * load:], activeTimeBin, \
                                 dt, Problem))
    
    particles_new = list()
    while len(result):
        done_id, result = ray.wait(result)
        particles_new += ray.get(done_id[0])
    
    t1 = time()
    Problem.Timer["DRIFTS"] += t1- t0
    return particles_new

def Kick(Particles, Problem, activeTimeBin):
    "kick all active velocities and entropies"
    active = list()
    done = list()
    for particle in Particles:
        if particle.timeBin >= activeTimeBin:
            active.append(Particles)
        else:
            done.append(Particles)
    
    load, ncpu = get_optimal_load(len(active))
    t0 = time()
    
    if ncpu > 1:
        #split work evenly among processes
        result = [process_half.remote(active[i * load:(i+1) * load]) \
                  for i in range(ncpu-1)]
        result.append(process_half.remote(active[(ncpu-1) * load:]))
        while len(result):
            done_id, result = ray.wait(result)
            done += ray.get(done_id[0])
    else:
        #do the remaining work locally
        for particle in active:
            kick(particle)
        done += active
    t1 = time()
    Problem.Timer["DRIFTS"] += t1- t0
    return done

def keep_inside_box(particle, Periodic):
    "Makes sure the particle is within the domain"
    for axis in range(NDIM):
        if Periodic[axis]:
            while particle.position[axis] < 0:
                particle.position[axis] += (1 << BITS_FOR_POSITIONS)
            while particle.position[axis] > MAX_INT:
                particle.position[axis] -= (1 << BITS_FOR_POSITIONS)
        else:
            while particle.position[axis] < 0 or \
                particle.position[axis] > MAX_INT:
                    if particle.position[axis] < 0:
                        if particle.position[axis] < -(1 << (BITS_FOR_POSITIONS-1)):
                            particle.position += (1 << BITS_FOR_POSITIONS)
                        else:
                            particle.position[axis] *= -1
                    else:
                        if particle.position[axis] > MAX_INT + (1 << (BITS_FOR_POSITIONS-1)):
                            particle.position[axis] -= (1 << BITS_FOR_POSITIONS)
                        else:
                            particle.position[axis] += 2 * (MAX_INT - \
                                                        particle.position[axis])