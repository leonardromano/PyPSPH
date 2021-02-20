#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:28:41 2020

@author: leonard
"""
from numpy import zeros
import ray
from time import time

from src.data.int_conversion import get_distance_vector
from src.data.parallel_utility import get_optimal_load
from src.data.utility import dot, norm
from src.parameters.Parameters import NDIM, Viscosity, AdiabaticIndex, \
    ViscositySoftening, ExternalForce, GravAcceleration, Floor, GravAxis
from src.sph.density import get_minimum_distance_from_wall, add_ghost
from src.sph.Kernel import kernel

###############################################################################
#parallel worker job

@ray.remote(num_cpus=1)
def process(Particles, NgbTree, ahead):
    "this function does the heavy lifting for the parallelization"
    for particle in Particles:
        compute_force(particle, NgbTree, ahead)
    return Particles

###############################################################################
#per particle processing

def compute_force(particle, NgbTree, ahead):
    "compute the SPH acceleration for a single particle"
    #initialize results
    particle.acceleration = zeros(NDIM, dtype=float)
    particle.entropyChange = 0
    #prepare boundary treatment
    if particle.CloseToWall:
        min_dist_from_wall = get_minimum_distance_from_wall(particle, NgbTree)
    
    #loop over neighbors
    for no in particle.neighbors:
        ngb = NgbTree.Tp[no]
        if ngb.index != particle.index:
            #we will need these multiple times
            dist = get_distance_vector(particle.position, ngb.position, NgbTree)
            if particle.CloseToWall and add_ghost(particle, ngb, dist, min_dist_from_wall, NgbTree):
                #particle and ghost contributions cancel out
                continue
            r = norm(dist)
            dkern_i = dist/r * kernel(r/particle.Hsml, particle.Hsml, True)
            dkern_j = dist/r * kernel(r/ngb.Hsml, ngb.Hsml, True)
            if ahead:
                vij = particle.velocity_ahead - ngb.velocity_ahead
            else:
                vij = particle.velocity - ngb.velocity
            visc = viscosity_tensor(particle, ngb, dist, vij)
            #add viscosity contribution
            particle.acceleration -= visc * (dkern_i + dkern_j)/2
            particle.entropyChange += 0.5 * NgbTree.Mpart * visc * \
                                          dot(vij, dkern_i + dkern_j)
            #add pressure force contribution
            dkern_i *= particle.VarHsmlFac * particle.pressure/particle.Rho**2
            dkern_j *= ngb.VarHsmlFac * ngb.pressure/ngb.Rho**2
            particle.acceleration -= (dkern_i +dkern_j)
    particle.entropyChange *= (AdiabaticIndex - 1)/2/particle.Rho**(AdiabaticIndex - 1)
    particle.acceleration  *= NgbTree.Mpart 
    if ExternalForce:
        height = particle.position[GravAxis] * NgbTree.FacIntToCoord[GravAxis]
        #check if the gravitational pull is countered by the normal force
        flag = Floor and height < particle.Hsml
        if not flag:
            #particle is not touching the ground: apply gravity
            particle.acceleration[GravAxis] -= GravAcceleration
    #update the timestep criterion and free the memory occupied by the neighbor list
    particle.update_timestep_criterion(NgbTree)
    particle.neighbors = list()

###############################################################################

def force_step(particles, NgbTree_ref, Problem, activeTimeBin, ahead = False):
    "Updates the active particles' force and rate of entropy change"
    t0 = time()
    active = list()
    done = list()
    for particle in particles:
        if particle.timeBin >= activeTimeBin:
            active.append(particle)
        else:
            done.append(particle)
    
    load, ncpu = get_optimal_load(len(active))
    
    if ncpu > 1:
        #split work evenly among processes
        result = [process.remote(active[i * load:(i+1) * load], NgbTree_ref, \
                                 ahead) for i in range(ncpu-1)]
        result.append(process.remote(active[(ncpu-1) * load:], NgbTree_ref, \
                                     ahead))
        while len(result):
            done_id, result = ray.wait(result)
            done += ray.get(done_id[0])
    else:
        NgbTree = ray.get(NgbTree_ref)
        #do the remaining work locally
        for particle in active:
            compute_force(particle, NgbTree, ahead)
        done += active
    t1 = time()
    Problem.Timer["FORCE"] += t1- t0
    return done
        
def viscosity_tensor(particle1, particle2, rij, vij):
    "Returns the viscosity tensor for two particles"
    if dot(rij, vij) < 0:
        hij = (particle1.Hsml + particle2.Hsml)/2
        rhoij = (particle1.Rho + particle2.Rho)/2
        cij = (particle1.csound + particle2.csound)/2
        muij = hij * dot(vij, rij)/(norm(rij)**2 + ViscositySoftening*hij**2)
        return Viscosity*muij*(2*muij - cij)/rhoij
    else:
        return 0

        