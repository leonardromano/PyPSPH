#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:52:07 2020

@author: leonard
"""
import h5py
from numpy import asarray
import ray
from time import time

from src.parameters.Constants import BITS_FOR_POSITIONS, NORM_COEFF, NCPU
from src.parameters.Parameters import ICfile, DESNNGBS, NDIM

from src.data.int_conversion import convert_to_int_position
from src.data.particle_class import particle
from src.data.problem_class import problem
from src.sph.density import density

###############################################################################
#parallel worker job

@ray.remote(num_cpus=1)
def process(Particles, NgbTree_ref):
    "this function does the heavy lifting for the parallelization"
    for p in Particles:
        guess_hsml(p, NgbTree_ref)
    return Particles

###############################################################################
#per particle processing

def guess_hsml(particle, NgbTree):
    i = particle.ID
    no = NgbTree.Father[i]
    while(10 * DESNNGBS * NgbTree.Mpart > NgbTree.get_nodep(no).Mass):
        p = NgbTree.get_nodep(no).father
        if p < 0:
            break
        no = p
        
    if(NgbTree.get_nodep(no).level > 0):
        length = (1 << (BITS_FOR_POSITIONS - NgbTree.get_nodep(no).level)) * \
            NgbTree.FacIntToCoord.max()
    else:
        length = NgbTree.Boxsize.max()

    particle.Hsml =  length * (DESNNGBS * NgbTree.Mpart / \
                               NgbTree.get_nodep(no).Mass / NORM_COEFF)**(1/NDIM)
        
###############################################################################

def read_ic_file():
    "Read the IC file specifying the problem and the particle data"
    t0 = time()
    Problem = problem()
    file = h5py.File(ICfile + ".hdf5", "r")
    header = file["Header"].attrs
    Problem.Mpart = header["Mpart"]
    Problem.BoxSize = header["Boxsize"]
    Problem.Periodic = header["Periodic"]
    Problem.update_int_conversion()
    positions  = asarray(file["PartData/Coordinates"])
    velocities = asarray(file["PartData/Velocity"]) 
    entropies  = asarray(file["PartData/Entropy"])
    Particles = [particle(convert_to_int_position(positions[i], Problem.FacIntToCoord), \
                          velocities[i], entropies[i], i) \
                 for i in range(header["NumPart"])]
    print("   Npart: %d \n"%(header["NumPart"]) + \
          "   Mpart: %g \n"%(Problem.Mpart) + \
          "   Boxsize:" + str(Problem.Boxsize) + "\n" + \
          "   Periodic:" + str(Problem.Periodic) + "\n\n")
    file.close()
    t1 = time()
    Problem.Timer["INIT"] += t1-t0
    return Particles, Problem
    
def sph_quantities(particles, NgbTree_ref, Problem):
    "This function initializes the sph properties of the particles"
    t0 = time()
    particles = initial_guess_hsml(particles, NgbTree_ref)
    #compute density, smoothing length and thermodynamic quantities and 
    #finds neighbors
    particles = density(particles, NgbTree_ref)
    t1 = time()
    Problem.Timer["DENSITY"] += t1-t0
    return particles
    
def initial_guess_hsml(Particles, NgbTree_ref):
    "computes an initial guess for the smoothing lengths"
    load = len(Particles)//NCPU
    result = [process.remote(Particles[i * load:(i+1) * load], NgbTree_ref) \
              for i in range(NCPU-1)]
    result.append(process.remote(Particles[(NCPU-1) * load:], NgbTree_ref))
    
    Particles_new = list()
    while len(result):
        done_id, result = ray.wait(result)
        Particles_new += ray.get(done_id[0])
    
    return Particles_new