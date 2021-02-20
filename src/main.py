#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:25:36 2020

@author: leonard
"""
import ray
from sys import exit, stdout
from time import time


from src.parameters.Constants import Dt, NCPU
from src.parameters.Parameters import NTimesteps, OutputFrequency, FinalTime

import src.init.Initialization as init
from src.forces.forceCalculation import force_step
from src.time_integration.Drift import Kick_and_Drift, Kick
from src.sph.density import update_sph_quantities
from src.time_integration.timesteps import assign_timestep_classes, \
    get_active_time_bin
from src.tree.tree import ngbtree, update_Tp
from src.writing.writing import write_data

def main():
    "This function initializes the simulation and contains the main loop"
    #turn of all running instances of ray to avoid interference with our task
    ray.shutdown()
    ray.init(num_cpus = NCPU)
    t0 = time()
    print("Starting SPH calculation using %d cores..."%NCPU)
    #Read IC file specifying the problem and the particle data
    Particles, Problem = init.read_ic_file()
    #initialize particles and perform first force calculation
    NgbTree_ref = ray.put(ngbtree(Particles, Problem))
    Particles = init.sph_quantities(Particles, NgbTree_ref, Problem)
    NgbTree_ref = update_Tp(Particles, NgbTree_ref, Problem)
    Particles = force_step(Particles, NgbTree_ref, Problem, 0)
    #assign particles to time bins
    Particles, tb_lowest = assign_timestep_classes(Particles, 0, Problem)
    #loop over all timesteps
    for i in range(NTimesteps):
        #Write a snapshot if required
        if i%OutputFrequency == 0:
            print("writing Snapshot no. %d" %(i//OutputFrequency))
            write_data(Particles, i//OutputFrequency, i*Dt, Problem)
        #j is the subcycle 'clock'
        j = 0
        activeTimeBin = 0
        #cycle integration step over all timebins
        while j < 1:
            #first kick all active particles by the current active timestep
            dt = Dt/2**tb_lowest
            #flush the output buffer
            print("Current Time: %g, timestep %g"%((i + j) * Dt, dt))
            stdout.flush()
            
            Particles = Kick_and_Drift(Particles, Problem, activeTimeBin, dt)
            
            #update the local clock to the next synchronization point and 
            #determine which particles are active next
            j += 2**(-tb_lowest)
            activeTimeBin = get_active_time_bin(j)
            
            #at the new synchronisation point update sph quantities and forces
            NgbTree_ref = ray.put(ngbtree(Particles, Problem))
            Particles = update_sph_quantities(Particles, NgbTree_ref, Problem)
            NgbTree_ref = update_Tp(Particles, NgbTree_ref, Problem)
            Particles = force_step(Particles, NgbTree_ref, Problem, \
                                   activeTimeBin, True)
            
            #kick the active particles by the current active timestep
            Particles = Kick(Particles, Problem, activeTimeBin)
            
            #determine the new smallest timestep
            Particles, tb_lowest = assign_timestep_classes(Particles, activeTimeBin, \
                                                             Problem)
    #after we're done we want to write the final results in a snapshot
    write_data(Particles, NTimesteps//OutputFrequency, FinalTime, Problem)
    t1 = time()
    T = t1 - t0
    print("Reached the final timestep! Took %g seconds.\n"%T)
    print("Compuational cost of individual parts:")
    print("INIT: %g (%g)"%(Problem.Timer["INIT"], Problem.Timer["INIT"]/T))
    print("TREE: %g (%g)"%(Problem.Timer["TREE"], Problem.Timer["TREE"]/T))
    print("DENSITY: %g (%g)"%(Problem.Timer["DENSITY"], Problem.Timer["DENSITY"]/T))
    print("TIMEBINS: %g (%g)"%(Problem.Timer["TIMEBINS"], Problem.Timer["TIMEBINS"]/T))
    print("FORCE: %g (%g)"%(Problem.Timer["FORCE"], Problem.Timer["FORCE"]/T))
    print("DRIFTS: %g (%g)"%(Problem.Timer["DRIFTS"], Problem.Timer["DRIFTS"]/T))
    print("OUTPUT: %g (%g)"%(Problem.Timer["OUTPUT"], Problem.Timer["OUTPUT"]/T))
    print("We are done now.\nBye.")
    ray.shutdown()
    exit()