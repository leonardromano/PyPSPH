#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:19:30 2020

@author: leonard
"""
from numpy import zeros, ones, asarray, minimum, maximum, copy
from math import ceil
import ray
from sys import exit, stdout
from time import time

from src.parameters.Constants import NORM_COEFF, MAX_INT, LARGE_NUM
from src.parameters.Parameters import NDIM, DESNNGBS, NNGBSDEV
from src.sph.Kernel import kernel, bias_correction
from src.data.int_conversion import get_distance_vector
from src.data.parallel_utility import get_optimal_load
from src.data.utility import norm

###############################################################################
#Parallel functions & classes

@ray.remote(num_cpus=1)
def process_treewalk(Particles, NgbTree):
    "this function does the heavy lifting for the parallelization"
    for particle in Particles:
        walk_tree(particle, NgbTree)
    return Particles

@ray.remote(num_cpus=1)
def process_finalize(Particles, NgbTree, Left, Right, ahead):
    "this function does the heavy lifting for the parallelization"
    Lower = copy(Left)
    Upper = copy(Right)
    done = list()
    work_left = list()
    for particle in Particles:
        i = particle.index
        #do some postprocessing on density
        finish_density_update(particle, NgbTree)
        numNgb = NORM_COEFF * particle.Hsml**(NDIM) * particle.Rho / NgbTree.Mpart
        if abs(numNgb-DESNNGBS) <= NNGBSDEV:
            #now update the thermodynamic quantities
            particle.update_pressure(ahead)
            particle.update_soundspeed()
            done.append(particle)
        else:    
            if Left[i] > 0 and Right[i] < LARGE_NUM and Right[i]-Left[i] < 1e-3 * Left[i]:
                #this one should be ok
                particle.update_pressure(ahead)
                particle.update_soundspeed()
                done.append(particle)
                continue
            #need to redo this one
            Lower[i], Upper[i] = update_bounds(Lower[i], Upper[i], numNgb, \
                                              particle.Hsml)
            update_smoothing_length(Lower[i], Upper[i], numNgb, particle)
            work_left.append(particle)
    return done, work_left, Lower, Upper

###############################################################################
#per particle processing

def walk_tree(particle, NgbTree):
    "Preparation, neighbor search and density computation"
    particle.Rho = 0
    particle.VarHsmlFac = 0
    particle.neighbors = list()
    sph_density_interact(particle, NgbTree.MaxPart, "node", NgbTree)
    evaluate_kernel(particle, NgbTree)

###############################################################################
#Treewalk related functions

def evaluate_particle_node_opening_criterion(particle, node, NgbTree):
    """
    This function checks whether there is a spatial overlap between the 
    (rectangular) enclosing box of the particles contained in a node, 
    and the search region.
    """
    if node.level <= 0:
        return 1
    
    #compute the offset from the particle position
    part_offset  = zeros(NDIM, dtype = int)
    part_offset += particle.position
    for i in range(NDIM):
        part_offset[i] -= ceil(particle.Hsml/NgbTree.FacIntToCoord[i])
        
    left  = get_distance_vector(node.center_offset_min + node.center, \
                                part_offset, NgbTree)
    right = get_distance_vector(node.center_offset_max + node.center, \
                                part_offset, NgbTree)
    for i in range(NDIM):
        if left[i] > 2 * particle.Hsml and right[i] > left[i]:
            return 0
    return 1

def sph_density_open_node(particle, nop, NgbTree):
    "Continues to walk the tree for the particle by opening a node."
    p = nop.nextnode
    while p != nop.sibling:
        if p < 0:
            print("p=%d < 0  node.sibling=%d node.nextnode=%d" \
                   %(p, nop.sibling, nop.nextnode))
            ray.shutdown()
            exit()
        nextp = 0
        typep = ""
        if p < NgbTree.MaxPart:
            nextp = NgbTree.Nextnode[p]
            typep = "particle"
        else:
            node = NgbTree.get_nodep(p)
            nextp = node.sibling
            typep = "node"
        sph_density_interact(particle, p, typep, NgbTree)
        
        p = nextp

def sph_density_interact(particle, no, no_type, NgbTree):
    """
    Take care of SPH density interaction between the particle, and the node
    referenced through no. The node can either be a node or a particle.
    """
    if no_type == "particle":
        #we have a particle check whether it's a neighbor
        ngb = NgbTree.Tp[no]
        r = norm(get_distance_vector(particle.position, ngb.position, NgbTree))
        if r > particle.Hsml:
            return
        particle.neighbors.append(no)
    else:
        node = NgbTree.get_nodep(no)
        if not node.notEmpty:
            return
        if evaluate_particle_node_opening_criterion(particle, node, NgbTree):
            sph_density_open_node(particle, node, NgbTree)

def densities_determine(NgbTree_ref, Workstack, npleft):
    """
    for each target walk the tree to determine the neighbors and then 
    compute density and thermodynamic quantities
    """
    #first get the needed number of processes and their load
    load, ncpu = get_optimal_load(npleft)
    
    if ncpu > 1:
        #split work evenly among processes
        result = [process_treewalk.remote(Workstack[i * load:(i+1) * load], \
                                          NgbTree_ref) for i in range(ncpu-1)]
        result.append(process_treewalk.remote(Workstack[(ncpu-1) * load:], \
                                              NgbTree_ref))
    
        Worklist = list()
        while len(result):
            done_id, result = ray.wait(result)
            Worklist += ray.get(done_id[0])
        Workstack = Worklist
    else:
        NgbTree = ray.get(NgbTree_ref)
        #do the remaining work locally
        for particle in Workstack:
            walk_tree(particle, NgbTree)
            
    return Workstack
        
###############################################################################
#main loop

def update_sph_quantities(Particles, NgbTree_ref, Problem):
    t0 = time()
    Particles = density(Particles, NgbTree_ref, True)
    t1 = time()
    Problem.Timer["DENSITY"] += t1-t0
    return Particles

def density(Workstack, NgbTree_ref, ahead = False):
    "For each particle compute density, smoothing length and thermodynamic variables"
    Npart = len(Workstack)
    Left = zeros(Npart)
    Right = ones(Npart) * LARGE_NUM
    Donestack = list()
    npleft = Npart
    niter = 0
    while True:
        # now do the primary work with this call
        Workstack = densities_determine(NgbTree_ref, Workstack, npleft)
        # do final operations on results
        Donestack, Workstack, \
        Left, Right, npleft = do_final_operations(NgbTree_ref, Workstack, \
                                                  Donestack, Left, Right, \
                                                  npleft, ahead)
        niter += 1
        
        #DEBUG
        print("DENSITY: Loop %d. npleft = %d, npdone = %d"\
              %(niter, npleft, len(Donestack)))
        stdout.flush()
        
        if npleft <= 0:
            break
    print("DENSITY: Finished after %d iterations."%niter)
    return Donestack

###############################################################################
#Bisection algorithm related functions

def update_bounds(lowerBound, upperBound, numNgb, h):
    "Update the bounds for the smoothing length in the bisection algorithm"
    if numNgb < DESNNGBS - NNGBSDEV:
        lowerBound = max(lowerBound, h)
    else:
        if upperBound != LARGE_NUM:
            upperBound = min(h, upperBound)
        else:
            upperBound = h
    return lowerBound, upperBound

def update_smoothing_length(lowerBound, upperBound, NumNgb, particle):
    "Perform the bisection part of the bisection algorithm"
    if lowerBound > 0 and upperBound < LARGE_NUM:
        particle.Hsml = ((lowerBound**3 + upperBound**3)/2)**(1/3)
    else:
        if upperBound == LARGE_NUM and lowerBound == 0:
            print("Upper and Lower bounds not updated!")
            ray.shutdown()
            exit()
            
        if upperBound == 0 and lowerBound>0:
            if abs(NumNgb - DESNNGBS) < 0.5*DESNNGBS:
                fac = 1 - (NumNgb - DESNNGBS)/NumNgb/NDIM * particle.VarHsmlFac
                if fac < 1.26:
                    particle.Hsml *= fac
                else:
                    particle.Hsml *= 1.26
            else:
                particle.Hsml *= 1.26
                    
        if upperBound > 0 and lowerBound  == 0:
            if abs(NumNgb - DESNNGBS) < 0.5*DESNNGBS:
                fac = 1 - (NumNgb - DESNNGBS)/NumNgb/NDIM * particle.VarHsmlFac
                if fac > 1/1.26:
                    particle.Hsml *= fac
                else:
                    particle.Hsml /= 1.26
            else:
                particle.Hsml /= 1.26

###############################################################################
#ghost particle related functions    

def close_to_wall(particle, NgbTree):
    for i in range(NDIM):
        if not NgbTree.Periodic[i]:
            dist_from_wall = get_distance_from_wall(particle, NgbTree, i)
            if particle.Hsml > dist_from_wall:
                particle.CloseToWall = 1
                return
    particle.CloseToWall = 0

def add_ghost(particle, ngb, dist, min_dist_from_wall, NgbTree):
    for i in range(NDIM):
        if not NgbTree.Periodic[i]:
            dist_from_wall = get_distance_from_wall(particle, NgbTree, i)
            if dist_from_wall < get_distance_from_wall(ngb, NgbTree, i) and \
               dist_from_wall < abs(dist[i]) + 0.75 * min_dist_from_wall[i]:
                   return 1
    return 0

def get_distance_from_wall(particle, NgbTree, axis):
    "returns the distance from the nearest non-periodic boundary"
    dx = min(particle.position[axis], MAX_INT - particle.position[axis]) 
    return NgbTree.FacIntToCoord[axis] * dx

def get_minimum_distance_from_wall(particle, NgbTree):
    "Determines the minimum distance from the wall among a list of particles"
    min_dist_from_wall = [*NgbTree.Boxsize]
    for i in range(NDIM):
        if not NgbTree.Periodic[i]:
            for n in particle.neighbors:
                dist = get_distance_from_wall(NgbTree.Tp[n], NgbTree, i)
                if dist < min_dist_from_wall[i]:
                    min_dist_from_wall[i] = dist
    return asarray(min_dist_from_wall)

###############################################################################
#density calculation

def do_final_operations(NgbTree_ref, Workstack, Donestack, Left, Right, \
                        npleft, ahead):
    "Postprocessing and check if done"
    load, ncpu = get_optimal_load(npleft)
    
    if ncpu > 1:
        #split work evenly among processes
        result = [process_finalize.remote(Workstack[i * load:(i+1) * load], \
                                          NgbTree_ref, Left, Right, ahead) \
                  for i in range(ncpu-1)]
        result.append(process_finalize.remote(Workstack[(ncpu-1) * load:], \
                                              NgbTree_ref, Left, Right, ahead))
    
        Worklist = list()
        while len(result):
            done_id, result = ray.wait(result)
            done, work, left, right = ray.get(done_id[0])
            Worklist += work
            Donestack += done
            Left = maximum(Left, left)
            Right = minimum(Right, right)
    else:
        #do the remaining work locally
        NgbTree = ray.get(NgbTree_ref)
        Worklist = list()
        for particle in Workstack:
            i = particle.index
            #do some postprocessing on density
            finish_density_update(particle, NgbTree)
            numNgb = NORM_COEFF * particle.Hsml**(NDIM) * particle.Rho / NgbTree.Mpart
            if abs(numNgb-DESNNGBS) <= NNGBSDEV:
                particle.update_pressure(ahead)
                particle.update_soundspeed()
                Donestack.append(particle)
            else:   
                #check whether we're done
                if Left[i] > 0 and Right[i] < LARGE_NUM and Right[i]-Left[i] < 1e-3 * Left[i]:
                    #this one should be ok
                    particle.update_pressure(ahead)
                    particle.update_soundspeed()
                    Donestack.append(particle)
                    continue
                #need to redo this one
                Left[i], Right[i] = update_bounds(Left[i], Right[i], numNgb, \
                                              particle.Hsml)
                update_smoothing_length(Left[i], Right[i], particle)
                Worklist.append(particle)
                
    npleft = len(Worklist)
    return Donestack, Worklist, Left, Right, npleft        

def evaluate_kernel(particle, NgbTree):
    "Perform the neighbor sum to compute density and SPH correction factor"
    close_to_wall(particle, NgbTree)
    if particle.CloseToWall:
        min_dist_from_wall = get_minimum_distance_from_wall(particle, NgbTree)
    for n in particle.neighbors:
        ngb = NgbTree.Tp[n]
        dist = get_distance_vector(particle.position, ngb.position, NgbTree)
        h = particle.Hsml
        r = norm(dist)
        wk  = kernel(r/h, h)
        dwk = kernel(r/h, h, True)
        if particle.CloseToWall and add_ghost(particle, ngb, dist, min_dist_from_wall, NgbTree):
            #if we have a ghost, double the neighbors contribution instead
            wk  *= 2.0
            dwk *= 2.0
        particle.Rho += wk
        particle.VarHsmlFac -= r * dwk
            
def finish_density_update(particle, NgbTree):
    "some final postprocessing steps in density calculation"
    if particle.Rho > 0:
        bias_correction(particle)
        #now if we have more than one neighbor update the dhsml factor
        if particle.VarHsmlFac > 0:
            particle.VarHsmlFac = NDIM * particle.Rho / particle.VarHsmlFac
        else:
            particle.VarHsmlFac = 1
        particle.Rho *= NgbTree.Mpart
