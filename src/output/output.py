#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:16:08 2020

@author: leonard
"""
import h5py
from numpy import zeros
from time import time

from src.parameters.Parameters import AdiabaticIndex, FinalTime, Viscosity, \
    ViscositySoftening, DESNNGBS, CourantParameter, output, NDIM


def write_data(Particles, label, Time, Problem):
    "writes all the particle data in a hdf5 file."
    t0 = time()
    
    f = h5py.File("%s/sph_%d.hdf5"%(output,label), "w")
    #first dump all the header info
    header = f.create_group("Header")
    Npart = len(Particles)
    
    h_att = header.attrs
    h_att.create("NumPart", Npart)
    h_att.create("FinalTime", FinalTime)
    h_att.create("Mass", Problem.Mpart)
    h_att.create("AdiabaticIndex", AdiabaticIndex)
    h_att.create("Viscosity", Viscosity)
    h_att.create("ViscositySoftening", ViscositySoftening)
    h_att.create("Time", Time)
    h_att.create("NumNeighbors", DESNNGBS)
    h_att.create("CourantParameter", CourantParameter)
    h_att.create("Boxsize", Problem.Boxsize)
    #now make the data sets for the particle data
    IDs        = zeros((Npart), dtype = int)
    positions  = zeros((Npart, NDIM), dtype = float)
    velocities = zeros((Npart, NDIM), dtype = float)
    entropies  = zeros((Npart), dtype = float)
    densities  = zeros((Npart), dtype = float)
    pressures  = zeros((Npart), dtype = float)
    hsml       = zeros((Npart), dtype = float)
    for particle in Particles:
        i = particle.index
        IDs[i]        += i
        positions[i]  += particle.position * Problem.FacIntToCoord
        velocities[i] += particle.velocity
        entropies[i]  += particle.entropy
        densities[i]  += particle.density
        pressures[i]  += particle.pressure
        hsml[i]       += particle.hsml
    f.create_dataset("PartData/IDs", data = IDs,  dtype = "u4")
    f.create_dataset("PartData/Coordinates", data = positions, dtype = "f4")
    f.create_dataset("PartData/Velocity", data = velocities, dtype = "f4")
    f.create_dataset("PartData/Entropy", data = entropies, dtype = "f4")
    f.create_dataset("PartData/Density", data = densities, dtype = "f4")
    f.create_dataset("PartData/Pressure", data = pressures, dtype = "f4")
    f.create_dataset("PartData/SmoothingLength", data = hsml, dtype = "f4")
    f.close()
    
    t1 = time()
    Problem.Timer["OUTPUT"] += t1-t0