#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:09:40 2021

@author: leonard
"""
"""
SPH simulation code by Leonard Romano
"""

"""
This file stores runtime parameters.
Specify the parameters of your simulation in this file 
before running the simulation with 'run'.
"""

#General parameters
##############################################################################
#path to IC-file without the .hdf5 tail
ICfile    = "/home/t30/all/ga87reg/Num_seminar/ICs/Rayleigh-Taylor"
#output directory
output    = "/home/t30/all/ga87reg/Num_seminar/output/"  
#Name of scheduling system
Scheduler = "SGE"
#Number of timesteps between Snapshots
OutputFrequency = 8     
##############################################################################

#Time parameters
##############################################################################
InitialTime = 0                    #Starting Time
FinalTime   = 1                    #Final Time
NTimesteps  = 128                  #Time refinement
NTimebins   = 16                   #Maximum depth of timestep hierarchy
##############################################################################

#Problem Parameters
##############################################################################
#Number of spatial dimensions
NDIM = 2
#Determines the domain and boundary
Problem_Specifier = "Rayleigh-Taylor"
#SPH parameters
AdiabaticIndex     = 1.4          #adiabatic Index gamma P~rho^gamma
Kernel             = "cubic"      #SPH Kernel function new Kernel options can be defined in Kernel.py
Viscosity          = 1.           #Viscosity Parameter \alpha
ViscositySoftening = 0.01         #Softening parameter to prevent blow up of viscosity force
DESNNGBS = 16                     #Desired number of neighbors
NNGBSDEV = 1                       #Limit how much the actual number of neighbors may deviate from the desired value
CourantParameter = 0.3            #Courant parameter
TimestepLimiter  = 0.01          #Limiter for kinematic timestep
#external force (gravity)
ExternalForce = True              #Enable external forces true/false
Floor = True                      #If true particles touching the floor don't experience gravity
GravAxis = 1                      #Axis along which gravitational field lies
GravAcceleration = 0.5            #magnitude of external force
##############################################################################
