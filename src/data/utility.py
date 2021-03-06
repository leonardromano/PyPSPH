#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:21:28 2020

@author: leonard
"""
from numpy import sqrt

from src.parameters.Parameters import NDIM

def factorial(n, lowerb = 2):
    result = 1
    for i in range(lowerb, n+1):
        result *= i
    return result

def dot(v1, v2):
    r = 0
    for i in range(NDIM):
        r += v1[i] * v2[i]
    return r

def norm(vector):
    r = 0
    for component in vector:
        r += component * component
    return sqrt(r)

