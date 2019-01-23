#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 11:55:09 2018

@author: dirac
"""

import numpy as np
import math

comb=lambda m,n : math.factorial(m)/(math.factorial(n)*math.factorial(m-n))

def kEmptyBins_outof_n(n,k,x):
    assert x>=n
    assert k<n
    p=comb(n,k)*((n-k)/float(n))**x
    return p

def p(x,n):
    summ=0
    for i in range(1,n):
        p=kEmptyBins_outof_n(n,i,x)
        print (p)
        summ+=p
    return 1-summ
    