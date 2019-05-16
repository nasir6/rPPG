# -*- coding: utf-8 -*-
"""
Implements
Color distortion Filtering preprocessing step for POS.
"""

import numpy as np
from numpy.linalg import inv
import math
#import scipy.io
#mat = scipy.io.loadmat('c.mat')

def CDF(C, B):
    C_ = np.matmul(inv(np.diag(np.mean(C,1))),C)-1
    F = np.fft.fft(C_)
    S = np.dot(np.expand_dims(np.array([-1/math.sqrt(6), 2/math.sqrt(6), -1/math.sqrt(6)]), axis=0), F)
    W = np.real((S* S.conj()) / np.sum((F * F.conj()),0)[None,:])
    W[:, 0:B[0]] = 0
    W[:, B[1]+1:] = 0
    
    F_ = F * W
    iF = (np.fft.ifft(F_) + 1).real
    C__ = np.matmul(np.diag(np.mean(C,1)) , iF)
    C = C__.astype(np.float)
    return C
    
#C = mat['C']
#CDF(C)