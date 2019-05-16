# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:15:49 2019

@author: nasir

Implements
Amplitude Selective Filtering preprocessing step for POS.

"""

import numpy as np
from numpy.linalg import inv
import math
# import scipy.io
# mat = scipy.io.loadmat('c.mat')

def ASF(C):
    alpha=.002;delta=0.0001            
    C_=np.dot(inv(np.diag(np.mean(C,1))),C)-1   
    L = C.shape[1]
    F=np.fft.fft(C_)/L        
    W=delta/(1e-12+np.abs(F[0,:]))
    W=W.astype(np.complex)        
    
    W[np.abs(F[0,:]) < alpha]= 1
    
    W=np.stack((W,W,W),axis=0)
    F_=F*W
    
    C__ = np.dot(np.diag(np.mean(C,1)) ,(np.fft.ifft(F_)+1))

    C__ = C__.astype(np.float)
    return C__
    
# C = mat['C']
# A = ASF(C)