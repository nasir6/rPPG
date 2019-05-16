# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:15:49 2019

@author: nasir

Implements
Wavelet for post processing of pulse signal

"""
from scipy.signal import convolve2d
import numpy as np
import math
import pywt
from scipy.signal import convolve2d
import pycwt as wavelet

class Wavelet():
    def __init__(self, sr):
        self.samplingRate = sr
        self.minFreq = 0.75 #
        self.maxFreq = 3 # 
#        
    def get_scales(self):
        MorletFourierFactor = 4* math.pi / (6+math.sqrt(2+ 6**2))
        """take default smallest scale (corresponding to Nyquist frequency)"""
        scMin = 1 / (0.5 * self.samplingRate * MorletFourierFactor)
        """32 scales per octave"""
        ds = 1/32
        """  we do not consider low freq components """
        scMax = 1/(0.5 * self.minFreq * MorletFourierFactor)
        nSc =  math.floor(math.log2(scMax/scMin) /ds)

        scales = scMin * (2 ** (np.arange(0, nSc+1) * ds))
        return scales

    def get_cwt(self, signal):
        scales = self.get_scales()
        
#        """get frequencies range of interest 0.67 ~~ 4 Hz"""        
        MorletFourierFactor = 4* math.pi / (6+math.sqrt(2+ 6**2))
        freqs = 1 / (scales * MorletFourierFactor)        

        coef, scales, _, coi, fft, fftfreqs = wavelet.cwt(signal, 1/self.samplingRate, wavelet='morlet', freqs=freqs)


        firstScaleIndex = np.where(freqs < self.maxFreq)[0][0]
        lastScaleIndex = np.where(freqs > self.minFreq)[0][-1]
        
        energyProfile = np.abs(coef)
        
        max_index = np.argmax(energyProfile[firstScaleIndex:lastScaleIndex,:], axis=0)
        instantPulseRate = 60 * freqs[firstScaleIndex + max_index]
     
        return coef, instantPulseRate
    
    def get_instant_beats(self, energyProfile):
        scales = self.get_scales()
        MorletFourierFactor = 4* math.pi / (6+math.sqrt(2+ 6**2))
        freqs = 1 / (scales * MorletFourierFactor)              
        firstScaleIndex = np.where(freqs < self.maxFreq)[0][0]
        lastScaleIndex = np.where(freqs < self.minFreq)[0][0]
        max_index = np.argmax(energyProfile[firstScaleIndex:lastScaleIndex,:], axis=0)
        instantPulseRate = 60 * freqs[firstScaleIndex + max_index - 1]
        return instantPulseRate
