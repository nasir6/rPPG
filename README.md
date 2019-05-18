# rPPG

This repo measures instantaneous heart rate of a person through remote photoplethysmography(rPPG) without any physical contact with sensor, by capturing video stream through webcam or a video file.

## Pre Processing 
Skin pixels have significant part in extraction of rPPG signal we trained first ever deep learning model for semantic 
segmentation of skin and non skin pixels. This is novel technique for regio of interst (ROI) selection and tracking. The model is robust to motion, multiple poses and segments skin pixels from non skin very accurately.
Waveform of rPPG signal is different when extracted from different rigion of skin pixels therefore to consistently sample ROI from same part of skin we detect face in frame as pre step to semantic segmentation.

## rPPG Signal Extraction 
After accurately sampling ROI for signal extraction we compute the spatial red, green and blue channel mean of skin segmented pixels to minimise camera quantization error. Averaged values of RGB channel are temporaly normalized and projected to plane orthogonal to skin-tone. The projected signal is alpha tuned to extract rPPG signal. 

## Post processing

We apply moving average filter of order 6 to remove noise from signal. To estimate heart rate we compute power spectral density PSD applying fast fourier transformation (FFT) on rPPG signal. It  is then band pass filtered to analyse only frequencies of interest. The maximum power spectrum represents the frequency of instant heart rate. 

This code runs on cuda enabled device at 30 FPS.


## Pipeline

![](images/pipeline.png)

## Requirements

* Python 3
* Numpy
* Pytorch
* OpenCv
* Matplotlib, Scipy, Pillow
* Git Lfs to track trained model parameters
* We have used deep learning for semantic segmentation of skin and non skin pixels from frames. The segmentation requires cuda enabled device


Clone this repository.

        git clone https://github.com/nasir6/rPPG.git

To run

        cd rPPG
        python3 run.py --source=0 --frame-rate=25


