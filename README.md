# rPPG
This repo measures instantaneous heart rate of a person through remote photoplethysmography(rPPG) without any physical contact with sensor, It captures video stream through webcam or video file. The captured frames are passed through a pre trained network to segment face skin pixels to consistently sample the RoI. Then we compute the spatial segmented pixels mean to minimise camera quantization error. Averaged values are temporaly normalized. It is projected to plane orthogonal to skin-tone to extract rPPG signal. We apply moving average filter of order 6 and compute power spectral density PSD applying fast fourier transformation (FFT) on rPPG signal. The maximum power spectrum represents the frequency of instant heart rate. 

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
* We have used semantic segmentation to generate face masks to remove non skin pixels from frames. The Segmentation requires cuda device

Clone this repository.

        git clone https://github.com/nasir6/rPPG.git

To run

        cd rPPG
        python3 run.py --source=0 --frame-rate=25


