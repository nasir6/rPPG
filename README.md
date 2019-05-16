# rPPG
This repo measures instantaneous heart rate of a person through remote photoplethysmography(rPPG) without any physical contact with sensor, It captures video stream through webcam or video file. The captured frames are passed through a pre trained network to segment face skin pixels to consistently sample the RoI. We have used POS to extract the rPPG signal from spatially averaged masked frames. The extraced signal is further processed to estimate the instant heart rate. This code runs on cuda enabled device at 30 FPS.


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


