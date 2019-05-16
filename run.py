
import cv2
import numpy as np
from pulse import Pulse
import time
from threading import Lock, Thread
from plot_cont import DynamicPlot
from capture_frames import CaptureFrames
from process_mask import ProcessMasks

from utils import *
import multiprocessing as mp
import sys
from optparse import OptionParser

class RunPOS():
    def __init__(self,  sz=270, fs=28, bs=30, plot=False):
        self.batch_size = bs
        self.frame_rate = fs
        self.signal_size = sz
        self.plot = plot

    def __call__(self, source):
        time1=time.time()
        
        mask_process_pipe, chil_process_pipe = mp.Pipe()
        self.plot_pipe = None
        if self.plot:
            self.plot_pipe, plotter_pipe = mp.Pipe()
            self.plotter = DynamicPlot(self.signal_size, self.batch_size)
            self.plot_process = mp.Process(target=self.plotter, args=(plotter_pipe,), daemon=True)
            self.plot_process.start()
        
        process_mask = ProcessMasks(self.signal_size, self.frame_rate, self.batch_size)

        mask_processer = mp.Process(target=process_mask, args=(chil_process_pipe, self.plot_pipe, source, ), daemon=True)
        mask_processer.start()
        
        capture = CaptureFrames(self.batch_size, source, show_mask=True)
        capture(mask_process_pipe, source)

        mask_processer.join()
        if self.plot:
            self.plot_process.join()
        time2=time.time()
        time2=time.time()
        print(f'time {time2-time1}')

def get_args():
    parser = OptionParser()
    parser.add_option('-s', '--source', dest='source', default=0,
                        help='Signal Source: 0 for webcam or file path')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=30,
                        type='int', help='batch size')
    parser.add_option('-f', '--frame-rate', dest='framerate', default=25,
                        help='Frame Rate')

    (options, _) = parser.parse_args()
    return options
        
if __name__=="__main__":
    args = get_args()
    source = args.source
    runPOS = RunPOS(270, args.framerate, args.batchsize, True)
    runPOS(source)
    