import cv2
import numpy as np
from pulse import Pulse
import time
from threading import Lock, Thread
from plot_cont import DynamicPlot
from capture_frames import CaptureFrames
import pandas as pd
from matplotlib import pyplot as plt

from utils import *
import multiprocessing as mp
import sys


class ProcessMasks():

    def __init__(self, sz=270, fs=30, bs=30, size=256):
        print('init')
        self.stop = False
        self.masked_batches = []
        self.batch_mean = []
        self.signal_size = sz
        self.batch_size = bs
        self.signal = np.zeros((sz, 3))
        self.pulse = Pulse(fs, sz, bs, size)
        self.hrs = []
        self.save_results = True

    def __call__(self, pipe, plot_pipe, source):
        self.pipe = pipe
        self.plot_pipe = plot_pipe
        self.source = source
        compute_mean_thread =  Thread(target=self.compute_mean)
        compute_mean_thread.start()
        
        extract_signal_thread =  Thread(target=self.extract_signal)
        extract_signal_thread.start()

        self.rec_frames()
        
        compute_mean_thread.join()
        extract_signal_thread.join()
  
    def rec_frames(self):
        while True and not self.stop:
            data = self.pipe.recv()

            if data is None:
                self.terminate()
                break
            batch = data[0]

            self.masked_batches.append(batch)
    
    def process_signal(self, batch_mean):
        size = self.signal.shape[0]
        b_size = batch_mean.shape[0]

        self.signal[0:size-b_size] = self.signal[b_size:size]
        self.signal[size-b_size:] = batch_mean
        p = self.pulse.get_pulse(self.signal)
        p = moving_avg(p, 6)
        hr = self.pulse.get_rfft_hr(p)
        if len(self.hrs) > 300: self.hrs.pop(0)

        self.hrs.append(hr)
        if self.plot_pipe is not None and self.stop:
            self.plot_pipe.send(None)
        elif self.plot_pipe is not None:
            self.plot_pipe.send([p, self.hrs])
        else:
            hr_fft = moving_avg(self.hrs, 3)[-1] if len(self.hrs) > 5 else self.hrs[-1]
            sys.stdout.write(f'\rHr: {round(hr_fft, 0)}')
            sys.stdout.flush()
    
    def extract_signal(self):
        signal_extracted = 0
        
        while True and not self.stop:
            if len(self.batch_mean) == 0:
                time.sleep(0.01)
                continue
            
            mean_dict = self.batch_mean.pop(0)
            mean = mean_dict['mean']

            if mean_dict['face_detected'] == False:
                if self.plot_pipe is not None:
                    self.plot_pipe.send('no face detected')
                continue
            if signal_extracted >= self.signal_size:
                self.process_signal(mean)
            else:
                self.signal[signal_extracted: signal_extracted + mean.shape[0]] = mean
            signal_extracted+=mean.shape[0]
    

    def compute_mean(self):
        curr_batch_size = 0
        batch = None
        while True and not self.stop:                
            if len(self.masked_batches) == 0:
                time.sleep(0.01)
                continue
            
            mask = self.masked_batches.pop(0)     
            if batch is None:
                batch = np.zeros((self.batch_size, mask.shape[0], mask.shape[1], mask.shape[2]))
            
            if curr_batch_size < (self.batch_size - 1):
                batch[curr_batch_size] = mask
                curr_batch_size+=1
                continue
            
            batch[curr_batch_size] = mask
            curr_batch_size = 0

            non_zero_pixels = (batch!=0).sum(axis=(1,2))
            total_pixels = batch.shape[1] * batch.shape[2]
            avg_skin_pixels = non_zero_pixels.mean()
            m = {'face_detected': True, 'mean': np.zeros((self.batch_size, 3))}
            if (avg_skin_pixels + 1) / (total_pixels) < 0.05:
                m['face_detected'] = False
            else:
                m['mean'] = np.true_divide(batch.sum(axis=(1,2)), non_zero_pixels+1e-6)

            self.batch_mean.append(m)

    def terminate(self):
        
        if self.plot_pipe is not None:
            self.plot_pipe.send(None)
        self.savePlot(self.source)
        self.saveresults()
        self.stop = True
    
    def saveresults(self):
        """
        saves numpy array of heart rates as hrs
        saves numpy array of power spectrum as fft_spec
        """
        np.save('hrs', np.array(self.hrs))
        np.save('fft_spec', np.array(self.pulse.fft_spec))

    def savePlot(self, path):
        if self.save_results == False:
            return
        
        # path = path.replace   ('/media/munawar/','/munawar-desktop/')
        # fig_path = path[40:].replace("/","_")
        
        # file_path = path.replace('video.avi','gt_HR.csv')
        # gt_HR = pd.read_csv(file_path, index_col=False).values
        if len(self.hrs) == 0:
            return

        ax1 = plt.subplot(1,1,1)
        ax1.set_title('HR')
        ax1.set_ylim([20, 180]) 
        ax1.plot(moving_avg(self.hrs, 6))
    
        # ax3 = plt.subplot(1,2,2)
        # ax3.set_title('GT')
        # ax3.set_ylim([20, 180])
        # ax3.plot(gt_HR[8:])

        plt.tight_layout() 
        plt.savefig(f'results.png')
        plt.close()

            
                

