import numpy as np 
from scipy.signal import medfilt, decimate
import cv2
from torchvision import transforms
import pdb
from PIL import Image
import time
import torch
from torch.autograd import Variable

def scale_pulse(p):
    p = p - np.min(p)
    p = p/np.max(p)
    p-=0.5
    p*=2
    return p

def moving_avg(signal, w_s):
    ones = np.ones(w_s) / w_s
    moving_avg = np.convolve(signal, ones, 'valid')
    return moving_avg

def compute_snr(coef):
    num_of_bins = len(coef)
    max_bin_index = np.argmax(coef)
    signal_bins = coef[max_bin_index] + (coef[max_bin_index * 2] if max_bin_index * 2 < num_of_bins else 0)
    noise_bins = np.sum(coef) - signal_bins
    snr = 20*(np.log10(signal_bins/noise_bins))
    return snr

def post_process(values, w_s, k_s):
    ones = np.ones(w_s) / w_s
    moving_avg = np.convolve(values, ones, 'valid')
    decimated = decimate(moving_avg, w_s)
    filterd =  medfilt(decimated, k_s)
    return filterd

def compute_mean(frames):
    mmm = np.true_divide(frames.sum(axis=(1,2)),(frames!=0).sum(axis=(1,2)))
    return mmm

def transform_frames(frames, device, size=256):
    
    frames_copy = np.copy(frames)

    frames_transposed = np.transpose(frames_copy, (0,3,1,2))

    mean = torch.tensor(np.array([0.485, 0.456, 0.406]), dtype=torch.float).to(device=device)
    std = torch.tensor(np.array([0.229, 0.224, 0.225]), dtype=torch.float).to(device=device)

    tensors = Variable(torch.tensor(frames_transposed, dtype=torch.float).to(device=device)).div(255)

    resized = torch.nn.functional.interpolate(tensors, size=(size, size))

    normalized = (resized - mean[None, :, None, None]) / std[None, :, None, None]

    
    return normalized
        



def get_transform(size=256):
    t = transforms.Compose([
        transforms.Resize(size=(size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return t

def transform_single_frame(frames, size=256):
    f = get_transform()
    shape = frames.shape

    tranformed_frames = np.zeros((shape[0], 3,  shape[1], shape[2]))

    for i in range(shape[0]):
        tranformed_frames[i] = f(Image.fromarray(frames[i]))
    return tranformed_frames
    