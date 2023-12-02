import numpy as np
import math as m
from collections import OrderedDict

import torch
from scipy.signal import butter, lfilter, filtfilt

import matplotlib.pyplot as plt



def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def scale(points, factor):
    return points * factor

def rotate(data, axis, angle):
    if axis == 0:
        return np.array(np.matmul(Rx(angle), data.T).T)
    if axis == 1:
        return np.array(np.matmul(Ry(angle), data.T).T)
    if axis == 2:
        return np.array(np.matmul(Rz(angle), data.T).T)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def rotate_origin(data):
    # -1 * ground reaction force extracted from QTM
    return -1*data



def translate_cop(data, X, Y):
    trans_data = data.copy().T
    trans_data[0] += X
    trans_data[1] += Y
    return trans_data.T
 
def extractData(data, string):
    x = data[string+'_X'].tolist()
    y = data[string+'_Y'].tolist()
    z = data[string+'_Z'].tolist()

    fs = 2000.0
    cutoff = 20.0
    order = 6
    x = butter_lowpass_filter(x, cutoff, fs, order)
    y = butter_lowpass_filter(y, cutoff, fs, order)
    z = butter_lowpass_filter(z, cutoff, fs, order)

    return np.transpose(np.array([x,y,z]))


def plotting(data, x_label, y_label):
    plot_loss = np.array(data)
    time = np.arange(len(data))
    plt.plot(time, plot_loss, label="Losses")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()