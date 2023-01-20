# -*- coding: utf-8 -*-

import mne
import numpy as np


def make_trace(timesteps: int, initial: np.ndarray,
               A: np.ndarray) -> np.ndarray:
    trace = np.zeros((len(initial), timesteps))
    trace[:, 0] = initial
    for i in range(1, timesteps):
        trace[:, i] = A@trace[:, i-1]

    return trace


def calcA(data: np.ndarray) -> np.ndarray:
    b = np.ravel(data[:, 1:].T)
    (channels, steps) = data.shape
    H = np.zeros((channels*(steps-1), channels*channels))
    for i in range(steps-1):
        roffset = channels*i
        for j in range(channels):
            coffset = channels*j
            H[roffset+j, coffset:coffset+channels] = data[:, i]

    (X, __, __, __) = np.linalg.lstsq(H, b, rcond=None)
    return X.reshape((channels, channels))
