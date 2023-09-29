# -*- coding: utf-8 -*-

from typing import Tuple

import mne
from mne_connectivity import vector_auto_regression as vecAR
import numpy as np
import scipy as sp


def make_trace(timesteps: int, initial: np.ndarray,
               A: np.ndarray) -> np.ndarray:
    trace = np.zeros((len(initial), timesteps))
    trace[:, 0] = initial
    for i in range(timesteps-1):
        trace[:, i+1] = np.dot(A, trace[:, i])

    return trace


def calcA(data: np.ndarray) -> np.ndarray:
    return vecAR(np.expand_dims(data, axis=0)).get_data('dense')[0, :, :]


def calc_Abar(eeg: mne.io.Raw, step: float = 0.5,
              overlap: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    epochs = mne.make_fixed_length_epochs(raw=eeg, duration=step, overlap=overlap)
    times = epochs.times
    ch_names = epochs.ch_names
    conn = vecAR(epochs.get_data(), times=times, names=ch_names,
                 model='avg-epochs')
    return conn.get_data('dense')


def calc_ranks(A:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rows = A.shape[0]
    cnorm = np.zeros(rows)
    rnorm = np.zeros(rows)
    for i in range(rows):
        cnorm[i] = np.linalg.norm(A[:, i], 1) - np.abs(A[i, i])
        rnorm[i] = np.linalg.norm(A[i, :], 1) - np.abs(A[i, i])

    cr = sp.stats.rankdata(cnorm)
    rr = sp.stats.rankdata(rnorm)
    return cr/rows, rr/rows
