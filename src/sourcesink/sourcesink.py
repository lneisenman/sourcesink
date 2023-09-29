# -*- coding: utf-8 -*-

import mne
from mne_connectivity import vector_auto_regression as vecAR
import numpy as np
from typing import Tuple


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
