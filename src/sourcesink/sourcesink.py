# -*- coding: utf-8 -*-

import mne
import numpy as np


def make_trace(timesteps: int, initial: np.ndarray,
               A: np.ndarray) -> np.ndarray:
    trace = np.zeros((len(initial), timesteps))
    trace[:, 0] = initial
    for i in range(1, timesteps):
        trace[:, i] = A*trace[:, i-1]

    return trace
