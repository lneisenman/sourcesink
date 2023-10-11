# -*- coding: utf-8 -*-

from typing import Tuple

import matplotlib.pyplot as plt
import mne
from mne_connectivity import EpochConnectivity
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


def calc_A(eeg: mne.io.Raw, step: float = 0.5,
           overlap: float = 0.25) -> EpochConnectivity:
    epochs = mne.make_fixed_length_epochs(raw=eeg, duration=step, overlap=overlap)
    times = epochs.times
    ch_names = epochs.ch_names
    return vecAR(epochs.get_data(), times=times, names=ch_names)


def calc_Abar(eeg: mne.io.Raw, step: float = 0.5,
              overlap: float = 0.25) -> EpochConnectivity:
    epochs = mne.make_fixed_length_epochs(raw=eeg, duration=step, overlap=overlap)
    times = epochs.times
    ch_names = epochs.ch_names
    conn = vecAR(epochs.get_data(), times=times, names=ch_names,
                 model='avg-epochs')
    return conn


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


def calc_sink_src(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cr, rr = calc_ranks(A)
    sinkidx = np.zeros_like(cr)
    srcidx = np.zeros_like(cr)
    for i in range(len(sinkidx)):
        sinkidx[i] = np.sqrt(2) - np.linalg.norm((rr[i]-1, cr[i]-(1/len(cr))))
        srcidx[i] = np.sqrt(2) - np.linalg.norm((rr[i]-(1/len(cr)), cr[i]-1))

    return sinkidx, srcidx


def calc_infl_conn(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sink, src = calc_sink_src(A)
    sink /= np.max(sink)
    src /= np.max(src)
    infl = np.zeros_like(sink)
    conn = np.zeros_like(sink)
    for i in range(len(infl)):
        for j in range(len(infl)):
            infl[i] += np.abs(A[i, j])*src[j]
            conn[i] += np.abs(A[i, j])*sink[j]

    return infl, conn


def calc_SSM(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
    sink, src = calc_sink_src(A)
    sink /= np.max(sink)
    src /= np.max(src)
    infl = np.zeros_like(sink)
    conn = np.zeros_like(sink)
    for i in range(len(infl)):
        for j in range(len(infl)):
            infl[i] += np.abs(A[i, j])*src[j]
            conn[i] += np.abs(A[i, j])*sink[j]

    infl /= np.max(infl)
    conn /= np.max(conn)
    SSI = sink*infl*conn
    return sink, src, infl, conn, SSI


def calc_SSI(conn) -> np.ndarray:
    A = conn.get_data()
    SSI = np.zeros((A.shape[1], A.shape[0]))
    for i in range(A.shape[0]):
        SSM = calc_SSM(A[i, :, :])
        SSI[:, i] = SSM[-1]

    return SSI


def plot_SSI(conn, electrodes):
    SSI = calc_SSI(conn)
    lengths = list()
    for electrode in electrodes:
        temp = [contact for contact in conn.names if electrode in contact]
        lengths.append(len(temp))
    
    xticks = [x for x in range(0, 121, 20)]
    xticklabels = [str(x) for x in range(0, 61, 10)]
    yticks = [0]
    yticks.extend(np.cumsum(lengths)[:-1])
    fig, ax = plt.subplots(1, 1)
    ax.imshow(SSI)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('time (sec)')
    ax.set_yticks(yticks)
    ax.set_yticklabels(electrodes)
    return fig, ax
