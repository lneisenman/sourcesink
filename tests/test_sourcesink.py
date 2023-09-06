# -*- coding: utf-8 -*-

import mne
import numpy as np
import numpy.testing as npt
import pytest

import sourcesink as ss


@pytest.fixture
def trace(testA):
    return ss.make_trace(100, np.ones(4), testA)


@pytest.fixture
def trace_eeg(testA):
    SAMPLES = 5000
    SFREQ = 500
    # time = np.arange(SAMPLES)/SFREQ
    signal = ss.make_trace(SAMPLES, np.ones(4), testA)
    info = mne.create_info(['A1', 'A2', 'A3', 'A4'], SFREQ, ch_types='seeg',
                           verbose='error')
    return mne.io.RawArray(signal, info)


def test_make_trace(trueA):
    trace = ss.make_trace(2, np.ones(2), trueA[:2, :2])
    result = np.array([[1., 0.2], [1., 0.3]])
    npt.assert_allclose(trace, result)


def test_calcA(testA, trace):
    A = ss.calcA(trace[:, 90:])
    npt.assert_allclose(A, testA, rtol=1e-6)


def test_calc_Abar(testA, trace_eeg):
    Avals, test = ss.calc_Abar(trace_eeg)
    print(f'test = {test}')
    print(f'testA = {testA}')
    npt.assert_allclose(Avals[:, :, 19], testA, rtol=1e-6)
