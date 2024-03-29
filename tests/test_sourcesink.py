# -*- coding: utf-8 -*-

import mne
import numpy as np
import numpy.testing as npt
import pytest

import sourcesink as ss


@pytest.fixture
def trace(trueA):
    return ss.make_trace(10, np.ones(4), trueA)


@pytest.fixture
def trace_eeg(testA):
    SAMPLES = 100
    SFREQ = 50
    # time = np.arange(SAMPLES)/SFREQ
    signal = ss.make_trace(SAMPLES, np.ones(4), testA)
    info = mne.create_info(['A1', 'A2', 'A3', 'A4'], SFREQ, ch_types='seeg',
                           verbose='error')
    return mne.io.RawArray(signal, info)


def test_make_trace(trueA):
    trace = ss.make_trace(2, np.ones(2), trueA[:2, :2])
    result = np.array([[1., 0.9], [1., 1.7]])
    npt.assert_allclose(trace, result)


def test_calcA(trueA, trace):
    A = ss.calcA(trace)
    npt.assert_allclose(A, trueA, rtol=1e-6)


def test_calc_A(testA, trace_eeg):
    conn = ss.calc_A(trace_eeg)
    data = conn.get_data()
    for i in range(data.shape[0]):
        npt.assert_allclose(data[i, :, :], testA)


def test_calc_Abar(testA, trace_eeg):
    conn = ss.calc_Abar(trace_eeg)
    npt.assert_allclose(conn.get_data(), testA, rtol=1e-6)


def test_calc_ranks(trueA):
    cr, rr = ss.calc_ranks(trueA)
    assert np.abs(cr[1] - 1/4) < 1e-6
    assert np.abs(rr[1] - 1) < 1e-6


def test_calc_sink_src(trueA):
    sink, src = ss.calc_sink_src(trueA)
    assert np.abs(sink[1] - np.sqrt(2)) < 1e-6
    assert np.abs(src[1] - np.sqrt(2)/4) < 1e-6


def test_calc_infl_conn(trueA):
    infl, conn = ss.calc_infl_conn(trueA)
    assert np.abs(infl[1] - 2.075) < 1e-6
    assert np.abs(conn[1] - 2.175) < 1e-6


def test_calc_SSM(trueA):
    sink, src, infl, conn, SSI = ss.calc_SSM(trueA)
    assert np.abs(sink[1] - 1) < 1e-6
    assert np.abs(src[1] - 0.25) < 1e-6
    assert np.abs(infl[1] - 1) < 1e-6
    assert np.abs(conn[1] - 1) < 1e-6
    assert np.abs(SSI[1] - 1) < 1e-6
