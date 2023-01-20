# -*- coding: utf-8 -*-

import numpy as np
import numpy.testing as npt
import sourcesink as ss


def test_make_trace(trueA):
    test = ss.make_trace(2, [1., 1.], trueA[:2, :2])
    result = np.array([[1., 0.9], [1., 1.7]])
    npt.assert_allclose(test, result)


def test_calcA(trueA):
    trace = ss.make_trace(10, np.ones(4), trueA)
    A = ss.calcA(trace)
    npt.assert_allclose(A, trueA)
