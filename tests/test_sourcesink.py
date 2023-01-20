# -*- coding: utf-8 -*-

import numpy as np
import numpy.testing as npt
import sourcesink as ss


def test_make_trace(trueA):
    result = np.array([1., 0.9], [1., 1.7])
    npt.assert_allclose(ss.make_trace(2, [1., 1.], trueA[:2, :2]), result)
