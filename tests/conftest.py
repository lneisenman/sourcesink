# -*- coding: utf-8 -*-

import numpy as np
import pytest


@pytest.fixture
def trueA():
    return np.array([[0.55, 0.35, 0.65, 0.4],
                     [0.7, 1., 0.9, 0.8],
                     [0.2, 0.1, 0.45, 0.1],
                     [0.6, 0.8, 0.9, 0.95]])


@pytest.fixture
def testA():
    return np.array([[0.98, -0.002, 0.002, -0.002],
                     [0.0005, 0.995, -0.000651, -0.0005],
                     [-0.0003, -0.005, 1, 0.001],
                     [-0.0003, 0.007, -0.002, 1]])
