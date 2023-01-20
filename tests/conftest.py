# -*- coding: utf-8 -*-

import numpy as np
import pytest


@pytest.fixture
def trueA():
    return np.array([0.55, 0.35, 0.65, 0.4], [0.7, 1., 0.9, 0.8],
                    [0.2, 0.1, 0.45, 0.1], [0.6, 0.8, 0.9, 0.95])
