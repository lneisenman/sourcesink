# -*- coding: utf-8 -*-

import modern_python_template as mpt


def test_template(tester):
    assert mpt.hello() == 'Hello Larry'
    assert mpt.hello(tester) == 'Hello Tester'
