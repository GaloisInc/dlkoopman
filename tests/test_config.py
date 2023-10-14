import os
import unittest

import numpy as np

from dlkoopman.config import *
from dlkoopman.state_pred import *


def test_config():
    cfg = Config(
        precision = "double"
    )
    dh = StatePredDataHandler(
        ttr = [100, 203, 298, 400],
        Xtr = np.array([
            [0.7, 2.1, 9.2],
            [1.1, 5. , 6.1],
            [4.3, 2. , 7.3],
            [6.1, 4.2, 0.3]
        ]),
        cfg = cfg
    )
    sp = StatePred(
        dh = dh,
        rank = 4,
        encoded_size = 50
    )
    
    assert dh.cfg.precision == "double"
    assert sp.cfg.precision == "double"
    assert dh.Xtr.dtype == torch.double

    os.system(f'rm -rf log_{sp.uuid}.log')


def test_config_validation_error():
    
    class _Test(unittest.TestCase):
        def _test(self):
            with self.assertRaises(ConfigValidationError):
                _ = Config(precision="foat")
            with self.assertRaises(ConfigValidationError):
                _ = Config(use_exact_eigenvectors="foat", sigma_threshold=False)
            with self.assertRaises(ConfigValidationError):
                _ = Config(torch_compile_backend="some garbage backend xxx")
    
    t = _Test()
    t._test()
