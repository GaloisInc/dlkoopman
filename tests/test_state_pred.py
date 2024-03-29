import os
import pickle

import numpy as np
import pytest
import torch

from dlkoopman import utils
from dlkoopman.config import Config
from dlkoopman.state_pred import *


def get_data():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'examples/state_pred_naca0012/data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data


def test_StatePredDataHandler():
    cfg = Config()
    data = get_data()
    dh = StatePredDataHandler(
        Xtr=data['Xtr'], ttr=data['ttr'],
        Xva=data['Xva'], tva=data['tva'],
        Xte=data['Xte'], tte=data['tte']
    )
    assert np.isclose(dh.Xscale, 5.5895)
    assert dh.tscale == 1.
    assert dh.tshift == 0.

    data['Xtr'][0][0] = -100.
    dh = StatePredDataHandler(
        Xtr=data['Xtr'], ttr=data['ttr']
    )
    assert np.isclose(dh.Xscale, 100.)
    assert dh.tscale == 1.
    assert dh.tshift == 0.
    assert torch.equal(dh.Xva, torch.tensor([], dtype=cfg.RTYPE, device=cfg.DEVICE))
    assert torch.equal(dh.tva, torch.tensor([], dtype=cfg.RTYPE, device=cfg.DEVICE))
    assert torch.equal(dh.Xte, torch.tensor([], dtype=cfg.RTYPE, device=cfg.DEVICE))
    assert torch.equal(dh.tte, torch.tensor([], dtype=cfg.RTYPE, device=cfg.DEVICE))


def test_StatePred():
    data = get_data()
    dh = StatePredDataHandler(
        Xtr=data['Xtr'], ttr=data['ttr'],
        Xva=data['Xva'], tva=data['tva'],
        Xte=data['Xte'], tte=data['tte']
    )

    utils.set_seed(10)
    #NOTE: We don't need to set a seed since torch results will anyway be different across versions (ref: https://pytorch.org/docs/stable/notes/randomness.html). However, it seems that the starting weights are the same across versions for the same seed, and the variability comes from the floating point computations performed during training (ref: https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047). So, it's good to set a seed so that the results do not stray too much.

    sp = StatePred(
        dh = dh,
        rank = 6,
        encoded_size = 50
    )
    sp.train_net(
        numepochs = 50
    )

    assert not sp.error_flag

    dom_eigval = torch.exp(sp.Omega)[0,0]
    assert 0.9 < dom_eigval.real < 1.1
    assert -0.1 < dom_eigval.imag < 0.1

    metric_moving_avg = utils.moving_avg(sp.stats['total_loss_va'], window_size=9)
    assert metric_moving_avg == sorted(metric_moving_avg, reverse=True)

    t = [-1,6.789,30]
    preds = sp.predict_new(t)
    assert preds.shape == (len(t), 200)

    logfile = f'log_{sp.uuid}.log'
    assert os.path.isfile(logfile)
    os.system(f'rm -rf {logfile}')


def round3(stats):
    for k in stats.keys():
        if type(stats[k]) != list:
            stats[k] = np.round(stats[k],3)
        else:
            stats[k] = [np.round(v,3) for v in stats[k]]
    return stats

@pytest.mark.skipif(torch.__version__ != '1.12.1', reason = "Exact comparisons require a specific torch version since the results of training may not match across versions (see https://pytorch.org/docs/stable/notes/randomness.html).")
def test_StatePred_exact():
    """
    Pin the torch version and test if a specific training run exactly matches provided results.
    NOTE: If tests still fail, consider pinning numpy==1.23.0 and Python==3.9.12.
    """
    data = get_data()
    dh = StatePredDataHandler(
        Xtr=data['Xtr'], ttr=data['ttr'],
        Xva=data['Xva'], tva=data['tva'],
        Xte=data['Xte'], tte=data['tte']
    )

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ref_sp_stats.pkl'), 'rb') as f:
        ref_stats = round3(pickle.load(f))

    utils.set_seed(10)

    sp = StatePred(
        dh = dh,
        rank = 6,
        encoded_size = 50
    )
    sp.train_net(
        numepochs = 50
    )
    sp.test_net()

    logfile = f'log_{sp.uuid}.log'
    assert os.path.isfile(logfile)
    os.system(f'rm -rf {logfile}')

    assert round3(sp.stats) == ref_stats
