import os
import pickle

import numpy as np
import pytest
import torch

from dlkoopman import utils
from dlkoopman.traj_pred import *


def get_data():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'examples/traj_pred_polynomial_manifold/data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data


def test_TrajPredDataHandler():
    data = get_data()
    dh = TrajPredDataHandler(
        Xtr=data['Xtr'],
        Xva=data['Xva'],
        Xte=data['Xte']
    )
    assert np.isclose(dh.Xscale, 0.4999852776527405)

    data['Xtr'][0][0][0] = -100.
    dh = TrajPredDataHandler(
        Xtr=data['Xtr']
    )
    assert np.isclose(dh.Xscale, 100.)
    assert torch.equal(dh.Xva, torch.tensor([]))
    assert torch.equal(dh.Xte, torch.tensor([]))


def test_TrajPred():
    data = get_data()
    dh = TrajPredDataHandler(
        Xtr=data['Xtr'],
        Xva=data['Xva'],
        Xte=data['Xte']
    )

    utils.set_seed(10)
    #NOTE: See note about setting seed in test_state_pred.py.

    tp = TrajPred(
        dh = dh,
        encoded_size = 10
    )
    tp.train_net(
        numepochs = 10,
        batch_size = 500
    )

    assert not tp.error_flag

    dom_eigval = tp.Lambda[0]
    assert 0.9 < dom_eigval.real < 1.1
    assert -0.1 < dom_eigval.imag < 0.1

    metric_moving_avg = utils.moving_avg(tp.stats['total_loss_va'], window_size=3)
    assert metric_moving_avg == sorted(metric_moving_avg, reverse=True)

    X0 = [[0.1,-0.1], [0.4,-0.45], [0.,0.]]
    preds = tp.predict_new(X0)
    assert preds.shape == (len(X0), 51, 2)

    logfile = f'log_{tp.uuid}.log'
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
def test_TrajPred_exact():
    """
    Pin the torch version and test if a specific training run exactly matches provided results.
    NOTE: If tests still fail, consider pinning numpy==1.23.0 and Python==3.9.12.
    """
    data = get_data()
    dh = TrajPredDataHandler(
        Xtr=data['Xtr'],
        Xva=data['Xva'],
        Xte=data['Xte']
    )

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ref_tp_stats.pkl'), 'rb') as f:
        ref_stats = round3(pickle.load(f))

    utils.set_seed(10)

    tp = TrajPred(
        dh = dh,
        encoded_size = 10
    )
    tp.train_net(
        numepochs = 2,
        batch_size = 250
    )
    tp.test_net()

    logfile = f'log_{tp.uuid}.log'
    assert os.path.isfile(logfile)
    os.system(f'rm -rf {logfile}')

    assert round3(tp.stats) == ref_stats
