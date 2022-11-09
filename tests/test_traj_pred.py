import pytest
import pickle
import os
import numpy as np
from dlkoopman.traj_pred import *
from dlkoopman import utils


def round3(stats):
    for k in stats.keys():
        if type(stats[k]) != list:
            stats[k] = np.round(stats[k],3)
        else:
            stats[k] = [np.round(v,3) for v in stats[k]]
    return stats


@pytest.fixture
def get_data():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'examples/traj_pred_polynomial_manifold/data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data

@pytest.fixture
def get_ref_stats_rounded3():
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ref_tp_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    return round3(stats)


def test_TrajPredDataHandler(get_data):
    data = get_data
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


def test_TrajPred(get_data, get_ref_stats_rounded3):
    data = get_data
    dh = TrajPredDataHandler(
        Xtr=data['Xtr'],
        Xva=data['Xva'],
        Xte=data['Xte']
    )

    ref_stats = get_ref_stats_rounded3

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
