import pytest
import pickle
import os
import numpy as np
from deepk.state_predictor import *
from deepk import utils


def round3(stats):
    for k in stats.keys():
        if type(stats[k]) != list:
            stats[k] = np.round(stats[k],3)
        else:
            stats[k] = [np.round(v,3) for v in stats[k]]
    return stats


@pytest.fixture
def get_data():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'examples/state_predictor_naca0012/data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data

@pytest.fixture
def get_ref_stats_rounded3():
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ref_sp_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    return round3(stats)


def test_StatePredictor_DataHandler(get_data):
    data = get_data
    dh = StatePredictor_DataHandler(
        Xtr=data['Xtr'], ttr=data['ttr'],
        Xva=data['Xva'], tva=data['tva'],
        Xte=data['Xte'], tte=data['tte']
    )
    assert np.isclose(dh.Xscale, 5.5895)
    assert dh.tscale == 1.
    assert dh.tshift == 0.

    data['Xtr'][0][0] = -100.
    dh = StatePredictor_DataHandler(
        Xtr=data['Xtr'], ttr=data['ttr']
    )
    assert np.isclose(dh.Xscale, 100.)
    assert dh.tscale == 1.
    assert dh.tshift == 0.
    assert torch.equal(dh.Xva, torch.tensor([]))
    assert torch.equal(dh.tva, torch.tensor([]))
    assert torch.equal(dh.Xte, torch.tensor([]))
    assert torch.equal(dh.tte, torch.tensor([]))


def test_StatePredictor(get_data, get_ref_stats_rounded3):
    data = get_data
    dh = StatePredictor_DataHandler(
        Xtr=data['Xtr'], ttr=data['ttr'],
        Xva=data['Xva'], tva=data['tva'],
        Xte=data['Xte'], tte=data['tte']
    )

    ref_stats = get_ref_stats_rounded3

    utils.set_seed(10)

    sp = StatePredictor(
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
