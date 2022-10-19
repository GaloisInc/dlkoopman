import pytest
import pickle
import os
import numpy as np
from deepk.core import *
from deepk.data import DataHandler
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
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'examples/naca0012/data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data

@pytest.fixture
def get_ref_dk_stats_rounded3():
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ref_dk_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    return round3(stats)


def test_core(get_data, get_ref_dk_stats_rounded3):
    data = get_data
    dh = DataHandler(
        Xtr=data['Xtr'], ttr=data['ttr'],
        Xva=data['Xva'], tva=data['tva'],
        Xte=data['Xte'], tte=data['tte']
    )

    ref_stats = get_ref_dk_stats_rounded3

    utils.set_seed(10)

    dk = DeepKoopman(
        dh = dh,
        rank = 6,
        num_encoded_states = 50
    )
    dk.train_net(
        numepochs = 50
    )
    dk.test_net()

    logfile = os.path.join(f'dk_{dk.uuid}.log')
    assert os.path.isfile(logfile)
    os.system(f'rm -rf {logfile}')

    assert round3(dk.stats) == ref_stats
