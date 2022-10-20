import pytest
import numpy as np
import os
import pickle
from deepk.data import *


@pytest.fixture
def get_data():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'examples/naca0012/data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data


def test_DataHandler(get_data):
    data = get_data
    dh = DataHandler(
        Xtr=data['Xtr'], ttr=data['ttr'],
        Xva=data['Xva'], tva=data['tva'],
        Xte=data['Xte'], tte=data['tte']
    )
    assert np.isclose(dh.Xscale, 5.5895)
    assert dh.tscale == 1.
    assert dh.tshift == 0.

    data['Xtr'][0][0] = -100.
    dh = DataHandler(
        Xtr=data['Xtr'], ttr=data['ttr']
    )
    assert np.isclose(dh.Xscale, 100.)
    assert dh.tscale == 1.
    assert dh.tshift == 0.
    assert torch.equal(dh.Xva, torch.tensor([]))
    assert torch.equal(dh.tva, torch.tensor([]))
    assert torch.equal(dh.Xte, torch.tensor([]))
    assert torch.equal(dh.tte, torch.tensor([]))
