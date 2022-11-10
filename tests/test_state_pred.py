import pickle
import os
import numpy as np
from dlkoopman.state_pred import *
from dlkoopman import utils


def get_data():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'examples/state_pred_naca0012/data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data


def test_StatePredDataHandler():
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
    assert torch.equal(dh.Xva, torch.tensor([]))
    assert torch.equal(dh.tva, torch.tensor([]))
    assert torch.equal(dh.Xte, torch.tensor([]))
    assert torch.equal(dh.tte, torch.tensor([]))


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
