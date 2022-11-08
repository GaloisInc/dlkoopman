import torch
from dlkoopman.metrics import *
from dlkoopman.metrics import _naae


def test_anae():
    assert torch.isclose(
        anae(
            ref = torch.tensor([[0.1,1],[100,200]]),
            new = torch.tensor([[1.1,2],[99,199]])
        ),
        torch.tensor(275.375)
    )

def test_naae():
    assert torch.isclose(
        _naae(
            ref = torch.tensor([[0.1,1],[100,200]]),
            new = torch.tensor([[1.1,2],[99,199]])
        ),
        torch.tensor(4000/3011)
    )


def test_overall_loss():
    losses = overall_loss(
        X = torch.tensor([[0.1,2],[97,202]]),
        Y = torch.tensor([[0.1,2],[97,202]]),
        Xr = torch.tensor([[0.1,2],[97,202]]),
        Ypred = torch.tensor([[1.1,2],[99,199]]),
        Xpred = torch.tensor([[1.1,2],[99,199]]),
        decoder_loss_weight = 0.1
    )
    assert torch.isclose(losses['recon'], torch.tensor(0.))
    assert torch.isclose(losses['lin'], torch.tensor(3.5))
    assert torch.isclose(losses['pred'], torch.tensor(3.5))
    assert torch.isclose(losses['total'], torch.tensor(3.85))
