import torch
from deepk.losses import *


def test_overall():
    losses = overall(
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
