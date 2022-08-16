import torch
from deepk.losses import *


def test_mse():
    assert torch.isclose(
        mse(
            ref = torch.tensor([[0.1,2],[97,202]]),
            new = torch.tensor([[1.1,2],[99,199]])
        ),
        torch.tensor(3.5)
    )

def test_l1():
    assert torch.isclose(
        l1(
            ref = torch.tensor([[0.1,2],[97,202]]),
            new = torch.tensor([[1.1,2],[99,199]])
        ),
        torch.tensor(1.5)
    )
