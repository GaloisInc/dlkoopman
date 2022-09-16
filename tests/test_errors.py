import torch
from deepk.errors import *
from deepk.errors import _naae # must import this separately since _ prefixed identifiers are not imported


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
