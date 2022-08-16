import torch
from deepk.utils import *


def test_stable_svd():
    x = torch.tensor([[1.,1.], [0.,2.]])
    U, S, V = stable_svd(x)
    assert torch.all(torch.isclose(
        torch.round(U, decimals=4),
        torch.tensor([[ 0.5257,-0.8507], [0.8507,0.5257]])
    ))
    assert torch.all(torch.isclose(
        torch.round(S, decimals=4),
        torch.tensor([2.2882,0.8740])
    ))
    assert torch.all(torch.isclose(
        torch.round(V, decimals=4),
        torch.tensor([[0.2298,-0.9732], [0.9732,0.2298]])
    ))
    assert torch.all(torch.isclose(
        U @ torch.diag(S) @ V.t(),
        x,
        atol=1e-6
    ))
