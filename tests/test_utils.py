import torch
from deepk.utils import *
from deepk.utils import _extract_item


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


def test_extract_item():
    assert _extract_item(torch.tensor(3.)) == 3.
    assert _extract_item(torch.tensor([3.])) == 3.
    assert torch.equal(
        _extract_item(torch.tensor([[1,2],[3,4]])),
        torch.tensor([[1,2],[3,4]])
    )
    assert _extract_item(3.) == 3.
    assert _extract_item([3.]) == [3.]
