"""Utilities"""


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import torch
from typing import Any


__pdoc__ = {
    'stable_svd': False,
    'moving_avg': False
}


def _safe_inverse(x, epsilon=1e-12):
    return x/(x**2 + epsilon)

class _SVD(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        U, S, V = torch.svd(A) #NOTE: torch.svd may be depreciated later, requiring the switch to torch.linalg.svd
        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = len(S)

        F = (S - S[:, None])
        F = _safe_inverse(F)
        F.diagonal().fill_(0)

        G = (S + S[:, None])
        G.diagonal().fill_(np.inf)
        G = 1/G 

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F+G)*(UdU-UdU.t())/2
        Sv = (F-G)*(VdV-VdV.t())/2

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt 
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt 
        if (N>NS):
            dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        return dA

def stable_svd(x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable Singular Value Decomposition (courtesy [this](https://github.com/wangleiphy/tensorgrad/blob/master/tensornets/adlib/svd.py), in response to [this](https://github.com/google/jax/issues/2311#issuecomment-984131512)) alternative to [`torch.linalg.svd`](https://pytorch.org/docs/stable/generated/torch.linalg.svd.html), which may encounter NaNs in gradients.

    The other solution is to zero out the NaN gradients as described [here](https://github.com/tensorflow/tensorflow/issues/17476#issue-302663705), however, we don't use this technique.

    ## Parameters
    **x** (*torch.Tensor*) - Matrix whose SVD will be computed. Assume shape to be (m,n).

    ## Returns
    - **U** (*torch.Tensor, shape=(m, min(m,n))*)
    - **Sigma** (*torch.Tensor, shape=(min(m,n),)*)
    - **V** (*torch.Tensor, shape=(n, min(m,n))*)
    """
    return _SVD.apply(x)


def plot_stats(model, perfs=['pred_anae'], start_epoch=1, fontsize=12):
    """Plot stats from a model.

    ## Parameters
    - **model** (*StatePred* or *TrajPred*) - A model with `stats` populated.

    - **perfs** (*list[str]*) - Which variables from `stats` to plot. For each variable, training data and validation data stats are plotted vs epochs, and the title of the plot is the test data stats value.
    
    - **start_epoch** (*int*) - Start plotting from this epoch. Setting this to higher than 1 may be useful when the first few epochs have weird values that skew the y axis scale.

    - **fontsize** (*int*) - Font size of plot title. Other font sizes are automatically adjusted relative to this.

    ## Effects
    Creates plots for each `perf` and saves their png file(s) to `"./plot_<model.uuid>_<perf>.png"`.
    """
    for perf in perfs:
        is_anae = 'anae' in perf

        tr_data = model.stats[perf+'_tr'][start_epoch-1:]
        if model.stats[perf+'_va']:
            va_data = model.stats[perf+'_va'][start_epoch-1:]
            tr_data = tr_data[:len(va_data)] # va_data should normally have size equal to tr_data, but will have lesser size if some error occurred during training. This slicing ensures that only that portion of tr_data is considered which corresponds to va_data.
        epoch_range = range(start_epoch,start_epoch+len(tr_data))

        plt.figure()
        if model.stats[perf+'_te']:
            plt.suptitle(f"Test performance = {model.stats[perf+'_te'][-1]}" + (' %' if is_anae else ''), fontsize=fontsize)

        plt.plot(epoch_range, model.stats[perf+'_tr'][start_epoch-1:], c='MediumBlue', label='Training')
        if model.stats[perf+'_va']:
            plt.plot(epoch_range, model.stats[perf+'_va'][start_epoch-1:], c='DeepPink', label='Validation')

        if is_anae:
            ylim_anae = plt.gca().get_ylim()
            plt.ylim(max(0,ylim_anae[0]), min(100,ylim_anae[1])) # keep ANAE limits between [0,100]
        plt.xlabel('Epochs', fontsize=fontsize)
        plt.ylabel(perf + (' (%)' if is_anae else ''), fontsize=fontsize)
        plt.xticks(fontsize=fontsize-2)
        plt.yticks(fontsize=fontsize-2)

        plt.grid()
        plt.legend(fontsize=fontsize)

        save_path = Path(f'./plot_{model.uuid}_{perf}.png').resolve()
        print(f'Saving figure {save_path}')
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)


def set_seed(seed):
    """Set a random seed to make results reproducible.

    ## Parameters
    **seed** (*int*) - The seed to be set.

    ## Effects
    Sets the random seed to `seed`.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _tensorize(arg, dtype, device) -> Any:
    return torch.as_tensor(arg, dtype=dtype, device=device) if arg is not None else torch.tensor([], dtype=dtype, device=device)

def _scale(arg, scale) -> Any:
    return arg/scale

def _shift(arg, shift) -> Any:
    return arg-shift


def _extract_item(v) -> Any:
    """ Given input, return its `.item()` if it can be extracted, otherwise return input. """
    try:
        ret = v.item()
    except (AttributeError, ValueError):
        ret = v
    return ret


def moving_avg(inp, window_size = 3):
    """Compute moving average of the elements of an ordered iterable.

    ## Parameters
    **inp** (*Ordered Array, e.g. list, tuple*) - Input ordered iterable.
    **window_size** (*int*) - How many elements to consider in each moving average.

    ## Return
    out (*Ordered Array, same type as input*) - Moving averaged elements of `inp`.

    Throws a `ValueError` if `window_size` > `len(inp)`.

    ## Example
    ```python
    inp = [1,2,3,5,7]
    out = moving_avg(inp, 3)
    # out = [2, 10/3, 5]
    ```
    """
    if window_size > len(inp):
        raise ValueError(f"'window_size' must be <= length of 'inp', but {window_size} > {len(inp)}")
    return type(inp)(np.convolve(inp, np.ones(window_size), 'valid') / window_size)
