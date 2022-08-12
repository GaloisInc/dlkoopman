import matplotlib.pyplot as plt
import numpy as np
import torch


def paramgrads_check_nan(parameters):
    """
    Check if there is any NAN in the gradients of all the parameters of a net
    parameters: Just pass net.parameters()
    """
    for parameter in parameters:
        if torch.any(torch.isnan(parameter.grad)).item():
            raise ValueError


########################################################################
# SVD alternative to avoid NaN gradient issues
# Courtesy https://github.com/wangleiphy/tensorgrad/blob/master/tensornets/adlib/svd.py
# In response to https://github.com/google/jax/issues/2311#issuecomment-984131512

def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + epsilon)

class SVD(torch.autograd.Function):
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
        F = safe_inverse(F)
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

stable_svd = SVD.apply

# Note that the other alternative is to zero out the NaN gradients
# Described here: https://github.com/tensorflow/tensorflow/issues/17476#issue-302663705
# We don't use this technique here
########################################################################


def plot_stats(stats, perfs=['pred_anae'], save_path='./plots/', start_epoch=1, fontsize=12):
    """
    Plot stats for a run
    
    Inputs
        perfs: Which variables to plot. For each, tr and va are plotted vs epochs, and the title of the plot is the te value.
        start_epoch: If not 1, start plotting from that epoch. This is useful when the first few epochs have weird values that skew the y axis scale.

    Returns:
        None
    Saves png file(s) to disk.
    """
    for perf in perfs:
        is_anae = 'anae' in perf

        tr_data = stats[perf+'_tr'][start_epoch-1:]
        if stats[perf+'_va']:
            va_data = stats[perf+'_va'][start_epoch-1:]
            tr_data = tr_data[:len(va_data)] # va_data should normally have size equal to tr_data, but will have lesser size if some error occurred during training. This slicing ensures that only that portion of tr_data is considered which corresponds to va_data.
        epoch_range = range(start_epoch,start_epoch+len(tr_data))
        
        plt.figure()
        if stats[perf+'_te']:
            plt.suptitle(f"Test performance = {stats[perf+'_te']}" + (' %' if is_anae else ''), fontsize=fontsize)
        
        if perf == 'loss':
            plt.plot(epoch_range, stats['loss_tr_before_K_reg'][start_epoch-1:], c='DarkSlateBlue', label='Training, before K_reg')
        plt.plot(epoch_range, stats[perf+'_tr'][start_epoch-1:], c='MediumBlue', label='Training')
        if stats[perf+'_va']:
            plt.plot(epoch_range, stats[perf+'_va'][start_epoch-1:], c='DeepPink', label='Validation')
        
        if is_anae:
            ylim_anae = plt.gca().get_ylim()
            plt.ylim(max(0,ylim_anae[0]), min(100,ylim_anae[1])) # keep ANAE limits between [0,100]
        plt.xlabel('Epochs', fontsize=fontsize)
        plt.ylabel(perf + (' (%)' if is_anae else ''), fontsize=fontsize)
        plt.xticks(fontsize=fontsize-2)
        plt.yticks(fontsize=fontsize-2)
        
        plt.grid()
        plt.legend(fontsize=fontsize)

        plt.savefig(f'{save_path}_{perf}.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
