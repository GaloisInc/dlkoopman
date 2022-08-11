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
