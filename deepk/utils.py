"""Utilities"""


import matplotlib.pyplot as plt
import numpy as np
import os
import torch


class MLP(torch.nn.Module):
    """Multi-layer perceptron neural net.

    ## Parameters
    - **input_size** (*int*) - Dimension of the input layer.

    - **output_size** (*int*) - Dimension of the output layer.

    - **hidden_sizes** (*list[int], optional*) - Dimensions of hidden layers, if any.

    - **batch_norm** (*bool, optional*) - Whether to use batch normalization.

    ## Attributes
    - **net** (*torch.nn.ModuleList*) - List of layers
    """
    def __init__(self, input_size, output_size, hidden_sizes=[], batch_norm=False):
        """ """
        super().__init__()
        self.net = torch.nn.ModuleList([])
        layers = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layers)-1):
            self.net.append(torch.nn.Linear(layers[i],layers[i+1]))
            if i != len(layers)-2: #all layers except last
                if batch_norm:
                    self.net.append(torch.nn.BatchNorm1d(layers[i+1]))
                self.net.append(torch.nn.ReLU())

    def forward(self, X) -> torch.Tensor:
        """Forward propagation of neural net.

        ## Parameters
        - **X** (*torch.Tensor, shape=(\\*,input_size)*) - Input data to net.

        ## Returns 
        - **X** (*torch.Tensor, shape=(\\*,output_size)*) - Output data from net.
        """
        for layer in self.net:
            X = layer(X)
        return X


class AutoEncoder(torch.nn.Module):
    """AutoEncoder neural net. Contains an encoder MLP connected to a decoder MLP.

    ## Parameters
    - **num_input_states** (*int*) - Number of dimensions in original data (`X`) and reconstructed data (`Xr`).

    - **num_encoded_states** (*int*) - Number of dimensions in encoded data (`Y`).

    - **encoder_hidden_layers** (*list[int], optional*) - Encoder will have layers = `[num_input_states, *encoder_hidden_layers, num_encoded_states]`. If not set, defaults to reverse of `decoder_hidden_layers`. If that is also not set, defaults to `[]`.

    - **decoder_hidden_layers** (*list[int], optional*) - Decoder has layers = `[num_encoded_states, *decoder_hidden_layers, num_input_states]`. If not set, defaults to reverse of `encoder_hidden_layers`. If that is also not set, defaults to `[]`.

    - **batch_norm** (*bool, optional*): Whether to use batch normalization.

    ## Attributes
    - **encoder** (*MLP*) - Encoder neural net.

    - **decoder** (*MLP*) - Decoder neural net.
    """
    def __init__(self, num_input_states, num_encoded_states, encoder_hidden_layers=[], decoder_hidden_layers=[], batch_norm=False):
        """ """
        super().__init__()

        if not decoder_hidden_layers and encoder_hidden_layers:
            decoder_hidden_layers = encoder_hidden_layers[::-1]
        elif not encoder_hidden_layers and decoder_hidden_layers:
            encoder_hidden_layers = decoder_hidden_layers[::-1]

        self.encoder = MLP(
            input_size = num_input_states,
            output_size = num_encoded_states,
            hidden_sizes = encoder_hidden_layers,
            batch_norm = batch_norm
        )

        self.decoder = MLP(
            input_size = num_encoded_states,
            output_size = num_input_states,
            hidden_sizes = decoder_hidden_layers,
            batch_norm = batch_norm
        )

    def forward(self, X) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation of neural net.

        ## Parameters
        - **X** (*torch.Tensor, shape=(\\*,num_input_states)*) - Input data to encoder.

        ## Returns 
        - **Y** (*torch.Tensor, shape=(\\*,num_encoded_states)*) - Encoded data, i.e. output from encoder, input to decoder.

        - **Xr** (*torch.Tensor, shape=(\\*,num_input_states)*) - Reconstructed data, i.e. output of decoder.
        """
        Y = self.encoder(X) # encoder complete output
        Xr = self.decoder(Y) # final reconstructed output
        return Y, Xr


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

    The other alternative is to zero out the NaN gradients as described [here](https://github.com/tensorflow/tensorflow/issues/17476#issue-302663705), however, we don't use this technique.

    ## Parameters
    **x** (*torch.Tensor*) - Matrix whose SVD will be computed. Assume shape to be (m,n).

    ## Returns
    - **U** (*torch.Tensor, shape=(m,min(m,n))*)
    - **Sigma** (*torch.Tensor, shape=(min(m,n),)*)
    - **V** (*torch.Tensor, shape=(n,min(m,n))*)
    """
    return _SVD.apply(x)


def plot_stats(dk, perfs=['pred_anae'], start_epoch=1, fontsize=12):
    """Plot stats from a DeepKoopman run.

    ## Parameters
    - **dk** (*core.DeepKoopman*) - A DeepKoopman object with `stats` populated.

    - **perfs** (*list[str]*) - Which variables from `stats` to plot. For each variable, training data and validation data stats are plotted vs epochs, and the title of the plot is the test data stats value.
    
    - **start_epoch** (*int*) - Start plotting from this epoch. Setting this to higher than 1 may be useful when the first few epochs have weird values that skew the y axis scale.

    - **fontsize** (*int*) - Font size of plot title. Other font sizes are automatically adjusted relative to this.

    ## Effects
    Creates plots for each `perf` and saves their png file(s) to `"<dk.results_folder>/<dk.uuid>_<perf>.png"`.
    """
    for perf in perfs:
        is_anae = 'anae' in perf

        tr_data = dk.stats[perf+'_tr'][start_epoch-1:]
        if dk.stats[perf+'_va']:
            va_data = dk.stats[perf+'_va'][start_epoch-1:]
            tr_data = tr_data[:len(va_data)] # va_data should normally have size equal to tr_data, but will have lesser size if some error occurred during training. This slicing ensures that only that portion of tr_data is considered which corresponds to va_data.
        epoch_range = range(start_epoch,start_epoch+len(tr_data))
        
        plt.figure()
        if dk.stats[perf+'_te']:
            plt.suptitle(f"Test performance = {dk.stats[perf+'_te']}" + (' %' if is_anae else ''), fontsize=fontsize)
        
        if perf == 'loss':
            plt.plot(epoch_range, dk.stats['loss_tr_before_K_reg'][start_epoch-1:], c='DarkSlateBlue', label='Training, before K_reg')
        plt.plot(epoch_range, dk.stats[perf+'_tr'][start_epoch-1:], c='MediumBlue', label='Training')
        if dk.stats[perf+'_va']:
            plt.plot(epoch_range, dk.stats[perf+'_va'][start_epoch-1:], c='DeepPink', label='Validation')
        
        if is_anae:
            ylim_anae = plt.gca().get_ylim()
            plt.ylim(max(0,ylim_anae[0]), min(100,ylim_anae[1])) # keep ANAE limits between [0,100]
        plt.xlabel('Epochs', fontsize=fontsize)
        plt.ylabel(perf + (' (%)' if is_anae else ''), fontsize=fontsize)
        plt.xticks(fontsize=fontsize-2)
        plt.yticks(fontsize=fontsize-2)
        
        plt.grid()
        plt.legend(fontsize=fontsize)

        plt.savefig(os.path.join(dk.results_folder, f'{dk.uuid}_{perf}.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)


def set_seed(seed):
    """Set a random seed to make results reproducible.

    ## Parameters
    **seed** (*int*) - The seed to be set.

    ## Effects
    Sets the random seed to `seed`.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
