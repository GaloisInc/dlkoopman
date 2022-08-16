"""Loss functions used for DeepKoopman neural net optimization."""


import torch


def mse(ref, new) -> torch.Tensor:
    """[Pytorch MSE loss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) with `reduction='mean'`.

    ## Parameters
    **ref** (*torch.Tensor*) and **new** (*torch.Tensor*) - Loss will be calculated between these tensors.

    ## Returns
    Pytorch MSE loss.
    """
    return torch.nn.MSELoss(reduction='mean')(ref, new)

def l1(ref, new) -> torch.Tensor:
    """[Pytorch L1 loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html) with `reduction='mean'`.

    ## Parameters
    **ref** (*torch.Tensor*) and **new** (*torch.Tensor*) - Loss will be calculated between these tensors.

    ## Returns
    Pytorch L1 loss.
    """
    return torch.nn.L1Loss(reduction='mean')(ref, new)


def overall(X, Y, Xr, Ypred, Xpred, decoder_loss_weight=0.01, func=mse) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes overall loss for the DeepK neural net.
    
    ## Parameters
    - **X** (*torch.Tensor, shape=(\\*,num_input_states)*) - Input data.

    - **Y** (*torch.Tensor, shape=(\\*,num_encoded_states)*) - Encoded data, i.e. output from encoder, input to decoder.

    - **Xr** (*torch.Tensor, shape=(\\*,num_input_states)*) - Reconstructed data, i.e. output of decoder.

    - **Ypred** (*torch.Tensor, shape=(\\*,num_encoded_states)*) - DeepK-predicted encoded data.

    - **Xpred** (*torch.Tensor, shape=(\\*,num_input_states)*) - DeepK-predicted encoded data passed through decoder.

    - **decoder_loss_weight** (*float, optional*) - Weight the losses between decoder outputs (`recon` and `pred`) by this number. This is to account for the scaling effect of the decoder.

    - **func** (*function*) - Which loss function to use. See loss functions above.

    ## Returns
    - **recon** - Reconstruction loss between `X` and `Xr`.

    - **lin** - Linearity loss between `Y` and `Ypred`.

    - **pred** - Prediction loss between `X` and `Xpred`.

    - **total** - Total loss = `lin + decoder_loss_weight*(recon+pred)`
    """
    recon = func(ref=X, new=Xr)
    lin = func(ref=Y, new=Ypred)
    pred = func(ref=X, new=Xpred)
    total = lin + decoder_loss_weight*(recon+pred)
    return recon, lin, pred, total
