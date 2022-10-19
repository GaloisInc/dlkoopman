"""Loss functions used for DeepKoopman neural net optimization."""


import torch


def overall(X, Y, Xr, Ypred, Xpred, decoder_loss_weight) -> dict[str, torch.Tensor]:
    """Computes overall loss for the DeepK neural net.
    
    ## Parameters
    - **X** (*torch.Tensor, shape=(\\*,num_input_states)*) - Input data.

    - **Y** (*torch.Tensor, shape=(\\*,num_encoded_states)*) - Encoded data, i.e. output from encoder, input to decoder.

    - **Xr** (*torch.Tensor, shape=(\\*,num_input_states)*) - Reconstructed data, i.e. output of decoder.

    - **Ypred** (*torch.Tensor, shape=(\\*,num_encoded_states)*) - DeepK-predicted encoded data.

    - **Xpred** (*torch.Tensor, shape=(\\*,num_input_states)*) - DeepK-predicted encoded data passed through decoder.

    - **decoder_loss_weight** (*float, optional*) - Weight the losses between decoder outputs (`recon` and `pred`) by this number. This is to account for the scaling effect of the decoder.

    ## Returns
    **losses** (*dict[str, torch.Tensor]*)
        - Key **'recon'**: (*torch.Tensor, scalar*) - Reconstruction loss between `X` and `Xr`.
        - Key **'lin'**: (*torch.Tensor, scalar*) - Linearity loss between `Y` and `Ypred`.
        - Key **'pred'**: (*torch.Tensor, scalar*) - Prediction loss between `X` and `Xpred`.
        - Key **'total'**: (*torch.Tensor, scalar*) - Total loss = `lin + decoder_loss_weight*(recon+pred)`
    """
    losses = {
        'recon': torch.nn.MSELoss(reduction='mean')(X, Xr),
        'lin': torch.nn.MSELoss(reduction='mean')(Y, Ypred),
        'pred': torch.nn.MSELoss(reduction='mean')(X, Xpred)
    }
    losses['total'] = losses['lin'] + decoder_loss_weight * (losses['recon'] + losses['pred'])
    return losses
