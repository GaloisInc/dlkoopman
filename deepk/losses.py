"""Loss functions used for DeepKoopman neural net optimization."""


import torch


def overall(X, Y, Xr, Ypred, Xpred, decoder_loss_weight) -> dict[str, torch.Tensor]:
    """Computes overall loss for the DeepK neural net.
    
    ## Parameters
    - **X** (*torch.Tensor, shape=(\\*, input_size)*) - Input states, i.e. input to encoder.

    - **Y** (*torch.Tensor, shape=(\\*, encoded_size)*) - Encoded states, i.e. output from encoder, input to decoder.

    - **Xr** (*torch.Tensor, shape=(\\*, input_size)*) - Reconstructed states, i.e. output of decoder.

    - **Ypred** (*torch.Tensor, shape=(\\*, encoded_size)*) - Predicted encoded states obtained from evolving baseline encoded state.

    - **Xpred** (*torch.Tensor, shape=(\\*, input_size)*) - Predicted input states, which are predicted encoded states passed through decoder.

    - **decoder_loss_weight** (*float, optional*) - Weight the losses between decoder outputs (`recon` and `pred`) by this number. This is to account for the scaling effect of the decoder.

    ## Returns
    **losses** (*dict[str, torch.Tensor]*)
        - Key **'recon'**: (*torch scalar*) - Reconstruction loss between `X` and `Xr`.
        - Key **'lin'**: (*torch scalar*) - Linearity loss between `Y` and `Ypred`.
        - Key **'pred'**: (*torch scalar*) - Prediction loss between `X` and `Xpred`.
        - Key **'total'**: (*torch scalar*) - Total loss = `lin + decoder_loss_weight*(recon+pred)`
    """
    losses = {
        'recon': torch.nn.MSELoss(reduction='mean')(X, Xr),
        'lin': torch.nn.MSELoss(reduction='mean')(Y, Ypred),
        'pred': torch.nn.MSELoss(reduction='mean')(X, Xpred)
    }
    losses['total'] = losses['lin'] + decoder_loss_weight * (losses['recon'] + losses['pred'])
    return losses
