"""Loss and error functions, used to optimize models and report their performance."""


import torch


def anae(ref, new) -> torch.Tensor:
    """Average Normalized Absolute Error (ANAE).
    
    ANAE first normalizes each absolute deviation by the corresponding absolute ground truth, then averages them. This is a useful thing to report since it tells us how much percentage deviation to expect for a new value. E.g. if prediction ANAE for a problem is around 10%, then one can expect a newly predicted value to have an error of around 10% from the actual.
    
    Example: Let
    ```python
    ref = torch.tensor([[-0.1,0.2,0],[100,200,300]])
    new = torch.tensor([[-0.11,0.15,0.01],[105,210,285]])
    ```

    $$ANAE = \\text{Avg}\\left(\\frac{|-0.1-(-0.11)|}{|-0.1|}, \\frac{|0.2-0.15|}{|0.15|}, \\frac{|100-105|}{|100|}, \\frac{|200-210|}{|200|}, \\frac{|300-285|}{|300|}\\right) = 10\\%$$
    Note that:
    
    - Ground truth value of \\(0\\) is ignored. 
    
    - ANAE heavily penalizes deviations for small values of ground truth.

    ## Parameters
    **ref** (*torch.Tensor*) and **new** (*torch.Tensor*) - Error will be calculated between these tensors.

    ## Returns
    **anae** (*torch scalar*) - In percentage.
    """
    anae = torch.abs(ref-new)/torch.abs(ref)
    return 100.*torch.mean(anae[anae!=torch.inf])

def _naae(ref, new) -> torch.Tensor:
    """Normalized Average Absolute Error (NAAE).
    
    NAAE first averages the absolute error, then normalizes that using the average value of the absolute ground truth. This reduces the chance of a few small ground truth values having a huge impact, but loses details by taking the average over all absolute error values, and likewise over all ground truth values.
    
    Example: Let `ref = torch.tensor([[-0.1,0.2,0],[100,200,300]]), new = torch.tensor([[-0.11,0.15,0.01],[105,210,285]])`.
    $$NAAE = \\frac{\\text{Avg}(|0.01|,|0.05|,|-0.01|,|-5|,|-10|,|15|)}{\\text{Avg}(|-0.1|,|0.2|,|0|,|100|,|200|,|300|)} = 5\\%$$

    ## Parameters
    **ref** (*torch.Tensor*) and **new** (*torch.Tensor*) - Error will be calculated between these tensors.

    ## Returns
    **naae** (*torch scalar*) - In percentage.

    ## Caution
    NAAE is experimental, so it is best to stick with ANAE for now.
    """
    return 100.*torch.mean(torch.abs(ref-new))/torch.mean(torch.abs(ref))


def overall_anae(X, Y, Xr, Ypred, Xpred) -> dict[str, torch.Tensor]:
    """Computes overall ANAE for a model.
    
    ## Parameters
    - **X** (*torch.Tensor, shape=(\\*, input_size)*) - Input states, i.e. input to encoder.

    - **Y** (*torch.Tensor, shape=(\\*, encoded_size)*) - Encoded states, i.e. output from encoder, input to decoder.

    - **Xr** (*torch.Tensor, shape=(\\*, input_size)*) - Reconstructed states, i.e. output of decoder.

    - **Ypred** (*torch.Tensor, shape=(\\*, encoded_size)*) - Predicted encoded states obtained from evolving baseline encoded state.

    - **Xpred** (*torch.Tensor, shape=(\\*, input_size)*) - Predicted input states, which are predicted encoded states passed through decoder.

    ## Returns
    **anaes** (*dict[str, torch.Tensor]*)

    - Key **'recon'**: (*torch scalar*) - Reconstruction ANAE between `X` and `Xr`.
    - Key **'lin'**: (*torch scalar*) - Linearity ANAE between `Y` and `Ypred`.
    - Key **'pred'**: (*torch scalar*) - Prediction ANAE between `X` and `Xpred`.
    """
    return {
        'recon': anae(ref=X, new=Xr),
        'lin': anae(ref=Y, new=Ypred),
        'pred': anae(ref=X, new=Xpred)
    }


def overall_loss(X, Y, Xr, Ypred, Xpred, decoder_loss_weight) -> dict[str, torch.Tensor]:
    """Computes overall loss for a model.
    
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
