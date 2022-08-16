"""Error functions used to report how well a DeepKoopman neural net has performed in a more human-readable way than loss."""


import torch


def anae(ref, new) -> torch.Tensor:
    """Average Normalized Absolute Error (ANAE).
    
    ANAE first normalizes each absolute deviation by the corresponding absolute ground truth, then averages them. This is a useful thing to report since it tells us how much percentage deviation to expect for a new value. E.g. if prediction ANAE for a problem is around 10%, then one can expect a newly predicted value to have an error of around 10% from the actual.
    
    Example: Let `ref = torch.tensor([[0.1,1],[100,200]]), new = torch.tensor([[1.1,2],[99,199]])`.
    $$ANAE = \\text{Avg}\\left(\\frac{|0.1-1.1|}{0.1}, \\frac{|1-2|}{1}, \\frac{|100-99|}{100}, \\frac{|200-199|}{200}\\right) = 275\\%$$
    Note that ANAE heavily penalizes deviations for small values of ground truth.

    ## Parameters
    **ref** (*torch.Tensor*) and **new** (*torch.Tensor*) - Error will be calculated between these tensors.

    ## Returns
    **anae** (*torch scalar*) - In percentage.
    """
    anae = torch.abs(ref-new)/torch.abs(ref)
    return 100.*torch.mean(anae[anae!=torch.inf])

def naae(ref, new) -> torch.Tensor:
    """Normalized Average Absolute Error (NAAE).
    
    NAAE first averages the absolute error, then normalizes that using the average value of the absolute ground truth. This reduces the chance of a few small ground truth values having a huge impact, but loses details by taking the average over all absolute error values, and likewise over all ground truth values.
    
    Example: Let `ref = torch.tensor([[0.1,1],[-100,200]]), new = torch.tensor([[1.1,2],[-99,199]])`.
    $$NAAE = \\frac{\\text{Avg}(1,1,1,1)}{\\text{Avg}(0.1,1,100,200)} = 1.33\\%$$

    ## Parameters
    **ref** (*torch.Tensor*) and **new** (*torch.Tensor*) - Error will be calculated between these tensors.

    ## Returns
    **naae** (*torch scalar*) - In percentage.

    ## Caution
    NAAE is experimental, so it is best to stick with ANAE for now.
    """
    return 100.*torch.mean(torch.abs(ref-new))/torch.mean(torch.abs(ref))


def overall(X, Y, Xr, Ypred, Xpred, func=anae) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes overall error for the DeepK neural net.
    
    ## Parameters
    - **X** (*torch.Tensor, shape=(\\*,num_input_states)*) - Input data.

    - **Y** (*torch.Tensor, shape=(\\*,num_encoded_states)*) - Encoded data, i.e. output from encoder, input to decoder.

    - **Xr** (*torch.Tensor, shape=(\\*,num_input_states)*) - Reconstructed data, i.e. output of decoder.

    - **Ypred** (*torch.Tensor, shape=(\\*,num_encoded_states)*) - DeepK-predicted encoded data.

    - **Xpred** (*torch.Tensor, shape=(\\*,num_input_states)*) - DeepK-predicted encoded data passed through decoder.

    - **func** (*function*) - Which error function to use. See error functions above.

    ## Returns
    - **recon** - Reconstruction error between `X` and `Xr`.

    - **lin** - Linearity error between `Y` and `Ypred`.

    - **pred** - Prediction error between `X` and `Xpred`.
    """
    recon = func(ref=X, new=Xr)
    lin = func(ref=Y, new=Ypred)
    pred = func(ref=X, new=Xpred)
    return recon, lin, pred
