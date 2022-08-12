import torch


def anae(ref, new):
    """
    Average Normalized Absolute Error, returned as percentage.
    ANAE first normalizes each absolute error by the corresponding absolute ground truth, then averages them.
    E.g: ref = [[0.1,1],[100,200]], new = [[1.1,2],[99,199]]. So, all deviations are 1. Then, ANAE = Avg(1/0.1, 1/1, 1/100, 1/200) = 275%. Note that ANAE heavily penalizes deviations for small values of ground truth. 
    """
    anae = torch.abs(ref-new)/torch.abs(ref)
    return 100.*torch.mean(anae[anae!=torch.inf])

def naae(ref, new):
    """
    Normalized Average Absolute Error, returned as percentage.
    NAAE first averages the absolute error, then normalizes that using the average value of the absolute ground truth. This reduces the chance of a few small ground truth values having a huge impact, but loses details by taking the average over all absolute error values, and likewise over all ground truth values.
    E.g.: ref = [[0.1,1],[-100,200]], new = [[1.1,2],[-99,199]]. So, all deviations are 1. Then, NAAE = Avg(1,1,1,1)/Avg(0.1,1,100,200) = 1.33%.
    """
    return 100.*torch.mean(torch.abs(ref-new))/torch.mean(torch.abs(ref))


def overall(X, Y, Xr, Ypred, Xpred, func=anae):
    """
    Find the corresponding ANAE for each of the losses.
    NOTE: MSE loss is a useful thing to optimize, but it is basically the squares of the values. ANAE is a more useful thing to report since it tells us how much percentage deviation to expect for a new value. E.g. if prediction ANAE for a DeepK motor current problem is around 10%, then one can expect a newly predicted motor current to have an error of around 10% from the actual.
    """
    recon = func(ref=X, new=Xr)
    lin = func(ref=Y, new=Ypred)
    pred = func(ref=X, new=Xpred)
    return recon, lin, pred
