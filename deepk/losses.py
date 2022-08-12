import torch


def mse(ref, new):
    return torch.nn.MSELoss(reduction='mean')(ref, new)

def l1(ref, new):
    return torch.nn.L1Loss(reduction='mean')(ref, new)


def overall(X, Y, Xr, Ypred, Xpred, func=mse, decoder_loss_weight=0.01):
    """
    Inputs:
        X: Some original input data, shape = (num_samples,n). It should not include the sample x0.
        Y, Xr: Results of forward passing the original input data through self.net. Respective shapes = (num_samples,p), (num_samples,n).
        Ypred: Results of dmd_predict() using t = timesteps covered by X, Y, Xr. Shape = (num_samples,p).
        Xpred: Results of passing Ypred through the decoder. Shape = (num_samples,n).
    
    Returns:
    (For the following discussion, note that the Lusch paper uses x_1 as the initial sample, while I use x_0. So, I've replaced the 1 subscript in the Lusch paper with 0 in the explanation here).
        recon_loss:
            Lusch:
                In Lusch, this is given in the right column of Pg 4 as ||x - phi_inv(phi(x))||.
                In Lusch eq (12), x is assumed to be only the first sample x_0.
            Here:
                Instead of using a single sample x_i, I compute recon loss for all samples in the input data X and then average out.
                So, `torch.nn.MSELoss(reduction='mean')(Xr,X)` is the reconstruction loss.
        lin_loss:
            Lusch:
                In Lusch, this is given in the right column of Pg 4 as ||phi(x_(k+m)) - K^m*phi(x_k)||. (Btw do not confuse this `m` in Lusch with the `m` I'm using for total number of timestamps).
                In Lusch eq (14), k is assumed to be 0 to get ||phi(x_m) - K^m*phi(x_0)||, where x_m is m steps advanced from x_0. This is repeated for different m from 1 to T-1, and then averaged out.
            Here:
                Note that the X which is input to this method is already like a bunch of x_m since it contains samples advanced by different amounts of time from x_0. **Remember this for pred loss.**
                When we do `Y, Xr = self.net(X)`, Y_ideal becomes phi(x_m) for a bunch of m.
                Ypred is obtained from dmd_predict() as K^m*phi(x_0) for a bunch of m. **Remember this for pred loss.**
                So, `torch.nn.MSELoss(reduction='mean')(Ypred,Y)` is the linearity loss. The difference from Lusch is that instead of considering samples from x_1 to x_T-1 (which is all samples in the experiment time horizon T), we consider samples in the input data X, which can be e.g. from x_11 to x_15.
        pred_loss:
            Lusch:
                In Lusch, this is given in the right column of Pg 4 as ||x_(k+m) - phi_inv(K^m*phi(x_k))||,.
                In Lusch eq (13), k is assumed to be 0 to get ||x_m - phi_inv(K^m*phi(x_0))||, where x_m is m steps advanced from x_0. This is repeated for different m from 1 to Sp, and then averaged out.
            Here:
                When we do `Xpred = self.net.decoder(Ypred)`, we obtain `phi_inv(K^m*phi(X_0))` for a bunch of m.
                Then, `torch.nn.MSELoss(reduction='mean')(Xpred,X)` is the prediction loss. The difference from Lusch is that instead of considering samples from x_1 to x_Sp (where Sp is a hyperparameter), we consider samples in the input data, which can be e.g. from x_11 to x_15.
        loss: Linear combination of the above losses weighted by the hyperparameters.
    
    Note:
        We don't use L_infinity loss as in Lusch because we do not want to specifically penalize outliers since we don't know the nature of our data beforehand.
    """
    recon = func(ref=X, new=Xr)
    lin = func(ref=Y, new=Ypred)
    pred = func(ref=X, new=Xpred)
    total = lin + decoder_loss_weight*(recon+pred)
    return recon, lin, pred, total
