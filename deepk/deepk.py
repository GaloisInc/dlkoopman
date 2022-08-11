from collections import defaultdict
import os

import matplotlib.pyplot as plt
import numpy as np
import shortuuid
import torch
from torch import nn
from tqdm import tqdm

from config import *
import utils


#NOTE:
## Uncomment the following line if there's weird behavior and a more detailed analysis is desired.
## Uncommenting catches runtime errors such as exploding nan gradients and displays where they happened in the forward pass.
## However, uncommenting slows down execution.
# torch.autograd.set_detect_anomaly(True)


class MLP(nn.Module):
    """
    This can serve as the encoder and the decoder
    """
    def __init__(self, input_size, output_size, hidden_sizes=[], batch_norm=False):
        super().__init__()
        self.net = nn.ModuleList([])
        layers = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layers)-1):
            self.net.append(nn.Linear(layers[i],layers[i+1]))
            if i != len(layers)-2: #all layers except last
                if batch_norm:
                    self.net.append(nn.BatchNorm1d(layers[i+1]))
                self.net.append(nn.ReLU())

    def forward(self,X):
        for layer in self.net:
            X = layer(X)
        return X


class AutoEncoder(nn.Module):
    def __init__(self, num_input_states, num_encoded_states, encoder_hidden_layers=[], decoder_hidden_layers=[], batch_norm=False):
        """
        Inputs:
            num_input_states: Number of dimensions in original data (X) and reconstructed data (Xr).
                This is set automatically by the data.
            num_encoded_states: Number of dimensions in encoded data (Y)
            encoder_hidden_layers: Encoder has layers = [num_input_states, encoder_hidden_layers, num_encoded_states]
                If not set, defaults to reverse of decoder_hidden_layers. If that is also not set, defaults to []
            decoder_hidden_layers: Decoder has layers = [num_encoded_states, decoder_hidden_layers, num_input_states]
                If not set, defaults to reverse of encoder_hidden_layers. If that is also not set, defaults to []
            batch_norm: Whether to insert batch normalization layers
        forward() method returns:
            Y: Encoder output
            Xr: Decoder output
        """
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

    def forward(self,X):
        Y = self.encoder(X) # encoder complete output
        Xr = self.decoder(Y) # final reconstructed output
        return Y,Xr


class DeepKoopman:
    
    def __init__(self, data, hyps, misc):
        """
        data: Input data. Can have:
            - Xtr, Xva, Xte: 2d arrays for training, validation and test data. Each has shape = (num_samples_xx,num_dimensions), where xx = tr, va, te
            - ttr, tva, tte: Lists / 1D arrays for training, validation and test independent variables. Each has length = num_samples_xx, where xx = tr, va, te
        hyps: Training hyperparameters (all these have defaults, see the code):
            - Dictionary of arguments as required by AutoEncoder().
            - numepochs: Number of epochs for which to train neural net.
            - early_stopping: (NOTE: uses prediction ANAE as the performance metric)
                If False, no early stopping. The net will run for the complete `numepochs` epochs
                If an integer, early stop if val perf doesn't improve for that many epochs
                If a float, early stop if val perf doesn't improve for that fraction of epochs, rounded up
            - lr: Learning rate for neural net optimizer, e.g. Adam.
            - weight_decay: L2 coefficient for weights of the neural net.
            - decoder_loss_weight: Weight of any losses that go through the decoder (i.e. reconstruction and prediction loss) w.r.t to those that don't (i.e. linearity loss). Papers use a mix from 1e-1 to 1e-3.
            - K_reg: L1 penalty for the Ktilde matrix obtained from DMD.
            - rank: Rank to which the data matrices are truncated, which is also the size of Ktilde. Use 0 for full rank.
            - cond_threshold: Condition number of the eigenvector matrix W greater than this will be reported, and singular values smaller than this fraction of the largest will be ignored for the pinv.
            - use_custom_stable_svd: If set, use the custom implementation from utils.stable_svd instead of torch.linalg.svd, which can blow up if multiple singular values are close to each other.
            - clip_grad_norm: If not None, clip the norm of gradients to this.
            - clip_grad_value: If not None, clip the values of gradients to [-clip_grad_value,clip_grad_value].
            - normalize_Xdata: If True, Xtr, Xva, Xte are all divided by the max abs value in Xtr.
               Theory:
                    Normalizing data is a generally good technique for ML, and is normally done as X_col = (X_col-offset_col)/scale_col, where (offset_col,scale_col) = (mu_col,sigma_col) for Gaussian normalization, and (min_col,max_col-min_col) for Minmax normalization.
                    However, this messes up the spectral techniques such as SVD and EVD. This is why we just use a single scale value for the whole data to get X = X/scale. This results in the spectra (i.e. the singular and eigen vectors) remaining the same, the only change is that the singular and eigen values get divided by scale.
                    NOTE: Setting this to False may end the run by leading to a) NaNs in the gradient, or b) imaginary parts of tensors reaching values where the loss function depends on the phase (in torch >= 1.11, this leads to "RuntimeError: linalg_eig_backward: The eigenvectors in the complex case are specified up to multiplication by e^{i phi}. The specified loss function depends on this quantity, so it is ill-defined.") Thus, even though it makes more sense to report the final ANAE numbers on the un-normalized data values, don't try that.
        misc: Other options (all these have defaults, see the code):
            - results_folder: Folder where logs, output figs, etc will be stored. Will be created if it doesn't exist.
            - sigma_threshold: Any singular value of the original SVD (prior to truncation to r values) lower than this will be reported.
        """
        ## Define results folder
        self.results_folder = misc.get('results_folder', './')
        os.makedirs(self.results_folder, exist_ok=True)
        
        ## Define UUID for the run
        self.uuid = shortuuid.uuid()

        ## Define log file
        self.log_file = os.path.join(self.results_folder, f'{self.uuid}.log')
        lf = open(self.log_file, 'a')
        lf.write(f'Log file for Deep Koopman object {self.uuid}\n\n')
        
        ## Define sigma_threshold
        self.sigma_threshold = misc.get('sigma_threshold',1e-25)

        ## Get X data
        Xtr = data.get('Xtr')
        Xva = data.get('Xva')
        Xte = data.get('Xte')

        ## Convert X data to torch tensors on device
        self.Xtr = torch.as_tensor(Xtr, dtype=RTYPE, device=DEVICE) if Xtr is not None else torch.tensor([])
        self.Xva = torch.as_tensor(Xva, dtype=RTYPE, device=DEVICE) if Xva is not None else torch.tensor([])
        self.Xte = torch.as_tensor(Xte, dtype=RTYPE, device=DEVICE) if Xte is not None else torch.tensor([])

        ## Define Xscale, and normalize X data if applicable
        self.Xscale = torch.max(torch.abs(self.Xtr))
        self.normalize_Xdata = hyps.get('normalize_Xdata',True)
        if self.normalize_Xdata:
            self.Xtr /= self.Xscale
            self.Xva /= self.Xscale
            self.Xte /= self.Xscale

        ## Get t data
        ttr = data.get('ttr')
        tva = data.get('tva')
        tte = data.get('tte')

        ## Convert t data to torch tensors on device
        self.ttr = torch.as_tensor(ttr, dtype=RTYPE, device=DEVICE) if ttr is not None else torch.tensor([])
        self.tva = torch.as_tensor(tva, dtype=RTYPE, device=DEVICE) if tva is not None else torch.tensor([])
        self.tte = torch.as_tensor(tte, dtype=RTYPE, device=DEVICE) if tte is not None else torch.tensor([])

        ## Shift t data to make ttr start from 0 if it doesn't
        self.tshift = self.ttr[0].item()
        self.ttr -= self.tshift
        self.tva -= self.tshift
        self.tte -= self.tshift

        ## Find differences between ttr values
        dts = defaultdict(int)
        for i in range(len(self.ttr)-1):
            dts[self.ttr[i+1].item()-self.ttr[i].item()] += 1
        lf.write("Timestamp difference | Frequency\n")
        for dt,freq in dts.items():
            lf.write(f"{dt} | {freq}\n")
        
        ## Define t scale as most common difference between ttr values, and normalize t data by it
        self.tscale = max(dts, key=dts.get)
        lf.write(f"Using timestamp difference = {self.tscale}. Training timestamp values not on this grid will be rounded.\n")
        self.ttr /= self.tscale
        self.tva /= self.tscale
        self.tte /= self.tscale

        ## Ensure that ttr now goes as [0,1,2,...], i.e. no gaps
        self.ttr = torch.round(self.ttr)
        dts = set(self.ttr[i+1].item()-self.ttr[i].item() for i in range(len(self.ttr)-1))
        assert dts == {1}, 'Training timestamps are not equally spaced.'

        ## Get hyps of AutoEncoder
        self.num_encoded_states = hyps.get('num_encoded_states',50)
        self.encoder_hidden_layers = hyps.get('encoder_hidden_layers',[100])
        self.batch_norm = hyps.get('batch_norm',False)
        
        ## Get other hyps used in training
        self.numepochs = hyps.get('numepochs',500)
        self.lr = hyps.get('lr',1e-3)
        self.weight_decay = hyps.get('weight_decay',0.)
        self.decoder_loss_weight = hyps.get('decoder_loss_weight',0.01)
        self.K_reg = hyps.get('K_reg',0.01)
        self.cond_threshold = hyps.get('cond_threshold',100)
        self.use_custom_stable_svd = hyps.get('use_custom_stable_svd', True)
        self.clip_grad_norm = hyps.get('clip_grad_norm', None)
        self.clip_grad_value = hyps.get('clip_grad_value', None)

        ## Get rank and ensure it's a valid value
        self.rank = hyps.get('rank',4)
        full_rank = min(self.num_encoded_states,len(self.ttr)-1) #this is basically min(Y.shape), where Y is defined in dmd_linearize() and has shape (p,m). p is num_encoded_states and length of ttr is m+1.
        if self.rank==0 or self.rank>=full_rank:
            self.rank = full_rank

        ## Early stopping logic
        self.early_stopping = hyps.get('early_stopping', False)
        if type(self.early_stopping)==float:
            self.early_stopping = int(np.ceil(self.early_stopping * self.numepochs))

        ## Define AutoEncoder
        self.net = AutoEncoder(
            num_input_states=self.Xtr.shape[1],
            num_encoded_states=self.num_encoded_states,
            encoder_hidden_layers=self.encoder_hidden_layers,
            batch_norm=self.batch_norm
        )
        self.net.to(dtype=RTYPE, device=DEVICE) #for variables, we need `X = X.to()`. For net, only `net.to()` is sufficient

        ## Define training optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay) #papers either use SGD or Adam

        ## Define other attributes to be used later
        self.stats = defaultdict(lambda: [])
        self.Omega = None
        self.W = None
        self.coeffs = None

        ## Define error flag
        self.error_flag = False

        lf.close()


    def dmd_linearize(self, Y, Yprime):
        """
        Get the rank-reduced Koopman matrix Ktilde.

        Inputs:
            Y: Matrix of states as columns. Shape = (p,m), where p is the dimensionality of a state and m is the number of states.
            Yprime: Matrix of states as columns, shifted forward by 1 from Y. Shape = (p,m).
        
        Returns:
            Ktilde: Rank-reduced Koopman matrix. Shape = (r,r).
            interm: Intermediate quantity Y'VS^-1. Shape = (p,r).

        Note that r is <= min(m,p)
        Also note that wherever we do `tensor.t()`, technically the Hermitian should be taken via `tensor.t().conj()`. But since we deal with real data, just the ordinary transpose is fine.
        """
        if self.use_custom_stable_svd:
            U, Sigma, V = utils.stable_svd(Y) #shapes: U = (p,min(m,p)), Sigma = (min(m,p),), V = (m,min(m,p))
        else:
            U, Sigma, Vt = torch.linalg.svd(Y) #shapes: U = (p,p), Sigma = (min(m,p),), Vt = (m,m)
            V = Vt.t() #shape = (m,m)
        
        # Left singular vectors
        U = U[:,:self.rank] #shape = (p,r)
        Ut = U.t() #shape = (r,p)
        
        # Singular values
        if Sigma[-1] < self.sigma_threshold:
            with open(self.log_file, 'a') as lf:
                lf.write(f"Smallest singular value prior to truncation = {Sigma[-1]} is smaller than threshold = {self.sigma_threshold}. This may lead to very large / nan values in backprop of SVD.\n")
        #NOTE: We can also do a check if any pair (or more) of consecutive singular values are so close that the difference of their squares is less than some threshold, since this will also lead to very large / nan values during backprop. But doing this check requires looping over the Sigma tensor (time complexity = O(len(Sigma))) and is not worth it.
        Sigma = torch.diag(Sigma[:self.rank]) #shape = (r,r)
        
        # Right singular vectors
        V = V[:,:self.rank] #shape = (m,r)
        
        # Outputs
        interm = Yprime @ V @ torch.linalg.inv(Sigma) #shape = (p,r)
        Ktilde = Ut @ interm #shape = (r,r)
        
        return Ktilde, interm


    def dmd_eigen(self, Ktilde, interm, y0):
        """
        Get the eigendecomposition of the Koopman matrix.

        Inputs:
            Ktilde: From dmd_linearize(). Shape = (r,r).
            interm: From dmd_linearize(). Shape = (p,r).
            y0: Initial state of the system (should be part of training data). Shape = (p,).
        
        Returns:
            Omega: Eigenvalues of **continuous time** Ktilde. Shape = (r,r).
            W: First r exact eigenvectors of the full (i.e. not tilde) system. Shape = (p,r).
            coeffs: Coefficients of the linear combination. Shape = (r,).
        """
        Lambda, Wtilde = torch.linalg.eig(Ktilde) #Lambda is (r,), Wtilde is (r,r)
        
        # Eigenvalues
        with open(self.log_file, 'a') as lf:
            lf.write(f"Largest magnitude among eigenvalues = {torch.max(torch.abs(Lambda))}\n")
        Omega = torch.diag(torch.log(Lambda)) #shape = (r,r) #NOTE: No need to divide by self.tscale here because spacings are normalized to 1
        #NOTE: We can do a check if any pair (or more) of consecutive eigenvalues are so close that their difference is less than some threshold, since this will lead to very large / nan values during backprop. But doing this check requires looping over the Omega tensor (time complexity = O(len(Omega))) and is not worth it.
        
        # Eigenvectors
        W = interm.to(CTYPE) @ Wtilde #shape = (p,r)
        S,s = torch.linalg.norm(W,2), torch.linalg.norm(W,-2)
        cond = S/s
        if cond > self.cond_threshold:
            with open(self.log_file, 'a') as lf:
                lf.write(f"Condition number = {cond} is greater than threshold = {self.cond_threshold}. This may lead to numerical instability in evaluating linearity loss. In an attempt to mitigate this, singular values smaller than 1/{self.cond_threshold} times the largest will be discarded from the pseudo-inverse computation.\n")
        
        # Coefficients
        coeffs = torch.linalg.pinv(W, rtol=1./self.cond_threshold) @ y0.to(CTYPE) #shape = (r,)
        
        return Omega, W, coeffs


    def dmd_predict(self, t, Omega,W,coeffs):
        """
        Predict the dynamics of a system.
        
        Inputs:
            t: Sequence of time steps for which dynamics are predicted. Shape = (num_samples,).
            Omega: From dmd_eigen(). Shape = (r,r).
            W: From dmd_eigen(). Shape = (p,r).
            coeffs: From dmd_eigen(). Shape = (r,).
        Note: Omega,W,coeffs can also be provided from outside for a new system.
        
        Returns:
            Ypred: Predictions for all the timestamps in t. Shape (num_samples,p).
                Predictions may be complex, however they are converted to real by taking absolute value. This is fine because, if things are okay, the imaginary parts should be negligible (several orders of magnitude less) than the real parts.
        """
        Ypred = torch.tensor([])
        for index,ti in enumerate(t):
            ypred = W @ torch.linalg.matrix_exp(Omega*ti) @ coeffs #shape = (p,)
            Ypred = ypred.view(1,-1) if index==0 else torch.vstack((Ypred,ypred)) #shape at end = (num_samples,p)
        return torch.abs(Ypred)


    def find_losses(self, X, Y, Xr, Ypred, Xpred):
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
                    So, `nn.MSELoss(reduction='mean')(Xr,X)` is the reconstruction loss.
            lin_loss:
                Lusch:
                    In Lusch, this is given in the right column of Pg 4 as ||phi(x_(k+m)) - K^m*phi(x_k)||. (Btw do not confuse this `m` in Lusch with the `m` I'm using for total number of timestamps).
                    In Lusch eq (14), k is assumed to be 0 to get ||phi(x_m) - K^m*phi(x_0)||, where x_m is m steps advanced from x_0. This is repeated for different m from 1 to T-1, and then averaged out.
                Here:
                    Note that the X which is input to this method is already like a bunch of x_m since it contains samples advanced by different amounts of time from x_0. **Remember this for pred loss.**
                    When we do `Y, Xr = self.net(X)`, Y_ideal becomes phi(x_m) for a bunch of m.
                    Ypred is obtained from dmd_predict() as K^m*phi(x_0) for a bunch of m. **Remember this for pred loss.**
                    So, `nn.MSELoss(reduction='mean')(Ypred,Y)` is the linearity loss. The difference from Lusch is that instead of considering samples from x_1 to x_T-1 (which is all samples in the experiment time horizon T), we consider samples in the input data X, which can be e.g. from x_11 to x_15.
            pred_loss:
                Lusch:
                    In Lusch, this is given in the right column of Pg 4 as ||x_(k+m) - phi_inv(K^m*phi(x_k))||,.
                    In Lusch eq (13), k is assumed to be 0 to get ||x_m - phi_inv(K^m*phi(x_0))||, where x_m is m steps advanced from x_0. This is repeated for different m from 1 to Sp, and then averaged out.
                Here:
                    When we do `Xpred = self.net.decoder(Ypred)`, we obtain `phi_inv(K^m*phi(X_0))` for a bunch of m.
                    Then, `nn.MSELoss(reduction='mean')(Xpred,X)` is the prediction loss. The difference from Lusch is that instead of considering samples from x_1 to x_Sp (where Sp is a hyperparameter), we consider samples in the input data, which can be e.g. from x_11 to x_15.
            loss: Linear combination of the above losses weighted by the hyperparameters.
        
        Note:
            We don't use L_infinity loss as in Lusch because we do not want to specifically penalize outliers since we don't know the nature of our data beforehand.
        """
        recon_loss = nn.MSELoss(reduction='mean')(Xr, X)
        lin_loss = nn.MSELoss(reduction='mean')(Ypred, Y)
        pred_loss = nn.MSELoss(reduction='mean')(Xpred, X)

        loss = lin_loss + self.decoder_loss_weight*(recon_loss+pred_loss)

        return recon_loss, lin_loss, pred_loss, loss


    @staticmethod
    def anae(ref,new):
        """
        Average Normalized Absolute Error, returned as percentage.
        ANAE first normalizes each absolute error by the corresponding absolute ground truth, then averages them.
        E.g: ref = [[0.1,1],[100,200]], new = [[1.1,2],[99,199]]. So, all deviations are 1. Then, ANAE = Avg(1/0.1, 1/1, 1/100, 1/200) = 275%. Note that ANAE heavily penalizes deviations for small values of ground truth. 
        """
        anae = torch.abs(ref-new)/torch.abs(ref)
        return 100.*torch.mean(anae[anae!=torch.inf])

    def find_anaes(self, X, Y, Xr, Ypred, Xpred):
        """
        Find the corresponding ANAE for each of the losses.
        NOTE: MSE loss is a useful thing to optimize, but it is basically the squares of the values. ANAE is a more useful thing to report since it tells us how much percentage deviation to expect for a new value. E.g. if prediction ANAE for a DeepK motor current problem is around 10%, then one can expect a newly predicted motor current to have an error of around 10% from the actual.
        """
        recon_anae = self.anae(ref=X, new=Xr)
        lin_anae = self.anae(ref=Y, new=Ypred)
        pred_anae = self.anae(ref=X, new=Xpred)
        return recon_anae, lin_anae, pred_anae
    
    @staticmethod
    def naae(ref,new):
        """
        Normalized Average Absolute Error, returned as percentage.
        NAAE first averages the absolute error, then normalizes that using the average value of the absolute ground truth. This reduces the chance of a few small ground truth values having a huge impact, but loses details by taking the average over all absolute error values, and likewise over all ground truth values.
        E.g.: ref = [[0.1,1],[-100,200]], new = [[1.1,2],[-99,199]]. So, all deviations are 1. Then, NAAE = Avg(1,1,1,1)/Avg(0.1,1,100,200) = 1.33%.
        """
        return 100.*torch.mean(torch.abs(ref-new))/torch.mean(torch.abs(ref))
    
    def find_naaes(self, X, Y, Xr, Ypred, Xpred):
        """ Find the corresponding NAAE for each of the losses """
        recon_naae = self.naae(ref=X, new=Xr)
        lin_naae = self.naae(ref=Y, new=Ypred)
        pred_naae = self.naae(ref=X, new=Xpred)
        return recon_naae, lin_naae, pred_naae


    def train(self):
        """
        Train a Deep Koopman model.
        This computes losses and the eigendecomposition.
        If validation data exists in self, then the eigendecomposition at the end of training each epoch is used to do validation after every epoch and compute losses.
        If test data exists in self, the final eigendecomposition is also used to do testing and compute losses.

        Effects:
            self.stats stores all the training (and val and test if those are done) stats.
            self.Omega, self.W and self.coeffs contain the eigendecomposition at the end of training. These can be used for future evaluation.
        
        Inputs: None
        Returns: None
        """
        print(f'UUID for this run = {self.uuid}')
        with open(self.log_file, 'a') as lf:
            lf.write("\nStarting training ...\n")
        
        no_improvement_epochs_count = 0
        best_val_perf = np.inf
        
        for epoch in tqdm(range(self.numepochs)):
            # NOTE: Do not do any kind of shuffling before/after epochs
            # The num_input_states dimension cannot be shuffled because that would mess up the MLP weights
            # The m dimension is typically shuffled for standard classification problems, here however, m captures time information, so order must be maintained
            
            with open(self.log_file, 'a') as lf:
                lf.write(f"\nEpoch {epoch+1}\n")
            
            ## Training ##
            self.net.train()
            self.opt.zero_grad()
            
            Ytr, Xrtr = self.net(self.Xtr)
            
            # Following 2 steps are unique to training
            Ktilde, interm = self.dmd_linearize(Y = Ytr[:-1,:].t(), Yprime = Ytr[1:,:].t())
            Omega, W, coeffs = self.dmd_eigen(Ktilde=Ktilde, interm=interm, y0=Ytr[0])

            # Record DMD variables
            self.Omega = Omega
            self.W = W
            self.coeffs = coeffs
            
            # Get predictions
            Ypredtr = self.dmd_predict(t=self.ttr[1:], Omega=Omega, W=W, coeffs=coeffs)
            Xpredtr = self.net.decoder(Ypredtr)

            # ANAEs
            with torch.no_grad():
                recon_anae_tr, lin_anae_tr, pred_anae_tr = self.find_anaes(X=self.Xtr[1:], Y=Ytr[1:], Xr=Xrtr[1:], Ypred=Ypredtr, Xpred=Xpredtr)
            self.stats['recon_anae_tr'].append(recon_anae_tr.item())
            self.stats['lin_anae_tr'].append(lin_anae_tr.item())
            self.stats['pred_anae_tr'].append(pred_anae_tr.item())
            
            # Losses
            recon_loss_tr, lin_loss_tr, pred_loss_tr, loss_tr = self.find_losses(X=self.Xtr[1:], Y=Ytr[1:], Xr=Xrtr[1:], Ypred=Ypredtr, Xpred=Xpredtr)
            self.stats['recon_loss_tr'].append(recon_loss_tr.item())
            self.stats['lin_loss_tr'].append(lin_loss_tr.item())
            self.stats['pred_loss_tr'].append(pred_loss_tr.item())
            self.stats['loss_tr_before_K_reg'].append(loss_tr.item())

            ## Following steps – adding K regularization – are unique to training
            K_reg_loss_tr = self.K_reg*torch.sum(torch.abs(Ktilde))/torch.numel(Ktilde)
            loss_tr += K_reg_loss_tr
            self.stats['loss_tr'].append(loss_tr.item())

            with open(self.log_file, 'a') as lf:
                lf.write(f"Training ANAEs: Reconstruction = {recon_anae_tr.item()}, Linearity = {lin_anae_tr.item()}, Prediction = {pred_anae_tr.item()}\nTraining losses: Reconstruction = {recon_loss_tr.item()}, Linearity = {lin_loss_tr.item()}, Prediction = {pred_loss_tr.item()}, Due to K_reg = {K_reg_loss_tr.item()}, Total = {loss_tr.item()}\n")
            
            # Backprop
            try:
                loss_tr.backward()
            except RuntimeError as e:
                self.error_flag = True
                message = f"Encountered RuntimeError: {e}\nStopping training!\n"
                with open(self.log_file, 'a') as lf:
                    lf.write(message)
                print(message)
                break
            
            try:
                utils.paramgrads_check_nan(self.net.parameters())
            except ValueError:
                self.error_flag = True
                message = "Encountered NaN in gradients\nStopping training!\n"
                with open(self.log_file, 'a') as lf:
                    lf.write(message)
                print(message)
                break
            
            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad_norm)
            if self.clip_grad_value:
                nn.utils.clip_grad_value_(self.net.parameters(), self.clip_grad_value)
            
            # Update
            self.opt.step()
    
            ## Validation ##
            if len(self.Xva):
                self.net.eval()
                with torch.no_grad():
                    Yva, Xrva = self.net(self.Xva)
                    Ypredva = self.dmd_predict(t=self.tva, Omega=Omega, W=W, coeffs=coeffs)
                    Xpredva = self.net.decoder(Ypredva)
                    recon_anae_va, lin_anae_va, pred_anae_va = self.find_anaes(X=self.Xva, Y=Yva, Xr=Xrva, Ypred=Ypredva, Xpred=Xpredva)
                    recon_loss_va, lin_loss_va, pred_loss_va, loss_va = self.find_losses(X=self.Xva, Y=Yva, Xr=Xrva, Ypred=Ypredva, Xpred=Xpredva)
                    #Note that for evaluation, there is no K_reg. This is because K_reg is only used for training.
                    
                self.stats['recon_anae_va'].append(recon_anae_va.item())
                self.stats['lin_anae_va'].append(lin_anae_va.item())
                self.stats['pred_anae_va'].append(pred_anae_va.item())
                self.stats['recon_loss_va'].append(recon_loss_va.item())
                self.stats['lin_loss_va'].append(lin_loss_va.item())
                self.stats['pred_loss_va'].append(pred_loss_va.item())
                self.stats['loss_va'].append(loss_va.item())

                with open(self.log_file, 'a') as lf:
                    lf.write(f"Validation ANAEs: Reconstruction = {recon_anae_va.item()}, Linearity = {lin_anae_va.item()}, Prediction = {pred_anae_va.item()}\nValidation losses: Reconstruction = {recon_loss_va.item()}, Linearity = {lin_loss_va.item()}, Prediction = {pred_loss_va.item()}, Total = {loss_va.item()}\n")

                ## Early stopping logic
                if self.early_stopping:
                    if pred_anae_va.item() < best_val_perf:
                        no_improvement_epochs_count = 0
                        best_val_perf = pred_anae_va.item()
                    else:
                        no_improvement_epochs_count += 1
                    if no_improvement_epochs_count == self.early_stopping:
                        with open(self.log_file, 'a') as lf:
                            lf.write(f"\nEarly stopped due to no improvement for {self.early_stopping} epochs.")
                        break


    def test(self):
        self.net.eval()
        with torch.no_grad():
            Yte, Xrte = self.net(self.Xte)
            Ypredte = self.dmd_predict(t=self.tte, Omega=self.Omega, W=self.W, coeffs=self.coeffs)
            Xpredte = self.net.decoder(Ypredte)
            recon_anae_te, lin_anae_te, pred_anae_te = self.find_anaes(X=self.Xte, Y=Yte, Xr=Xrte, Ypred=Ypredte, Xpred=Xpredte)
            recon_loss_te, lin_loss_te, pred_loss_te, loss_te = self.find_losses(X=self.Xte, Y=Yte, Xr=Xrte, Ypred=Ypredte, Xpred=Xpredte)
            
        self.stats['recon_anae_te'] = recon_anae_te.item()
        self.stats['lin_anae_te'] = lin_anae_te.item()
        self.stats['pred_anae_te'] = pred_anae_te.item()
        self.stats['recon_loss_te'] = recon_loss_te.item()
        self.stats['lin_loss_te'] = lin_loss_te.item()
        self.stats['pred_loss_te'] = pred_loss_te.item()
        self.stats['loss_te'] = loss_te.item()

        with open(self.log_file, 'a') as lf:
            lf.write(f"\nTest ANAEs: Reconstruction = {recon_anae_te.item()}, Linearity = {lin_anae_te.item()}, Prediction = {pred_anae_te.item()}\nTest losses: Reconstruction = {recon_loss_te.item()}, Linearity = {lin_loss_te.item()}, Prediction = {pred_loss_te.item()}, Total = {loss_te.item()}\n")


    def predict_new(self, t):
        """
        Use the trained Deep Koopman model to predict the X values for new t.
            This is different from testing because there we have the ground truth Xte values and we use those to find loss. Here, we do not know the X values because the t values are new (e.g. they can be extrapolated in either direction (forward / backward) or interpolated between the existing t values used for training / val / test).

        Inputs:
            t: List / 1D array containing new timestamps. Shape = (num_samples,).
        
        Returns:
            Xpred: Tensor with predictions for the new X values. Shape = (num_samples,n)
        """
        t_processed = (torch.as_tensor(t, dtype=RTYPE, device=DEVICE) - self.tshift) / self.tscale
        with torch.no_grad():
            Ypred = self.dmd_predict(t=t_processed, Omega=self.Omega, W=self.W, coeffs=self.coeffs)
            Xpred = self.net.decoder(Ypred)
            if self.normalize_Xdata:
                Xpred *= self.Xscale
        with open(self.log_file, 'a') as lf:
            lf.write("\nNew predictions:\n")
            for i in range(len(t)):
                lf.write(f'Independent variable = {t[i]}, Dependent variable =\n')
                lf.write(f'{Xpred[i]}\n')
        return Xpred


    def plot_stats(self, perfs = ['pred_anae'], start_epoch=1, fontsize=12):
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

            tr_data = self.stats[perf+'_tr'][start_epoch-1:]
            if self.stats[perf+'_va']:
                va_data = self.stats[perf+'_va'][start_epoch-1:]
                tr_data = tr_data[:len(va_data)] # va_data should normally have size equal to tr_data, but will have lesser size if some error occurred during training. This slicing ensures that only that portion of tr_data is considered which corresponds to va_data.
            epoch_range = range(start_epoch,start_epoch+len(tr_data))
            
            plt.figure()
            if self.stats[perf+'_te']:
                plt.suptitle(f"Test performance = {self.stats[perf+'_te']}" + (' %' if is_anae else ''), fontsize=fontsize)
            
            if perf == 'loss':
                plt.plot(epoch_range,self.stats['loss_tr_before_K_reg'][start_epoch-1:], c='DarkSlateBlue', label='Training, before K_reg')
            plt.plot(epoch_range,self.stats[perf+'_tr'][start_epoch-1:], c='MediumBlue', label='Training')
            if self.stats[perf+'_va']:
                plt.plot(epoch_range,self.stats[perf+'_va'][start_epoch-1:], c='DeepPink', label='Validation')
            
            if is_anae:
                ylim_anae = plt.gca().get_ylim()
                plt.ylim(max(0,ylim_anae[0]), min(100,ylim_anae[1])) # keep ANAE limits between [0,100]
            plt.xlabel('Epochs', fontsize=fontsize)
            plt.ylabel(perf + (' (%)' if is_anae else ''), fontsize=fontsize)
            plt.xticks(fontsize=fontsize-2)
            plt.yticks(fontsize=fontsize-2)
            
            plt.grid()
            plt.legend(fontsize=fontsize)

            plt.savefig(os.path.join(self.results_folder, f'{self.uuid}_{perf}.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)
