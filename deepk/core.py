from collections import defaultdict
import os

import numpy as np
import shortuuid
import torch
from tqdm import tqdm

from deepk import config as cfg
from deepk.subnets import AutoEncoder
from deepk import utils, losses, errors


#NOTE:
## Uncomment the following line if there's weird behavior and a more detailed analysis is desired.
## Uncommenting catches runtime errors such as exploding nan gradients and displays where they happened in the forward pass.
## However, uncommenting slows down execution.
# torch.autograd.set_detect_anomaly(True)


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
        self.Xtr = torch.as_tensor(Xtr, dtype=cfg.RTYPE, device=cfg.DEVICE) if Xtr is not None else torch.tensor([])
        self.Xva = torch.as_tensor(Xva, dtype=cfg.RTYPE, device=cfg.DEVICE) if Xva is not None else torch.tensor([])
        self.Xte = torch.as_tensor(Xte, dtype=cfg.RTYPE, device=cfg.DEVICE) if Xte is not None else torch.tensor([])

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
        self.ttr = torch.as_tensor(ttr, dtype=cfg.RTYPE, device=cfg.DEVICE) if ttr is not None else torch.tensor([])
        self.tva = torch.as_tensor(tva, dtype=cfg.RTYPE, device=cfg.DEVICE) if tva is not None else torch.tensor([])
        self.tte = torch.as_tensor(tte, dtype=cfg.RTYPE, device=cfg.DEVICE) if tte is not None else torch.tensor([])

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
        self.net.to(dtype=cfg.RTYPE, device=cfg.DEVICE) #for variables, we need `X = X.to()`. For net, only `net.to()` is sufficient

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
        W = interm.to(cfg.CTYPE) @ Wtilde #shape = (p,r)
        S,s = torch.linalg.norm(W,2), torch.linalg.norm(W,-2)
        cond = S/s
        if cond > self.cond_threshold:
            with open(self.log_file, 'a') as lf:
                lf.write(f"Condition number = {cond} is greater than threshold = {self.cond_threshold}. This may lead to numerical instability in evaluating linearity loss. In an attempt to mitigate this, singular values smaller than 1/{self.cond_threshold} times the largest will be discarded from the pseudo-inverse computation.\n")
        
        # Coefficients
        coeffs = torch.linalg.pinv(W, rtol=1./self.cond_threshold) @ y0.to(cfg.CTYPE) #shape = (r,)
        
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


    def train_net(self):
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
                recon_anae_tr, lin_anae_tr, pred_anae_tr = errors.overall(X=self.Xtr[1:], Y=Ytr[1:], Xr=Xrtr[1:], Ypred=Ypredtr, Xpred=Xpredtr)
            self.stats['recon_anae_tr'].append(recon_anae_tr.item())
            self.stats['lin_anae_tr'].append(lin_anae_tr.item())
            self.stats['pred_anae_tr'].append(pred_anae_tr.item())
            
            # Losses
            recon_loss_tr, lin_loss_tr, pred_loss_tr, loss_tr = losses.overall(X=self.Xtr[1:], Y=Ytr[1:], Xr=Xrtr[1:], Ypred=Ypredtr, Xpred=Xpredtr, decoder_loss_weight=self.decoder_loss_weight)
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
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad_norm)
            if self.clip_grad_value:
                torch.nn.utils.clip_grad_value_(self.net.parameters(), self.clip_grad_value)
            
            # Update
            self.opt.step()
    
            ## Validation ##
            if len(self.Xva):
                self.net.eval()
                with torch.no_grad():
                    Yva, Xrva = self.net(self.Xva)
                    Ypredva = self.dmd_predict(t=self.tva, Omega=Omega, W=W, coeffs=coeffs)
                    Xpredva = self.net.decoder(Ypredva)
                    recon_anae_va, lin_anae_va, pred_anae_va = errors.overall(X=self.Xva, Y=Yva, Xr=Xrva, Ypred=Ypredva, Xpred=Xpredva)
                    recon_loss_va, lin_loss_va, pred_loss_va, loss_va = losses.overall(X=self.Xva, Y=Yva, Xr=Xrva, Ypred=Ypredva, Xpred=Xpredva, decoder_loss_weight=self.decoder_loss_weight)
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


    def test_net(self):
        self.net.eval()
        with torch.no_grad():
            Yte, Xrte = self.net(self.Xte)
            Ypredte = self.dmd_predict(t=self.tte, Omega=self.Omega, W=self.W, coeffs=self.coeffs)
            Xpredte = self.net.decoder(Ypredte)
            recon_anae_te, lin_anae_te, pred_anae_te = errors.overall(X=self.Xte, Y=Yte, Xr=Xrte, Ypred=Ypredte, Xpred=Xpredte)
            recon_loss_te, lin_loss_te, pred_loss_te, loss_te = losses.overall(X=self.Xte, Y=Yte, Xr=Xrte, Ypred=Ypredte, Xpred=Xpredte, decoder_loss_weight=self.decoder_loss_weight)
            
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
        t_processed = (torch.as_tensor(t, dtype=cfg.RTYPE, device=cfg.DEVICE) - self.tshift) / self.tscale
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
