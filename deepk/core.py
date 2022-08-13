"""**The core DeepKoopman class**""" 


from collections import defaultdict
import os

import numpy as np
import shortuuid
import torch
from tqdm import tqdm

from deepk import config as cfg
from deepk import utils, losses, errors


class DeepKoopman:
    """Main DeepKoopman class.

    ## Parameters
    - **data** (*dict[str,torch.Tensor]*)
        - Key **'Xtr'**: Training data (*torch.Tensor, shape=(num_training_samples, num_input_states)*).
        - Key **'Xva'** (*optional*): Validation data (*torch.Tensor, shape=(num_validation_samples, num_input_states)*).
        - Key **'Xte'** (*optional*): Test data (*torch.Tensor, shape=(num_test_samples, num_input_states)*).
        - Key **'ttr'**: Training indices (*torch.Tensor, shape=(num_training_samples,)*). **Must be in ascending order.**
        - Key **'tva'** (*optional*): Validation indices (*torch.Tensor, shape=(num_validation_samples,)*).
        - Key **'tte'** (*optional*): Test indices (*torch.Tensor, shape=(num_test_samples,)*).
    
    - **rank** (*int*) - Rank of DeepK operation. Use 0 for full rank. Good values are 4-10.
    
    - Parameters required by [utils.AutoEncoder](TODO):
        - **num_encoded_states** (*int*)
        - **encoder_hidden_layers** (*list[int], optional*)
        - **decoder_hidden_layers** (*list[int], optional*)
        - **batch_norm** (*bool, optional*)
    
    - **numepochs** (*int, optional*) - Number of epochs for which to train neural net.
    
    - **early_stopping** (*bool/int/float, optional*) - Whether to terminate training early due to no improvement in validation metric.
        - If `False`, no early stopping. The net will run for the complete `numepochs` epochs.
        - If an integer, early stop if validation metric doesn't improve for that many epochs.
        - If a float, early stop if validation performance doesn't improve for that fraction of epochs, rounded up.
    - **early_stopping_metric** (*str, optional*) - Which metric to use for early stopping (metrics are given below in the documentation for `stats`). Irrelevant if `early_stopping=False`.
    
    - **lr** (*float, optional*) - Learning rate for neural net optimizer.
    
    - **weight_decay** (*float, optional*) - L2 coefficient for weights of the neural net.
    
    - **decoder_loss_weight** (*float, optional*) - Weight the losses between decoder outputs (`recon` and `pred`) by this number. This is to account for the scaling effect of the decoder.
    
    - **K_reg** (*float, optional*) - L1 penalty for the Ktilde matrix.
    
    - **cond_threshold** (*float, optional*) - Condition number of the eigenvector matrix greater than this will be reported, and singular values smaller than this fraction of the largest will be ignored for the pseudo-inverse operation.
    
    - **clip_grad_norm** (*float, optional*) - If not None, clip the norm of gradients to this value.
    - **clip_grad_value** (*float, optional*) - If not None, clip the values of gradients to [-`clip_grad_value`,`clip_grad_value`].
    
    - **results_folder** (*str, optional*) - Save results to this folder.

    - **verbose** (*bool, optional*) - Trigger verbose output.

    ## Attributes
    - **uuid** (*str*) - Unique ID for this object. Results are prefixed with `uuid`.

    - **RTYPE**, **CTYPE**, **DEVICE** - Torch tensor attributes - Real data precision, complex data precision, and device.

    - **Xscale** (*torch scalar*) - Max of absolute value of `Xtr`. All X data is scaled by `Xscale`.
    - **tshift** (*float*) - First value of `ttr`. All t data is shifted backwards by `tshift` so that `ttr` starts from 0.
    - **tscale** (*float*) - All t data is scaled by `tscale` so that `ttr` becomes `[0,1,...]`
    
    - **net** (*utils.AutoEncoder*) - DeepKoopman neural network. `num_input_states` is set by num_input_states in X data, while other parameters are passed via this class.
    - **opt** (*torch optimizer*) - Optimizer used to train DeepKoopman neural net.

    - **stats** (*dict[list]*) - Stores different metrics like loss and error values for training and validation data for each epoch, and for test data.

    - **Omega** (*torch.Tensor*), **W** (*torch.Tensor*), **coeffs** (*torch.Tensor*) - Used to make predictions using a trained DeepKoopman model.

    - **error_flag** (*bool*) - Signals if any error has occurred in training.
    """
    
    def __init__(
        self, data, rank,
        num_encoded_states, encoder_hidden_layers=[], decoder_hidden_layers=[], batch_norm=False,
        numepochs=500, early_stopping=False, early_stopping_metric='pred_anae',
        lr=1e-3, weight_decay=0., decoder_loss_weight=1e-2, K_reg=1e-3,
        cond_threshold=100., clip_grad_norm=None, clip_grad_value=None,
        results_folder='./', verbose=True
    ):
        ## Define precision and device
        self.RTYPE = torch.float16 if cfg.precision=="half" else torch.float32 if cfg.precision=="float" else torch.float64
        self.CTYPE = torch.complex32 if cfg.precision=="half" else torch.complex64 if cfg.precision=="float" else torch.complex128
        self.DEVICE = torch.device("cuda" if cfg.use_cuda and torch.cuda.is_available() else "cpu")
        
        ## Define verbose, results folder, UUID, and log file
        self.verbose = verbose

        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)

        self.uuid = shortuuid.uuid()
        print(f'UUID for this run = {self.uuid}')
        
        self.log_file = os.path.join(results_folder, f'{self.uuid}.log')
        lf = open(self.log_file, 'a')
        lf.write(f'Log file for Deep Koopman object {self.uuid}\n\n')

        ## Get X data
        Xtr = data.get('Xtr')
        Xva = data.get('Xva')
        Xte = data.get('Xte')

        ## Convert X data to torch tensors on device
        self.Xtr = torch.as_tensor(Xtr, dtype=self.RTYPE, device=self.DEVICE) if Xtr is not None else torch.tensor([])
        self.Xva = torch.as_tensor(Xva, dtype=self.RTYPE, device=self.DEVICE) if Xva is not None else torch.tensor([])
        self.Xte = torch.as_tensor(Xte, dtype=self.RTYPE, device=self.DEVICE) if Xte is not None else torch.tensor([])

        ## Define Xscale, and normalize X data if applicable
        self.Xscale = torch.max(torch.abs(self.Xtr))
        if cfg.normalize_Xdata:
            self.Xtr /= self.Xscale
            self.Xva /= self.Xscale
            self.Xte /= self.Xscale

        ## Get t data
        ttr = data.get('ttr')
        tva = data.get('tva')
        tte = data.get('tte')

        ## Convert t data to torch tensors on device
        self.ttr = torch.as_tensor(ttr, dtype=self.RTYPE, device=self.DEVICE) if ttr is not None else torch.tensor([])
        self.tva = torch.as_tensor(tva, dtype=self.RTYPE, device=self.DEVICE) if tva is not None else torch.tensor([])
        self.tte = torch.as_tensor(tte, dtype=self.RTYPE, device=self.DEVICE) if tte is not None else torch.tensor([])

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
        self.num_encoded_states = num_encoded_states
        self.encoder_hidden_layers = encoder_hidden_layers
        self.decoder_hidden_layers = decoder_hidden_layers
        self.batch_norm = batch_norm
        
        ## Get other hyps used in training
        self.numepochs = numepochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.decoder_loss_weight = decoder_loss_weight
        self.K_reg = K_reg
        self.cond_threshold = cond_threshold
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        ## Get rank and ensure it's a valid value
        self.rank = rank
        full_rank = min(self.num_encoded_states,len(self.ttr)-1) #this is basically min(Y.shape), where Y is defined in _dmd_linearize() and has shape (p,m). p is num_encoded_states and length of ttr is m+1.
        if self.rank==0 or self.rank>=full_rank:
            self.rank = full_rank

        ## Early stopping logic
        self.early_stopping = early_stopping
        if type(self.early_stopping)==float:
            self.early_stopping = int(np.ceil(self.early_stopping * self.numepochs))
        self.early_stopping_metric = early_stopping_metric+'_va'

        ## Define AutoEncoder
        self.net = utils.AutoEncoder(
            num_input_states=self.Xtr.shape[1],
            num_encoded_states=self.num_encoded_states,
            encoder_hidden_layers=self.encoder_hidden_layers,
            batch_norm=self.batch_norm
        )
        self.net.to(dtype=self.RTYPE, device=self.DEVICE) #for variables, we need `X = X.to()`. For net, only `net.to()` is sufficient

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


    def _dmd_linearize(self, Y, Yprime):
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
        if cfg.use_custom_stable_svd:
            U, Sigma, V = utils.stable_svd(Y) #shapes: U = (p,min(m,p)), Sigma = (min(m,p),), V = (m,min(m,p))
        else:
            U, Sigma, Vt = torch.linalg.svd(Y) #shapes: U = (p,p), Sigma = (min(m,p),), Vt = (m,m)
            V = Vt.t() #shape = (m,m)
        
        # Left singular vectors
        U = U[:,:self.rank] #shape = (p,r)
        Ut = U.t() #shape = (r,p)
        
        # Singular values
        if self.verbose and Sigma[-1] < cfg.sigma_threshold:
            with open(self.log_file, 'a') as lf:
                lf.write(f"Smallest singular value prior to truncation = {Sigma[-1]} is smaller than threshold = {cfg.sigma_threshold}. This may lead to very large / nan values in backprop of SVD.\n")
        #NOTE: We can also do a check if any pair (or more) of consecutive singular values are so close that the difference of their squares is less than some threshold, since this will also lead to very large / nan values during backprop. But doing this check requires looping over the Sigma tensor (time complexity = O(len(Sigma))) and is not worth it.
        Sigma = torch.diag(Sigma[:self.rank]) #shape = (r,r)
        
        # Right singular vectors
        V = V[:,:self.rank] #shape = (m,r)
        
        # Outputs
        interm = Yprime @ V @ torch.linalg.inv(Sigma) #shape = (p,r)
        Ktilde = Ut @ interm #shape = (r,r)
        
        return Ktilde, interm


    def _dmd_eigen(self, Ktilde, interm, y0):
        """
        Get the eigendecomposition of the Koopman matrix.

        Inputs:
            Ktilde: From _dmd_linearize(). Shape = (r,r).
            interm: From _dmd_linearize(). Shape = (p,r).
            y0: Initial state of the system (should be part of training data). Shape = (p,).
        
        Returns:
            Omega: Eigenvalues of **continuous time** Ktilde. Shape = (r,r).
            W: First r exact eigenvectors of the full (i.e. not tilde) system. Shape = (p,r).
            coeffs: Coefficients of the linear combination. Shape = (r,).
        """
        Lambda, Wtilde = torch.linalg.eig(Ktilde) #Lambda is (r,), Wtilde is (r,r)
        
        # Eigenvalues
        if self.verbose:
            with open(self.log_file, 'a') as lf:
                lf.write(f"Largest magnitude among eigenvalues = {torch.max(torch.abs(Lambda))}\n")
        Omega = torch.diag(torch.log(Lambda)) #shape = (r,r) #NOTE: No need to divide by self.tscale here because spacings are normalized to 1
        #NOTE: We can do a check if any pair (or more) of consecutive eigenvalues are so close that their difference is less than some threshold, since this will lead to very large / nan values during backprop. But doing this check requires looping over the Omega tensor (time complexity = O(len(Omega))) and is not worth it.
        
        # Eigenvectors
        W = interm.to(self.CTYPE) @ Wtilde #shape = (p,r)
        S,s = torch.linalg.norm(W,2), torch.linalg.norm(W,-2)
        cond = S/s
        if self.verbose and cond > self.cond_threshold:
            with open(self.log_file, 'a') as lf:
                lf.write(f"Condition number = {cond} is greater than threshold = {self.cond_threshold}. This may lead to numerical instability in evaluating linearity loss. In an attempt to mitigate this, singular values smaller than 1/{self.cond_threshold} times the largest will be discarded from the pseudo-inverse computation.\n")
        
        # Coefficients
        coeffs = torch.linalg.pinv(W, rtol=1./self.cond_threshold) @ y0.to(self.CTYPE) #shape = (r,)
        
        return Omega, W, coeffs


    def _dmd_predict(self, t, Omega,W,coeffs):
        """
        Predict the dynamics of a system.
        
        Inputs:
            t: Sequence of time steps for which dynamics are predicted. Shape = (num_samples,).
            Omega: From _dmd_eigen(). Shape = (r,r).
            W: From _dmd_eigen(). Shape = (p,r).
            coeffs: From _dmd_eigen(). Shape = (r,).
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
        """Train (and optionally validate) the Deep Koopman model.

        ## Effects
        - self.stats is populated.
        - self.Omega, self.W and self.coeffs contain the eigendecomposition at the end of training. These can be used for future evaluation.
        """
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
            Ktilde, interm = self._dmd_linearize(Y = Ytr[:-1,:].t(), Yprime = Ytr[1:,:].t())
            Omega, W, coeffs = self._dmd_eigen(Ktilde=Ktilde, interm=interm, y0=Ytr[0])

            # Record DMD variables
            self.Omega = Omega
            self.W = W
            self.coeffs = coeffs
            
            # Get predictions
            Ypredtr = self._dmd_predict(t=self.ttr[1:], Omega=Omega, W=W, coeffs=coeffs)
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
                for parameter in self.net.parameters():
                    if torch.any(torch.isnan(parameter.grad)).item():
                        raise ValueError
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
                    Ypredva = self._dmd_predict(t=self.tva, Omega=Omega, W=W, coeffs=coeffs)
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
                    if self.stats[self.early_stopping_metric][-1] < best_val_perf:
                        no_improvement_epochs_count = 0
                        best_val_perf = self.stats[self.early_stopping_metric][-1]
                    else:
                        no_improvement_epochs_count += 1
                    if no_improvement_epochs_count == self.early_stopping:
                        with open(self.log_file, 'a') as lf:
                            lf.write(f"\nEarly stopped due to no improvement in {self.early_stopping_metric} for {self.early_stopping} epochs.")
                        break


    def test_net(self):
        """Run the trained DeepKoopman model on test data.
        
        ## Effects
        - self.stats is populated further.
        """
        self.net.eval()
        with torch.no_grad():
            Yte, Xrte = self.net(self.Xte)
            Ypredte = self._dmd_predict(t=self.tte, Omega=self.Omega, W=self.W, coeffs=self.coeffs)
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


    def predict_new(self, t) -> torch.Tensor:
        """Use the trained Deep Koopman model to predict the X values for new indices.
        
        This is different from testing because the ground truth values are not present, thus losses and errors are not computed. This method can be used to make predictions for interpolated and extrapolated indices.

        ## Parameters
        **t** (*list[int/float]*) - Array containing new indices.

        ## Returns
        **Xpred** (*torch.Tensor, shape=(len(t),num_input_states)*) - Predicted X values for the new indices.
        """
        t_processed = (torch.as_tensor(t, dtype=self.RTYPE, device=self.DEVICE) - self.tshift) / self.tscale
        with torch.no_grad():
            Ypred = self._dmd_predict(t=t_processed, Omega=self.Omega, W=self.W, coeffs=self.coeffs)
            Xpred = self.net.decoder(Ypred)
            if cfg.normalize_Xdata:
                Xpred *= self.Xscale
        with open(self.log_file, 'a') as lf:
            lf.write("\nNew predictions:\n")
            for i in range(len(t)):
                lf.write(f'Independent variable = {t[i]}, Dependent variable =\n')
                lf.write(f'{Xpred[i]}\n')
        return Xpred
