"""**The core DeepKoopman class**""" 


from collections import defaultdict
import numpy as np
from pathlib import Path
import shortuuid
import torch
from tqdm import tqdm

from deepk import config as cfg
from deepk import utils, losses, errors, nets

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class DeepKoopman:
    """Main DeepKoopman class.

    ## Parameters
    
    - **rank** (*int*) - Rank of DeepK operation. Use 0 for full rank. Will be set to `min(num_encoded_states, num_training_samples-1)` if the provided value is greater.
    
    - Parameters required by [AutoEncoder](https://galoisinc.github.io/deep-koopman/nets.html#deepk.nets.AutoEncoder):
        - **num_encoded_states** (*int*).
        
        - **encoder_hidden_layers** (*list[int], optional*).
        
        - **decoder_hidden_layers** (*list[int], optional*).
        
        - **batch_norm** (*bool, optional*).
    
    - **numepochs** (*int, optional*) - Number of epochs for which to train neural net.
    
    - **early_stopping** (*bool/int/float, optional*) - Whether to terminate training early due to no improvement in validation metric.
        - If `False`, no early stopping. The net will run for the complete `numepochs` epochs.
        - If an integer, early stop if validation metric doesn't improve for that many epochs.
        - If a float, early stop if validation performance doesn't improve for that fraction of epochs, rounded up.
    - **early_stopping_metric** (*str, optional*) - Which validation metric to use for early stopping (metrics are given below in the documentation for `stats`). Ignored if `early_stopping=False`.
    
    - **lr** (*float, optional*) - Learning rate for neural net optimizer.
    
    - **weight_decay** (*float, optional*) - L2 coefficient for weights of the neural net.
    
    - **decoder_loss_weight** (*float, optional*) - Weight the losses between decoder outputs (`recon` and `pred`) by this number. This is to account for the scaling effect of the decoder.
    
    - **K_reg** (*float, optional*) - L1 penalty for the Ktilde matrix.
    
    - **cond_threshold** (*float, optional*) - Condition number of the eigenvector matrix greater than this will be reported, and singular values smaller than this fraction of the largest will be ignored for the pseudo-inverse operation.
    
    - **clip_grad_norm** (*float, optional*) - If not None, clip the norm of gradients to this value.
    - **clip_grad_value** (*float, optional*) - If not None, clip the values of gradients to [-`clip_grad_value`,`clip_grad_value`].

    ## Attributes
    - **uuid** (*str*) - Unique ID for this Deep Koopman run.
    - **log_file** (*Path*) - Path to log file for this Deep Koopman run = `./dk_<uuid>.log`.
    
    - **net** (*nets.AutoEncoder*) - DeepKoopman neural network. `num_input_states` is set by num_input_states in X data, while other parameters are passed via this class.
    - **opt** (*torch optimizer*) - Optimizer used to train DeepKoopman neural net.

    - **Omega** (*torch.Tensor*), **eigvecs** (*torch.Tensor*), **coeffs** (*torch.Tensor*) - Used to make predictions using a trained DeepKoopman model.

    - **error_flag** (*bool*) - Signals if any error has occurred in training.

    - **stats** (*dict[list]*) - Stores different metrics like loss and error values for training and validation data for each epoch, and for test data. Possible stats are `'<recon/lin/pred>_<loss/anae>_<tr/va/te>'`, and `'loss_<tr/va/te>'`.
    """
    
    def __init__(
        self, dh, rank,
        num_encoded_states, encoder_hidden_layers=[100], decoder_hidden_layers=[], batch_norm=False,
        numepochs=500, early_stopping=False, early_stopping_metric='pred_anae',
        lr=1e-3, weight_decay=0., decoder_loss_weight=1e-2, K_reg=1e-3,
        cond_threshold=100., clip_grad_norm=None, clip_grad_value=None
    ):
        self.dh = dh
        
        ## Define UUID and log file
        self.uuid = shortuuid.uuid()
        self.log_file = Path(f'./dk_{self.uuid}.log').resolve()
        print(f'Deep Koopman log file = {self.log_file}')

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
        full_rank = min(self.num_encoded_states,len(self.dh.ttr)-1) #this is basically min(Y.shape), where Y is defined in _dmd_linearize()
        self.rank = full_rank if (rank==0 or rank>=full_rank) else rank

        ## Early stopping logic
        if early_stopping:
            EARLY_STOPPING_METRIC_CHOICES = ['recon_loss', 'lin_loss', 'pred_loss', 'loss', 'recon_anae', 'lin_anae', 'pred_anae']
            assert early_stopping_metric in EARLY_STOPPING_METRIC_CHOICES, f"`early_stopping_metric` must be in {EARLY_STOPPING_METRIC_CHOICES}, instead found '{early_stopping_metric}'"
        self.early_stopping = int(np.ceil(early_stopping * self.numepochs)) if type(early_stopping)==float else early_stopping
        self.early_stopping_metric = early_stopping_metric + ('_va' if not early_stopping_metric.endswith('va') else '')

        ## Define AutoEncoder
        self.net = nets.AutoEncoder(
            num_input_states=self.dh.Xtr.shape[1],
            num_encoded_states=self.num_encoded_states,
            encoder_hidden_layers=self.encoder_hidden_layers,
            batch_norm=self.batch_norm
        )
        self.net.to(dtype=cfg._RTYPE, device=cfg._DEVICE) #for variables, we need `X = X.to()`. For net, only `net.to()` is sufficient.

        ## Define training optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        ## Define other attributes to be used later
        self.stats = defaultdict(lambda: [])
        self.Omega = None
        self.eigvecs = None
        self.coeffs = None

        ## Define error flag
        self.error_flag = False

        ## Finally, if no errors have occurred, write info to log file
        with open(self.log_file, 'a') as lf:
            lf.write("Timestamp difference | Frequency\n")
            for dt,freq in self.dh.dts.items():
                lf.write(f"{dt} | {freq}\n")
            lf.write(f"Using timestamp difference = {self.dh.tscale}. Training timestamp values not on this grid will be rounded.\n")


    def _dmd_linearize(self, Y, Yprime) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the rank-reduced Koopman matrix Ktilde.

        Inputs:
            Y: Matrix of states as columns. Contains all states except for last. Shape = (num_encoded_states, num_training_samples-1).
            Yprime: Matrix of states as columns. Shifted forward by 1 from Y, i.e. contains all states except for first. Shape = (num_encoded_states, num_training_samples-1).
        
        Returns:
            Ktilde: Rank-reduced Koopman matrix. Shape = (rank, rank).
            interm: Intermediate quantity, which is Y'VS^-1 for exact eignvectors, or U for projected eigenvectors. Shape = (num_encoded_states, rank).

        Note that wherever we do `tensor.t()`, technically the Hermitian should be taken via `tensor.t().conj()`. But since we deal with real data, just the ordinary transpose is fine.
        """
        if cfg.use_custom_stable_svd:
            U, Sigma, V = utils.stable_svd(Y) #shapes: U = (num_encoded_states, min(num_training_samples-1,num_encoded_states)), Sigma = (min(num_training_samples-1,num_encoded_states),), V = (num_training_samples-1, min(num_training_samples-1,num_encoded_states))
        else:
            U, Sigma, Vt = torch.linalg.svd(Y) #shapes: U = (num_encoded_states, num_encoded_states), Sigma = (min(num_training_samples-1,num_encoded_states),), Vt = (num_training_samples-1, num_training_samples-1)
            V = Vt.t() #shape = (num_training_samples-1, num_training_samples-1)
        
        # Left singular vectors
        U = U[:,:self.rank] #shape = (num_encoded_states, rank)
        Ut = U.t() #shape = (rank, num_encoded_states)
        
        # Singular values
        if Sigma[-1] < cfg.sigma_threshold:
            with open(self.log_file, 'a') as lf:
                lf.write(f"Smallest singular value prior to truncation = {Sigma[-1]} is smaller than threshold = {cfg.sigma_threshold}. This may lead to very large / nan values in backprop of SVD.\n")
        #NOTE: We can also do a check if any pair (or more) of consecutive singular values are so close that the difference of their squares is less than some threshold, since this will also lead to very large / nan values during backprop. But doing this check requires looping over the Sigma tensor (time complexity = O(len(Sigma))) and is not worth it.
        Sigma = torch.diag(Sigma[:self.rank]) #shape = (rank, rank)
        
        # Right singular vectors
        V = V[:,:self.rank] #shape = (num_training_samples-1, rank)
        
        # Outputs
        interm = Yprime @ V @ torch.linalg.inv(Sigma) #shape = (num_encoded_states, rank)
        Ktilde = Ut @ interm #shape = (rank, rank)

        if not cfg.use_exact_eigenvectors:
            interm = U
        
        return Ktilde, interm


    def _dmd_eigen(self, Ktilde, interm, y0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the eigendecomposition of the Koopman matrix.

        Inputs:
            Ktilde: From _dmd_linearize(). Shape = (rank, rank).
            interm: From _dmd_linearize(). Shape = (num_encoded_states, rank).
            y0: Initial state of the system (should be part of training data). Shape = (num_encoded_states,).
        
        Returns:
            Omega: Eigenvalues of **continuous time** Ktilde. Shape = (rank, rank).
            eigvecs: First 'rank' exact eigenvectors of the full (i.e. not tilde) system. Shape = (num_encoded_states, rank).
            coeffs: Coefficients of the linear combination. Shape = (rank,).
        """
        Lambda, Wtilde = torch.linalg.eig(Ktilde) #shapes: Lambda = (rank,), Wtilde is (rank, rank)
        
        # Eigenvalues
        with open(self.log_file, 'a') as lf:
            lf.write(f"Largest magnitude among eigenvalues = {torch.max(torch.abs(Lambda))}\n")
        Omega = torch.diag(torch.log(Lambda)) #shape = (rank, rank) #NOTE: No need to divide by self.dh.tscale here because spacings are normalized to 1
        #NOTE: We can do a check if any pair (or more) of consecutive eigenvalues are so close that their difference is less than some threshold, since this will lead to very large / nan values during backprop. But doing this check requires looping over the Omega tensor (time complexity = O(len(Omega))) and is not worth it.
        
        # Eigenvectors
        eigvecs = interm.to(cfg._CTYPE) @ Wtilde #shape = (num_encoded_states, rank)
        S,s = torch.linalg.norm(eigvecs,2), torch.linalg.norm(eigvecs,-2)
        cond = S/s
        if cond > self.cond_threshold:
            with open(self.log_file, 'a') as lf:
                lf.write(f"Condition number = {cond} is greater than threshold = {self.cond_threshold}. This may lead to numerical instability in evaluating linearity loss. In an attempt to mitigate this, singular values smaller than 1/{self.cond_threshold} times the largest will be discarded from the pseudo-inverse computation.\n")
        
        # Coefficients
        coeffs = torch.linalg.pinv(eigvecs, rtol=1./self.cond_threshold) @ y0.to(cfg._CTYPE) #shape = (rank,)
        
        return Omega, eigvecs, coeffs


    def _dmd_predict(self, t, Omega,eigvecs,coeffs) -> torch.Tensor:
        """
        Predict the dynamics of a system.
        
        Inputs:
            t: Sequence of time steps for which dynamics are predicted. Shape = (num_samples,).
            Omega: From _dmd_eigen(). Shape = (rank, rank).
            eigvecs: From _dmd_eigen(). Shape = (num_encoded_states, rank).
            coeffs: From _dmd_eigen(). Shape = (rank,).
        Note: Omega,eigvecs,coeffs can also be provided from outside for a new system.
        
        Returns:
            Ypred: Predictions for all the timestamps in t. Shape (num_samples, num_encoded_states).
                Predictions may be complex, however they are converted to real by taking absolute value. This is fine because, if things are okay, the imaginary parts should be negligible (several orders of magnitude less) than the real parts.
        """
        Ypred = torch.tensor([])
        for index,ti in enumerate(t):
            ypred = eigvecs @ torch.linalg.matrix_exp(Omega*ti) @ coeffs #shape = (num_encoded_states,)
            Ypred = ypred.view(1,-1) if index==0 else torch.vstack((Ypred,ypred)) #shape at end = (num_samples, num_encoded_states)
        return torch.abs(Ypred)


    def train_net(self):
        """Train (and optionally validate) the Deep Koopman model.

        ## Effects
        - self.stats is populated.
        - self.Omega, self.eigvecs and self.coeffs contain the eigendecomposition at the end of training. These can be used for future evaluation.
        """
        with open(self.log_file, 'a') as lf:
            lf.write("\nStarting training ...\n")
        
        no_improvement_epochs_count = 0
        best_val_perf = np.inf
        
        for epoch in tqdm(range(self.numepochs)):
            # NOTE: Do not do any kind of shuffling before/after epochs. The samples dimension is typically shuffled for standard classification problems, but here that corresponds to the index (such as time), which should be ordered.
            
            with open(self.log_file, 'a') as lf:
                lf.write(f"\nEpoch {epoch+1}\n")
            
            ## Training ##
            self.net.train()
            self.opt.zero_grad()
            
            Ytr, Xrtr = self.net(self.dh.Xtr)
            
            # Following 2 steps are unique to training
            Ktilde, interm = self._dmd_linearize(Y = Ytr[:-1,:].t(), Yprime = Ytr[1:,:].t())
            Omega, eigvecs, coeffs = self._dmd_eigen(Ktilde=Ktilde, interm=interm, y0=Ytr[0])

            # Record DMD variables
            self.Omega = Omega
            self.eigvecs = eigvecs
            self.coeffs = coeffs
            
            # Get predictions
            Ypredtr = self._dmd_predict(t=self.dh.ttr[1:], Omega=Omega, eigvecs=eigvecs, coeffs=coeffs)
            Xpredtr = self.net.decoder(Ypredtr)

            # ANAEs
            with torch.no_grad():
                recon_anae_tr, lin_anae_tr, pred_anae_tr = errors.overall(X=self.dh.Xtr[1:], Y=Ytr[1:], Xr=Xrtr[1:], Ypred=Ypredtr, Xpred=Xpredtr)
            self.stats['recon_anae_tr'].append(recon_anae_tr.item())
            self.stats['lin_anae_tr'].append(lin_anae_tr.item())
            self.stats['pred_anae_tr'].append(pred_anae_tr.item())
            
            # Losses
            recon_loss_tr, lin_loss_tr, pred_loss_tr, loss_tr = losses.overall(X=self.dh.Xtr[1:], Y=Ytr[1:], Xr=Xrtr[1:], Ypred=Ypredtr, Xpred=Xpredtr, decoder_loss_weight=self.decoder_loss_weight)
            self.stats['recon_loss_tr'].append(recon_loss_tr.item())
            self.stats['lin_loss_tr'].append(lin_loss_tr.item())
            self.stats['pred_loss_tr'].append(pred_loss_tr.item())
            self.stats['loss_before_K_reg_tr'].append(loss_tr.item())

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
            if len(self.dh.Xva):
                self.net.eval()
                with torch.no_grad():
                    Yva, Xrva = self.net(self.dh.Xva)
                    Ypredva = self._dmd_predict(t=self.dh.tva, Omega=Omega, eigvecs=eigvecs, coeffs=coeffs)
                    Xpredva = self.net.decoder(Ypredva)
                    recon_anae_va, lin_anae_va, pred_anae_va = errors.overall(X=self.dh.Xva, Y=Yva, Xr=Xrva, Ypred=Ypredva, Xpred=Xpredva)
                    recon_loss_va, lin_loss_va, pred_loss_va, loss_va = losses.overall(X=self.dh.Xva, Y=Yva, Xr=Xrva, Ypred=Ypredva, Xpred=Xpredva, decoder_loss_weight=self.decoder_loss_weight)
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
            Yte, Xrte = self.net(self.dh.Xte)
            Ypredte = self._dmd_predict(t=self.dh.tte, Omega=self.Omega, eigvecs=self.eigvecs, coeffs=self.coeffs)
            Xpredte = self.net.decoder(Ypredte)
            recon_anae_te, lin_anae_te, pred_anae_te = errors.overall(X=self.dh.Xte, Y=Yte, Xr=Xrte, Ypred=Ypredte, Xpred=Xpredte)
            recon_loss_te, lin_loss_te, pred_loss_te, loss_te = losses.overall(X=self.dh.Xte, Y=Yte, Xr=Xrte, Ypred=Ypredte, Xpred=Xpredte, decoder_loss_weight=self.decoder_loss_weight)
            
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
        **Xpred** (*torch.Tensor, shape=(len(t), num_input_states)*) - Predicted X values for the new indices.
        """
        _t = utils._tensorize(t, dtype=cfg._RTYPE, device=cfg._DEVICE)
        _t = utils._shift(_t, shift=self.dh.tshift)
        _t = utils._scale(_t, scale=self.dh.tscale)
        with torch.no_grad():
            Ypred = self._dmd_predict(t=_t, Omega=self.Omega, eigvecs=self.eigvecs, coeffs=self.coeffs)
            Xpred = self.net.decoder(Ypred)
            if cfg.normalize_Xdata:
                Xpred = utils._scale(Xpred, scale=1/self.dh.Xscale) # inverse operation, hence 1/
        with open(self.log_file, 'a') as lf:
            lf.write("\nNew predictions:\n")
            for i in range(len(t)):
                lf.write(f'Independent variable = {t[i]}, Dependent variable =\n')
                lf.write(f'{Xpred[i]}\n')
        return Xpred
