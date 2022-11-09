"""**State Predictor**.

`StatePred` can be used to train on given states of a system at given indexes, then predict unknown states of the system at new indexes. See a specific example and tutorial [here](https://github.com/GaloisInc/dlkoopman/blob/main/examples/state_pred_naca0012/run.ipynb).
"""


from collections import defaultdict
import numpy as np
from pathlib import Path
import shortuuid
import torch
from tqdm import tqdm

from dlkoopman import config as cfg
from dlkoopman import utils, metrics, nets

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


__pdoc__ = {
    'StatePred.decoder_loss_weight': False,
    'StatePred.cond_threshold': False,
    'StatePred.Omega': False,
    'StatePred.eigvecs': False,
    'StatePred.y0': False
}


class StatePredDataHandler:
    """State predictor data handler. Used to provide data to train (and optionally validate and test) the `StatePred` model.

    ## Parameters
    - **'Xtr'** (*Array[float], shape=(num_training_indexes, input_size)*) - Input states to be used as training data. *Array* can be any data type such as *numpy.array*, *torch.Tensor*, *list* etc.

    - **'ttr'** (*Array[int,float], shape=(num_training_indexes,)*) - Indexes of the states in `Xtr`. *Array* can be any data type such as *numpy.array*, *torch.Tensor*, *list*, *range*, etc.
        - **`ttr` must be in ascending order and should ideally be equally spaced**. The 1st value of `ttr` will be used as the *baseline index*.
        - Small deviations are okay, e.g. `[100, 203, 298, 400, 500]` will become `[100, 200, 300, 400, 500]`, but larger deviations that cannot be unambiguously rounded will lead to errors.

    - **'Xva'** (*Array[float], shape=(num_validation_indexes, input_size), optional*) - Input states to be used as validation data. Same data type requirements as `Xtr`.

    - **'tva'** (*Array[int,float], shape=(num_validation_indexes,), optional*): Indexes of the states in `Xva`. Same data type requirements as `ttr`.
        - The order and spacing restrictions on `ttr` do *not* apply. The values of these indexes can be anything.
        
    - **'Xte'** (*Array[float], shape=(num_test_indexes, input_size), optional*) - Input states to be used as test data. Same data type requirements as `Xtr`.

    - **'tte'** (*Array[int,float], shape=(num_test_indexes,), optional*): Indexes of the states in `Xte`. Same data type requirements as `ttr`.
        - The order and spacing restrictions on `ttr` do *not* apply. The values of these indexes can be anything.

    ## Example
    ```python
    # Provide data of a system with 3-dimensional states (i.e. input_size=3)
    # Provide data at 4 indexes for training, and 2 indexes each for validation and testing

    dh = StatePredDataHandler(
        ttr = [100, 203, 298, 400], # ascending order, (almost) equally spaced
        Xtr = numpy.array([
            [0.7, 2.1, 9.2], # state at index 100
            [1.1, 5. , 6.1], # state at index 203
            [4.3, 2. , 7.3], # state at index 298
            [6.1, 4.2, 0.3]  # state at index 400
        ]),
        tva = [66, 238], # anything
        Xva = numpy.array([
            [3.8, 1.7, 0.4], # state at index 66
            [7.6, 6.5, 9.3]  # state at index 238
        ]),
        tte = [-32, 784], # anything
        Xte = numpy.array([
            [9.8, 8.9, 0.2], # state at index -32
            [7.3, 4.8, 7.5]  # state at index 784
        ])
    )
    ```
    """

    def __init__(self, Xtr, ttr, Xva=None, tva=None, Xte=None, tte=None):
        self.Xtr = utils._tensorize(Xtr, dtype=cfg._RTYPE, device=cfg._DEVICE)
        self.Xva = utils._tensorize(Xva, dtype=cfg._RTYPE, device=cfg._DEVICE)
        self.Xte = utils._tensorize(Xte, dtype=cfg._RTYPE, device=cfg._DEVICE)
        self.ttr = utils._tensorize(ttr, dtype=cfg._RTYPE, device=cfg._DEVICE)
        self.tva = utils._tensorize(tva, dtype=cfg._RTYPE, device=cfg._DEVICE)
        self.tte = utils._tensorize(tte, dtype=cfg._RTYPE, device=cfg._DEVICE)

        ## Check sizes
        assert len(self.Xtr) == len(self.ttr), f"Expected 'Xtr' and 'ttr' to have same length of 1st dimension, instead found {len(self.Xtr)} and {len(self.ttr)}"
        assert len(self.Xva) == len(self.tva), f"Expected 'Xva' and 'tva' to have same length of 1st dimension, instead found {len(self.Xva)} and {len(self.tva)}"
        assert len(self.Xte) == len(self.tte), f"Expected 'Xte' and 'tte' to have same length of 1st dimension, instead found {len(self.Xte)} and {len(self.tte)}"

        ## Define Xscale, and normalize X data if applicable
        self.Xscale = torch.max(torch.abs(self.Xtr)).item()
        if cfg.normalize_Xdata:
            self.Xtr = utils._scale(self.Xtr, scale=self.Xscale)
            self.Xva = utils._scale(self.Xva, scale=self.Xscale)
            self.Xte = utils._scale(self.Xte, scale=self.Xscale)

        ## Shift t data to make ttr start from 0 if it doesn't
        self.tshift = self.ttr[0].item()
        self.ttr = utils._shift(self.ttr, shift=self.tshift)
        self.tva = utils._shift(self.tva, shift=self.tshift)
        self.tte = utils._shift(self.tte, shift=self.tshift)

        ## Find differences between ttr values
        self.dts = defaultdict(int) # define this as a class attribute because it will be accessed outside for writing to log file
        for i in range(len(self.ttr)-1):
            self.dts[self.ttr[i+1].item()-self.ttr[i].item()] += 1

        ## Define t scale as most common difference between ttr values, and normalize t data by it
        self.tscale = max(self.dts, key=self.dts.get)
        self.ttr = utils._scale(self.ttr, scale=self.tscale)
        self.tva = utils._scale(self.tva, scale=self.tscale)
        self.tte = utils._scale(self.tte, scale=self.tscale)

        ## Ensure that ttr now goes as [0,1,2,...], i.e. no gaps
        self.ttr = torch.round(self.ttr)
        dts_set = set(self.ttr[i+1].item()-self.ttr[i].item() for i in range(len(self.ttr)-1))
        if dts_set != {1}:
            raise ValueError(f"Training indexes are not equally spaced and cannot be rounded to get equal spacing. Please check 'ttr' = {ttr}")


class StatePred:
    """State predictor. Used to train on given states of a system at given indexes, then predict unknown states of the system at new indexes.

    ## Parameters
    - **dh** (*StatePredDataHandler*) - Data handler that feeds data.

    - **rank** (*int*) - Rank of SVD operation to compute Koopman matrix. Use `0` for full rank. Will be set to min(`encoded_size`, `num_training_indexes`-1) if the provided value is greater.

    - Parameters for [AutoEncoder](https://galoisinc.github.io/dlkoopman/nets.html#dlkoopman.nets.AutoEncoder):
        - **encoded_size** (*int*).

        - **encoder_hidden_layers** (*list[int], optional*).

        - **decoder_hidden_layers** (*list[int], optional*).

        - **batch_norm** (*bool, optional*).

    ## Attributes
    - **uuid** (*str*) - Unique ID assigned to this instance. Results will include `uuid` in their filename.
    - **log_file** (*Path*) - Path to log file = `./log_<uuid>.log`.

    - **input_size** (*int*) - Dimensionality of input states. Inferred from `dh.Xtr`.
    - **encoded_size** (*int*) - Dimensionality of encoded states. As given in input.

    - **ae** (*nets.AutoEncoder*) - AutoEncoder neural network to encode input states into a linearizable domain where the Koopman matrix can be learnt, then decode them back into original domain.

    - **Omega** (*torch.Tensor*), **eigvecs** (*torch.Tensor*) - Eigenvalues, and eigenvectors of the trained Koopman matrix that characterizes the continuous index system \\(\\frac{dy}{di} = Ky_i\\). These are used to make predictions.
    - **y0** (*torch.Tensor*) - Encoded state at the baseline index, which is evolved using `Omega` and `eigvecs` to get predictions for any index.

    - **stats** (*dict[list]*) - Stores different metrics from training and testing. Useful for checking performance and [plotting](https://galoisinc.github.io/dlkoopman/utils.html#dlkoopman.utils.plot_stats).

    - **error_flag** (*bool*) - Signals if any error has occurred in training.
    """
    
    def __init__(self,
        dh, rank, encoded_size,
        encoder_hidden_layers=[100], decoder_hidden_layers=[], batch_norm=False
    ):
        ## Define UUID and log file
        self.uuid = shortuuid.uuid()
        self.log_file = Path(f'./log_{self.uuid}.log').resolve()
        print(f'Log file = {self.log_file}')

        ## Get data handler and sizes
        self.dh = dh
        self.input_size = self.dh.Xtr.shape[1]
        self.encoded_size = encoded_size

        ## Define AutoEncoder
        self.ae = nets.AutoEncoder(
            input_size = self.input_size,
            encoded_size = self.encoded_size,
            encoder_hidden_layers = encoder_hidden_layers,
            decoder_hidden_layers = decoder_hidden_layers,
            batch_norm = batch_norm
        )
        self.ae.to(dtype=cfg._RTYPE, device=cfg._DEVICE)

        ## Get rank and ensure it's a valid value
        full_rank = min(self.encoded_size,len(self.dh.ttr)-1) #this is basically min(Y.shape), where Y is defined in _dmd_linearize()
        self.rank = full_rank if (rank==0 or rank>=full_rank) else rank

        ## Define results
        self.stats = defaultdict(list)

        ## Define error flag
        self.error_flag = False

        ## Define other attributes to be used later (getter/setters should exist for each of these)
        self.decoder_loss_weight = None
        self.cond_threshold = None
        self.Omega = None
        self.eigvecs = None
        self.y0 = None

        ## Write info to log file
        with open(self.log_file, 'a') as lf:
            lf.write("Index difference | Frequency\n")
            for dt,freq in self.dh.dts.items():
                lf.write(f"{dt} | {freq}\n")
            lf.write(f"Using index difference = {self.dh.tscale}. Training index values not on this grid will be rounded.\n")


    @property
    def decoder_loss_weight(self):
        if self._decoder_loss_weight is None:
            raise ValueError("'decoder_loss_weight' is not set. Please call 'train_net()' first.")
        return self._decoder_loss_weight

    @decoder_loss_weight.setter
    def decoder_loss_weight(self, val):
        self._decoder_loss_weight = val

    @property
    def cond_threshold(self):
        if self._cond_threshold is None:
            raise ValueError("'cond_threshold' is not set. Please call 'train_net()' first.")
        return self._cond_threshold

    @cond_threshold.setter
    def cond_threshold(self, val):
        self._cond_threshold = val

    @property
    def Omega(self):
        if self._Omega is None:
            raise ValueError("'Omega' is not set. Please call 'train_net()' first.")
        return self._Omega

    @Omega.setter
    def Omega(self, val):
        self._Omega = val

    @property
    def eigvecs(self):
        if self._eigvecs is None:
            raise ValueError("'eigvecs' is not set. Please call 'train_net()' first.")
        return self._eigvecs

    @eigvecs.setter
    def eigvecs(self, val):
        self._eigvecs = val

    @property
    def y0(self):
        if self._y0 is None:
            raise ValueError("'y0' is not set. Please call 'train_net()' first.")
        return self._y0

    @y0.setter
    def y0(self, val):
        self._y0 = val


    def _dmd_linearize(self, Y, Yprime) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the rank-reduced Koopman matrix Ktilde.

        Inputs:
            Y: Matrix of states as columns. Contains all states except for last. Shape = (encoded_size, num_training_indexes-1).
            Yprime: Matrix of states as columns. Shifted forward by 1 from Y, i.e. contains all states except for first. Shape = (encoded_size, num_training_indexes-1).
        
        Returns:
            Ktilde: Rank-reduced Koopman matrix. Shape = (rank, rank).
            interm: Intermediate quantity, which is Y'VS^-1 for exact eignvectors, or U for projected eigenvectors. Shape = (encoded_size, rank).

        Note that wherever we do `tensor.t()`, technically the Hermitian should be taken via `tensor.t().conj()`. But since we deal with real data, just the ordinary transpose is fine.
        """
        U, Sigma, V = utils.stable_svd(Y) #shapes: U = (encoded_size, min(num_training_indexes-1,encoded_size)), Sigma = (min(num_training_indexes-1,encoded_size),), V = (num_training_indexes-1, min(num_training_indexes-1,encoded_size))
        
        ## Alternative - use torch's native SVD, which is numerically unstable
        # U, Sigma, Vt = torch.linalg.svd(Y) #shapes: U = (encoded_size, encoded_size), Sigma = (min(num_training_indexes-1,encoded_size),), Vt = (num_training_indexes-1, num_training_indexes-1)
        # V = Vt.t() #shape = (num_training_indexes-1, num_training_indexes-1)
        
        # Left singular vectors
        U = U[:,:self.rank] #shape = (encoded_size, rank)
        Ut = U.t() #shape = (rank, encoded_size)
        
        # Singular values
        if Sigma[-1] < cfg.sigma_threshold:
            with open(self.log_file, 'a') as lf:
                lf.write(f"Smallest singular value prior to truncation = {Sigma[-1]} is smaller than threshold = {cfg.sigma_threshold}. This may lead to very large / nan values in backprop of SVD.\n")
        #NOTE: We can also do a check if any pair (or more) of consecutive singular values are so close that the difference of their squares is less than some threshold, since this will also lead to very large / nan values during backprop. But doing this check requires looping over the Sigma tensor (time complexity = O(len(Sigma))) and is not worth it.
        Sigma = torch.diag(Sigma[:self.rank]) #shape = (rank, rank)
        
        # Right singular vectors
        V = V[:,:self.rank] #shape = (num_training_indexes-1, rank)
        
        # Outputs
        interm = Yprime @ V @ torch.linalg.inv(Sigma) #shape = (encoded_size, rank)
        Ktilde = Ut @ interm #shape = (rank, rank)

        if not cfg.use_exact_eigenvectors:
            interm = U
        
        return Ktilde, interm


    def _dmd_eigen(self, Ktilde, interm) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the eigendecomposition of the Koopman matrix.

        Inputs:
            Ktilde: From _dmd_linearize(). Shape = (rank, rank).
            interm: From _dmd_linearize(). Shape = (encoded_size, rank).

        Returns:
            Omega: **Continuous index** eigenvalues of Ktilde. Shape = (rank, rank).
            eigvecs: First 'rank' exact eigenvectors of the full (i.e. not tilde) system. Shape = (encoded_size, rank).
        """
        Lambda, eigvecstilde = torch.linalg.eig(Ktilde) #shapes: Lambda = (rank,), eigvecstilde is (rank, rank)
        
        # Eigenvalues
        with open(self.log_file, 'a') as lf:
            lf.write(f"Largest magnitude among eigenvalues = {torch.max(torch.abs(Lambda))}\n")
        Omega = torch.diag(torch.log(Lambda)) #shape = (rank, rank) #NOTE: No need to divide by self.dh.tscale here because spacings are normalized to 1
        #NOTE: We can do a check if any pair (or more) of consecutive eigenvalues are so close that their difference is less than some threshold, since this will lead to very large / nan values during backprop. But doing this check requires looping over the Omega tensor (time complexity = O(len(Omega))) and is not worth it.
        
        # Eigenvectors
        eigvecs = interm.to(cfg._CTYPE) @ eigvecstilde #shape = (encoded_size, rank)
        S,s = torch.linalg.norm(eigvecs,2), torch.linalg.norm(eigvecs,-2)
        cond = S/s
        if cond > self.cond_threshold:
            with open(self.log_file, 'a') as lf:
                lf.write(f"Condition number = {cond} is greater than threshold = {self.cond_threshold}. This may lead to numerical instability in evaluating linearity loss. In an attempt to mitigate this, singular values smaller than 1/{self.cond_threshold} times the largest will be discarded from the pseudo-inverse computation.\n")

        return Omega, eigvecs


    def _dmd_predict(self, t,y0, Omega,eigvecs) -> torch.Tensor:
        """
        Predict the dynamics of a system.
        
        Inputs:
            t: Sequence of index steps for which dynamics are predicted. Shape = (num_samples,).
            y0: Initial state of the system from which to make predictions. Shape = (encoded_size,).
            Omega: From _dmd_eigen(). Shape = (rank, rank).
            eigvecs: From _dmd_eigen(). Shape = (encoded_size, rank).
        Note: Omega,eigvecs can also be provided from outside for a new system.
        
        Returns:
            Ypred: Predictions for all the indexes in t. Shape (num_samples, encoded_size). Predictions may be complex, however they are converted to real by taking absolute value. This is fine because, if things are okay, the imaginary parts should be negligible (several orders of magnitude less) than the real parts.
        """
        coeffs = torch.linalg.pinv(eigvecs, rtol=1./self.cond_threshold) @ y0.to(cfg._CTYPE) #shape = (rank,)
        Ypred = torch.tensor([])
        for index,ti in enumerate(t):
            ypred = eigvecs @ torch.linalg.matrix_exp(Omega*ti) @ coeffs #shape = (encoded_size,)
            Ypred = ypred.view(1,-1) if index==0 else torch.vstack((Ypred,ypred)) #shape at end = (num_samples, encoded_size)
        return torch.abs(Ypred)


    def train_net(self,
        numepochs=500, early_stopping=0, early_stopping_metric='pred_anae',
        lr=1e-3, weight_decay=0., decoder_loss_weight=1e-2, Kreg=1e-3,
        cond_threshold=100., clip_grad_norm=None, clip_grad_value=None
    ):
        """Train the model using `dh.Xtr` and `dh.ttr`, and validate it on `dh.Xva` and `dh.tva`.

        ## Parameters
        - **numepochs** (*int, optional*) - Number of epochs for which to train. Each epoch uses the complete training data to learn the Koopman matrix.
    
        - **early_stopping** (*int/float, optional*) - Whether to terminate training early due to no improvement in validation metric.
            - If `0`, no early stopping. The model will run for the complete `numepochs` epochs.
            - If an integer, early stop if validation metric doesn't improve for that many epochs.
            - If a float, early stop if validation performance doesn't improve for that fraction of epochs, rounded up.
        - **early_stopping_metric** (*str, optional*) - Which validation metric to use for early stopping. Ignored if `early_stopping=False`. Possible metrics are `'<recon/lin/pred/total>_loss'`, and `'<recon/lin/pred>_anae'`.
        
        - **lr** (*float, optional*) - Learning rate for optimizer.
        
        - **weight_decay** (*float, optional*) - L2 coefficient for weights of the neural nets.
        
        - **decoder_loss_weight** (*float, optional*) - Weight the losses between autoencoder decoder outputs (`recon` and `pred`) by this number. This is to account for the scaling effect of the decoder.
        
        - **Kreg** (*float, optional*) - L1 penalty for the Ktilde matrix.
        
        - **cond_threshold** (*float, optional*) - Condition number of the eigenvector matrix greater than this will be reported, and singular values smaller than this fraction of the largest will be ignored for the pseudo-inverse operation.
        
        - **clip_grad_norm** (*float, optional*) - If not `None`, clip the norm of gradients to this value.
        - **clip_grad_value** (*float, optional*) - If not `None`, clip the values of gradients to [-`clip_grad_value`,`clip_grad_value`].

        ## Effects
        - `self.stats` is populated.
        - `self.Omega` and `self.eigvecs` hold the eigendecomposition of the Koopman matrix learnt during training, which can be used to predict any state by evolving `self.y0`.
        """
        ## Define instance attributes for those inputs which are used in other methods
        self.decoder_loss_weight = decoder_loss_weight
        self.cond_threshold = cond_threshold

        # Define validation condition
        do_val = len(self.dh.Xva) > 0

        # Check if early stopping is moot
        if not do_val and early_stopping:
            print("WARNING: You have specified 'early_stopping=True' without providing validation data. As a result, early stopping will not occur.")

        # Define optimizer
        opt = torch.optim.Adam(self.ae.parameters(), lr=lr, weight_decay=weight_decay)

        with open(self.log_file, 'a') as lf:
            lf.write("\nStarting training ...\n")

        # Start epochs
        for epoch in tqdm(range(numepochs)):
            with open(self.log_file, 'a') as lf:
                lf.write(f"\nEpoch {epoch+1}\n")

            # NOTE: Do not do any kind of shuffling before/after epochs. The samples dimension is typically shuffled for standard classification problems, but here that corresponds to the index (such as time), which should be ordered.
            
            ## Training ##
            self.ae.train()
            opt.zero_grad()
            
            Ytr, Xrtr = self.ae(self.dh.Xtr)
            
            # Following 2 steps are unique to training
            Ktilde, interm = self._dmd_linearize(Y = Ytr[:-1,:].t(), Yprime = Ytr[1:,:].t())
            Omega, eigvecs = self._dmd_eigen(Ktilde=Ktilde, interm=interm)

            # Record variables that will be used for predictions
            self.Omega = Omega
            self.eigvecs = eigvecs
            self.y0 = Ytr[0]
            
            # Get predictions
            Ypredtr = self._dmd_predict(t=self.dh.ttr[1:], y0=self.y0, Omega=self.Omega, eigvecs=self.eigvecs)
            Xpredtr = self.ae.decoder(Ypredtr)

            # ANAEs
            with torch.no_grad():
                anaes_tr = metrics.overall_anae(X=self.dh.Xtr[1:], Y=Ytr[1:], Xr=Xrtr[1:], Ypred=Ypredtr, Xpred=Xpredtr)

            # Losses
            losses_tr = metrics.overall_loss(X=self.dh.Xtr[1:], Y=Ytr[1:], Xr=Xrtr[1:], Ypred=Ypredtr, Xpred=Xpredtr, decoder_loss_weight = self.decoder_loss_weight)
            losses_tr['Kreg'] = Kreg*torch.sum(torch.abs(Ktilde))/torch.numel(Ktilde) # this is unique to training
            losses_tr['total'] += losses_tr['Kreg']

            # Collect epoch training stats and record
            for k,v in anaes_tr.items():
                self.stats[f'{k}_anae_tr'].append(utils._extract_item(v))
            for k,v in losses_tr.items():
                self.stats[f'{k}_loss_tr'].append(utils._extract_item(v))
            with open(self.log_file, 'a') as lf:
                lf.write(', '.join([f'{k} = {v[-1]}' for k,v in self.stats.items() if k.endswith('_tr')]) + '\n')

            # Backprop
            loss_tr = losses_tr['total']
            try:
                loss_tr.backward()
            except RuntimeError as e:
                self.error_flag = True
                message = f"Encountered RuntimeError: {e}\nStopping training!\n"
                with open(self.log_file, 'a') as lf:
                    lf.write(message)
                print(message)
                break

            # Check for NaN gradients
            try:
                for parameter in self.ae.parameters():
                    if torch.any(torch.isnan(parameter.grad)).item():
                        raise ValueError
            except ValueError:
                self.error_flag = True
                message = "Encountered NaN in gradients\nStopping training!\n"
                with open(self.log_file, 'a') as lf:
                    lf.write(message)
                print(message)
                break

            # Gradient clipping
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.ae.parameters(), clip_grad_norm)
            if clip_grad_value:
                torch.nn.utils.clip_grad_value_(self.ae.parameters(), clip_grad_value)

            # Update
            opt.step()

            ## Validation ##
            if do_val:
                self.ae.eval()
                with torch.no_grad():
                    Yva, Xrva = self.ae(self.dh.Xva)
                    Ypredva = self._dmd_predict(t=self.dh.tva, y0=self.y0, Omega=Omega, eigvecs=eigvecs)
                    Xpredva = self.ae.decoder(Ypredva)

                    anaes_va = metrics.overall_anae(X=self.dh.Xva, Y=Yva, Xr=Xrva, Ypred=Ypredva, Xpred=Xpredva)

                    losses_va = metrics.overall_loss(X=self.dh.Xva, Y=Yva, Xr=Xrva, Ypred=Ypredva, Xpred=Xpredva, decoder_loss_weight=self.decoder_loss_weight)

                # Collect epoch validation stats and record
                for k,v in anaes_va.items():
                    self.stats[f'{k}_anae_va'].append(utils._extract_item(v))
                for k,v in losses_va.items():
                    self.stats[f'{k}_loss_va'].append(utils._extract_item(v))
                with open(self.log_file, 'a') as lf:
                    lf.write(', '.join([f'{k} = {v[-1]}' for k,v in self.stats.items() if k.endswith('_va')]) + '\n')

                # Early stopping
                if early_stopping:

                    # Initialize early stopping in first epoch
                    if epoch==0:
                        EARLY_STOPPING_METRIC_CHOICES = [k for k in self.stats.keys() if k.endswith('_va')]
                        early_stopping_metric = early_stopping_metric + ('_va' if not early_stopping_metric.endswith('va') else '')
                        if early_stopping_metric not in EARLY_STOPPING_METRIC_CHOICES:
                            raise ValueError(f"'early_stopping_metric' must be in {EARLY_STOPPING_METRIC_CHOICES}, instead found '{early_stopping_metric}'")
                        early_stopping = int(np.ceil(early_stopping * numepochs)) if type(early_stopping)==float else early_stopping

                        no_improvement_epochs_count = 0
                        best_val_perf = np.inf

                    # Perform early stopping
                    if self.stats[early_stopping_metric][-1] < best_val_perf:
                        no_improvement_epochs_count = 0
                        best_val_perf = self.stats[early_stopping_metric][-1]
                    else:
                        no_improvement_epochs_count += 1
                    if no_improvement_epochs_count == early_stopping:
                        with open(self.log_file, 'a') as lf:
                            lf.write(f"\nEarly stopped due to no improvement in {early_stopping_metric} for {early_stopping} epochs.\n")
                        break


    def test_net(self):
        """Run the trained model on test data - `dh.Xte` and `dh.tte`.
        
        ## Effects
        - `self.stats` is populated further.
        """
        do_test = len(self.dh.Xte) > 0

        if not do_test:
            print("WARNING: You have called 'test_net()', but there is no test data. Please pass a 'DataHandler' object containing 'Xte' and 'Yte'.")

        else:
            self.ae.eval()
            with torch.no_grad():
                Yte, Xrte = self.ae(self.dh.Xte)
                Ypredte = self._dmd_predict(t=self.dh.tte, y0=self.y0, Omega=self.Omega, eigvecs=self.eigvecs)
                Xpredte = self.ae.decoder(Ypredte)

                anaes_te = metrics.overall_anae(X=self.dh.Xte, Y=Yte, Xr=Xrte, Ypred=Ypredte, Xpred=Xpredte)
                
                losses_te = metrics.overall_loss(X=self.dh.Xte, Y=Yte, Xr=Xrte, Ypred=Ypredte, Xpred=Xpredte, decoder_loss_weight=self.decoder_loss_weight)

            # Collect test stats and record
            for k,v in anaes_te.items():
                self.stats[f'{k}_anae_te'].append(utils._extract_item(v))
            for k,v in losses_te.items():
                self.stats[f'{k}_loss_te'].append(utils._extract_item(v))
            with open(self.log_file, 'a') as lf:
                lf.write(', '.join([f'{k} = {v[-1]}' for k,v in self.stats.items() if k.endswith('_te')]) + '\n')


    def predict_new(self, t) -> torch.Tensor:
        """Use the trained model to predict the states for new indexes that are unknown.

        This is different from testing because the ground truth values are not present, thus losses and errors are not computed. This method can be used to make predictions for interpolated and extrapolated indexes.

        ## Parameters
        - **t** (*Array[int,float], shape=(num_new_indexes,)*) - Indexes for which unknown states should be predicted. *Array* can be any data type such as *numpy.array*, *torch.Tensor*, *list*, *range*, etc.

        ## Returns
        **Xpred** (*torch.Tensor, shape=(len(t), input_size)*) - Predicted states for the new indexes.
        """
        _t = utils._tensorize(t, dtype=cfg._RTYPE, device=cfg._DEVICE)
        _t = utils._shift(_t, shift=self.dh.tshift)
        _t = utils._scale(_t, scale=self.dh.tscale)

        self.ae.eval()
        with torch.no_grad():
            Ypred = self._dmd_predict(t=_t, y0=self.y0, Omega=self.Omega, eigvecs=self.eigvecs)
            Xpred = self.ae.decoder(Ypred)

            if cfg.normalize_Xdata:
                Xpred = utils._scale(Xpred, scale=1/self.dh.Xscale) # unscale back to original domain

        with open(self.log_file, 'a') as lf:
            lf.write("\nNew predictions:\n")
            for i in range(len(t)):
                lf.write(f'Independent variable = {t[i]}, Dependent variable =\n')
                lf.write(f'{Xpred[i]}\n')

        return Xpred
