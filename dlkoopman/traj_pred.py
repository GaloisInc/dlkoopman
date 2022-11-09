"""**Trajectory Predictor**.

`TrajPred` can be used to train on given equal-length trajectories of a system, then predict unknown trajectories of the system starting from new initial states. See a specific example and tutorial [here](https://github.com/GaloisInc/dlkoopman/blob/main/examples/traj_pred_polynomial_manifold/run.ipynb).
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
    'TrajPred.decoder_loss_weight': False,
    'TrajPred.Lambda': False,
    'TrajPred.eigvecs': False
}


class TrajPredDataHandler:
    """Trajectory predictor data handler. Used to provide data to train (and optionally validate and test) the `TrajPred` model.

    ## Parameters
    - **'Xtr'** (*Array[float], shape=(num_training_trajectories, num_indexes, input_size)*) - Input trajectories to be used as training data. *Array* can be any data type such as *numpy.array*, *torch.Tensor*, *list* etc.

    - **'Xva'** (*Array[float], shape=(num_validation_trajectories, num_indexes, input_size), optional*) - Input trajectories to be used as validation data. Same data type requirements as `Xtr`.

    - **'Xte'** (*Array[float], shape=(num_test_trajectories, num_indexes, input_size), optional*) - Input trajectories to be used as test data. Same data type requirements as `Xtr`.

    ## Example
    ```python
    # Provide data of a system with 2-dimensional states (i.e. input_size=3)
    # containing trajectories of length 4 (i.e. num_indexes=3)
    # Provide 3 trajectories for training, 1 for validation, and none for testing

    dh = TrajPredDataHandler(
        Xtr = numpy.array([
            [ # 1st trajectory
                [8.4, 6.7],
                [4.2, 4.2],
                [3.4, 4.6],
                [7.7, 1.8]
            ],
            [ # 2nd trajectory
                [5.9, 1.5],
                [8.3, 5.1],
                [9.9, 0.2],
                [1.6, 7.2]
            ],
            [ # 3rd trajectory
                [5.4, 9.8],
                [2.4, 6.9],
                [8.9, 0.5],
                [9.3, 0.7]
            ]
        ]),
        Xva = numpy.array([
            [ # 1st trajectory
                [4.7, 1.4],
                [1.1, 6.2],
                [7.8, 6.9],
                [6.1, 8. ]
            ]
        ])
    )
    ```
    """

    def __init__(self, Xtr, Xva=None, Xte=None):
        self.Xtr = utils._tensorize(Xtr, dtype=cfg._RTYPE, device=cfg._DEVICE)
        self.Xva = utils._tensorize(Xva, dtype=cfg._RTYPE, device=cfg._DEVICE)
        self.Xte = utils._tensorize(Xte, dtype=cfg._RTYPE, device=cfg._DEVICE)

        ## Check sizes
        if len(self.Xva):
            assert self.Xva.shape[1:] == self.Xtr.shape[1:], f"Shape of 'Xva' and 'Xtr' must match except for 0th dimension, instead found 'Xva.shape' = {self.Xva.shape} and 'Xtr.shape' = {self.Xtr.shape}"
        if len(self.Xte):
            assert self.Xte.shape[1:] == self.Xtr.shape[1:], f"Shape of 'Xte' and 'Xtr' must match except for 0th dimension, instead found 'Xte.shape' = {self.Xte.shape} and 'Xtr.shape' = {self.Xtr.shape}"

        ## Define Xscale, and normalize X data if applicable
        self.Xscale = torch.max(torch.abs(self.Xtr)).item()
        if cfg.normalize_Xdata:
            self.Xtr = utils._scale(self.Xtr, scale=self.Xscale)
            self.Xva = utils._scale(self.Xva, scale=self.Xscale)
            self.Xte = utils._scale(self.Xte, scale=self.Xscale)


class TrajPred:
    """Trajectory predictor. Used to train on given equal-length trajectories of a system, then predict unknown trajectories of the system starting from new initial states.

    ## Parameters
    - **dh** (*TrajPredDataHandler*) - Data handler that feeds data.

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
    
    - **Knet** (*nets.Knet*) - Linear layer to approximate the Koopman matrix. This is used to evolve states in the encoded domain so as to generate their trajectories.

    - **Lambda** (*torch.Tensor*), **eigvecs** (*torch.Tensor*) - Eigenvalues, and eigenvectors of the trained Koopman matrix that characterizes the discrete index system \\(y_{i+1} = Ky_i\\). The system is discrete since specific trajectory indexes are not provided, so they are always assumed to be \\([0,1,2,\\cdots]\\). The eigendecomposition is not used in computations since the trained `Knet` layer performs all predictions, but is still calculated to characterize the system.

    - **stats** (*dict[list]*) - Stores different metrics from training and testing. Useful for checking performance and [plotting](https://galoisinc.github.io/dlkoopman/utils.html#dlkoopman.utils.plot_stats).

    - **error_flag** (*bool*) - Signals if any error has occurred in training.
    """
    
    def __init__(self,
        dh, encoded_size,
        encoder_hidden_layers=[100], decoder_hidden_layers=[], batch_norm=False
    ):
        ## Define UUID and log file
        self.uuid = shortuuid.uuid()
        self.log_file = Path(f'./log_{self.uuid}.log').resolve()
        print(f'Log file = {self.log_file}')

        ## Get data handler and sizes
        self.dh = dh
        self.input_size = self.dh.Xtr.shape[2]
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

        ## Define linear layer
        self.Knet = nets.Knet(
            size = encoded_size
        )
        self.Knet.to(dtype=cfg._RTYPE, device=cfg._DEVICE)

        ## Define params
        self.params = list(self.ae.parameters()) + list(self.Knet.parameters())

        ## Define results
        self.stats = defaultdict(list)

        ## Define error flag
        self.error_flag = False

        ## Define other attributes to be used later (getter/setters should exist for each of these)
        self.decoder_loss_weight = None
        self.Lambda = None
        self.eigvecs = None


    @property
    def decoder_loss_weight(self):
        if self._decoder_loss_weight is None:
            raise ValueError("'decoder_loss_weight' is not set. Please call 'train_net()' first.")
        return self._decoder_loss_weight

    @decoder_loss_weight.setter
    def decoder_loss_weight(self, val):
        self._decoder_loss_weight = val

    @property
    def Lambda(self):
        if self._Lambda is None:
            raise ValueError("'Lambda' is not set. Please call 'train_net()' first.")
        return self._Lambda

    @Lambda.setter
    def Lambda(self, val):
        self._Lambda = val

    @property
    def eigvecs(self):
        if self._eigvecs is None:
            raise ValueError("'eigvecs' is not set. Please call 'train_net()' first.")
        return self._eigvecs

    @eigvecs.setter
    def eigvecs(self, val):
        self._eigvecs = val


    def _evolve(self, Y0) -> torch.Tensor:
        """
        Y0: (num_trajectories, encoded_size) - Initial state for all trajectories.
        """
        Ypred = torch.zeros(Y0.shape[0], self.dh.Xtr.shape[1], Y0.shape[1]) # shape = (num_trajectories, num_indexes, encoded_size)
        Ypred[:, 0, :] = Y0
        for index in range(1, Ypred.shape[1]):
            Ypred[:, index] = self.Knet(Ypred[:, index-1].clone()) #NOTE: .clone() since we are in-place modifying a variable needed for gradient computation
        return Ypred


    def train_net(self,
        numepochs=10, batch_size=250, early_stopping=0, early_stopping_metric='total_loss',
        lr=1e-3, weight_decay=0., decoder_loss_weight=1e-2,
        clip_grad_norm=None, clip_grad_value=None
    ):
        """Train the model using `dh.Xtr`, and validate it on `dh.Xva`.

        ## Parameters
        - **numepochs** (*int, optional*) - Number of epochs for which to train. Each epoch uses the complete training data to learn the Koopman matrix.

        - **batch_size** (*int, optional*) - How many trajectories to train on per batch. Set to 0 to train on all trajectories per batch (not recommended when number of training trajectories is large).
    
        - **early_stopping** (*int/float, optional*) - Whether to terminate training early due to no improvement in validation metric.
            - If `0`, no early stopping. The model will run for the complete `numepochs` epochs.
            - If an integer, early stop if validation metric doesn't improve for that many epochs.
            - If a float, early stop if validation performance doesn't improve for that fraction of epochs, rounded up.
        - **early_stopping_metric** (*str, optional*) - Which validation metric to use for early stopping. Ignored if `early_stopping=False`. Possible metrics are `'<recon/lin/pred/total>_loss'`, and `'<recon/lin/pred>_anae'`.
        
        - **lr** (*float, optional*) - Learning rate for optimizer.
        
        - **weight_decay** (*float, optional*) - L2 coefficient for weights of the neural nets.
        
        - **decoder_loss_weight** (*float, optional*) - Weight the losses between autoencoder decoder outputs (`recon` and `pred`) by this number. This is to account for the scaling effect of the decoder.


        - **cond_threshold** (*float, optional*) - Condition number of the eigenvector matrix greater than this will be reported, and singular values smaller than this fraction of the largest will be ignored for the pseudo-inverse operation.
        
        - **clip_grad_norm** (*float, optional*) - If not `None`, clip the norm of gradients to this value.
        - **clip_grad_value** (*float, optional*) - If not `None`, clip the values of gradients to [-`clip_grad_value`,`clip_grad_value`].

        ## Effects
        - `self.stats` is populated.
        """
        ## Define instance attributes for those inputs which are used in other methods
        self.decoder_loss_weight = decoder_loss_weight

        # Define validation condition
        do_val = len(self.dh.Xva) > 0

        # Check if early stopping is moot
        if not do_val and early_stopping:
            print("WARNING: You have specified 'early_stopping=True' without providing validation data. As a result, early stopping will not occur.")

        # Define optimizer
        opt = torch.optim.Adam(self.params, lr=lr, weight_decay=weight_decay)

        # Define numbatches
        if batch_size <= 0 or batch_size > self.dh.Xtr.shape[0]:
            batch_size = self.dh.Xtr.shape[0]
            numbatches = 1
        else:
            numbatches = int(np.ceil(self.dh.Xtr.shape[0]/batch_size))

        with open(self.log_file, 'a') as lf:
            lf.write("\nStarting training ...\n")

        # Start epochs
        for epoch in tqdm(range(numepochs)):
            with open(self.log_file, 'a') as lf:
                lf.write(f"\nEpoch {epoch+1}\n")

            anaes_tr = defaultdict(float)
            losses_tr = defaultdict(float)

            # Get current snapshot of Lambda and eigvecs (for record-keeping only, these are not used in any computations since linear layer directly does all computation)
            with torch.no_grad():
                self.Lambda, self.eigvecs = torch.linalg.eig(self.Knet.net.weight)
            with open(self.log_file, 'a') as lf:
                lf.write(f"Largest magnitude among eigenvalues = {torch.max(torch.abs(self.Lambda))}\n")

            # Shuffle
            self.dh.Xtr = self.dh.Xtr[torch.randperm(self.dh.Xtr.shape[0])]

            ## Training ##
            self.ae.train()
            self.Knet.train()

            # Start batches
            for batch in range(numbatches):
                opt.zero_grad()

                Ytr, Xrtr = self.ae(self.dh.Xtr[batch*batch_size : (batch+1)*batch_size]) # shapes: Ytr = (batch_size, num_indexes, encoded_size), Xrtr = (batch_size, num_indexes, input_size)

                # Get predictions
                Ypredtr = self._evolve(Ytr[:,0,:]) # shape = (batch_size, num_indexes, encoded_size)
                Xpredtr = self.ae.decoder(Ypredtr) # shape = (batch_size, num_indexes, input_size)

                # ANAEs
                with torch.no_grad():
                    batch_anaes_tr = metrics.overall_anae(X = self.dh.Xtr[batch*batch_size : (batch+1)*batch_size, 1:], Y=Ytr[:,1:], Xr=Xrtr[:,1:], Ypred=Ypredtr[:,1:], Xpred=Xpredtr[:,1:])

                # Losses
                batch_losses_tr = metrics.overall_loss(X = self.dh.Xtr[batch*batch_size : (batch+1)*batch_size, 1:], Y=Ytr[:,1:], Xr=Xrtr[:,1:], Ypred=Ypredtr[:,1:], Xpred=Xpredtr[:,1:], decoder_loss_weight = self.decoder_loss_weight)

                # Collect batch training stats
                for k,v in batch_anaes_tr.items():
                    anaes_tr[k] += utils._extract_item(v)
                for k,v in batch_losses_tr.items():
                    losses_tr[k] += utils._extract_item(v)

                # Backprop
                loss_tr = batch_losses_tr['total']
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
                    for parameter in self.params:
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
                    torch.nn.utils.clip_grad_norm_(self.params, clip_grad_norm)
                if clip_grad_value:
                    torch.nn.utils.clip_grad_value_(self.params, clip_grad_value)

                # Update
                opt.step()

            # Collect epoch training stats and record
            for k,v in anaes_tr.items():
                self.stats[f'{k}_anae_tr'].append(utils._extract_item(v/numbatches))
            for k,v in losses_tr.items():
                self.stats[f'{k}_loss_tr'].append(utils._extract_item(v/numbatches))
            with open(self.log_file, 'a') as lf:
                lf.write(', '.join([f'{k} = {v[-1]}' for k,v in self.stats.items() if k.endswith('_tr')]) + '\n')


            ## Validation ##
            if do_val:
                self.ae.eval()
                self.Knet.eval()

                with torch.no_grad():
                    Yva, Xrva = self.ae(self.dh.Xva) # shapes: Yva = (num_va_trajectories, num_indexes, encoded_size), Xrva = (num_va_trajectories, num_indexes, input_size)
                    Ypredva = self._evolve(Yva[:,0,:]) # shape = (num_va_trajectories, num_indexes, encoded_size)
                    Xpredva = self.ae.decoder(Ypredva) # shape = (num_va_trajectories, num_indexes, input_size)

                    anaes_va = metrics.overall_anae(X=self.dh.Xva[:,1:], Y=Yva[:,1:], Xr=Xrva[:,1:], Ypred=Ypredva[:,1:], Xpred=Xpredva[:,1:])

                    losses_va = metrics.overall_loss(X=self.dh.Xva[:,1:], Y=Yva[:,1:], Xr=Xrva[:,1:], Ypred=Ypredva[:,1:], Xpred=Xpredva[:,1:], decoder_loss_weight=self.decoder_loss_weight)

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
        """Run the trained model on test data - `dh.Xte`.
        
        ## Effects
        - `self.stats` is populated further.
        """
        do_test = len(self.dh.Xte) > 0

        if not do_test:
            print("WARNING: You have called 'test_net()', but there is no test data. Please pass a 'DataHandler' object containing 'Xte' and 'Yte'.")

        else:
            self.ae.eval()
            self.Knet.eval()

            with torch.no_grad():
                Yte, Xrte = self.ae(self.dh.Xte) # shapes: Yte = (num_te_trajectories, num_indexes, encoded_size), Xrte = (num_te_trajectories, num_indexes, input_size)
                Ypredte = self._evolve(Yte[:,0,:]) # shape = (num_te_trajectories, num_indexes, encoded_size)
                Xpredte = self.ae.decoder(Ypredte) # shape = (num_te_trajectories, num_indexes, input_size)

                anaes_te = metrics.overall_anae(X=self.dh.Xte[:,1:], Y=Yte[:,1:], Xr=Xrte[:,1:], Ypred=Ypredte[:,1:], Xpred=Xpredte[:,1:])

                losses_te = metrics.overall_loss(X=self.dh.Xte[:,1:], Y=Yte[:,1:], Xr=Xrte[:,1:], Ypred=Ypredte[:,1:], Xpred=Xpredte[:,1:], decoder_loss_weight=self.decoder_loss_weight)

            # Collect test stats and record
            for k,v in anaes_te.items():
                self.stats[f'{k}_anae_te'].append(utils._extract_item(v))
            for k,v in losses_te.items():
                self.stats[f'{k}_loss_te'].append(utils._extract_item(v))
            with open(self.log_file, 'a') as lf:
                lf.write(', '.join([f'{k} = {v[-1]}' for k,v in self.stats.items() if k.endswith('_te')]) + '\n')


    def predict_new(self, X0) -> torch.Tensor:
        """Use the trained model to predict complete trajectories for new starting states.

        This is different from testing because the ground truth values are not present, thus losses and errors are not computed.

        ## Parameters
        - **'X0'** (*Array[float], shape=(num_new_trajectories, input_size)*) - The starting states for the new trajectories that are to be predicted. *Array* can be any data type such as *numpy.array*, *torch.Tensor*, *list*, *range*, etc.

        ## Returns
        **Xpred** (*torch.Tensor, shape=(num_new_trajectories, num_indexes, input_size)*) - Predicted trajectories for the new starting states.
        """
        X0 = utils._tensorize(X0, dtype=cfg._RTYPE, device=cfg._DEVICE)
        if cfg.normalize_Xdata:
            _X0 = utils._scale(X0, scale=self.dh.Xscale)

        self.ae.eval()
        self.Knet.eval()
        with torch.no_grad():
            Y0 = self.ae.encoder(_X0)
            Ypred = self._evolve(Y0)
            Xpred = self.ae.decoder(Ypred)

            if cfg.normalize_Xdata:
                Xpred = utils._scale(Xpred, scale=1/self.dh.Xscale)

        with open(self.log_file, 'a') as lf:
            lf.write("\nNew predictions:\n\n")
            Xpred[:,0,:] = X0 # Start predicted trajectories from given starting points instead of reconstructed starting points. This helps in the user identifying each trajectory.
            for i in range(Xpred.shape[0]):
                lf.write(f'{Xpred[i]}\n\n')

        return Xpred
