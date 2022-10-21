"""Run hyperparameter search across inputs to DeepKoopman.""" 

import csv
import inspect
import itertools
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
import shortuuid
from tqdm import tqdm

from deepk.state_predictor import StatePredictor


def run_hyp_search(data, hyp_options, numruns=None, avg_ignore_initial_epochs=100, sort_key='avg_pred_anae_va') -> str:
    """Perform hyperparameter search by running DeepKoopman multiple times on given data, and save the loss and ANAE statistics for each run.

    **The method can be interrupted at any time and the intermediate results will be saved**.

    ## Saved statistics
    For each of the following `<x>`:

    - 'recon_loss', 'recon_anae' - Reconstruction loss and ANAE.
    - 'lin_loss', 'lin_anae' - Linearity loss and ANAE.
    - 'pred_loss', 'pred_anae' - Predicton loss and ANAE.
    - 'loss' - Total loss.

    save the following statistics:

    - `avg_<x>_tr` - Average training `<x>` across all epochs.
    - `final_<x>_tr` - Final training `<x>` after last epoch.
    - `avg_<x>_va` - Average validation `<x>` across all epochs.
    - `best_<x>_va` - Best validation `<x>` across epochs.
    - `bestep_<x>_va` - Epoch number after which best validation `<x>` was achieved.

    For more details, refer to [losses](https://galoisinc.github.io/deep-koopman/losses.html) and [ANAEs](https://galoisinc.github.io/deep-koopman/errors.html).

    ## Parameters
    - **data** (*dict[str,torch.Tensor]*) - Data to be used for all runs. Same format as `data` for the [`DeepKoopman` class](https://galoisinc.github.io/deep-koopman/core.html#deepk.core.DeepKoopman).
    
    - **hyp_options** (*dict[str,list]*) - Input hyperparameters to `DeepKoopman` will be swept over these values across runs. Set a key to have a single value to keep it constant across runs. Possible keys are any input to [`DeepKoopman`](https://galoisinc.github.io/deep-koopman/core.html) except for `data`. Example:
    ```python
    hyp_options = {
        'rank': [6,8],
        'encoded_size': 50, # this input stays constant across runs
        'encoder_hidden_layers': [[100,100],[200,100,50]]
        # other inputs to DeepKoopman not defined here are set to default values
    }
    
    # this results in 4 possible runs:
    DeepKoopman(rank=6, encoded_size=50, encoder_hidden_layers=[100,100])
    DeepKoopman(rank=6, encoded_size=50, encoder_hidden_layers=[200,100,50])
    DeepKoopman(rank=8, encoded_size=50, encoder_hidden_layers=[100,100])
    DeepKoopman(rank=8, encoded_size=50, encoder_hidden_layers=[200,100,50])
    ```

    - **numruns** (*int, optional*) - The total number of possible DeepKoopman runs is \\(N =\\) the number of elements in the Cartesian product of the values of `hyp_options` (in the above example, this is \\(2\\times1\\times2 = 4\\)). If `numruns` is `None` or \\(>N\\), run \\(N\\) runs, otherwise run `numruns` runs.<br>Since each run takes time, it is recommended to set `numruns` to a reasonably smaller value when \\(N\\) is large.

    - **avg_ignore_initial_epochs** (*int, optional*) - When collecting average statistics across epochs, ignore this many initial epochs. This is useful because the loss / ANAE values can be quite high early on, which can skew the averages and make it meaningless to compare runs based on the averages.

    - **sort_key** (*str, optional*) - Results in the final CSV will be sorted in ascending order of this column. For possible options, see **Saved statistics**. Note that sorting only happens at the end, and thus will not take place if execution is interrupted. Set to `None` to skip sorting.

    ## Returns
    **output_csv_path** (*Path*) - The path to a newly created results CSV file = `hyp_search_<uuid>.csv`. This contains results in the format: 
    ```
    -------------------------------------------------------------------------
    | UUID | <hyperparameters swept over> | <loss results> | <anae results> |
    -------------------------------------------------------------------------
    | run1 |              ...             |       ...      |       ...      |
    | run2 |              ...             |       ...      |       ...      |
    | run3 |              ...             |       ...      |       ...      |
    -------------------------------------------------------------------------
    ```
    """
    ## Pre-process hyp options
    POSSIBLE_KEYS = [arg for arg in inspect.getfullargspec(StatePredictor).args if arg not in ['self', 'data']]
    ignore_hyps = []
    for k,v in hyp_options.items():
        if k not in POSSIBLE_KEYS:
            print(f"Key '{k}' in `hyp_options` is not a valid input that can be swept for DeepKoopman. This key will be ignored. Keys must be in {POSSIBLE_KEYS}.")
            ignore_hyps.append(k)
        if type(v) not in [list,set,tuple]:
            hyp_options[k] = [v]
    
    ## Ignore early stopping metric if early_stopping=False
    if not hyp_options.get('early_stopping', [False])[0] and 'early_stopping_metric' in hyp_options:
        del hyp_options['early_stopping_metric']
    
    ## Get hyp variables and constants
    hyp_variables = {k:v for k,v in hyp_options.items() if k not in ignore_hyps and len(v)>1}
    hyp_constants = {k:v[0] for k,v in hyp_options.items() if k not in ignore_hyps and len(v)==1}
    del hyp_options

    ## Get hyps_list_all
    hyps_list_all = list(itertools.product(*[hyp_variables[key] for key in hyp_variables.keys()]))
    if numruns and len(hyps_list_all) > numruns:
        hyps_list_all = random.sample(hyps_list_all, numruns)

    ## Get results file path
    output_csv_path = Path(f'./hyp_search_{shortuuid.uuid()}.csv').resolve()

    ## Print initial info
    print('********************************************************************************')
    print('********************************************************************************')
    print(f'Starting DeepKoopman hyperparameter search. Results will be stored in {output_csv_path}.')
    print()
    print(f'Performing total {len(hyps_list_all)} runs. You can interrupt the script at any time (e.g. Ctrl+C), and intermediate results will be available in the above file.')
    print()
    print('Hyperparameters swept over:')
    for k,v in hyp_variables.items():
        print(f"{k} : {', '.join([str(vv) for vv in v])}")
    print()
    print('Constant hyperparameters:')
    for k,v in hyp_constants.items():
        print(f"{k} : {v}")
    print()
    print('Other hyperparameters have default values, see https://galoisinc.github.io/deep-koopman/core.html')
    print('********************************************************************************')
    print('********************************************************************************')
    print()

    ## Open CSV and write header row
    with open(output_csv_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        header_row = [
            'UUID',
            *list(hyp_variables.keys()), # columns will only be created for inputs that vary, because it doesn't make sense to have a constant column
            'avg_recon_loss_tr', 'final_recon_loss_tr', 'avg_recon_loss_va', 'best_recon_loss_va', 'bestep_recon_loss_va',
            'avg_lin_loss_tr', 'final_lin_loss_tr', 'avg_lin_loss_va', 'best_lin_loss_va', 'bestep_lin_loss_va',
            'avg_pred_loss_tr', 'final_pred_loss_tr', 'avg_pred_loss_va', 'best_pred_loss_va', 'bestep_pred_loss_va',
            'avg_loss_tr', 'final_loss_tr', 'avg_loss_va', 'best_loss_va', 'bestep_loss_va',
            'avg_recon_anae_tr', 'final_recon_anae_tr', 'avg_recon_anae_va', 'best_recon_anae_va', 'bestep_recon_anae_va',
            'avg_lin_anae_tr', 'final_lin_anae_tr', 'avg_lin_anae_va', 'best_lin_anae_va', 'bestep_lin_anae_va',
            'avg_pred_anae_tr', 'final_pred_anae_tr', 'avg_pred_anae_va', 'best_pred_anae_va', 'bestep_pred_anae_va',
        ]
        csvwriter.writerow(header_row)

        ## Do DK runs
        for hyps_list in tqdm(hyps_list_all):
            hyps = dict(zip(hyp_variables.keys(),hyps_list))
            print(f'Hyperparameters: {hyps}')
            
            try:
                dk = StatePredictor(
                    data = data,
                    **hyps,
                    **hyp_constants
                )
            
            except AssertionError as e:
                print(f'Encountered error, skipping this run\n"{e}"')
            
            else:
                row = [
                    dk.uuid,
                    *[hyps[key] for key in hyps.keys()]
                ]
                
                dk.train_net()
                
                if dk.error_flag:
                    row += (len(header_row)-len(row))*[np.nan]
                    #TODO have a way of signaling this to user?
                else:
                    if avg_ignore_initial_epochs >= len(dk.stats['loss_tr']): # can pick any key
                        print(f"WARNING: The value of `avg_ignore_initial_epochs` = {avg_ignore_initial_epochs} is greater than the actual number of epochs for which the current DeepKoopman instance has run for = {len(dk.stats['loss_tr'])}. Thus, 'avg' statistics will be equal to the last epoch's value.")
                        _avg_ignore_initial_epochs = len(dk.stats['loss_tr'])-1
                    else:
                        _avg_ignore_initial_epochs = avg_ignore_initial_epochs
                    row += [
                        np.mean(dk.stats['recon_loss_tr'][_avg_ignore_initial_epochs:]),
                        dk.stats['recon_loss_tr'][-1],
                        np.mean(dk.stats['recon_loss_va'][_avg_ignore_initial_epochs:]),
                        np.min(dk.stats['recon_loss_va']),
                        np.argmin(dk.stats['recon_loss_va'])+1,

                        np.mean(dk.stats['lin_loss_tr'][_avg_ignore_initial_epochs:]),
                        dk.stats['lin_loss_tr'][-1],
                        np.mean(dk.stats['lin_loss_va'][_avg_ignore_initial_epochs:]),
                        np.min(dk.stats['lin_loss_va']),
                        np.argmin(dk.stats['lin_loss_va'])+1,

                        np.mean(dk.stats['pred_loss_tr'][_avg_ignore_initial_epochs:]),
                        dk.stats['pred_loss_tr'][-1],
                        np.mean(dk.stats['pred_loss_va'][_avg_ignore_initial_epochs:]),
                        np.min(dk.stats['pred_loss_va']),
                        np.argmin(dk.stats['pred_loss_va'])+1,

                        np.mean(dk.stats['loss_tr'][_avg_ignore_initial_epochs:]),
                        dk.stats['loss_tr'][-1],
                        np.mean(dk.stats['loss_va'][_avg_ignore_initial_epochs:]),
                        np.min(dk.stats['loss_va']),
                        np.argmin(dk.stats['loss_va'])+1,

                        np.mean(dk.stats['recon_anae_tr'][_avg_ignore_initial_epochs:]),
                        dk.stats['recon_anae_tr'][-1],
                        np.mean(dk.stats['recon_anae_va'][_avg_ignore_initial_epochs:]),
                        np.min(dk.stats['recon_anae_va']),
                        np.argmin(dk.stats['recon_anae_va'])+1,

                        np.mean(dk.stats['lin_anae_tr'][_avg_ignore_initial_epochs:]),
                        dk.stats['lin_anae_tr'][-1],
                        np.mean(dk.stats['lin_anae_va'][_avg_ignore_initial_epochs:]),
                        np.min(dk.stats['lin_anae_va']),
                        np.argmin(dk.stats['lin_anae_va'])+1,

                        np.mean(dk.stats['pred_anae_tr'][_avg_ignore_initial_epochs:]),
                        dk.stats['pred_anae_tr'][-1],
                        np.mean(dk.stats['pred_anae_va'][_avg_ignore_initial_epochs:]),
                        np.min(dk.stats['pred_anae_va']),
                        np.argmin(dk.stats['pred_anae_va'])+1
                    ]

                csvwriter.writerow(row)

                os.system(f"rm -rf {dk.log_file}")
            
            finally:
                print('\n\n')

    ## Sort results
    if sort_key:
        df = pd.read_csv(output_csv_path)
        try:
            df.sort_values(sort_key, ascending=True, inplace=True)
        except KeyError:
            print(f"WARNING: `sort_key` = '{sort_key}' not found among the columns of the output CSV '{output_csv_path}'. Results will be unsorted.")
        df.to_csv(output_csv_path, index=False)

    return output_csv_path
