"""Ready-to-use hyperparameter search module.""" 

import csv
from collections import OrderedDict
import itertools
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
import shortuuid
import sys
from tqdm import tqdm

from dlkoopman.state_pred import StatePred, StatePredDataHandler
from dlkoopman.traj_pred import TrajPred, TrajPredDataHandler


def _gen_colnames(perf, do_val=True) -> list[str]:
    """
    Inputs:
        perf (str) - Either of <recon/lin/pred>_<loss/anae> or total_loss
    
    Return a list of strings that are column names of output CSV - see code for details
    """
    ret = [
        f'avg_{perf}_tr',
        f'final_{perf}_tr'
    ]
    if do_val:
        ret.extend([
            f'avg_{perf}_va',
            f'best_{perf}_va',
            f'bestep_{perf}_va'
        ])
    return ret

def _compute_stats(stats, perf, do_val=True, avg_ignore_initial_epochs=0) -> list[float]:
    """
    Inputs:
        stats - A MODEL_CLASS.stats object
        perf (str) - Either of <recon/lin/pred>_<loss/anae> or total_loss
        do_val (bool) - Whether to return validation stats or not
        avg_ignore_initial_epochs (bool) - See doc of run_hyp_search()
    
    Return:
        avg(perf_tr)
        last(perf_tr)
        avg(perf_va) (if do_val is True)
        best(perf_va) (if do_val is True)
        bestep(perf_va) (if do_val is True)
    """
    ret = [
        np.mean(stats[f'{perf}_tr'][avg_ignore_initial_epochs:]),
        stats[f'{perf}_tr'][-1]
    ]
    if do_val:
        ret.extend([
            np.mean(stats[f'{perf}_va'][avg_ignore_initial_epochs:]),
            np.min(stats[f'{perf}_va']),
            np.argmin(stats[f'{perf}_va'])+1
    ])
    return ret



def run_hyp_search(
    dh, hyp_options,
    numruns=None, avg_ignore_initial_epochs=0, sort_key='avg_pred_anae_va'
) -> Path:
    """Perform many runs of a type of predictor model (either `StatePred` or `TrajPred`) with different configurations on given data, and save the metrics for each run.

    Use the results to pick a good predictor configuration.

    **The method can be interrupted at any time and the intermediate results will be saved**.

    ## Saved statistics
    For each of the following `<x>`:

    - 'recon_loss', 'recon_anae' - Reconstruction loss and ANAE.
    - 'lin_loss', 'lin_anae' - Linearity loss and ANAE.
    - 'pred_loss', 'pred_anae' - Predicton loss and ANAE.
    - 'total_loss' - Total loss.

    save the following statistics:

    If only training data is provided:

    - `avg_<x>_tr` - Average training `<x>` across all epochs.
    - `final_<x>_tr` - Final training `<x>` after last epoch.

    If both training and validation data is provided (recommended), save the above, and additionally:

    - `avg_<x>_va` - Average validation `<x>` across all epochs.
    - `best_<x>_va` - Best validation `<x>` across all epochs.
    - `bestep_<x>_va` - Epoch number at which best validation `<x>` was achieved.

    For more details on losses and ANAEs, refer to [metrics](https://galoisinc.github.io/dlkoopman/metrics.html).

    ## Parameters
    - **dh** (*StatePredDataHandler, or TrajPredDataHandler*) - Data handler providing data. **Model to be run is inferred from data, i.e. either `StatePred` or `TrajPred`.**
    
    - **hyp_options** (*dict[str,list]*) - Input arguments to model and its methods will be swept over these values across runs. As an example, when `dh` is a `StatePredDataHandler`:
    ```python
    hyp_options = {
        ## arguments to __init__()
        'rank': [6,8], # must be provided since it's a required argument
        'encoded_size': 50, # must be provided since it's a required argument; this input stays constant across runs
        'encoder_hidden_layers': [[100,100],[200,100,50]]
        
        ## arguments to train_net()
        'numepochs': list(range(100,501,100)), # ranges must be provided as lists
        'clip_grad_norm': 5 # this input stays constant across runs
    }
    
    # this results in 2*2*5=20 possible runs
    ```

    - **numruns** (*int, optional*) - The total number of possible model runs is \\(N =\\) the number of elements in the Cartesian product of the values of `hyp_options` (in the above example, this is 20). If `numruns` is `None` or \\(>N\\), run \\(N\\) runs, otherwise run `numruns` runs.<br>Since each run takes time, it is recommended to set `numruns` to a reasonably smaller value when \\(N\\) is large.

    - **avg_ignore_initial_epochs** (*int, optional*) - When collecting average statistics across epochs, ignore this many initial epochs. This is useful because the loss / ANAE values can be quite high early on, which can skew the averages and make it meaningless to compare runs based on the averages.

    - **sort_key** (*str, optional*) - Results in the final CSV will be sorted in ascending order of this column. For possible options, see **Saved statistics**. Note that sorting only happens at the end, and thus will not take place if execution is interrupted. Set to `None` to skip sorting.

    ## Returns
    **output_folder** (*Path*) - The path to a newly created folder containing:
    
    - Results CSV file = `hyp_search_results.csv`
    - Log file = `hyp_search_log.log`
    - If any of the individual model runs resulted in errors, their log files will be stored as well so that you can take a closer look at what went wrong.

    The results CSV contains:
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
    ## Define do_val
    do_val = len(dh.Xva) > 0
    
    ## Define perfs
    PERFS = [
        'recon_loss', 'lin_loss', 'pred_loss', 'total_loss',
        'recon_anae', 'lin_anae', 'pred_anae'
    ]

    ## Define possible hyp options and MODEL_CLASS
    if isinstance(dh, StatePredDataHandler):
        REQD_CLASS_KEYS = ['rank', 'encoded_size']
        CLASS_KEYS = ['rank', 'encoded_size', 'encoder_hidden_layers', 'decoder_hidden_layers', 'batch_norm']
        TRAIN_KEYS = ['numepochs', 'early_stopping', 'early_stopping_metric', 'lr', 'weight_decay', 'decoder_loss_weight', 'Kreg', 'cond_threshold', 'clip_grad_norm', 'clip_grad_value']
        MODEL_CLASS = StatePred
    elif isinstance(dh, TrajPredDataHandler):
        REQD_CLASS_KEYS = ['encoded_size']
        CLASS_KEYS = ['encoded_size', 'encoder_hidden_layers', 'decoder_hidden_layers', 'batch_norm']
        TRAIN_KEYS = ['numepochs', 'batch_size', 'early_stopping', 'early_stopping_metric', 'lr', 'weight_decay', 'decoder_loss_weight', 'clip_grad_norm', 'clip_grad_value']
        MODEL_CLASS = TrajPred
    else:
        raise ValueError(f"Invalid 'dh' of type {type(dh)}")
    ALL_KEYS = CLASS_KEYS + TRAIN_KEYS

    ## Check that required hyp options are provided
    msg = ""
    for k in REQD_CLASS_KEYS:
        if k not in hyp_options:
            msg += f"'{k}' is a required argument to {MODEL_CLASS.__name__}, so 'hyp_options' must include a key-value pair for {{'{k}': <{k}_values>}}"
    if msg: 
        raise ValueError(msg)

    ## Order key-value pairs into a new dict where CLASS_KEYS comes first, then TRAIN_KEYS
    hyp_options_ordered = OrderedDict()

    USED_CLASS_KEYS = []
    for k in CLASS_KEYS:
        if k in hyp_options:
            USED_CLASS_KEYS.append(k)
            v = hyp_options[k]
            hyp_options_ordered[k] = [v] if type(v) not in [list,set,tuple] else v

    USED_TRAIN_KEYS = []
    for k in TRAIN_KEYS:
        if k in hyp_options:
            USED_TRAIN_KEYS.append(k)
            v = hyp_options[k]
            hyp_options_ordered[k] = [v] if type(v) not in [list,set,tuple] else v

    ## Report invalid hyps
    invalid_hyps = set(hyp_options.keys()).difference(set(hyp_options_ordered.keys()))
    for k in invalid_hyps:
        print(f"WARNING: Key '{k}' in 'hyp_options' is not a valid argument to {MODEL_CLASS.__name__} or any of its methods. This key will be ignored. Keys must be in {ALL_KEYS}.")

    ## Delete hyp_options as workflow has completely moved over to hyp_options_ordered
    del hyp_options

    ## Ignore early stopping metric if early stopping is only False
    if hyp_options_ordered.get('early_stopping', [0]) == [0] and 'early_stopping_metric' in hyp_options_ordered:
        del hyp_options_ordered['early_stopping_metric']
        USED_TRAIN_KEYS.remove('early_stopping_metric')

    ## Get hyps_list_all
    hyps_list_all = list(itertools.product(*[hyp_options_ordered[key] for key in hyp_options_ordered]))
    if numruns and len(hyps_list_all) > numruns:
        hyps_list_all = random.sample(hyps_list_all, numruns)

    ## Process results folder and its files
    output_folder = Path(f'./hyp_search_{shortuuid.uuid()}').resolve()
    output_folder.mkdir()
    results_file = output_folder.joinpath('hyp_search_results.csv')
    log_file = output_folder.joinpath('hyp_search_log.log')

    ## Print initial info
    print('\n********************************************************************************')
    print(f'Starting {MODEL_CLASS.__name__} hyperparameter search. Results will be stored in {results_file}.\n')
    print(f'Performing total {len(hyps_list_all)} runs. You can interrupt the script at any time (e.g. Ctrl+C), and intermediate results will be available in the above file.\n')
    print(f'Log of the entire hyperparameter search, as well as logs of failed {MODEL_CLASS.__name__} runs will also be stored in the same folder.\n')
    print("Hyperparameters' sweep ranges:")
    for k,v in hyp_options_ordered.items():
        print(f"{k} : {', '.join([str(vv) for vv in v])}")
    print('********************************************************************************\n')

    ## Change directory
    original_cwd = os.getcwd()
    os.chdir(output_folder)

    ## Write outputs to log file from here on
    original_stdout = sys.stdout
    sys.stdout = open(log_file, 'w')

    ## Open CSV and write header row
    with open(results_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        header_row = [
            'UUID',
            *list(hyp_options_ordered.keys())
        ]
        for perf in PERFS:
            header_row += _gen_colnames(perf=perf, do_val=do_val)
        csvwriter.writerow(header_row)

        ## Do runs
        for hyps_list in tqdm(hyps_list_all):
            class_hyps = OrderedDict(zip(USED_CLASS_KEYS, hyps_list[:len(USED_CLASS_KEYS)]))
            train_hyps = OrderedDict(zip(USED_TRAIN_KEYS, hyps_list[len(USED_CLASS_KEYS):]))
            all_hyps = {**class_hyps, **train_hyps}
            print(f'Hyperparameters: {all_hyps}')

            model = MODEL_CLASS(
                dh = dh,
                **class_hyps
            )

            row = [
                model.uuid,
                *list(class_hyps.values()),
                *list(train_hyps.values()),
            ]

            model.train_net(**train_hyps)

            if model.error_flag:
                row += (len(header_row)-len(row))*[np.nan]
                print('WARNING: Encountered error. The log file of this model will be saved for you to take a closer look.')

            else:
                if avg_ignore_initial_epochs >= len(model.stats['total_loss_tr']): # can pick any key
                    print(f"WARNING: The value of `avg_ignore_initial_epochs` = {avg_ignore_initial_epochs} is greater than the actual number of epochs for which the current model instance has run for = {len(model.stats['total_loss_tr'])}. Thus, 'avg' statistics will be equal to the last epoch's value.")
                    _avg_ignore_initial_epochs = len(model.stats['total_loss_tr'])-1
                else:
                    _avg_ignore_initial_epochs = avg_ignore_initial_epochs
                for perf in PERFS:
                    row += _compute_stats(stats=model.stats, perf=perf, do_val=do_val, avg_ignore_initial_epochs=_avg_ignore_initial_epochs)
                os.system(f"rm -rf {model.log_file}")

            csvwriter.writerow(row)
            print()

    ## Back to writing in terminal
    sys.stdout = original_stdout

    ## Delete constant columns and sort results
    df = pd.read_csv(results_file)
    df.drop(columns = [k for k,v in hyp_options_ordered.items() if len(v)==1], inplace=True)
    
    if sort_key:
        try:
            df.sort_values(sort_key, ascending=True, inplace=True)
        except KeyError:
            print(f"WARNING: `sort_key` = '{sort_key}' not found among the columns of the output CSV '{results_file}'. Results will be unsorted.")
        df.to_csv(results_file, index=False)

    ## Change back to original directory
    os.chdir(original_cwd)

    return output_folder
