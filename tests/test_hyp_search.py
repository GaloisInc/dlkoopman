import pytest
import pickle
import os
from deepk.hyp_search import *


@pytest.fixture
def get_data():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'examples/naca0012/data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data


def test_run_hyp_search(get_data):
    data = get_data
    
    
    output_csv_path = run_hyp_search(
        data = data,
        hyp_options = {
            'rank': [6,8],
            'num_encoded_states': 50,
            'encoder_hidden_layers': [[100,100],[200,100,50]],
            'numepochs': [50],
            'early_stopping': False,
            'early_stopping_metric': ['pred_anae','loss']
        },
        avg_ignore_initial_epochs = 10
    )
    
    df = pd.read_csv(output_csv_path)
    assert len(df) == 4
    
    df_sortcol = list(df['avg_pred_anae_va'])
    assert df_sortcol == sorted(df_sortcol)
    
    results_folder = '/'.join(output_csv_path.split('/')[:-1])
    files = [file for file in os.listdir(results_folder) if not file.startswith('.')]
    output_csv_file = output_csv_path.split('/')[-1]
    assert files == [output_csv_file]
    
    os.system(f'rm -rf {results_folder}')

    
    output_csv_path = run_hyp_search(
        data = data,
        hyp_options = {
            'rank': [6,8],
            'num_encoded_states': 50,
            'encoder_hidden_layers': [[100,100],[200,100,50]],
            'numepochs': [50],
            'early_stopping': 5,
            'early_stopping_metric': ['pred_anae','loss']
        },
        numruns = 5,
        avg_ignore_initial_epochs = 10,
        sort_key = 'avg_lin_loss_va',
        delete_logs = False
    )
    df = pd.read_csv(output_csv_path)
    assert len(df) == 5
    
    df_sortcol = list(df['avg_lin_loss_va'])
    assert df_sortcol == sorted(df_sortcol)
    
    results_folder = '/'.join(output_csv_path.split('/')[:-1])
    files = [file for file in os.listdir(results_folder) if not file.startswith('.')]
    output_csv_file = output_csv_path.split('/')[-1]
    assert output_csv_file in files
    logfiles = [file for file in files if file.endswith('.log')]
    assert len(logfiles) == 5
    
    os.system(f'rm -rf {results_folder}') 
