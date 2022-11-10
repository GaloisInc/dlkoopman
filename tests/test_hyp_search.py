import pickle
import os
import shutil
from dlkoopman.hyp_search import *
from dlkoopman.state_pred import StatePredDataHandler


def test_run_hyp_search():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'examples/state_pred_naca0012/data.pkl'), 'rb') as f:
        data = pickle.load(f)

    PERFS = [
        'recon_loss', 'lin_loss', 'pred_loss', 'total_loss',
        'recon_anae', 'lin_anae', 'pred_anae'
    ]

    ## Test sort_key and tr+va columns
    dh = StatePredDataHandler(
        Xtr=data['Xtr'], ttr=data['ttr'],
        Xva=data['Xva'], tva=data['tva']
    )
    sort_key = 'avg_lin_loss_va'
    output_folder = run_hyp_search(
        dh = dh,
        hyp_options = {
            'rank': [6,8],
            'encoded_size': 50,
            'encoder_hidden_layers': [[100],[200,50]],
            'numepochs': [10],
            'early_stopping_metric': ['pred_anae','total_loss']
        },
        sort_key = sort_key
    )
    df = pd.read_csv(output_folder.joinpath('hyp_search_results.csv'))

    assert len(df) == 4

    assert list(df[sort_key]) == sorted(list(df[sort_key]))

    expected_columns = ['UUID', 'rank', 'encoder_hidden_layers']
    for perf in PERFS:
        expected_columns.extend([f'avg_{perf}_tr', f'final_{perf}_tr', f'avg_{perf}_va', f'best_{perf}_va', f'bestep_{perf}_va'])
    assert set(df.columns) == set(expected_columns)

    shutil.rmtree(output_folder)

    ## Test numruns and tr columns
    dh = StatePredDataHandler(
        Xtr=data['Xtr'], ttr=data['ttr']
    )
    output_folder = run_hyp_search(
        dh = dh,
        hyp_options = {
            'rank': 4,
            'encoded_size': [10,20,50],
            'numepochs': [10],
            'weight_decay': [1e-4,1e-5,1e-6,0.],
            'clip_grad_value': [5,None]
        },
        numruns = 2
    )
    df = pd.read_csv(output_folder.joinpath('hyp_search_results.csv'))

    assert len(df) == 2

    expected_columns = ['UUID', 'encoded_size', 'weight_decay', 'clip_grad_value']
    for perf in PERFS:
        expected_columns.extend([f'avg_{perf}_tr', f'final_{perf}_tr'])
    assert set(df.columns) == set(expected_columns)

    shutil.rmtree(output_folder)
