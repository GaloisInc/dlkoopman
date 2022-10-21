import pytest
import pickle
import os
from deepk.hyp_search import *


@pytest.fixture
def get_data():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'examples/naca0012/data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data


@pytest.mark.skip(reason='Will come back to this later')
def test_run_hyp_search(get_data):
    data = get_data

    sort_key = 'avg_lin_loss_va'

    output_csv_path = run_hyp_search(
        data = data,
        hyp_options = {
            'rank': [6,8],
            'encoded_size': 50,
            'encoder_hidden_layers': [[100,100],[200,100,50]],
            'numepochs': [50],
            'early_stopping': 5,
            'early_stopping_metric': ['pred_anae','loss']
        },
        numruns = 5,
        avg_ignore_initial_epochs = 10,
        sort_key = sort_key
    )

    df = pd.read_csv(output_csv_path)
    assert len(df) == 5

    df_sortcol = list(df[sort_key])
    assert df_sortcol == sorted(df_sortcol)

    os.system(f'rm -rf {output_csv_path}')
