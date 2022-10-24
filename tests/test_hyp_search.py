import pickle
import os
import shutil
from deepk.hyp_search import *
from deepk.state_predictor import StatePredictor_DataHandler
from deepk import utils


def test_run_hyp_search():
    utils.set_seed(10)

    ref_df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ref_hyp_search_results.csv'))
    ref_df.drop(columns=['UUID'], inplace=True)

    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'examples/naca0012/data.pkl'), 'rb') as f:
        data = pickle.load(f)
    dh = StatePredictor_DataHandler(
        Xtr=data['Xtr'], ttr=data['ttr'],
        Xva=data['Xva'], tva=data['tva']
    )

    output_folder = run_hyp_search(
        dh = dh,
        hyp_options = {
            'rank': [6,8],
            'encoded_size': 50,
            'encoder_hidden_layers': [[100],[200,50]],
            'numepochs': [10],
            'early_stopping_metric': ['pred_anae','total_loss']
        },
        sort_key = 'avg_lin_loss_va'
    )

    df = pd.read_csv(output_folder.joinpath('hyp_search_results.csv'))
    df.drop(columns=['UUID'], inplace=True)
    assert df.equals(ref_df)

    shutil.rmtree(output_folder)
