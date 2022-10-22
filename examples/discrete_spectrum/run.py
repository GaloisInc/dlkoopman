import pickle

from deepk.trajectory_predictor import TrajectoryPredictor_DataHandler, TrajectoryPredictor
from deepk import utils


with open('/Users/sourya/work/Essence/deep-koopman/examples/discrete_spectrum/data.pkl', 'rb') as f:
    data = pickle.load(f)

dh = TrajectoryPredictor_DataHandler(
    Xtr=data['Xtr'],
    Xva=data['Xva'],
    Xte=data['Xte']
)

utils.set_seed(10)

tp = TrajectoryPredictor(
    dh = dh,
    encoded_size = 10
)

tp.train_net(numepochs=20, batch_size=250)
tp.test_net()

utils.plot_stats(tp, ['pred_loss', 'total_loss', 'pred_anae'])

tp.predict_new([[0.5,0.5], [-0.4,0.6]])
