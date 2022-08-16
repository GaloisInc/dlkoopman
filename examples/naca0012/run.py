import os
import pickle
from deepk.core import DeepKoopman
from deepk import utils

os.chdir(os.path.dirname(os.path.realpath(__file__)))


with open('./data.pkl', 'rb') as f:
    data = pickle.load(f)

utils.set_seed(10)


# ============ #
# Un-optimized
# ============ #
dk = DeepKoopman(
    data = data,
    rank = 6,
    num_encoded_states = 50
)


# ========= #
# Optimized
# ========= #

# dk = DeepKoopman(
#     data = data,
#     rank = 6,
#     num_encoded_states = 500,
#     encoder_hidden_layers = [1000,500],
#     numepochs = 1000,
#     decoder_loss_weight = 0.1,
#     K_reg = 0.,
#     clip_grad_value = 2.
# )

# ============================== #
# Optimized, with early stopping
# ============================== #

# dk = DeepKoopman(
#     data = data,
#     rank = 6,
#     num_encoded_states = 500,
#     encoder_hidden_layers = [1000,500],
#     numepochs = 1000,
#     early_stopping = 50,
#     early_stopping_metric = 'pred_anae',
#     decoder_loss_weight = 0.1,
#     K_reg = 0.,
#     clip_grad_value = 2.
# )


dk.train_net()
dk.test_net()

utils.plot_stats(dk, ['pred_loss', 'loss', 'pred_anae'])

# print(dk.predict_new([3.75,21]))
