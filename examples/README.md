# Examples and Tutorials
Tutorials are in the form of Jupyter notebooks. You can view them online. If you want to run them interactively, please perform the following steps first:
- Clone the repo: `git clone https://github.com/GaloisInc/deep-koopman.git`
- Install requirements: `pip install -r requirements.txt`
- Add the location to your Python path: `export PYTHONPATH="<clone_location>/deep-koopman:$PYTHONPATH"`

You are now ready to go!

## State Predictor (on NACA0012 data)
The folder [`state_predictor_naca0012`](./state_predictor_naca0012/) contains tutorials on:
1. [Running `StatePredictor`](./state_predictor_naca0012/run.ipynb).
2. [Performing hyperparameter search](./state_predictor_naca0012/hyp_search.ipynb).

## Trajectory Predictor (on discrete spectrum data)
The folder [`trajectory_predictor_discrete_spectrum`](./trajectory_predictor_discrete_spectrum/) contains tutorials on:
1. [Running `TrajectoryPredictor`](./trajectory_predictor_discrete_spectrum/run.ipynb).
2. [Performing hyperparameter search](./trajectory_predictor_discrete_spectrum/hyp_search.ipynb).
