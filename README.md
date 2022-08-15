# deep-koopman
Koopman theory is a mathematical technique to achieve data-driven approximations of dynamical systems. This package implements Deep Koopman – a method to achieve Koopman approximations using deep neural networks to learn the dynamics of any system and predict its unknown states.

Key features:
- TODO write key differences from `pydmd` and `pykoopman`.
- other features

## Installation
```
git clone https://github.com/GaloisInc/deep-koopman.git
pip install -r requirements.txt
```

PyPI installation details coming soon!

## API Reference
Available at https://galoisinc.github.io/deep-koopman/.

## Theory
This section gives a brief overview. For a thorough mathematical treatment, refer to [`koopman_theory.pdf`](./koopman_theory.pdf).

Assume a dynamical system:
$$x_{t+1} = F(x_t)$$
- $x$ is the state of the system described by a collection of dependent variables.
- $t$ is the independent indexing variable.
- $F$ is the evolution rule describing the dynamics of the system, which is in general non-linear.

Koopman theory attempts to transform $x$ into a different space $g(x)$, where the dynamics of the system are linear, i.e.:
$$g(x_{t+1}) = Kg(x_t)$$
where $K$ is the Koopman matrix.

Linearizing the system is incredibly powerful since the state $x_T$ at any $t=T$ can be predicted from $K$ and the initial state $x_0$ as:
$$x_T = g^{-1}\left(K^Tg(x_0)\right)$$

## Quick Tutorial
We will walk through a tutorial on predicting the $200$-dimensional pressure vector $x$ across the surface of a [NACA0012 airfoil](https://en.wikipedia.org/wiki/NACA_airfoil) at varying angles of attack $t$. Run:
```bash
cd examples/naca0012
python run.py
```
The script is broken down below:

### Data
```python
with open('./data.pkl', 'rb') as f:
    data = pickle.load(f)
```
The resulting `data` dictionary is documented [here](https://galoisinc.github.io/deep-koopman/core.html#deepk.core.DeepKoopman). Its keys are:
- `ttr = range(15)`. Angle of attack values used for training. Note that the training indices must be in ascending order and should ideally be equally spaced.
- `Xtr` of shape `(15,200)`. Each row is the 200-dimensional pressure vector $x$ for the corresponding angle of attack.
- `tva = [0.5,3.5,4.5,7.5,16,17,20]`, and `tte = [1.5,2.5,5.5,6.5,15,18,19]`. Angle of attack values used for validation and testing, respectively. Note that these indices can be anything, order and spacing is not important.
- `Xva` and `Xte`. Corresponding pressure vectors. E.g. `Xva[0]` corresponds to $x_{0.5}$, `Xte[-1]` corresponds to $x_{19}$, etc.

### Deep Koopman run
```python
utils.set_seed(10)
```
This is used to seed the run. You may omit it, we have included it so that your tutorial results are the same as ours.

```python
dk = DeepKoopman(
    data = data,
    rank = 10,
    num_encoded_states = 100,
    encoder_hidden_layers = [100,50]
)
```
This creates the `DeepKoopman` object (documented [here](https://galoisinc.github.io/deep-koopman/core.html#deepk.core.DeepKoopman)). The `rank` is $10$, i.e. the Koopman matrix will be of dimension $10\times10$. The encoded vector $g(x)$ will be $100$-dimensional, and the encoder has 2 hidden layers of 100 and 50 nodes. Thus, the overall network looks like:
TODO diagram of encoder and decoder with pretty shapes

```python
dk.train_net()
dk.test_net()
```
Train and test the net. This uses all the default training settings, such as `numepochs=500`. All the defaults can be found [here](https://galoisinc.github.io/deep-koopman/core.html#deepk.core.DeepKoopman). When training starts, the UUID for the `DeepKoopman` object is printed. All resulting files are prefixed with this UUID.

### Results
The goal of Deep Koopman is three-fold, which correspond to three types of losses and errors:
- *Reconstruction* (`recon`): Learn $g(\cdot)$ and $g^{-1}(\cdot)$ as an autoencoder neural network which can successfully reconstruct its input.
- *Linearity* (`lin`): Learn $K$ such that dynamics in the $g(x)$ space are linear.
- *Prediction* (`pred`): Learn the overall system such that predictions for states in the training data match their actual values.

```python
utils.plot_stats(dk, ['recon_loss', 'recon_anae', 'lin_loss', 'lin_anae', 'pred_loss', 'pred_anae', 'loss'])
```
This plots the [L2 loss](https://galoisinc.github.io/deep-koopman/losses.html#deepk.losses.mse) and [average normalized absolute error (ANAE)](https://galoisinc.github.io/deep-koopman/errors.html#deepk.errors.anae) for all the 3 types above, as well as the [overall loss](https://galoisinc.github.io/deep-koopman/losses.html#deepk.losses.overall), which is a linear combination of the above losses and is optimized during training via gradient descent. We have included these plots with the `ref_` prefix. (TODO new sims and ref prefix)

```python
print(dk.predict_new([3.75,21]))
```
This uses the trained DeepKoopman model to print predictions for $x_{3.75}$ and $x_{21}$, i.e. the unknown $200$-dimensional pressure vector for angles of attack $3.75^{\circ}$ and $21^{\circ}$.

## Numerical instabilities
The mathematical theory behind Deep Koopman involves operations such as singular value decomposition, eigenvalue decomposition, and matrix inversion. These can lead to the gradients becoming numerically unstable.

Some common numerical instabilities are described in the [issues](https://github.com/GaloisInc/deep-koopman/issues?q=is%3Aissue+is%3Aclosed).

Specifically, to debug gradient issues, insert the following line in your script:
```python
torch.autograd.set_detect_anomaly(True)
```
This catches runtime errors such as exploding NaN gradients and displays where they happened in the forward pass. The cost is slower execution. 

## Acknowledgements
This material is based upon work supported by the United States Air Force and DARPA under Contract No. FA8750-20-C-0534. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the United States Air Force and DARPA. Distribution Statement A, "Approved for Public Release, Distribution Unlimited."

The authors also referred to the following sources while creating this package:
- J. N. Kutz, S. L. Brunton, B. W. Brunton, and J. L. Proctor, "Dynamic Mode Decomposition: Data Driven Modeling of Complex Systems", published by Society for Industrial and Applied Mathematics (2016). DOI https://doi.org/10.1137/1.9781611974508.
- B. Lusch, J. N. Kutz, and S. L. Brunton, "Deep learning for universal linear embeddings of nonlinear dynamics" in Nature Communication 9, 4950 (2018). DOI https://doi.org/10.1038/s41467-018-07210-0.
