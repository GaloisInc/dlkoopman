<figure>
<img src="https://raw.githubusercontent.com/GaloisInc/dlkoopman/36108ffcfd9608a393985ac9af431d3910fe2fc5/logo.png" height=150/>
</figure>

**DLKoopman: A general-purpose Python package for Koopman theory using deep learning**.

Koopman theory is a technique to encode sampled data (aka states) of a nonlinear dynamical system into a linear domain. This is very powerful as a linear model can:
- Give insight into the dynamics via eigenvalues and eigenvectors.
- Leverage linear algebra techniques to easily analyze the system and predict its behavior under unknown conditions.


## Why DLKoopman?
*DLKoopman uses deep learning to learn an encoding of a nonlinear dynamical system into a linear domain, while simultaneously learning the dynamics of the linear model*. DLKoopman bridges the gap between:
- Software packages that restrict the learning of a good encoding (e.g. [`pykoopman`](https://github.com/dynamicslab/pykoopman)), and
- Efforts that learn encodings for specific applications instead of being a general-purpose tool (e.g. [`DeepKoopman`](https://github.com/BethanyL/DeepKoopman)).

### Key DLKoopman features
- State prediction (`StatePred`) - Train on individual states of a system, then predict unknown states.
    - E.g: What is the pressure vector on this aircraft for $23.5^{\circ}$ angle of attack?
- Trajectory prediction (`TrajPred`) - Train on generated trajectories of a system, then predict unknown trajectories for new initial states.
    - E.g: What is the behavior of this pendulum if I start from the point $[1,-1]$?
- General-purpose and reusable - supports data from any dynamical system.
- Novel error function Average Normalized Absolute Error (ANAE) for visualizing performance.
- Extensive options and a ready-to-use hyperparameter search module to improve performance.
- Built using [Pytorch](https://pytorch.org/), supports both CPU and GPU platforms.

Read more about DLKoopman in this [blog article](https://galois.com/blog/2023/01/dl-koopman/).


## Installation

### With pip (for regular users)
`pip install dlkoopman`

### From source (for development)
```
git clone https://github.com/GaloisInc/dlkoopman.git
cd dlkoopman
pip install .
```

### Running as a Docker container
DLKoopman can also be run as a docker container by pulling the image from `galoisinc/dlkoopman:<version>`, e.g. `docker pull galoisinc/dlkoopman:v1.2.0`.


## Tutorials and examples
Available in the [`examples`](https://github.com/GaloisInc/dlkoopman/tree/ed11bef92b90112d9ca90722942a6789e6af7d5a/examples) folder.


## Documentation and API Reference
Available at https://galoisinc.github.io/dlkoopman/.

## Changelog
See [Releases](https://github.com/GaloisInc/dlkoopman/releases) and their notes.


## Description 

### Koopman theory
Assume a dynamical system $x_{i+1} = F(x_i)$, where $x$ is the (genrally multi-dimensional) state of the system at index $i$, and $F$ is the (generally nonlinear) evolution rule describing the dynamics of the system. Koopman theory attempts to *encode* $x$ into a different space $y = g(x)$ where the dynamics are linear, i.e. $y_{i+1} = Ky_i$, where $K$ is the Koopman matrix. This is incredibly powerful since the state $y_i$ at any index $i$ can be predicted from the initial state $y_0$ as $y_i = K^iy_0$. This is then *decoded* back into the original space as $x = g^{-1}(y)$.

For a thorough mathematical treatment, see [this technical report](https://arxiv.org/abs/2211.07561).

### dlkoopman training
<figure>
<img src="https://raw.githubusercontent.com/GaloisInc/dlkoopman/ed11bef92b90112d9ca90722942a6789e6af7d5a/training_architecture.png" width=750/>
</figure>

This is a small example with three input states $\left[x_0, x_1, x_2\right]$. These are passed through an encoder neural network to get encoded states $\left[y_0, y_1, y_2\right]$. These are passed through a decoder neural network to get $\left[\hat{x}_0, \hat{x}_1, \hat{x}_2\right]$, and also used to learn $K$. This is used to derive predicted encoded states $\left[\mathsf{y}_1, \mathsf{y}_2\right]$, which are then passed through the same decoder to get predicted approximations $\left[\hat{\mathsf{x}}_1, \hat{\mathsf{x}}_2\right]$ to the original input states.

Errors mimimized during training:
- Train the autoencoder - Reconstruction `recon` between $x$ and $\hat{x}$.
- Train the Koopman matrix - Linearity `lin` between $y$ and $\mathsf{y}$.
- Combine the above - Prediction `pred` between $x$ and $\hat{\mathsf{x}}$.

### dlkoopman prediction
<figure>
<img src="https://raw.githubusercontent.com/GaloisInc/dlkoopman/ed11bef92b90112d9ca90722942a6789e6af7d5a/prediction_architecture.png" width=750/>
</figure>

Prediction happens after training.

(a) State prediction - Compute predicted states for new indexes such as $i'$. This uses the eigendecomposition of $K$, so $i'$ can be any real number - positive (forward extapolation), negative (backward extrapolation), or fractional (interpolation).

(b) Trajectory prediction - Generate predicted trajectories $j'$ for new starting states such as $x^{j'}_0$. This uses a linear neural net layer to evolve the initial state.


## Known issues
Some common issues and ways to overcome them are described in the [known issues](https://github.com/GaloisInc/dlkoopman/issues?q=is%3Aissue+is%3Aclosed+label%3Aknown-issue).


## How to cite
Please cite the [accompanying paper](https://proceedings.mlr.press/v211/dey23a.html):
```
@inproceedings{Dey2023_L4DC,
    author = {Sourya Dey and Eric William Davis},
    title = {{DLKoopman: A deep learning software package for Koopman theory}},
    booktitle = {Proceedings of The 5th Annual Learning for Dynamics and Control Conference},
    pages = {1467--1479},
    volume = {211},
    series = {Proceedings of Machine Learning Research},
    publisher = {PMLR},
    year = {2023},
    month = {Jun}
}
```


## References
- B. O. Koopman - Hamiltonian systems and transformation in Hilbert space
- J. Nathan Kutz, Steven L. Brunton, Bingni Brunton, Joshua L. Proctor - Dynamic Mode Decomposition
- Bethany Lusch, J. Nathan Kutz & Steven L. Brunton - Deep learning for universal linear embeddings of nonlinear dynamics


## Distribution Statement
This material is based upon work supported by the United States Air Force and DARPA under Contract No. FA8750-20-C-0534. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the United States Air Force and DARPA. Distribution Statement A, "Approved for Public Release, Distribution Unlimited."
