# deep-koopman
Koopman theory is a mathematical technique to achieve data-driven approximations of dynamical systems. This repository implements a complete software tool to achieve Koopman theory using *machine learning and deep neural networks* to learn the dynamics of any system and predict its unknown states.

We acknowledge previous implementations Koopman theory via code. Our tool `deep-koopman` bridges the gap between two schools of prior work -- a) software packages that implement Dynamic Mode Decomposition without *learning* Koopman observables (e.g. [`pykoopman`](https://github.com/dynamicslab/pykoopman)), and b) efforts that learn Koopman observables, but are not generalized software tools (e.g. [Lusch et al.](https://github.com/BethanyL/DeepKoopman)).

## Key features of `deep-koopman`
- A generalized software tool to apply deep learning based Koopman theory to any system with arbitrary input data / dimensionality.
- Eigendecomposition to predict unknown states of any system in the future (*forward extrapolation*), past (*backward extrapolation*), and in-between (*interpolation*).
- Novel error functions for visualizing performance.
- Extensive options and a ready-to-use *hyperparameter search module* to customize training and improve performance.
- Built using [Pytorch](https://pytorch.org/), supports both CPU and GPU platforms.


## Installation

### From source
```
git clone https://github.com/GaloisInc/deep-koopman.git
pip install -r requirements.txt
```
Add the location to your Python path, i.e. `export PYTHONPATH="<clone_location>/deep-koopman:$PYTHONPATH"`

### With pip
Details coming soon!


## Tutorials and examples
Available in the [`examples`](./examples/) folder.


## API Reference
Available at https://galoisinc.github.io/deep-koopman/.


## Background
This section gives a brief overview. For a thorough mathematical treatment, refer to [`koopman_theory.pdf`](./koopman_theory.pdf).

Assume a dynamical system $x_{t+1} = F(x_t)$, where $x$ is the (multi-dimensional, i.e. vector) state of the system at index $t$, and $F$ is the evolution rule describing the dynamics of the system. Koopman theory attempts to transform $x$ into a different space $y = g(x)$ where the dynamics are linear, i.e. $y_{t+1} = Ky_t$, where $K$ is the Koopman matrix. Linearizing the system is incredibly powerful since the state $x_t$ at any $t$ can be predicted from $K$ and the initial state $x_0$ as $x_t = g^{-1}\left(K^tg(x_0)\right)$.

The Deep Koopman system in this package performs three tasks: (TODO rework)
- *Reconstruction* (`recon`): Learn an autoencoder architecture to create the pipeline $\hat{x} = g^{-1}(y) = g^{-1}\left(g(x)\right)$.
- *Linearity* (`lin`): Learn a Koopman matrix which can operate on the initial encoded state $y_0$ to yield approximations $\{y_1',y_2',\cdots\}$ to the actual values $\{y_1,y_2,\cdots\}$, as well as predict unknown $y_t'$ for values of $t$ not in the given data.
- *Prediction* (`pred`): $\{y_1',y_2',\cdots\}$ are decoded to predict approximations $\{\hat{x}_1',\hat{x}_2',\cdots\}$ to the actual values $\{x_1,x_2,\cdots\}$, as well as predict unknown $\hat{x}_t'$ for values of $t$ not in the given data. This is the task we care about the most.
<figure><center>
<img src="deepk_system.png" width=750/>
</center></figure>


## Known issues
Some common issues and ways to overcome them are described in the [known issues](https://github.com/GaloisInc/deep-koopman/issues?q=is%3Aissue+label%3Aknown-issue+is%3Aclosed).


## How to cite
Please cite the accompanying paper:
```
@article{Dey2022,
    author = {Sourya Dey and Ethan Lew and Eric Davis},
    title = {A complete deep learning software package for Koopman theory},
    year = {2022},
    note = {To be submitted to 5th Annual Learning for Dynamics & Control Conference (L4DC)}
}
```
TODO arXiv


## Acknowledgements and Distribution Statement
This material is based upon work supported by the United States Air Force and DARPA under Contract No. FA8750-20-C-0534. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the United States Air Force and DARPA. Distribution Statement A, "Approved for Public Release, Distribution Unlimited."
