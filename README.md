# deep-koopman
Koopman theory is a mathematical technique to achieve data-driven approximations of dynamical systems. This package implements Deep Koopman – a method to achieve Koopman approximations using deep neural networks to learn the dynamics of any system and predict its unknown states.

Key features:
- TODO write key differences from `pydmd` and `pykoopman`.
- other features

## Installation
`git clone https://github.com/GaloisInc/deep-koopman.git`

Python package coming soon!

## Quick Tutorial
```
cd examples/naca0012
```
TODO

## API Reference
Full API reference is available at https://galoisinc.github.io/deep-koopman/.

## Brief Theory
Assume a dynamical system:
$$x_{t+1} = F(x_t)$$
- $x$ is the state of the system described by a collection of dependent variables.
- $t$ is the independent indexing variable.
- $F$ is the evolution rule describing the dynamics of the system, which is in general non-linear.

Koopman theory attempts to transform $x$ into a different space $g(x)$, where the dynamics of the system are linear, i.e.:
$$g(x_{t+1}) = Kg(x_t)$$
where $K$ is a matrix, also known as the Koopman operator.

This is incredibly powerful since the state $x_T$ at any $t=T$ can be predicted from $K$ and the initial state $x_0$ as:
$$x_T = g^{-1}\left(K^Tg(x_0)\right)$$
The goal of Deep Koopman is three-fold:
- Learn $g(\cdot)$ and $g^{-1}(\cdot)$ as an autoencoder which can *reconstruct* its input.
- Learn $K$ such that dynamics in the $g(x)$ space are *linear*.
- Learn the overall system such that *predictions* for states in the validation and test sets match their actual values.

For the complete mathematical theory, refer to [`koopman_theory.pdf`](./koopman_theory.pdf).

## Numerical instabilities
The mathematical theory behind Deep Koopman involves operations such as singular value decomposition, eigenvalue decomposition, and matrix inversion. These can lead to the gradients becoming numerically unstable.

Some common numerical instabilities are described in the [issues](https://github.com/GaloisInc/deep-koopman/issues?q=is%3Aissue+is%3Aclosed).

Specifically, to debug gradient issues, insert the following line in your script:
```python
torch.autograd.set_detect_anomaly(True)
```
This catches runtime errors such as exploding NaN gradients and displays where they happened in the forward pass. The cost is slower execution. 

## References
- J. N. Kutz, S. L. Brunton, B. W. Brunton, and J. L. Proctor, "Dynamic Mode Decomposition: Data Driven Modeling of Complex Systems", published by Society for Industrial and Applied Mathematics (2016). DOI https://doi.org/10.1137/1.9781611974508.
- B. Lusch, J. N. Kutz, and S. L. Brunton, "Deep learning for universal linear embeddings of nonlinear dynamics" in Nature Communication 9, 4950 (2018). DOI https://doi.org/10.1038/s41467-018-07210-0.

## Acknowledgements
TODO



