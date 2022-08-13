# deep-koopman
Koopman theory is a mathematical technique to achieve data-driven approximations of dynamical systems. This package implements Deep Koopman – a method to achieve Koopman approximations by using deep neural networks to learn the dynamics of a system and make predictions about its states that are unknown.

## Installation
TODO

## Documentation
Full API reference is available at https://galoisinc.github.io/deep-koopman/.

## Key features
TODO

Write key differences from `pydmd` and `pykoopman`.

## Brief Theory
Given a dynamical system:
$$\bm{x}_{t+1} = f(\bm{x}_t)$$
- $\bm{x}$ is the state of the system described by a collection of dependent variables
- $t$ is the independent indexing variable
- $f$ is the evolution rule describing the dynamics of the system, which Koopman theory attempts to model.

TODO brief theory, and link to Overleaf for complete theory

## Examples
TODO examples folder, maybe use CFD as an example

## Numerical instabilities
The mathematical theory behind Deep Koopman involves operations such as singular values, eigenvalue decomposition, and matrix inversion. These can lead to the gradients becoming numerically unstable. To debug gradient issues, insert the following line in your script:
```python
torch.autograd.set_detect_anomaly(True)
```
This catches runtime errors such as exploding NaN gradients and displays where they happened in the forward pass. The cost is slower execution. Some common numerical instabilities are described in the issues. TODO write the following as Github issues.

### Ill-defined loss function depending on eigenvector phase
Specifically: `RuntimeError: linalg_eig_backward: The eigenvectors in the complex case are specified up to multiplication by e^{i phi}. The specified loss function depends on this quantity, so it is ill-defined.` This check was added in `torch >= 1.11`.

Causes:
- Dominant eigenvalue has magnitude greater than 1. **This is the most common cause for training to fail**.
- `normalize_Xdata` is set to False, see its note in the documentation of `DeepKoopman.__init__()`.
- Rank is too high, in particular, `>=` than `num_encoded_states`. Obviously this will lead to errors because rank is supposed to strictly *reduce* the effective number of encoded states to a lower value than `num_encoded_states`. Note that numerical issues can also occur if rank is higher than any of the numbers in `encoder_hidden_layers`.

### Gradients become NaN
Causes:
- Some singular values of the `Y` matrix are [very close to each other](https://github.com/tensorflow/tensorflow/issues/17476#issue-302663705), or [very small](https://github.com/google/jax/issues/2311#issue-571251263) during the original SVD computation prior to truncation in `dmd_linearize()`. This makes their gradients explode.
    - To avoid NaNs, use the custom `SVD()` in `utils.py`.
    - To avoid large values, use the gradient clipping methods in `utils.py`.
- Rank is too high - makes condition number and largest eigenvalue very large
    - This can be partially mitigated by discarding low singular values for the pinv computation by decreasing `cond_threshold` to a low value such as 10 or 2.

### Losses and/or ANAEs become very high
Causes:
- Rank is too low and `K_reg` is too high - makes the entries of the `Ktilde` matrix and correspondingly its largest eigenvalue very small.
- Non-dominant eigenvalues of the `Ktilde` matrix can be much less than 1, even if the dominant one is 1. This makes the system implode backwards in time (since something between 0 and 1 to the power of a negative integer is very large, e.g. 0.1**(-5) = 100000, and for such values, even the non-dominant eigenvalues affect the system).
    - This can be mitigated (albeit in a cheating sort of way) by not predicting any past values. Either include early values in the training data or discard them, so that at the end of the day predictions are only done on intermediate and future values.

## References
TODO

## Acknowledgements
TODO



