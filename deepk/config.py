"""Configuration options"""


import torch


precision = "float"
"""Numerical precision of tensors.

**Options** (in increasing order of precision): `"half"` / `"float"` / `"double"`.

**Default**: `"float"`

## Note
Setting `precision = "double"` may help slightly with numerical instabilities, however, may also lead to inefficient GPU runtimes.
"""

use_cuda = True
"""If `True`, tensor computations when take place on CuDA GPUs whenever available.

**Options**: `True` / `False`

**Default**: `True`
"""

normalize_Xdata = True
"""If `True`, all input states (training, validation, test) are divided by the maximum absolute value in the training data.

**Options**: `True` / `False`

**Default**: `True`

## Notes

Normalizing data is a generally good technique for deep learning, and is normally done for each feature \\(f\\) in the input data as
$$X_f = \\frac{X_f-\\text{offset}_f}{\\text{scale}_f}$$
(where offset and scale are mean and standard deviation for Gaussian normalization, or minimum value and range for Minmax normalization.) However, *this messes up the spectral techniques such as singular value and eigen value decomposition required in Koopman theory*. Hence, setting `normalize_Xdata=True` will just use a single scale value for normalizing the whole data to get
$$X = \\frac{X}{\\text{scale}}$$
This results in the singular and eigen vectors remaining the same.

## Caution
Setting this to `normalize_Xdata=False` may end the run by leading to imaginary parts of tensors reaching values where the loss function depends on the phase (in `torch >= 1.11`, this leads to `"RuntimeError: linalg_eig_backward: The eigenvectors in the complex case are specified up to multiplication by e^{i phi}. The specified loss function depends on this quantity, so it is ill-defined"`). The only benefit to setting this to `False` is that if the run successfully completes, the final error metrics such as ANAE are slightly more accurate since they are reported on the un-normalized data values.
"""

use_exact_eigenvectors = True
"""If `True`, the exact eigenvectors of the Koopman matrix are used in `StatePredictor`, if `False`, the projected eigenvectors are used.

**Options**: `True` / `False`

**Default**: `True`

## Notes

For a discussion on exact and projected eigenvectors, see [Tu et al](https://arxiv.org/abs/1312.0041) or [Chapter 1 of Kutz et al](https://epubs.siam.org/doi/pdf/10.1137/1.9781611974508.ch1). The basic idea is that using exact eigenvectors is more accurate, but their computation may become less numerically stable than that of projected eigenvectors.
"""

sigma_threshold = 1e-25
"""When computing the SVD in `StatePredictor`, singular values lower than this will be reported, since they can be a possible cause of unstable gradients.

**Options**: Any numerical value.

**Default**: `1e-25`
"""

###############################################################################
# Validate config
# Run automatically whenever config is imported
###############################################################################
import sys
_error = False

try:
    assert precision in ["half", "float", "double"], '`precision` must be either of "half" / "float" / "double"'
except AssertionError as e:
    print(f'Config Validation Error: {e}')
    _error = True
try:
    assert use_cuda in [True, False], '`use_cuda` must be either True or False'
except AssertionError as e:
    print(f'Config Validation Error: {e}')
    _error = True
try:
    assert normalize_Xdata in [True, False], '`normalize_Xdata` must be either True or False'
except AssertionError as e:
    print(f'Config Validation Error: {e}')
    _error = True
try:
    assert use_exact_eigenvectors in [True, False], '`use_exact_eigenvectors` must be either True or False'
except AssertionError as e:
    print(f'Config Validation Error: {e}')
    _error = True
try:
    assert type(sigma_threshold) in [int, float], '`sigma_threshold` must be a number'
except AssertionError as e:
    print(f'Config Validation Error: {e}')
    _error = True

if _error:
    print('\nConfig validation failed, exiting!')
    sys.exit()
###############################################################################


###############################################################################
# Set other constants from config values
###############################################################################
_RTYPE = torch.half if precision=="half" else torch.float if precision=="float" else torch.double
_CTYPE = torch.chalf if precision=="half" else torch.cfloat if precision=="float" else torch.cdouble
_DEVICE = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
###############################################################################
