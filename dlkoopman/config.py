"""Configuration options"""


###############################################################################
# Change config options in this section as desired
###############################################################################

precision = "float"
"""Numerical precision of tensors.

**Options** (in increasing order of precision): `"half"` / `"float"` / `"double"`.

**Default**: `"float"`

## Note
Setting `precision = "double"` may make predictions slightly more accurate, however, may also lead to inefficient GPU runtimes.
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
"""If `True`, the exact eigenvectors of the Koopman matrix are used in `StatePred`, if `False`, the projected eigenvectors are used.

**Options**: `True` / `False`

**Default**: `True`

## Notes

For a discussion on exact and projected eigenvectors, see [Tu et al](https://arxiv.org/abs/1312.0041) or [Chapter 1 of Kutz et al](https://epubs.siam.org/doi/pdf/10.1137/1.9781611974508.ch1). The basic idea is that using exact eigenvectors is more accurate, but their computation may become less numerically stable than that of projected eigenvectors.
"""

sigma_threshold = 1e-25
"""When computing the SVD in `StatePred`, singular values lower than this will be reported, since they can be a possible cause of unstable gradients.

**Options**: Any numerical value.

**Default**: `1e-25`
"""

###############################################################################
# End of config options
###############################################################################




###############################################################################
# Do not change anything from here on
###############################################################################


__pdoc__ = {
    'ConfigValidationError': False
}

# Config validation - runs automatically whenever config is imported

class ConfigValidationError(Exception):
    """Raised when config does not validate."""

if precision not in ["half", "float", "double"]:
    raise ConfigValidationError(f'`precision` must be either of "half" / "float" / "double", instead found {precision}')
if use_cuda not in [True, False]:
    raise ConfigValidationError(f'`use_cuda` must be either True or False, instead found {use_cuda}')
if normalize_Xdata not in [True, False]:
    raise ConfigValidationError(f'`normalize_Xdata` must be either True or False, instead found {normalize_Xdata}')
if use_exact_eigenvectors not in [True, False]:
    raise ConfigValidationError(f'`use_exact_eigenvectors` must be either True or False, instead found {use_exact_eigenvectors}')
if type(sigma_threshold) not in [int, float]:
    raise ConfigValidationError(f'`sigma_threshold` must be a number, instead found {sigma_threshold}')


# Set other constants from config values

import torch

_RTYPE = torch.half if precision=="half" else torch.float if precision=="float" else torch.double
_CTYPE = torch.chalf if precision=="half" else torch.cfloat if precision=="float" else torch.cdouble
_DEVICE = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
