"""Configuration options"""


precision = "float"
"""Numerical precision of tensors.

Options (in increasing order of precision) = `"half"` / `"float"` / `"double"`.

**Default**: `"float"`

## Note
Setting `precision = "double"` may help sloghtly with numerical instabilities, however, may also lead to inefficient GPU runtimes.
"""

use_cuda = True
"""If `True`, tensor computations when take place on CuDA GPUs whenever available.

**Default**: `True`
"""

normalize_Xdata = True
"""If `True`, all input data (training, validation, test) are divided by the maximum absolute value in the training data.

**Default**: `True`

## Notes

Normalizing data is a generally good technique for deep learning, and is normally done for each feature \\(f\\) in the input data as
$$X_f = \\frac{X_f-\\text{offset}_f}{\\text{scale}_f}$$
(where offset and scale are mean and standard deviation for Gaussian normalization, or minimum value and range for Minmax normalization.) However, *this messes up the spectral techniques such as singular value and eigen value decomposition required in Deep Koopman*. Hence, setting `normalize_Xdata=True` will just use a single scale value for normalizing the whole data to get
$$X = \\frac{X}{\\text{scale}}$$
This results in the spectra (i.e. the singular and eigen vectors) remaining the same, the only change is that the singular and eigen values get divided by scale.

## Caution
Setting this to `normalize_Xdata=False` may end the run by leading to imaginary parts of tensors reaching values where the loss function depends on the phase (in `torch >= 1.11`, this leads to `"RuntimeError: linalg_eig_backward: The eigenvectors in the complex case are specified up to multiplication by e^{i phi}. The specified loss function depends on this quantity, so it is ill-defined"`). The only benefit to setting this to `False` is that if the run successfully completes, the final error metrics such as ANAE are slightly more accurate since they are reported on the un-normalized data values.
"""

use_custom_stable_svd = True
""" The singular value decomposition used is `utils.stable_svd` if `True`, and `torch.linalg.svd` if `False`.

**Default**: `True`

## Caution
Setting this to `False` may end the run due to encountering NaNs in gradients, as explained [here](https://pytorch.org/docs/stable/generated/torch.linalg.svd.html).
"""

sigma_threshold = 1e-25
"""Any singular value lower than this will be reported when in debug mode, since this is a possible cause of unstable gradients.

**Default**: `1e-25`
"""
