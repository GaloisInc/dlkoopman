"""Configuration options"""


import torch

from dlkoopman.utils import is_torch_2

__pdoc__ = {
    'ConfigValidationError': False
}


class ConfigValidationError(Exception):
    """Raised when config does not validate."""


class Config():
    """Configuration options.

    ## Parameters
    - **precision** (*str, optional*) - Numerical precision of tensors. Must be one of `"half"` / `"float"` / `"double"`.
        - Note that setting `precision = "double"` may make predictions slightly more accurate, however, may also lead to inefficient GPU runtimes.

    - **use_cuda** (*bool, optional*) - If `True`, tensor computations will take place on CuDA GPUs if available.

    - **torch_compile_backend** (*str / None, optional*) - The backend to use for `torch.compile()`, which is a feature added in torch major version 2 to potentially speed up computation.
        - If you are using `torch 1.x` or you set `torch_compile_backend = None`, `torch.compile()` will not be invoked on the DLKoopman neural nets.
        - If you are using `torch 2.x`, full lists of possible backends can be obtained by running `torch._dynamo.list_backends()` and `torch._dynamo.list_backends(None)`. See the [`torch.compile()` documentation](https://pytorch.org/docs/stable/generated/torch.compile.html) for more details.

    - **normalize_Xdata** (*bool, optional*) - If `True`, all input states (training, validation, test) are divided by the maximum absolute value in the training data.
        - Note that normalizing data is a generally good technique for deep learning, and is normally done for each feature \\(f\\) in the input data as
        $$X_f = \\frac{X_f-\\text{offset}_f}{\\text{scale}_f}$$
        (where offset and scale are mean and standard deviation for Gaussian normalization, or minimum value and range for Minmax normalization.)<br>However, *this messes up the spectral techniques such as singular value and eigen value decomposition required in Koopman theory*. Hence, setting `normalize_Xdata=True` will just use a single scale value for normalizing the whole data to get
        $$X = \\frac{X}{\\text{scale}}$$
        which results in the singular and eigen vectors remaining the same.
        - *Caution*: Setting `normalize_Xdata=False` may end the run by leading to imaginary parts of tensors reaching values where the loss function depends on the phase (in `torch >= 1.11`, this leads to `"RuntimeError: linalg_eig_backward: The eigenvectors in the complex case are specified up to multiplication by e^{i phi}. The specified loss function depends on this quantity, so it is ill-defined"`). The only benefit to setting `normalize_Xdata=False` is that if the run successfully completes, the final error metrics such as [ANAE](https://galoisinc.github.io/dlkoopman/metrics.html#dlkoopman.metrics.anae) are slightly more accurate since they are reported on the un-normalized data values.

    - **use_exact_eigenvectors** (*bool, optional*) - If `True`, the exact eigenvectors of the Koopman matrix are used in `StatePred`, if `False`, the projected eigenvectors are used.
        - For a discussion on exact and projected eigenvectors, see [Tu et al](https://arxiv.org/abs/1312.0041) or [Chapter 1 of Kutz et al](https://epubs.siam.org/doi/pdf/10.1137/1.9781611974508.ch1). The basic idea is that using exact eigenvectors is more accurate, but their computation may become less numerically stable than that of projected eigenvectors.
        
    - **sigma_threshold** (*float, optional*) - When computing the SVD in `StatePred`, singular values lower than this will be reported, since they can be a possible cause of unstable gradients.

    ## Attributes
    - **RTYPE** (*torch.dtype*) - Data type of real tensors. Is automatically set to `torch.<precision>` (e.g. `torch.float` if `precision="float"`).

    - **CTYPE** (*torch.dtype*) - Data type of complex tensors. Is automatically set to `torch.c<precision>` (e.g. `torch.cfloat` if `precision="float"`).

    - **DEVICE** (*torch.device*) - Device where tensors reside. Is automatically set to `"cpu"` if `use_cuda=False` or CuDA is not available, otherwise `"cuda"`.
    """
    
    def __init__(self,
        precision = "float",
        use_cuda = True,
        torch_compile_backend = "aot_eager",
        normalize_Xdata = True,
        use_exact_eigenvectors = True,
        sigma_threshold = 1e-25
    ):
        self.precision = precision
        self.use_cuda = use_cuda
        self.torch_compile_backend = torch_compile_backend
        self.normalize_Xdata = normalize_Xdata
        self.use_exact_eigenvectors = use_exact_eigenvectors
        self.sigma_threshold = sigma_threshold

        if precision not in ["half", "float", "double"]:
            raise ConfigValidationError(f'`precision` must be either of "half" / "float" / "double", instead found {precision}')
        if use_cuda not in [True, False]:
            raise ConfigValidationError(f'`use_cuda` must be either True or False, instead found {use_cuda}')
        if is_torch_2() and torch_compile_backend not in torch._dynamo.list_backends() + torch._dynamo.list_backends(None) + [None]:
            raise ConfigValidationError(f'`torch_compile_backend` must be either None or one out the options obtained from running `torch._dynamo.list_backends()` or `torch._dynamo.list_backends(None)`, instead found {torch_compile_backend}')
            #NOTE: This test will not occur if is_torch_2() = False (i.e. torch major version is < 2), so `torch_compile_backend` can be anything. But this is okay because `torch_compile_backend` won't be used anywhere inside the code if is_torch_2() = False.
        if normalize_Xdata not in [True, False]:
            raise ConfigValidationError(f'`normalize_Xdata` must be either True or False, instead found {normalize_Xdata}')
        if use_exact_eigenvectors not in [True, False]:
            raise ConfigValidationError(f'`use_exact_eigenvectors` must be either True or False, instead found {use_exact_eigenvectors}')
        if type(sigma_threshold) not in [int, float]:
            raise ConfigValidationError(f'`sigma_threshold` must be a number, instead found {sigma_threshold}')

        self.RTYPE = torch.half if self.precision=="half" else torch.float if self.precision=="float" else torch.double
        self.CTYPE = torch.chalf if self.precision=="half" else torch.cfloat if self.precision=="float" else torch.cdouble
        self.DEVICE = torch.device("cuda" if self.use_cuda and torch.cuda.is_available() else "cpu")
