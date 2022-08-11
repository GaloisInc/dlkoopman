import torch


########################################################################
# Set dtype in the following line
########################################################################
dtype = "float" #either 'half' or 'float' or 'double'


########################################################################
# Do not change
########################################################################
RTYPE = torch.float16 if dtype=="half" else torch.float32 if dtype=="float" else torch.float64
CTYPE = torch.complex32 if dtype=="half" else torch.complex64 if dtype=="float" else torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
