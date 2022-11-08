"""Neural nets used inside the models."""


import torch


class _MLP(torch.nn.Module):
    """Multi-layer perceptron neural net.

    ## Parameters
    - **input_size** (*int*) - Dimension of the input layer.

    - **output_size** (*int*) - Dimension of the output layer.

    - **hidden_sizes** (*list[int], optional*) - Dimensions of hidden layers, if any.

    - **batch_norm** (*bool, optional*) - Whether to use batch normalization.

    ## Attributes
    - **net** (*torch.nn.ModuleList*) - List of layers
    """
    def __init__(self, input_size, output_size, hidden_sizes=[], batch_norm=False):
        """ """
        super().__init__()
        self.net = torch.nn.ModuleList([])
        layers = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layers)-1):
            self.net.append(torch.nn.Linear(layers[i],layers[i+1]))
            if i != len(layers)-2: #all layers except last
                if batch_norm:
                    self.net.append(torch.nn.BatchNorm1d(layers[i+1]))
                self.net.append(torch.nn.ReLU())

    def forward(self, X) -> torch.Tensor:
        """Forward propagation of neural net.

        ## Parameters
        - **X** (*torch.Tensor, shape=(\\*,input_size)*) - Input data to net.

        ## Returns 
        - **X** (*torch.Tensor, shape=(\\*,output_size)*) - Output data from net.
        """
        for layer in self.net:
            X = layer(X)
        return X


class AutoEncoder(torch.nn.Module):
    """AutoEncoder neural net. Contains an encoder connected to a decoder, both are multi-layer perceptrons.

    ## Parameters
    - **input_size** (*int*) - Number of dimensions in original data (encoder input) and reconstructed data (decoder output).

    - **encoded_size** (*int*) - Number of dimensions in encoded data (encoder output and decoder input).

    - **encoder_hidden_layers** (*list[int], optional*) - Encoder will have layers = `[input_size, *encoder_hidden_layers, encoded_size]`. If not set, defaults to reverse of `decoder_hidden_layers`. If that is also not set, defaults to `[]`.

    - **decoder_hidden_layers** (*list[int], optional*) - Decoder has layers = `[encoded_size, *decoder_hidden_layers, input_size]`. If not set, defaults to reverse of `encoder_hidden_layers`. If that is also not set, defaults to `[]`.

    - **batch_norm** (*bool, optional*): Whether to use batch normalization.

    ## Attributes
    - **encoder** - Encoder neural net.

    - **decoder** - Decoder neural net.
    """
    def __init__(self, input_size, encoded_size, encoder_hidden_layers=[], decoder_hidden_layers=[], batch_norm=False):
        """ """
        super().__init__()

        if not decoder_hidden_layers and encoder_hidden_layers:
            decoder_hidden_layers = encoder_hidden_layers[::-1]
        elif not encoder_hidden_layers and decoder_hidden_layers:
            encoder_hidden_layers = decoder_hidden_layers[::-1]

        self.encoder = _MLP(
            input_size = input_size,
            output_size = encoded_size,
            hidden_sizes = encoder_hidden_layers,
            batch_norm = batch_norm
        )

        self.decoder = _MLP(
            input_size = encoded_size,
            output_size = input_size,
            hidden_sizes = decoder_hidden_layers,
            batch_norm = batch_norm
        )

    def forward(self, X) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation of neural net.

        ## Parameters
        - **X** (*torch.Tensor, shape=(\\*,input_size)*) - Input data to encoder.

        ## Returns 
        - **Y** (*torch.Tensor, shape=(\\*,encoded_size)*) - Encoded data, i.e. output from encoder, input to decoder.

        - **Xr** (*torch.Tensor, shape=(\\*,input_size)*) - Reconstructed data, i.e. output of decoder.
        """
        Y = self.encoder(X) # encoder complete output
        Xr = self.decoder(Y) # final reconstructed output
        return Y, Xr


class Knet(torch.nn.Module):
    """Linear neural net to approximate the Koopman matrix.
    
    Contains identically sized input and output layers, no hidden layers, no bias vector, and no activation function.

    ## Parameters
    - **size** (*int*) - Dimension of the input and output layer.

    ## Attributes
    - **net** (*torch.nn.ModuleList*) - The neural net.
    """
    def __init__(self, size):
        """ """
        super().__init__()
        self.net = torch.nn.Linear(
            in_features = size,
            out_features = size,
            bias = False
        )

    def forward(self, X) -> torch.Tensor:
        """Forward propagation of neural net.

        ## Parameters
        - **X** (*torch.Tensor, shape=(\\*, size)*) - Input data to net.

        ## Returns 
        - **X** (*torch.Tensor, shape=(\\*, size)*) - Output data from net.
        """
        return self.net(X)
