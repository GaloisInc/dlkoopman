import torch


class MLP(torch.nn.Module):
    """Multi-layer perceptron neural net.

    ## Parameters
    - **input_size** (*int*) - Dimension of the input layer.

    - **output_size** (*int*) - Dimension of the output layer.

    - **hidden_sizes** (*list(int), optional*) - Dimensions of hidden layers, if any.

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
    """AutoEncoder neural net. Contains an encoder MLP connected to a decoder MLP.

    ## Parameters
    - **num_input_states** (*int*) - Number of dimensions in original data (`X`) and reconstructed data (`Xr`).

    - **num_encoded_states** (*int*) - Number of dimensions in encoded data (`Y`).

    - **encoder_hidden_layers** (*list(int), optional*) - Encoder will have layers = `[num_input_states, *encoder_hidden_layers, num_encoded_states]`. If not set, defaults to reverse of `decoder_hidden_layers`. If that is also not set, defaults to `[]`.

    - **decoder_hidden_layers** (*list(int), optional*) - Decoder has layers = `[num_encoded_states, *decoder_hidden_layers, num_input_states]`. If not set, defaults to reverse of `encoder_hidden_layers`. If that is also not set, defaults to `[]`.

    - **batch_norm** (*bool, optional*): Whether to use batch normalization.

    ## Attributes
    - **encoder** (*MLP*) - Encoder neural net.

    - **decoder** (*MLP*) - Decoder neural net.
    """
    def __init__(self, num_input_states, num_encoded_states, encoder_hidden_layers=[], decoder_hidden_layers=[], batch_norm=False):
        """ """
        super().__init__()

        if not decoder_hidden_layers and encoder_hidden_layers:
            decoder_hidden_layers = encoder_hidden_layers[::-1]
        elif not encoder_hidden_layers and decoder_hidden_layers:
            encoder_hidden_layers = decoder_hidden_layers[::-1]

        self.encoder = MLP(
            input_size = num_input_states,
            output_size = num_encoded_states,
            hidden_sizes = encoder_hidden_layers,
            batch_norm = batch_norm
        )

        self.decoder = MLP(
            input_size = num_encoded_states,
            output_size = num_input_states,
            hidden_sizes = decoder_hidden_layers,
            batch_norm = batch_norm
        )

    def forward(self, X) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation of neural net.

        ## Parameters
        - **X** (*torch.Tensor, shape=(\\*,num_input_states)*) - Input data to encoder.

        ## Returns 
        - **Y** (*torch.Tensor, shape=(\\*,num_encoded_states)*) - Encoded data, i.e. output from encoder, input to decoder.

        - **Xr** (*torch.Tensor, shape=(\\*,num_input_states)*) - Reconstructed data, i.e. output of decoder.
        """
        Y = self.encoder(X) # encoder complete output
        Xr = self.decoder(Y) # final reconstructed output
        return Y, Xr
