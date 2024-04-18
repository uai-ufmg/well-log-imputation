'''
Neural Networks implementation files
It contains the implementation of a simple NeuralNet with Fully-Connected layers and a 
Autoencoder net (AENN) based on this NeuralNet class, with a NN encoder and decoder
'''

import torch
import numpy as np
from torch import nn

class NeuralNet(nn.Module):
    '''
    A torch.nn.Module that represents a simple Neural Network with Fully-Connected layers
    
    @attr net: A sequential module with the defined FC layers and activation fuctions
    '''
    
    def __init__(self, inp_size: int, num_class: int, layers: list[int], activation: str = 'relu') -> None:
        '''
        Construct a NeuralNet object.
        
        @param inp_size: size of the input data. In a forward-pass the input tensor has to be of shape (batch_size, inp_size)
        @param num_class: number of neurons of the last layer
        @param layers: list with the number of neurons in each layer excluding the last one
        @param activation: str with the type of non-linear activation function to be placed after each layer. Can be `relu`, `tanh` or `sigmoid`
        '''
        
        super(NeuralNet, self).__init__()

        layers_ = []

        in_ = inp_size
        for i, l in enumerate(layers):
            layers_.append(nn.Linear(in_, l))
            
            if activation == 'relu': layers_.append(nn.ReLU())
            elif activation == 'tanh': layers_.append(nn.Tanh())
            else: layers_.append(nn.Sigmoid())

            in_ = l
        layers_.append(nn.Linear(in_, num_class))
        
        self.net = nn.Sequential(*layers_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''A forward pass through the model'''
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        '''A prediction of the class of input x'''
        p = self.net(x)
        return torch.max(p, 1)[1]

    def score(self, x, y):
        '''The accuracy of the model with input x'''
        p = self.predict(x)
        err = torch.sum((p == y)*(1.0/len(y))).item()
        return err
    

class AENN(nn.Module):
    '''
    A torch.nn.Module that represents a simple autoencoder construct with Neural Network modules
    
    @attr enc: A NeuralNet module that represents the encoder part of the autoencoder
    @attr dec: A NeuralNet module that represents the decoder part of the autoencoder
    '''
    
    def __init__(self, enc_layers=[256], enc_activation='relu',
                       dec_layers=None, dec_activation=None, mirrored=True,
                       input_size=28*28, latent_dim=128):
        '''
        @param enc_layers: the number of neurons in each layer of the encoder
        @param enc_activation: type of non linear activation of the encoder layers (relu, tanh or sigmoid)
        @param dec_layers: the number of neurons in each layer of the decoder
        @param dec_activation: type of non linear activation of the decoder layers (relu, tanh or sigmoid)
        @param mirrored: if true, the decoder will be a mirror of the encoder 
                          i.e. the number of neurons in the layers will be in the reverse order of the encoder layers. It will also use the same activation function            
        @param input_size: size of the input data. In a forward-pass the input tensor has to be of shape (batch_size, inp_size)
        @param latent_dim: the size of latent space of the autoencoder
        '''
        
        super(AENN, self).__init__()
        
        self.enc = NeuralNet(input_size, latent_dim, enc_layers, enc_activation)
        
        _dec_layers = dec_layers
        _dec_activation = dec_activation
        if mirrored:
            _dec_layers = reversed(enc_layers)
            _dec_activation = enc_activation
        
        self.dec = NeuralNet(latent_dim, input_size, _dec_layers, _dec_activation)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''A forward pass through the model. It encodes the input then decodes it'''
        x_enc = self.enc(x)
        x_rec = self.dec(x_enc)
                
        return x_rec
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        '''A forward pass through the encoder module'''
        return self.enc(x)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        '''A forward pass through the decoder module'''
        return self.dec(x)

    
def _reconstruction_loss(y_pred: np.ndarray | torch.Tensor, y_true: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor = None) -> torch.Tensor:
    '''
    Computes the MSE loss of a prediction with optional mask
    
    @param y_pred: predicted values
    @param y_true: true values
    @param mask: mask array. Only where it is equal to 1 the error is computed
    
    @return MSE error
    '''
    y_true = y_true[mask == 1.]
    y_pred = y_pred[mask == 1.]

    return torch.nn.functional.mse_loss(y_pred, y_true)