from collections import OrderedDict
import os
import torch
from torch.nn import Linear, LayerNorm, ReLU, Sigmoid, BatchNorm1d, Dropout, Sequential, Identity, Tanh

import torch_geometric
from torch_geometric.nn.sequential import Sequential as PYG_Sequential
from torch_geometric.nn import global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv, GINConv

from config import config


class GeneralGNN(torch.nn.Module):
    def __init__(self, model_config): 
        super(GeneralGNN, self).__init__()
        self.builder_GNN_conv = model_config.get('builder_GNN_conv', None)
        self.builder_MLP = model_config.get('builder_MLP', None)
        self._check_existance({
            'builder_GNN_conv': self.builder_GNN_conv,
            'builder_MLP': self.builder_MLP
        })

        self.GNN_conv = self.builder_GNN_conv(model_config)
        self.MLP = self.builder_MLP(model_config)
        self._check_existance({
            'conv': self.GNN_conv,
            'MLP': self.MLP
        })

    def forward(self, data):
        embedding = self.GNN_conv(data)
        return self.MLP(embedding)
    
    def embedding(self, data):
        return self.GNN_conv(data)

    def _check_existance(self, kwargs):
        for key, val in kwargs.items():
            assert val, f'{key} does not exist'

class TwoLayerGIN(torch_geometric.nn.MessagePassing):
    def __init__(self, input_size, output_size, activation=ReLU, batch_norm=False, dropout_p=0.0): 
        super(TwoLayerGIN, self).__init__()
        self.batch_norm = BatchNorm1d if batch_norm else Identity
        self.GIN = GINConv(
            Sequential(
                Linear(input_size, output_size),
                self.batch_norm(output_size), 
                activation(),
                Dropout(p=dropout_p),
                Linear(output_size, output_size), 
                self.batch_norm(output_size), 
                activation(),
                Dropout(p=dropout_p)
            ),
            eps=0.3
        )

    def forward(self, x, edge_index):
        return self.GIN(x, edge_index)

class StandardEncoderGNN(torch.nn.Module):
    def __init__(self, model_config): 
        super(StandardEncoderGNN, self).__init__()
        self.input_size = model_config.get('input_size', -1) # 13
        self.hidden_channels = model_config.get('hidden_channels', 64)
        self.num_conv_layers = model_config.get('num_conv_layers', 4)
        self.type_conv = model_config.get('type_conv', GCNConv)
        assert self.type_conv in (GCNConv, TwoLayerGIN, ), f'Cannot build {self.type_conv}'
        self.batch_norm = BatchNorm1d if model_config.get('batch_norm', False) else Identity
        self.dropout_p = model_config.get('dropout_p', 0.0)
        self.activation = model_config.get('activation', ReLU)
        self.conv = self._geo_sequential()
        self.misc_kwargs = model_config

    def _geo_sequential(self):
        if self.num_conv_layers == 1:
            return PYG_Sequential('x, edge_index',
                [(self.type_conv(self.input_size, self.hidden_channels), 'x, edge_index -> x')]
        )

        return PYG_Sequential('x, edge_index',
            [
                (self.type_conv(self.input_size, self.hidden_channels), 'x, edge_index -> x'),
                (self.batch_norm(self.hidden_channels), 'x -> x'),
                (self.activation(), 'x -> x'),
                (Dropout(p=self.dropout_p), 'x -> x')
            ] \
            + [self._convolution_block(i) for i in range((self.num_conv_layers - 2) * 4)] \
            + [(self.type_conv(self.hidden_channels, self.hidden_channels), 'x, edge_index -> x')]
        )
    
    def _convolution_block(self, i):
        func_type = i % 4
        if func_type == 0:
                return (self.type_conv(self.hidden_channels, self.hidden_channels), 'x, edge_index -> x')
        if func_type == 1:
                return (self.batch_norm(self.hidden_channels), 'x -> x')
        if func_type == 2:
                return (self.activation(), 'x -> x')
        if func_type == 3:
                return (Dropout(p=self.dropout_p), 'x -> x')
        assert False, 'Something went wrong in _convolution_block with the func_type'

    def forward(self, data):
        raise NotImplementedError

class BaseMLP(torch.nn.Module):
    def __init__(self, model_config):
        super(BaseMLP, self).__init__()
        self.decoder_input = model_config.get('decoder_input', model_config.get('hidden_channels', 64))
        self.hidden_dimensions = model_config.get('hidden_dimensions', 64)
        self.batch_norm = LayerNorm if model_config.get('batch_norm', False) else Identity
        self.dropout_p = model_config.get('dropout_p', 0.0)
        self.activation = model_config.get('activation', ReLU)
        self.num_linear_layers = model_config.get('num_linear_layers', 2)

        self.MLP = self._MLP()
        self.last_layer = self._last_layer()

    def _MLP(self):
        if self.num_linear_layers == 0:
             return Identity()
        return Sequential(OrderedDict(
                    [('lin_in', Linear(self.decoder_input, self.hidden_dimensions)),
                     ('norm_in',self.batch_norm(self.hidden_dimensions)),
                     ('act_in', self.activation())] \
                        + 
                    [self._MLP_block(i) for i in range((self.num_linear_layers - 2) * 3)  ] 
                ))  
    
    def _MLP_block(self, i):
        func_type = i % 3
        block_nr = i // 3 + 1
        if func_type == 0:
                return (f'lin{block_nr}', Linear(self.hidden_dimensions, self.hidden_dimensions))
        if func_type == 1: 
                return (f'norm{block_nr}', self.batch_norm(self.hidden_dimensions))
        if func_type == 2:
                return (f'act{block_nr}', self.activation())
        assert False, 'Something went wrong in _MLP_block with the func_type'

    def _last_layer(self):
        raise NotImplementedError

    def forward(self, x):
        x = self.MLP(x)
        return self.last_layer(x)   
    
class SoftmaxWithTemperature(torch.nn.Module):
    def __init__(self, temperature):
        super(SoftmaxWithTemperature, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x / self.temperature
        return self.softmax(x)
    
class MultiClassificationMLP(BaseMLP):
    def __init__(self, model_config):
        
        self.temperature = model_config.get('temperature', 1)
        self.n_classes = model_config.get('n_classes', None)
        assert self.n_classes, f'n_classes must be specified'
        super(MultiClassificationMLP, self).__init__(model_config)

    def _last_layer(self):
        return Sequential(OrderedDict([('lin_out', Linear(self.hidden_dimensions, self.n_classes)), ('tempered_softmax', SoftmaxWithTemperature(self.temperature))])) 

class BinaryClassificationMLP(BaseMLP):
    def __init__(self, model_config):
        super(BinaryClassificationMLP, self).__init__(model_config)

    def _last_layer(self):
        return Sequential(OrderedDict([('lin_out', Linear(self.hidden_dimensions, 1)), ('sigmoid_out', Sigmoid())]))  


class MaxPoolingGNN(StandardEncoderGNN):
    def __init__(self, model_config): 
        super(MaxPoolingGNN, self).__init__(model_config)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv(x, edge_index)
        batch = torch.zeros(x.shape[0], dtype=int) if batch is None else batch
        x_max = global_max_pool(x, batch)
        return x_max
    
class AddPoolingGNN(StandardEncoderGNN):
    def __init__(self, model_config): 
        super(AddPoolingGNN, self).__init__(model_config)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv(x, edge_index)
        batch = torch.zeros(x.shape[0], dtype=int) if batch is None else batch
        x_sum = global_add_pool(x, batch)
        return x_sum

class NeuralFingerprintGNN(StandardEncoderGNN):
    def __init__(self, model_config):
        super(NeuralFingerprintGNN, self).__init__(model_config)
     
        self.fp_length = model_config.get('fp_length', 1024)
        MLP_config = {
            'n_classes': self.fp_length, 
            'temperature': 0.05, 
            'num_linear_layers': 0,
            'decoder_input': model_config.get('hidden_channels', 64)
        }

        self.MLPs = [
            MultiClassificationMLP(MLP_config) \
            for layer in self.conv if isinstance(layer, self.activation)
        ] + [MultiClassificationMLP(MLP_config)] # last layer had no activation


    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        device = 'cpu' if x.get_device() == -1 else config["DEVICE"]
        fingerprint = torch.zeros((torch.unique(batch).shape[0], self.fp_length)).to(device)
        i = 0
        for layer in self.conv:
            if isinstance(layer, torch_geometric.nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
            if isinstance(layer, self.activation):
                out = self.MLPs[i].to(device)(x)
                fingerprint += global_add_pool(out, batch)
                i += 1

        # last layer
        x = self.activation().to(device)(x)
        out = self.MLPs[-1].to(device)(x)
        fingerprint += global_add_pool(out, batch)

        return fingerprint