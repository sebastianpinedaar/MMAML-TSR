import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import _VF
from torch.nn.utils.rnn import PackedSequence
from collections import  namedtuple

class CustomLSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super(CustomLSTM, self).__init__( *args, **kwargs)  
        
    def forward(self, input, params = None, hx=None, embeddings = None):  # noqa: F811
        
            if params is None:
                params = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn)] 
                
            
            orig_input = input
            # xxx: isinstance check needs to be in conditional for TorchScript to compile
            if isinstance(orig_input, PackedSequence):
                input, batch_sizes, sorted_indices, unsorted_indices = input
                max_batch_size = batch_sizes[0]
                max_batch_size = int(max_batch_size)
            else:
                batch_sizes = None
                max_batch_size = input.size(0) if self.batch_first else input.size(1)
                sorted_indices = None
                unsorted_indices = None

            if hx is None:
                num_directions = 2 if self.bidirectional else 1
                zeros = torch.zeros(self.num_layers * num_directions,
                                    max_batch_size, self.hidden_size,
                                    dtype=input.dtype, device=input.device)
                hx = (zeros, zeros)
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)

            self.check_forward_args(input, hx, batch_sizes)
            if batch_sizes is None:
                result = _VF.lstm(input, hx, params, self.bias, self.num_layers,
                                  self.dropout, self.training, self.bidirectional, self.batch_first)
            else:
                result = _VF.lstm(input, batch_sizes, hx, params, bias,
                                  self.num_layers, self.dropout, self.training, self.bidirectional)
            output = result[0]
            hidden = result[1:]
            # xxx: isinstance check needs to be in conditional for TorchScript to compile
            if isinstance(orig_input, PackedSequence):
                output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
                return output_packed, self.permute_hidden(hidden, unsorted_indices)
            else:
                return output, self.permute_hidden(hidden, unsorted_indices)



class LinearModel(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.layer_name = "linear"

    def forward(self, x, params):

        if params is None:
            params = OrderedDict(self.named_parameters())

        input = x

        weight = params.get( self.layer_name + '.weight', None)
        bias = params.get(self.layer_name + '.bias', None)
        x = F.linear(x, weight=weight, bias=bias)

        return x

    @property
    def param_dict(self):
        return OrderedDict(self.named_parameters())

class LSTMModel(nn.Module):
    
    def __init__(self, batch_size, seq_len, input_dim, n_layers, hidden_dim, output_dim, lin_hidden_dim = 100):
        super(LSTMModel, self).__init__()

        #self.lstm = nn.CustomLSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        #self.linear = nn.Linear(hidden_dim, output_dim)#
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_layers = n_layers
        #self.hidden = self.init_hidden()
        self.input_dim = input_dim
        self.features = torch.nn.Sequential(OrderedDict([
            ("lstm",  CustomLSTM(input_dim, hidden_dim, n_layers, batch_first=True)),
            ("linear", nn.Linear(hidden_dim, output_dim))]))
        

    def forward(self, x, params):
        
        if params is None:
            params = OrderedDict(self.named_parameters())


        input = x
        for layer_name, layer in self.features.named_children():

            if layer_name=="lstm":
                #names = ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']
                
                temp_params = []
                for name, _ in layer.named_parameters():
                   temp_params.append(params.get("features."+ layer_name +"."+name))

                output, (hn, cn) = layer(x, temp_params)
                x = hn[-1].view(len(input),-1)
            
            elif layer_name=="linear":
                weight = params.get('features.' + layer_name + '.weight', None)
                bias = params.get('features.' + layer_name + '.bias', None)
                x = F.linear(x, weight = weight, bias = bias)
        

        return x

    @property
    def param_dict(self):
        return OrderedDict(self.named_parameters())

Task = namedtuple('Task', ['x', 'y'])