import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockFCNConv(nn.Module):
    #based on https://dzlab.github.io/timeseries/2018/11/25/LSTM-FCN-pytorch-part-1/
    def __init__(self, in_channel=1, out_channel=128, kernel_size=8, momentum=0.99, epsilon=0.001, squeeze=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channel, eps=epsilon, momentum=momentum)
        self.relu = nn.ReLU()
    def forward(self, x):
        # input (batch_size, num_variables, time_steps), e.g. (128, 1, 512)
        x = self.conv(x)
        # input (batch_size, out_channel, L_out)
        x = self.batch_norm(x)
        # same shape as input
        y = self.relu(x)
        return y
    
    
class FCN(nn.Module):
    def __init__(self, time_steps, channels=[1, 128, 256, 128], kernels=[8, 5, 3], output_dim =1, mom=0.99, eps=0.001):
        super().__init__()
        self.conv1 = BlockFCNConv(channels[0], channels[1], kernels[0], momentum=mom, epsilon=eps, squeeze=True)
        self.conv2 = BlockFCNConv(channels[1], channels[2], kernels[1], momentum=mom, epsilon=eps, squeeze=True)
        self.conv3 = BlockFCNConv(channels[2], channels[3], kernels[2], momentum=mom, epsilon=eps)
        output_size = time_steps - sum(kernels) + len(kernels)
        self.global_pooling = nn.AvgPool1d(kernel_size=output_size)
        self.linear = nn.Linear(channels[2], output_dim )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # apply Global Average Pooling 1D
        x = self.global_pooling(x)
        x = self.linear(x.squeeze())
        return x

    def encoder(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # apply Global Average Pooling 1D
        x = self.global_pooling(x)
        x = x.squeeze()
        return x    #output size = 128

class ExtendedFCN(nn.Module):
    def __init__(self, fcn, hidden, output_dim):
        super().__init__()
        self.hidden1 = fcn.linear.in_features
        self.hidden2 = hidden
        self.linear1 = nn.Linear(self.hidden1, self.hidden2)
        self.linear2 = nn.Linear(self.hidden2, output_dim)
        self.fcn = fcn
    def forward(self, x):
        x = self.fcn.conv1(x)
        x = self.fcn.conv2(x)
        x = self.fcn.conv3(x)
        # apply Global Average Pooling 1D
        x = self.fcn.global_pooling(x)
        x = self.linear1(x.squeeze())
        return x  

    def encoder(self, x):
        x = self.fcn.conv1(x)
        x = self.fcn.conv2(x)
        x = self.fcn.conv3(x)
        # apply Global Average Pooling 1D
        x = self.fcn.global_pooling(x)
        x = x.squeeze()
        return x      

class LSTMModel(nn.Module):
    
    def __init__(self, batch_size, seq_len, input_dim, n_layers, hidden_dim, output_dim, lin_hidden_dim = 100):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)#
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_layers = n_layers
        #self.hidden = self.init_hidden()
        self.input_dim = input_dim
        
    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.n_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.n_layers, self.batch_size, self.hidden_dim))
        
    def forward(self, x):
        
        output, (hn, cn) = self.lstm(x)
        out1 = self.linear(hn[-1].view(len(x),-1))

        return out1

    def encoder(self, x):
        output, (hn, cn) = self.lstm(x)
        return torch.reshape(hn[-1],(len(x),-1))



    
class LSTMModel_MRA(nn.Module):
    
    def __init__(self, batch_size, seq_len, input_dim, n_layers, hidden_dim, output_dim, lin_hidden_dim = 100):
        super(LSTMModel_MRA, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)#
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_layers = n_layers
        #self.hidden = self.init_hidden()
        self.input_dim = input_dim
        
        self.hidden_to_mean = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.hidden_to_logvar = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.n_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.n_layers, self.batch_size, self.hidden_dim))
        
    def encoder(self, x):

        _, (hn, _) = self.lstm(x)
        out = hn[-1].view(len(x),-1)
        latent_mean = self.hidden_to_mean(out)
        latent_logvar = self.hidden_to_logvar(out)
        self.kld = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        
        if self.training:
            std = torch.exp(0.5 * latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(latent_mean)
        else:
            return latent_mean

    def get_kld(self):
        
        return self.kld


    def forward(self, x):

        return x