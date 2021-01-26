import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

def get_task_encoder_input(data_ML):

    task_encoder_input = np.concatenate((data_ML.x[: ,: ,0 ,:], data_ML.y), axis=2)

    return task_encoder_input



class LSTMDecoder(nn.Module):

    def __init__(self, batch_size, seq_len, output_dim, n_layers, hidden_dim, latent_dim, device):

        super(LSTMDecoder, self).__init__()

        self.batch_size = batch_size
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = seq_len

        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True)

        self.latent_to_hidden = nn.Linear(self.latent_dim, self.hidden_dim)
        self.hidden_to_output = nn.Linear(self.hidden_dim, self.output_dim)


        self.decoder_inputs = torch.zeros( self.batch_size, self.sequence_length, 1, requires_grad=True).to(device)
        self.c_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim, requires_grad=True).to(device)

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

        self.to(device)


    def forward(self, latent):

        h_state = self.latent_to_hidden(latent).unsqueeze(0)
        h_0 = torch.cat([h_state for _ in range(self.n_layers)], axis=0)
        decoder_output, _ = self.lstm(self.decoder_inputs, (h_0, self.c_0))
        out = self.hidden_to_output(decoder_output)

        return out


class Lambda(nn.Module):

    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    https://github.com/abhmalik/timeseries-clustering-vae/blob/master/vrae/vrae.py

    """

    def __init__(self, hidden_dim, latent_dim):

        super(Lambda, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.hidden_to_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.hidden_to_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)


    def forward(self, cell_output):

        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector

        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean


class MultimodalLearner(nn.Module):

    def __init__(self, task_net, task_encoder, task_decoder, lmbd, modulate_task_net = True):
        super(MultimodalLearner, self).__init__()

        self.task_net = task_net
        self.task_encoder = task_encoder
        self.task_decoder = task_decoder
        self.lmbd = lmbd
        self.modulation_layer = nn.Linear(task_encoder.hidden_dim, task_net.hidden_dim * 2)
        self.output_layer = nn.Linear(task_net.hidden_dim, 1)
        self.task_decoder = task_decoder
        self.rec_loss = nn.SmoothL1Loss()
        self.modulate_task_net = modulate_task_net

    def conditional_layer(self, x, embedding):
        ###apply by deffault the affine transformation -- FiLM layer

        gammas, betas = torch.split(embedding, x.size(1), dim=-1)
        gammas = gammas + torch.ones_like(gammas)
        x = x * gammas + betas

        return x

    def compute_loss(self, x_decoded, x):
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar
        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        recon_loss = self.rec_loss(x_decoded, x)

        return recon_loss + kl_loss, kl_loss, recon_loss

    def forward(self, x, task, output_encoding = False, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        x = self.task_net.encoder(x)

        if self.modulate_task_net:

            encoding = self.task_encoder.encoder(task)
            latent = self.lmbd(encoding)
            task_rec = self.task_decoder(latent)
            loss = self.compute_loss(task_rec, task)

            modulation_embeddings = self.modulation_layer(encoding)
            modulated_output = self.conditional_layer(x, modulation_embeddings)

            if output_encoding:
                return modulated_output, loss

            output = self.output_layer(modulated_output)

        else :

            if output_encoding:
                return x, (0.0, 0.0, 0.0)

            output = self.output_layer(x)
            loss = (0.0, 0.0, 0.0)

        return output, loss

    def encoder (self, x, task, params=None, embeddings=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        x = self.task_net.encoder(x)