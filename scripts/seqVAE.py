import torch
from torch import nn
import torch.nn.functional as F


class SeqVAE(nn.Module):
    def __init__(self, z_dim=2, input_dim=2):
        super().__init__()

        self.LSTM_dim = 100
        self.LSTM_layer_num = 1

        # encoder
        self.enc_lstm = nn.LSTM(
            input_dim,
            self.LSTM_dim,
            num_layers=self.LSTM_layer_num,
            batch_first=True,
            bidirectional=True,
        )
        self.enc_dense_mean = nn.Linear(2 * self.LSTM_dim, z_dim)
        self.enc_dense_var = nn.Linear(2 * self.LSTM_dim, z_dim)

        # decoder
        self.dec_lstm = nn.LSTM(
            input_dim + z_dim,
            self.LSTM_dim,
            num_layers=self.LSTM_layer_num,
            batch_first=True,
        )
        self.dec_dense = nn.Linear(self.LSTM_dim, input_dim)
        self.dec_dense_h = nn.Linear(z_dim, self.LSTM_layer_num * self.LSTM_dim)
        self.dec_dense_c = nn.Linear(z_dim, self.LSTM_layer_num * self.LSTM_dim)

    def encoder(self, x):
        hs, (h, c) = self.enc_lstm(x)
        x = hs[:, -1]
        mean = self.enc_dense_mean(x)
        std = F.relu(self.enc_dense_var(x))
        return mean, std

    def _sample_z(self, mean, std):
        epsilon = torch.randn(mean.shape).to(mean.device)
        return mean + std * epsilon

    def decoder(self, x, z):
        h = self.dec_dense_h(z)
        c = self.dec_dense_c(z)
        h = h.reshape(h.shape[0], self.LSTM_layer_num, self.LSTM_dim)
        c = c.reshape(c.shape[0], self.LSTM_layer_num, self.LSTM_dim)
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()

        z = z.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([x, z], dim=2)
        x, (h, c) = self.dec_lstm(x, (h, c))
        x = self.dec_dense(x)
        return x

    def forward(self, x):
        mean, std = self.encoder(x)
        z = self._sample_z(mean, std)
        zeros = torch.zeros_like(x[:, 0].unsqueeze(1))
        x = torch.cat([zeros, x[:, :-1]], dim=1)
        y = self.decoder(x, z)

        return y, mean, std


class VAELoss(nn.Module):
    def __init__(self, weight_mse=10000.0):
        super().__init__()

        self.weight_mse = weight_mse

    def forward(self, x, y, mean, std, eps=1e-10):
        # Mean Squared Error
        mse = self.weight_mse * F.mse_loss(x, y)

        # Kullback–Leibler divergence
        kld = -0.5 * (1 + torch.log(std**2 + eps) - mean**2 - std**2).mean()

        return mse, kld
