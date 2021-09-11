import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)


class TransformerVAE(nn.Module):
    def __init__(self, z_dim: int, input_dim: int):
        super().__init__()

        self.z_dim = z_dim
        self.input_dim = input_dim

        # encoder
        self.enc_pos_encoder = PositionalEncoding(d_model=input_dim)
        self.enc_transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=1,
            dim_feedforward=256,
            # batch_first=True,
        )
        self.enc_transformer = nn.TransformerEncoder(
            encoder_layer=self.enc_transformer_layer,
            num_layers=2,
        )
        self.enc_dense_mean = nn.Linear(input_dim, z_dim)
        self.enc_dense_var = nn.Linear(input_dim, z_dim)

        # decoder
        self.dec_pos_encoder = PositionalEncoding(d_model=input_dim + z_dim)
        self.dec_transformer_layer = nn.TransformerDecoderLayer(
            d_model=input_dim + z_dim,
            nhead=1,
            dim_feedforward=256,
            # batch_first=True,
        )
        self.dec_transformer = nn.TransformerDecoder(
            decoder_layer=self.dec_transformer_layer,
            num_layers=2,
        )
        self.dec_dense = nn.Linear(input_dim + z_dim, input_dim)

    def encoder(self, x: Tensor):
        x = self.enc_pos_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.enc_transformer(x)
        x = x.permute(1, 0, 2)
        x = x[:, -1]
        mean = torch.tanh(self.enc_dense_mean(x))
        std = F.relu(self.enc_dense_var(x))
        return mean, std

    def _sample_z(self, mean: Tensor, std: Tensor):
        epsilon = torch.randn(mean.shape).to(mean.device)
        return mean + std * epsilon

    def decoder(self, x: Tensor, z: Tensor):
        z = z.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([x, z], dim=2)
        x = self.dec_pos_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.dec_transformer(x, x)
        x = x.permute(1, 0, 2)
        x = self.dec_dense(x)
        return x

    def forward(self, x: Tensor):
        mean, std = self.encoder(x)
        z = self._sample_z(mean, std)
        zeros = torch.zeros_like(x[:, 0].unsqueeze(1))
        x = torch.cat([zeros, x[:, :-1]], dim=1)
        y = self.decoder(x, z)

        return y, mean, std

    def generate(self, z: Tensor = None, length: int = 100,
                 device: str = 'cpu', batch_size: int = 1):
        if z is None:
            z = torch.randn(size=(batch_size, self.z_dim)).to(device)
        else:
            batch_size = z.shape[0]

        x = torch.zeros(size=(batch_size, 1, self.input_dim)).to(device)
        z = z.unsqueeze(1)

        x_list = []
        i = 0
        while i < length:
            x = torch.cat([x, z], dim=2)
            x = self.dec_pos_encoder(x)
            x = x.permute(1, 0, 2)
            x = self.dec_transformer(x, x)
            x = x.permute(1, 0, 2)
            x = self.dec_dense(x)
            x_list.append(x)
            i += 1
        x = torch.cat(x_list, dim=1)
        return x


class VAELoss(nn.Module):
    def __init__(self, weight_mse: float = 10000.0):
        super().__init__()

        self.weight_mse = weight_mse

    def forward(self, x: Tensor, y: Tensor, mean: Tensor, std: Tensor,
                eps: float = 1e-10):
        # Mean Squared Error
        mse = self.weight_mse * F.mse_loss(x, y)

        # Kullback–Leibler divergence
        kld = -0.5 * (1 + torch.log(std**2 + eps) - mean**2 - std**2).mean()

        return mse, kld
