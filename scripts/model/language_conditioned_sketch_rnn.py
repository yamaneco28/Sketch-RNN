import torch
from torch import nn
import torch.nn.functional as F
from torch.tensor import Tensor
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPTokenizerFast
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LanguageConditionedSketchRNN(nn.Module):
    def __init__(self, z_dim=2, input_dim=2, device='cuda'):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.LSTM_dim = 100
        self.LSTM_layer_num = 1

        # label
        # label_vec_dim = 0
        # if label_dim != 0:
        #     self.label_dim = label_dim
        #     label_vec_dim = 2
        #     self.dense_label = nn.Linear(label_dim, label_vec_dim)

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

        # language encoder
        # self.tokenizer = torch.hub.load(
        #     'huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
        # self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")

        self = self.to(device)

    def encoder(self, x, label=None):
        # if label is not None:
        #     label = torch.eye(self.label_dim)[label].to(label.device)
        #     label = label.unsqueeze(1)
        #     v = self.dense_label(label.float())
        #     v = v.repeat(1, x.shape[1], 1)
        #     x = torch.cat([x, v], dim=2)
        hs, (h, c) = self.enc_lstm(x)
        x = hs[:, -1]
        mean = self.enc_dense_mean(x)
        std = F.relu(self.enc_dense_var(x))
        return mean, std

    def _sample_z(self, mean, std):
        epsilon = torch.randn(mean.shape).to(mean.device)
        return mean + std * epsilon

    def decoder(self, x, z, label=None):
        h = self.dec_dense_h(z)
        c = self.dec_dense_c(z)
        h = h.reshape(h.shape[0], self.LSTM_layer_num, self.LSTM_dim)
        c = c.reshape(c.shape[0], self.LSTM_layer_num, self.LSTM_dim)
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()

        z = z.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([x, z], dim=2)

        if label is not None:
            label = torch.eye(self.label_dim)[label].to(label.device)
            label = label.unsqueeze(1)
            v = self.dense_label(label.float())
            v = v.repeat(1, x.shape[1], 1)
            x = torch.cat([x, v], dim=2)

        x, (h, c) = self.dec_lstm(x, (h, c))
        x = self.dec_dense(x)
        return x

    def forward(self, x, label=None):
        x = x.to(self.device)
        mean, std = self.encoder(x, label)
        z = self._sample_z(mean, std)
        zeros = torch.zeros_like(x[:, 0].unsqueeze(1))
        x = torch.cat([zeros, x[:, :-1]], dim=1)
        y = self.decoder(x, z, label)

        return y, mean, std

    def generate(self, z=None, length=100, batch_size=1, label=None):
        if z is None:
            z = torch.randn(size=(batch_size, self.z_dim)).to(self.device)
        else:
            batch_size = z.shape[0]
        h = self.dec_dense_h(z)
        c = self.dec_dense_c(z)
        h = h.reshape(h.shape[0], self.LSTM_layer_num, self.LSTM_dim)
        c = c.reshape(c.shape[0], self.LSTM_layer_num, self.LSTM_dim)
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()

        x = torch.zeros(size=(batch_size, 1, self.input_dim)).to(self.device)
        z = z.unsqueeze(1)

        if label is not None:
            label = torch.eye(self.label_dim)[label].to(self.device)
            v = self.dense_label(label.float())
            v = v.repeat(batch_size, 1, 1)

        x_list = []
        i = 0
        while i < length:
            x = torch.cat([x, z], dim=2)
            if label is not None:
                x = torch.cat([x, v], dim=2)
            x, (h, c) = self.dec_lstm(x, (h, c))
            x = self.dec_dense(x)
            x_list.append(x)
            i += 1
        x = torch.cat(x_list, dim=1)
        return x

    def encode_text(self, ids: Tensor):
        return self.clip_model.get_text_features(input_ids=ids.to(self.device))

    def loss(self, x, y, mean, std, label, eps=1e-10, weight_mse=100,
             weight_language=100):
        x = x.to(self.device)
        y = y.to(self.device)
        mean = mean.to(self.device)
        std = std.to(self.device)
        label = label.to(self.device)

        # Mean Squared Error
        self.loss_mse = weight_mse * F.mse_loss(x, y)

        # Kullback–Leibler divergence
        self.loss_kld = -0.5 * (1 + torch.log(std**2 + eps) - mean**2 - std**2).mean()

        # language loss
        # text_features = self.clip_model.get_text_features(
        #     input_ids=label.to(self.device))
        text_features = self.encode_text(label)
        self.loss_language = weight_language * F.mse_loss(mean, text_features)

        return self.loss_mse + self.loss_kld + self.loss_language
