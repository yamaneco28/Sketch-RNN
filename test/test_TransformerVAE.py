import unittest

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.seqVAE import SeqVAE, VAELoss
from scripts.TransformerVAE import TransformerVAE, VAELoss
from scripts.quick_draw_dataset import QuickDrawDataset
from scripts.plot_result import *
from dataset_path import datafolder


class TestVAE(unittest.TestCase):
    def test_VAE(self):
        print('\n========== test SeqVAE model ==========')
        dataset = QuickDrawDataset(datafolder, split='valid')
        dataloader = DataLoader(
            dataset,
            batch_size=100,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        model = TransformerVAE(
        # model = SeqVAE(
            z_dim=6,
            input_dim=2,
        )
        model.train()
        # print(model)

        # device setting
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device:', device)
        model.to(device)

        loss_fn = VAELoss()

        for x, label in tqdm(dataloader):
            x = x.to(device)
            y, mean, std = model(x)
            # print('x:', x.shape)
            # print('y:', y.shape)
            # print('mean:', mean.shape)
            loss_mse, loss_kld = loss_fn(x, y, mean, std)
            loss = loss_mse + loss_kld
            loss.backward()
        print('loss_mse:', loss_mse.item())
        print('loss_kld:', loss_kld.item())

        generated = model.generate(device=device, batch_size=16)
        print('generated shape:', generated.shape)


if __name__ == "__main__":
    unittest.main()
