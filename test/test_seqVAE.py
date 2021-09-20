import unittest

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.seqVAE import SeqVAE, VAELoss
from scripts.quick_draw_dataset import QuickDrawDataset
from scripts.plot_result import *
from dataset_path import datafolder


class TestVAE(unittest.TestCase):
    def test_VAE(self):
        print('\n========== test SeqVAE model ==========')
        dataset = QuickDrawDataset(datafolder, split='valid')
        dataloader = DataLoader(
            dataset,
            batch_size=1000,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        x, label = dataset[0]
        model = SeqVAE(
            z_dim=5,
            input_dim=x.shape[-1],
            label_dim=10,
        )
        model.train()
        print(model)

        # device setting
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device:', device)
        model.to(device)

        loss_fn = VAELoss()

        for x, label in tqdm(dataloader):
            x = x.to(device)
            label = label.to(device)
            y, mean, std = model(x, label)
            # print('x:', x.shape)
            # print('y:', y.shape)
            # print('mean:', mean.shape)
            loss_mse, loss_kld = loss_fn(x, y, mean, std)
            loss = loss_mse + loss_kld
            loss.backward()
        print('loss_mse:', loss_mse.item())
        print('loss_kld:', loss_kld.item())


if __name__ == "__main__":
    unittest.main()
