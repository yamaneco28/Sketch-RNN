import unittest

import os
import time

import torch

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.plot_result import *
from scripts.quick_draw_dataset import QuickDrawDataset
from dataset_path import datafolder
from scripts.seqVAE import SeqVAE


class TestPlotResult(unittest.TestCase):
    def test_plot_result(self):
        print('\n========== test plot result ==========')

        dataset = QuickDrawDataset(datafolder, split='valid')

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            # num_workers=8,
            # pin_memory=True,
        )

        folder_name = 'test_image'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        for x, label in dataloader:
            fig = plt.figure(figsize=(20, 10))
            plot_reconstructed(fig, x, x, col=4)
            plt.savefig(folder_name + '/test_reconstructed.png')
            break

        start = time.time()
        fig = plt.figure(figsize=(10, 10))
        point_num = 1000
        z_dim = 5
        zs = torch.randn(point_num, z_dim)
        labels = torch.randint(high=10, size=(point_num,))
        plot_latent_space(fig, zs, labels)
        plt.savefig(folder_name + '/test_latent_space.png')
        end = time.time()
        print('elasped time:', end - start)

        model = SeqVAE(z_dim)
        model.load_state_dict(torch.load('../model_param/model_param_best.pt'))
        model.eval()

        # device setting
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device:', device)
        model.to(device)

        fig = plt.figure(figsize=(10, 10))
        plot_2D_Manifold(fig, model, z_sumple=zs, device=device)
        plt.savefig(folder_name + '/test_2D_Manifold.png')

        start = time.time()
        fig = plt.figure(figsize=(10, 10))
        plot_latent_traversal(fig, model, row=z_dim, device=device)
        plt.savefig(folder_name + '/test_latent_traversal.png')


if __name__ == "__main__":
    unittest.main()
