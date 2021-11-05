import unittest

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.model.language_conditioned_sketch_rnn import LanguageConditionedSketchRNN
from scripts.dataset.quick_draw_dataset import QuickDrawDataset
from scripts.plot_result import *
from dataset_path import datafolder


class TestLanguageConditionedSketchRNN(unittest.TestCase):
    def test_LanguageConditionedSketchRNN(self):
        print('\n========== test LanguageConditionedSketchRNN model ==========')
        dataset = QuickDrawDataset(datafolder, split='valid')
        dataloader = DataLoader(
            dataset,
            batch_size=1000,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        # device setting
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device:', device)

        x, label = dataset[0]
        model = LanguageConditionedSketchRNN(
            z_dim=512,
            input_dim=x.shape[-1],
            # label_dim=10,
            device=device
        )
        model.train()
        # print(model)

        loss_fn = model.loss

        for x, label in tqdm(dataloader):
            # x = x.to(device)
            y, mean, std = model(x)
            # print('x:', x.shape)
            # print('y:', y.shape)
            # print('mean:', mean.shape)
            loss = loss_fn(x, y, mean, std, label)
            loss.backward()
        print('loss_mse:', model.loss_mse)
        print('loss_kld:', model.loss_kld)
        print('loss_language:', model.loss_language)


if __name__ == "__main__":
    unittest.main()
