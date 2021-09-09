import unittest
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.quick_draw_dataset import QuickDrawDataset
from dataset_path import datafolder


class TestQuickDrawDataset(unittest.TestCase):
    def test_dataset(self):
        print('\n---------- quick draw dataset test ----------')
        dataset = QuickDrawDataset(datafolder, split='valid')
        torchdataloader = DataLoader(
            dataset,
            batch_size=10000,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        for e in range(3):
            start = time.time()
            for i, (x, label) in enumerate(tqdm(torchdataloader)):
                pass
                # print('#', i)
                print('shape:', x.shape)
                import torch
                print(torch.max(x))
                print(torch.min(x))
            end = time.time()
            print('elapsed time:', end - start)


if __name__ == "__main__":
    unittest.main()
