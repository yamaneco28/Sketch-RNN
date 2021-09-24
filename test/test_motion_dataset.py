import unittest
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.motion_dataset import MotionDataset
from dataset_path import datafolder


class TestStateDataset(unittest.TestCase):
    def test_state_dataset(self):
        print('\n========== test state dataset and fast data loader ==========')
        datafolder = '../../datasets/position_conditioned'
        dataset = MotionDataset(datafolder, data_num=50)
        print('data length:', len(dataset))
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            pin_memory=True,
        )
        for e in range(3):
            start = time.time()
            for i, (x, label) in enumerate(tqdm(dataloader)):
                pass
                # print('#', i)
                # print('x shape:', x.shape)
                # print('label shape:', label.shape)
                # end = time.time()
                # print('elapsed time:', end - start)
                # start = time.time()
                dataset.denormalization(x)


if __name__ == '__main__':
    unittest.main()
