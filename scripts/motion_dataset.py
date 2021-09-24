import torch
from torch.utils.data import Dataset
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import re


class MotionDataset(Dataset):
    def __init__(self, datafolder, data_num=None, normalization=True):
        state_list = []
        label_list = []

        folders = glob.glob('{}/*'.format(datafolder))
        for i, folder in enumerate(folders):
            paths = glob.glob('{}/*.csv'.format(folder))
            filenames = [os.path.splitext(os.path.basename(path))[0] for path in paths]
            filenum = [int(re.sub(r'\D', '', filename)) for filename in filenames]

            if data_num != None and data_num < len(filenum):
                filenum = filenum[:data_num]

            print('loading {} data from {}'.format(len(filenum), folder))
            for filenum in tqdm(filenum):
                df = pd.read_csv('{}/data{}.csv'.format(folder, filenum))
                df = df.set_index('time')
                index = [
                    'S_Angle[0]','S_Angle[1]','S_Angle[2]',
                    'S_AngularVelocity[0]','S_AngularVelocity[1]','S_AngularVelocity[2]',
                    'S_Torque[0]','S_Torque[1]','S_Torque[2]',
                    'M_Angle[0]','M_Angle[1]','M_Angle[2]',
                    'M_AngularVelocity[0]','M_AngularVelocity[1]','M_AngularVelocity[2]',
                    'M_Torque[0]','M_Torque[1]','M_Torque[2]',
                ]
                df = df.loc[:, index]
                state = np.array(df)

                # decimation
                skip_num = 20
                for start in range(skip_num):
                    state_list.append(state[start::skip_num])
                    label_list.append(i)

        self.label = np.array(label_list)

        length = [len(data_part) for data_part in state_list]
        self.max_length = int(np.mean(length) + 2 * np.std(length))
        state_list = [self._padding(data_part) for data_part in state_list]
        self.state = np.array(state_list).astype(np.float32)

        self.mean = 0
        self.std = 0
        if normalization:
            batch_size, steps, _ = self.state.shape
            state = self.state.reshape(batch_size * steps, -1)
            state = self._normalization(state, axis=0)
            self.state = state.reshape(batch_size, steps, -1).copy()

        print('state data size: {} [MiB]'.format(self.state.__sizeof__()/1.049e+6))
        print('label data size: {} [MiB]'.format(self.label.__sizeof__()/1.049e+6))
        print('state shape:', self.state.shape)
        print('label shape:', self.label.shape)

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        state = self.state[idx]
        label = self.label[idx]
        return state, label

    def _padding(self, x):
        input_length = len(x)
        if input_length < self.max_length:
            pad = self.max_length - input_length
            zeros = [x[-1]] * pad
            zeros = np.array(zeros)
            x = np.concatenate([x, zeros])
        elif input_length > self.max_length:
            x = x[:self.max_length]
        return x

    def _normalization(self, x, axis=None):
        self.mean = np.mean(x, axis=axis, keepdims=True)
        self.std = np.std(x, axis=axis, keepdims=True)
        zscore = (x - self.mean) / self.std
        return zscore

    def denormalization(self, x):
        return x * self.std + self.mean
