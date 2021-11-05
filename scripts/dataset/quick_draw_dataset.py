from torch.utils.data import Dataset
import glob
import numpy as np
import os
from concurrent import futures
from transformers import CLIPTokenizerFast
import torch


class QuickDrawDataset(Dataset):
    def __init__(self, datafolder, split='train', max_length=None):
        paths = glob.glob('{}/*'.format(datafolder))
        self.data = self._load_datas(paths, split=split)

        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        label_list = []
        for i in range(len(self.data)):
            label_str = os.path.splitext(os.path.basename(paths[i]))[0][10:]
            label_token = tokenizer([f'Sketching {label_str}'], padding=True, return_tensors='pt')
            label_list.extend([label_token['input_ids']] * self.data[i].shape[0])
        self.label = torch.cat(label_list)

        self.data = np.concatenate(self.data, axis=0)
        data_size = sum([data_part.__sizeof__() for data_part in self.data])
        print('data size: {} [MiB]'.format(data_size / 1.049e+6))

        if max_length is None:
            length = [len(data_part) for data_part in self.data]
            self.max_length = int(np.mean(length) + 2 * np.std(length))
        else:
            self.max_length = max_length

    def __len__(self):
        return len(self.data)-1

    def __getitem__(self, idx):
        data = self.data[idx].astype(np.float32)
        data = self.strokes_to_lines(data)
        data[:, :-1] /= np.std(data[:, :-1])
        data = self._padding(data)
        label = self.label[idx]
        return data, label

    def _padding(self, x):
        input_length = len(x)
        if input_length < self.max_length:
            n_pad = self.max_length - input_length
            pad = np.array([np.zeros_like(x[0])] * n_pad)
            x = np.concatenate([x, pad], axis=0)
        elif input_length > self.max_length:
            x = x[:self.max_length]
        return x

    def _load_datas(self, paths, split='train'):
        def load_data(idx):
            if not os.path.exists(paths[idx]):
                print(f'{paths[idx]} Not Found')
                return
            data = np.load(paths[idx], encoding='latin1', allow_pickle=True)
            data = data[split]
            print(f'load {data.shape[0]} data from {paths[idx]}')
            return idx, data

        data_list = []
        length = len(paths)
        data_list = [0] * length
        with futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_datas = [
                executor.submit(
                    load_data,
                    idx) for idx in range(length)]
            for future in futures.as_completed(future_datas):
                idx, data = future.result()
                data_list[idx] = data
        return np.array(data_list, dtype=object)

    def strokes_to_lines(self, strokes):
        x, y, z = 0, 0, 0
        line = []
        for stroke in strokes:
            x += stroke[0]
            y += stroke[1]
            z = stroke[2]
            line.append([x, y, z])
        return np.array(line, dtype=np.float32)
