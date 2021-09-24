import torch
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.seqVAE import SeqVAE
from scripts.plot_result import *


def load_model_param(filepath, device='cpu'):
    state_dict = torch.load(filepath, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    return new_state_dict


def main():
    z_dim = 5
    model = SeqVAE(
        z_dim=z_dim,
        input_dim=3,
    )
    state_dict = load_model_param('./model_param/model_param_best.pt')
    model.load_state_dict(state_dict)
    model.train()
    print(model)

    # device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model.to(device)

    seq = model.generate(device=device, batch_size=16)
    seq2 = model.generate(device=device, batch_size=16)
    seq2[:, :, 0] += 4
    seq = torch.cat([seq, seq2], dim=1)
    print(seq.shape)

    fig = plt.figure(figsize=(20, 10))
    plot_reconstructed(fig, seq, seq2)
    fig.savefig('valid_image/generated.png')

    fig_latent_traversal = plt.figure(figsize=(10, 10))
    plot_latent_traversal(fig_latent_traversal, model, device, z_dim)
    fig_latent_traversal.savefig('valid_image/latent_traversal.png')


if __name__ == '__main__':
    main()
