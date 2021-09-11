import torch
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.seqVAE import SeqVAE
from scripts.plot_result import *


def main():
    z_dim = 5
    model = SeqVAE(
        z_dim=z_dim,
        input_dim=2,
    )
    model.load_state_dict(torch.load('./model_param/model_param_best.pt'))
    model.train()
    print(model)

    # device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model.to(device)

    seq = model.generate(device=device, batch_size=16)

    fig = plt.figure(figsize=(20, 10))
    plot_reconstructed(fig, seq, seq)
    fig.savefig('valid_image/generated.png')

    fig_latent_traversal = plt.figure(figsize=(10, 10))
    plot_latent_traversal(fig_latent_traversal, model, device, z_dim)
    fig_latent_traversal.savefig('valid_image/latent_traversal.png')


if __name__ == '__main__':
    main()
