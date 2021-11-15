import torch
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.model.language_conditioned_sketch_rnn import LanguageConditionedSketchRNN
from scripts.plot_result import *
from transformers import CLIPTokenizerFast


def main():
    # device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # model.to(device)
    model = LanguageConditionedSketchRNN(
        z_dim=512,
        # input_dim=x.shape[-1],
        device=device,
    )

    # model.load_state_dict(torch.load('./model_param/model_param_best.pt'))
    model.train()
    # print(model)

    label_str = 'apple'
    tokenizer = CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch32')
    label_token = tokenizer([f'Sketching {label_str}'], padding=True, return_tensors='pt')
    text_feature = model.encode_text(label_token)
    seq = model.generate(z=text_feature, batch_size=16)

    fig = plt.figure(figsize=(20, 10))
    plot_reconstructed(fig, seq, seq)
    fig.savefig('valid_image/generated.png')

    # fig_latent_traversal = plt.figure(figsize=(10, 10))
    # plot_latent_traversal(fig_latent_traversal, model, device, z_dim)
    # fig_latent_traversal.savefig('valid_image/latent_traversal.png')


if __name__ == '__main__':
    main()
