import numpy as np
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
import torch


def torch2numpy(tensor):
    return tensor.cpu().detach().numpy().copy()


def strokes_to_lines(strokes):
    """Convert stroke-3 format to polyline format."""
    x = 0
    y = 0
    lines = []
    line = []
    for i in range(len(strokes)):
        if strokes[i, 2] > 0.5:
            x = float(strokes[i, 0])
            y = float(strokes[i, 1])
            line.append([x, y])
            lines.append(np.array(line))
            line = []
        else:
            x = float(strokes[i, 0])
            y = float(strokes[i, 1])
            line.append([x, y])
    if len(lines) == 0:
        lines.append(np.array(line))
    return lines


def plot_reconstructed(fig, seq_ans, seq_hat, col=4, epoch=None):
    seq_ans = torch2numpy(seq_ans)
    seq_hat = torch2numpy(seq_hat)

    seqnum = int(col**2)
    seq_ans = seq_ans[:seqnum]
    seq_hat = seq_hat[:seqnum]

    row = -(-len(seq_ans) // col)
    for i, (seq_ans, seq_hat) in enumerate(zip(seq_ans, seq_hat)):
        ax = fig.add_subplot(row, 2 * col, 2 * i + 1)
        seq_ans = strokes_to_lines(seq_ans)
        for seq_ans in seq_ans:
            ax.plot(seq_ans[:, 0], -seq_ans[:, 1])
        ax.axis('off')
        ax.set_aspect('equal')

        ax = fig.add_subplot(row, 2 * col, 2 * i + 2)
        seq_hat = strokes_to_lines(seq_hat)
        for seq_hat in seq_hat:
            ax.plot(seq_hat[:, 0], -seq_hat[:, 1])
        ax.axis('off')
        ax.set_aspect('equal')

    if epoch is not None:
        fig.suptitle(f'{epoch} epoch')
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)


def plot_latent_space(fig, zs, labels, epoch=0):
    zs = torch2numpy(zs)
    labels = torch2numpy(labels)

    if zs.shape[1] > 2:
        pca = PCA()
        pca.fit(zs)
        zs = pca.transform(zs)
    # zs = TSNE(n_components=2, random_state=0).fit_transform(zs)

    ax = fig.add_subplot(111)
    im = ax.scatter(zs[:, 0], zs[:, 1], c=labels, cmap='jet', marker='.')
    lim = np.max(np.abs(zs)) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im, ax=ax, orientation='vertical', cax=cax)
    ax.set_title('{} epoch'.format(epoch))


def plot_loss(ax, train_loss, valid_loss):
    ax.plot(train_loss, label='train', alpha=0.8)
    ax.plot(valid_loss, label='valid', alpha=0.8)
    train_max = np.mean(train_loss) + 2 * np.std(train_loss)
    valid_max = np.mean(valid_loss) + 2 * np.std(valid_loss)
    y_max = max(train_max, valid_max)
    y_min = min(min(train_loss), min(valid_loss))
    ax.set_ylim(0.9 * y_min, 1.1 * y_max)
    ax.set_yscale('log')


def plot_losses(fig, train_loss, valid_loss,
                train_loss_mse, valid_loss_mse,
                train_loss_kl, valid_loss_kl):
    ax = fig.add_subplot(311)
    plot_loss(ax, train_loss, valid_loss)
    ax.set_ylabel('Total Loss')
    ax.tick_params(bottom=False, labelbottom=False)
    ax.legend()

    ax = fig.add_subplot(312)
    plot_loss(ax, train_loss_mse, valid_loss_mse)
    ax.set_ylabel('Mean Squared Error')
    ax.tick_params(bottom=False, labelbottom=False)

    ax = fig.add_subplot(313)
    plot_loss(ax, train_loss_kl, valid_loss_kl)
    ax.set_ylabel('KL Divergence')
    ax.set_xlabel('epoch')

    fig.align_labels()


def plot_2D_Manifold(fig, model, device, z_sumple,
                     col=10, epoch=None, label=None):
    row = col

    x = np.tile(np.linspace(-2, 2, col), row)
    y = np.repeat(np.linspace(2, -2, row), col)
    z = np.stack([x, y]).transpose()
    zeros = np.zeros(shape=(z.shape[0], z_sumple.shape[1] - z.shape[1]))
    z = np.concatenate([z, zeros], axis=1)

    if z_sumple.shape[1] > 2:
        z_sumple = torch2numpy(z_sumple)
        pca = PCA()
        pca.fit(z_sumple)
        z = pca.inverse_transform(z)
    z = torch.from_numpy(z.astype(np.float32)).to(device)

    seq = model.generate(z, device=device, label=label)
    seq = torch2numpy(seq)

    for i, seq in enumerate(seq):
        ax = fig.add_subplot(row, col, i + 1)
        seq = strokes_to_lines(seq)
        for seq in seq:
            ax.plot(seq[:, 0], -seq[:, 1])
        ax.axis('off')
        ax.set_aspect('equal')

    if epoch is not None:
        fig.suptitle(f'{epoch} epoch')
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)


def plot_latent_traversal(fig, model, device, row, col=10,
                          epoch=None, label=None):
    gradation = np.linspace(-2, 2, col)
    z = np.zeros(shape=(row, col, row))
    for i in range(row):
        z[i, :, i] = gradation
    z = z.reshape(-1, row)
    z = torch.from_numpy(z.astype(np.float32)).to(device)

    seq = model.generate(z, device=device, label=label)
    seq = torch2numpy(seq)

    for i, seq in enumerate(seq):
        ax = fig.add_subplot(row, col, i + 1)
        seq = strokes_to_lines(seq)
        for seq in seq:
            ax.plot(seq[:, 0], -seq[:, 1])
        ax.axis('off')
        ax.set_aspect('equal')

    if epoch is not None:
        fig.suptitle(f'{epoch} epoch')
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
