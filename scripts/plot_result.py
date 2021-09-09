import numpy as np
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
import torch


def torch2numpy(tensor):
    return tensor.cpu().detach().numpy().copy()


def strokes_to_lines(strokes):
    x = 0
    y = 0
    line = []
    for i in range(len(strokes)):
        x += float(strokes[i, 0])
        y += float(strokes[i, 1])
        line.append([x, y])
    return np.array(line)


def plot_reconstructed(fig, seq_ans, seq_hat, col=4, epoch=0):
    seq_ans = torch2numpy(seq_ans)
    seq_hat = torch2numpy(seq_hat)

    seqnum = int(col**2)
    seq_ans = seq_ans[:seqnum]
    seq_hat = seq_hat[:seqnum]

    row = -(-len(seq_ans) // col)
    for i, (seq_ans, seq_hat) in enumerate(zip(seq_ans, seq_hat)):
        lines_ans = strokes_to_lines(seq_ans)
        ax = fig.add_subplot(row, 2 * col, 2 * i + 1)
        ax.plot(lines_ans[:, 0], -lines_ans[:, 1], color='black')
        ax.axis('off')
        ax.set_aspect('equal')

        lines_hat = strokes_to_lines(seq_hat)
        ax = fig.add_subplot(row, 2 * col, 2 * i + 2)
        ax.plot(lines_hat[:, 0], -lines_hat[:, 1], color='tab:blue')
        ax.axis('off')
        ax.set_aspect('equal')

    fig.suptitle('{} epoch'.format(epoch))
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
