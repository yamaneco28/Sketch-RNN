import sys
import os
import time
import datetime
import wandb

import torch
from torch import nn
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

sys.path.append('.')
sys.path.append('..')
# from scripts.seqVAE import SeqVAE, VAELoss
# from scripts.TransformerVAE import TransformerVAE, VAELoss
from scripts.model.language_conditioned_sketch_rnn import LanguageConditionedSketchRNN
from scripts.dataset.quick_draw_dataset import QuickDrawDataset
from scripts.plot_result import *
from scripts.print_progress_bar import print_progress_bar


def train_language_conditioned_sketch_rnn(
        n_epochs, train_loader, valid_loader, model, loss_fn, device,
        out_dir='', lr=0.001, optimizer_cls=optim.Adam,
        wandb_flag=False, gpu_num=0, conditional=False):
    train_losses, valid_losses = [], []
    train_losses_mse, valid_losses_mse = [], []
    train_losses_kl, valid_losses_kl = [], []
    train_losses_language, valid_losses_language = [], []
    total_elapsed_time = 0
    early_stopping_counter = 0
    best_test = 1e10
    optimizer = optimizer_cls(model.parameters(), lr=lr)

    print(f'Let\'s use {torch.cuda.device_count()} GPUs!')
    if len(gpu_num) > 1:
        model = nn.DataParallel(model, device_ids=gpu_num)

    # acceleration
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True

    # figure
    fig_reconstructed = plt.figure(figsize=(20, 10))
    fig_latent_space = plt.figure(figsize=(10, 10))
    fig_2D_Manifold = plt.figure(figsize=(10, 10))
    fig_latent_traversal = plt.figure(figsize=(10, 10))
    fig_loss = plt.figure(figsize=(10, 10))

    for epoch in range(n_epochs + 1):
        start = time.time()

        running_loss = 0.0
        running_loss_mse = 0.0
        running_loss_kl = 0.0
        running_loss_language = 0.0
        model.train()
        for i, (x, label) in enumerate(train_loader):
            with torch.cuda.amp.autocast():
                y, mean, std = model(x)
                loss = loss_fn(x, y, mean, std, label)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_loss_mse += model.loss_mse.item()
            running_loss_kl += model.loss_kld.item()
            running_loss_language += model.loss_language.item()

            header = f'epoch: {epoch}'
            print_progress_bar(i, len(train_loader), end='', header=header)

        train_loss = running_loss / len(train_loader)
        train_loss_mse = running_loss_mse / len(train_loader)
        train_loss_kl = running_loss_kl / len(train_loader)
        train_loss_language = running_loss_language / len(train_loader)
        train_losses.append(train_loss)
        train_losses_mse.append(train_loss_mse)
        train_losses_kl.append(train_loss_kl)
        train_losses_language.append(train_loss_language)

        running_loss = 0.0
        running_loss_mse = 0.0
        running_loss_kl = 0.0
        running_loss_language = 0.0
        valid_mean = []
        valid_label = []
        model.eval()
        for x, label in valid_loader:
            with torch.cuda.amp.autocast():
                y, mean, std = model(x)
                loss = loss_fn(x, y, mean, std, label)

            running_loss += loss.item()
            running_loss_mse += model.loss_mse.item()
            running_loss_kl += model.loss_kld.item()
            running_loss_language += model.loss_language.item()

            if len(valid_mean) * mean.shape[0] < 1000:
                valid_mean.append(mean)
                valid_label.append(label)

        valid_loss = running_loss / len(valid_loader)
        valid_loss_mse = running_loss_mse / len(valid_loader)
        valid_loss_kl = running_loss_kl / len(valid_loader)
        valid_loss_language = running_loss_language / len(train_loader)
        valid_losses.append(valid_loss)
        valid_losses_mse.append(valid_loss_mse)
        valid_losses_kl.append(valid_loss_kl)
        valid_losses_language.append(valid_loss_language)
        valid_mean = torch.cat(valid_mean, dim=0)
        valid_label = torch.cat(valid_label, dim=0)

        end = time.time()
        elapsed_time = end - start
        total_elapsed_time += elapsed_time

        log = '\r\033[K' + f'epoch: {epoch}'
        log += '  train loss: {:.6f} ({:.6f}, {:.6f}, {:.6f})'.format(
            train_loss, train_loss_mse, train_loss_kl, train_loss_language)
        log += '  valid loss: {:.6f} ({:.6f}, {:.6f}, {:.6f})'.format(
            valid_loss, valid_loss_mse, valid_loss_kl, valid_loss_language)
        log += '  elapsed time: {:.3f}'.format(elapsed_time)
        log += '  early stopping: {}'.format(early_stopping_counter)
        print(log)

        if epoch % 100 == 0:
            model_param_dir = os.path.join(out_dir, 'model_param')
            if not os.path.exists(model_param_dir):
                os.mkdir(model_param_dir)
            path_model_param = os.path.join(
                model_param_dir,
                'model_param_{:06d}.pt'.format(epoch))
            torch.save(model.state_dict(), path_model_param)
            if wandb_flag:
                wandb.save(path_model_param)

        # save checkpoint
        path_checkpoint = os.path.join(out_dir, 'checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path_checkpoint)

        # plot loss
        fig_loss.clf()
        plot_losses(fig_loss, train_losses, valid_losses,
                    train_losses_mse, valid_losses_mse,
                    train_losses_kl, valid_losses_kl)
        fig_loss.savefig(os.path.join(out_dir, 'loss.png'))

        # show output
        if epoch % 10 == 0:
            fig_reconstructed.clf()
            plot_reconstructed(fig_reconstructed, x, y, col=10, epoch=epoch)
            fig_reconstructed.savefig(
                os.path.join(out_dir, 'reconstructed.png'))

            fig_latent_space.clf()
            plot_latent_space(fig_latent_space, valid_mean,
                              valid_label, epoch=epoch)
            fig_latent_space.savefig(
                os.path.join(out_dir, 'latent_space.png'))

            # fig_2D_Manifold.clf()
            # # plot_2D_Manifold(fig_2D_Manifold, model.module, device,
            # plot_2D_Manifold(fig_2D_Manifold, model, device,
            #                  z_sumple=valid_mean, col=20,
            #                  epoch=epoch)
            # fig_2D_Manifold.savefig(
            #     os.path.join(out_dir, '2D_Manifold.png'))

            # fig_latent_traversal.clf()
            # # plot_latent_traversal(fig_latent_traversal, model.module, device,
            # plot_latent_traversal(fig_latent_traversal, model, device,
            #                       row=valid_mean.shape[1], col=10,
            #                       epoch=epoch)
            # fig_latent_traversal.savefig(
            #     os.path.join(out_dir, 'latent_traversal.png'))

            if wandb_flag:
                wandb.log({
                    'epoch': epoch,
                    'reconstructed': wandb.Image(fig_reconstructed),
                    'latent_space': wandb.Image(fig_latent_space),
                    # '2D_Manifold': wandb.Image(fig_2D_Manifold),
                    # 'latent_traversal': wandb.Image(fig_latent_traversal),
                })

        # wandb
        if wandb_flag:
            wandb.log({
                'epoch': epoch,
                'iteration': len(train_loader) * epoch,
                'train_loss': train_loss,
                'train_loss_mse': train_loss_mse,
                'train_loss_kl': train_loss_kl,
                'train_loss_language': train_loss_language,
                'valid_loss': valid_loss,
                'valid_loss_mse': valid_loss_mse,
                'valid_loss_kl': valid_loss_kl,
                'valid_loss_language': valid_loss_language,
            })
            wandb.save(path_checkpoint)

        if valid_loss < best_test:
            best_test = valid_loss
            early_stopping_counter = 0

            # save model
            path_model_param_best = os.path.join(
                out_dir, 'model_param_best.pt')
            torch.save(model.state_dict(), path_model_param_best)
            if wandb_flag:
                wandb.save(path_model_param_best)

        else:
            # Early Stopping
            early_stopping_counter += 1
            if early_stopping_counter >= 1000:
                print('Early Stopping!')
                break

    print('total elapsed time: {} [s]'.format(total_elapsed_time))


def main(args):
    train_dataset = QuickDrawDataset(args.data_path, split='train')
    valid_dataset = QuickDrawDataset(args.data_path, split='valid',
                                     max_length=train_dataset.max_length)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        # pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        # pin_memory=True,
        drop_last=True,
    )

    # device setting
    cuda_flag = torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu_num[0]}' if cuda_flag else 'cpu')
    print('device:', device)

    x, label = train_dataset[0]
    model = LanguageConditionedSketchRNN(z_dim=512, input_dim=x.shape[-1], device=device)

    if not os.path.exists('results'):
        os.mkdir('results')
    out_dir = 'results/' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.mkdir(out_dir)

    if args.wandb:
        wandb.init(project='SeqVAE')
        wandb.watch(model)

        config = wandb.config

        config.data_path = args.data_path
        config.epoch = args.epoch
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.gpu_num = args.gpu_num

        config.train_data_num = len(train_dataset)
        config.valid_data_num = len(valid_dataset)

    train_language_conditioned_sketch_rnn(
        n_epochs=args.epoch,
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        loss_fn=model.loss,
        out_dir=out_dir,
        lr=args.learning_rate,
        wandb_flag=args.wandb,
        gpu_num=args.gpu_num,
        device=device,
    )


def argparse():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../datasets/QuickDrawDataset/')
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--wandb', action='store_true')
    tp = lambda x:list(map(int, x.split(',')))
    parser.add_argument('--gpu_num', type=tp, default='0')
    return parser.parse_args()


if __name__ == '__main__':
    args = argparse()
    main(args)
