import os
import sys
import time
import argparse
import logging
import numpy as np
from data import dataset
import util
from data.noise import noise_torch
from data import fr
from data.loss import fnr
import os
from OGFreq_model import OGFreq
from torch import nn

import torch
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--output_dir', type=str, default='./checkpoint/model', help='output directory')
    parser.add_argument('--no_cuda', action='store_true', help="avoid using CUDA when available")
    # dataset parameters
    parser.add_argument('--batch_size', type=int, default=64, help='batch size used during training')
    parser.add_argument('--signal_dim', type=int, default=64, help='dimensionof the input signal')
    parser.add_argument('--fr_size', type=int, default=128, help='size of the frequency representation')
    parser.add_argument('--max_n_freq', type=int, default=10,
                        help='for each signal the number of frequencies is uniformly drawn between 1 and max_n_freq')
    parser.add_argument('--min_sep', type=float, default=1,
                        help='minimum separation between spikes, normalized by signal_dim')
    parser.add_argument('--distance', type=str, default='normal', help='distance distribution between spikes')
    parser.add_argument('--amplitude', type=str, default='uniform', help='spike amplitude distribution')
    parser.add_argument('--floor_amplitude', type=float, default=0.01, help='minimum amplitude of spikes')
    parser.add_argument('--noise', type=str, default='gaussian_blind', help='kind of noise to use')
    parser.add_argument('--snr', type=float, default=-10, help='snr parameter')
    # frequency-representation (fr) module parameters
    parser.add_argument('--fr_module_type', type=str, default='fr', help='type of the fr module: [fr | psnet]')
    parser.add_argument('--fr_n_layers', type=int, default=7, help='number of convolutional layers in the fr module')
    parser.add_argument('--fr_n_filters', type=int, default=32, help='number of filters per layer in the fr module')
    parser.add_argument('--fr_kernel_size', type=int, default=3,
                        help='filter size in the convolutional blocks of the fr module')
    parser.add_argument('--fr_kernel_out', type=int, default=25, help='size of the conv transpose kernel')
    parser.add_argument('--fr_inner_dim', type=int, default=128, help='dimension after first linear transformation')
    parser.add_argument('--fr_upsampling', type=int, default=1,
                        help='stride of the transposed convolution, upsampling * inner_dim = fr_size')

    # kernel parameters used to generate the ideal frequency representation
    parser.add_argument('--kernel_type', type=str, default='gaussian',
                        help='type of kernel used to create the ideal frequency representation [gaussian, triangle or closest]')
    parser.add_argument('--triangle_slope', type=float, default=4000,
                        help='slope of the triangle kernel normalized by signal_dim')
    parser.add_argument('--gaussian_std', type=float, default=0.12,
                        help='std of the gaussian kernel normalized by signal_dim')
    # training parameters
    parser.add_argument('--n_training', type=int, default=50000, help='# of training data')
    parser.add_argument('--n_validation', type=int, default=1000, help='# of validation data')
    parser.add_argument('--lr_fr', type=float, default=0.001,
                        help='initial learning rate for adam optimizer used for the frequency-representation module')
    parser.add_argument('--n_epochs_fr', type=int, default=4090, help='number of epochs used to train the fr module')
    parser.add_argument('--save_epoch_freq', type=int, default=20,
                        help ='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--numpy_seed', type=int, default=100)
    parser.add_argument('--torch_seed', type=int, default=76)

    args = parser.parse_args()


    if torch.cuda.is_available() and not args.no_cuda:
        args.use_cuda = True
    else:
        args.use_cuda = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, 'run.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )


    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)

    train_loader = dataset.make_train_data(args)
    val_loader = dataset.make_eval_data(args)


    device = torch.device("cuda:0" if args.use_cuda else "cpu")
    fr_module = OGFreq(args.fr_n_layers,args.signal_dim,args.fr_n_filters,args.fr_inner_dim,device,args.fr_size)
    fr_module = nn.DataParallel(fr_module)
    fr_module = fr_module.to(device)
    fr_optimizer, fr_scheduler = util.set_optim(args, fr_module, 'fr')
    start_epoch = 1
    logger.info('[Network] Number of parameters in the frequency-representation module : %.3f M' % (
                util.model_parameters(fr_module) / 1e6))

    xgrid = np.linspace(-0.5, 0.5, args.fr_size, endpoint=False)
    for epoch in range(start_epoch, args.n_epochs_fr + 1):

        if epoch < args.n_epochs_fr:
            epoch_start_time = time.time()
            fr_module.train()
            loss_val_total,loss_train_total=0,0
            loss_train_fr2, loss_val_fr2, loss_train_fr1, loss_val_fr1, loss_train_fr3, loss_val_fr3 = 0, 0, 0, 0, 0, 0
            if args.use_cuda:
                reslu = torch.tensor(xgrid[1] - xgrid[0]).cuda()
            else:
                reslu = torch.tensor(xgrid[1] - xgrid[0])

            lr_mult = 1.2
            lr = []
            losses = []
            best_loss = 1e18
            for batch_idx, (clean_signal, target_fr, freq, fr_ground) in enumerate(train_loader):
                if args.use_cuda:
                    clean_signal, target_fr, fr_ground, freq = clean_signal.cuda(), target_fr.cuda(), fr_ground.cuda(), freq.cuda()
                noisy_signal = noise_torch(clean_signal, args.snr, args.noise)

                for i in range(noisy_signal.size()[0]):
                    mv = torch.max(torch.sqrt(pow(noisy_signal[i][0], 2) + pow(noisy_signal[i][1], 2)))
                    noisy_signal[i][0] = noisy_signal[i][0] / mv
                    noisy_signal[i][1] = noisy_signal[i][1] / mv

                [x_output, loss_layers_sym,beta_output,beta_loss_layers_sym] = fr_module(noisy_signal)


                # Compute and print loss
                train_loss_discrepancy = torch.sum(torch.pow(x_output.abs() - target_fr, 2))
                train_beta_loss_discrepancy = torch.sum(torch.pow(beta_output.real - fr_ground, 2))
                loss_constraint = torch.sum(torch.pow(loss_layers_sym[0].abs(), 2))
                beta_loss_constraint = torch.sum(torch.pow(beta_loss_layers_sym[0].abs(), 2))
                for k in range(args.fr_n_layers - 1):
                    loss_constraint += torch.sum(torch.pow(loss_layers_sym[k + 1].abs(), 2))
                    beta_loss_constraint += torch.sum(torch.pow(beta_loss_layers_sym[k + 1].abs(), 2))

                gamma = torch.Tensor([0.01]).to(device)

                loss_all_train = train_loss_discrepancy + torch.mul(gamma, loss_constraint)+train_beta_loss_discrepancy + torch.mul(gamma, beta_loss_constraint)
                fr_optimizer.zero_grad()
                loss_all_train.backward()
                fr_optimizer.step()
                loss_train_total+=loss_all_train.data.item()
                loss_train_fr1+=train_loss_discrepancy.data.item()
                loss_train_fr2+=train_beta_loss_discrepancy.data.item()

            # mxnet
            #     lr.append(fr_optimizer.param_groups[0]['lr'])
            #     losses.append(loss_all_train.data.item())
            #     fr_optimizer.param_groups[0]['lr']=(fr_optimizer.param_groups[0]['lr'] * lr_mult)
            #
            #     if loss_all_train.data.item() < best_loss:
            #         best_loss = loss_all_train.data.item()
            #
            #     if loss_all_train.data.item() > 4 * best_loss or fr_optimizer.param_groups[0]['lr'] > 1.:
            #         break
            #
            # plt.plot(lr, losses)
            # plt.show()

            fr_module.eval()
            loss_val_fr, fnr_val = 0, 0
            for batch_idx, (noisy_signal, _, target_fr, freq, fr_ground) in enumerate(val_loader):
                if args.use_cuda:
                    noisy_signal, target_fr, fr_ground, freq = noisy_signal.cuda(), target_fr.cuda(), fr_ground.cuda(), freq.cuda()

                for i in range(noisy_signal.size()[0]):
                    mv = torch.max(torch.sqrt(pow(noisy_signal[i][0], 2) + pow(noisy_signal[i][1], 2)))
                    noisy_signal[i][0] = noisy_signal[i][0] / mv
                    noisy_signal[i][1] = noisy_signal[i][1] / mv

                with torch.no_grad():
                    [x_output, loss_layers_sym, beta_output, beta_loss_layers_sym] = fr_module(noisy_signal)

                    # Compute and print loss
                    val_loss_discrepancy = torch.sum(torch.pow(x_output.abs() - target_fr, 2))
                    val_beta_loss_discrepancy = torch.sum(torch.pow(beta_output.real - fr_ground, 2))
                    loss_constraint = torch.sum(torch.pow(loss_layers_sym[0].abs(), 2))
                    beta_loss_constraint = torch.sum(torch.pow(beta_loss_layers_sym[0].abs(), 2))
                    for k in range(args.fr_n_layers - 1):
                        loss_constraint += torch.sum(torch.pow(loss_layers_sym[k + 1].abs(), 2))
                        beta_loss_constraint += torch.sum(torch.pow(beta_loss_layers_sym[k + 1].abs(), 2))

                    gamma = torch.Tensor([0.01]).to(device)

                    loss_all_val = val_loss_discrepancy + torch.mul(gamma, loss_constraint) + val_beta_loss_discrepancy + torch.mul(
                        gamma, beta_loss_constraint)

                nfreq = (freq >= -0.5).sum(dim=1)
                f_hat = fr.find_freq(x_output.abs().cpu().detach().numpy(), nfreq, xgrid)
                fnr_val += fnr(f_hat, freq.cpu().numpy(), args.signal_dim)
                loss_val_total+=loss_all_val.data.item()
                loss_val_fr1+=val_loss_discrepancy.data.item()
                loss_val_fr2+=val_beta_loss_discrepancy.data.item()

            fnr_val *= 100 / args.n_validation
            fr_scheduler.step(loss_val_total)
            logger.info(
                "Epochs: %d / %d, Time: %.1f, FR training L2 loss %.2f, FR validation L2 loss %.2f, FNR %.2f %% (fr %.2f) (beta %.2f) (LR %e)",
                epoch, args.n_epochs_fr, time.time() - epoch_start_time, loss_train_total, loss_val_total,
                fnr_val, loss_train_fr1, loss_train_fr2,fr_optimizer.param_groups[0]['lr'])


        if epoch % args.save_epoch_freq == 0 or epoch == args.n_epochs_fr :
            util.save(fr_module, fr_optimizer, fr_scheduler, args, epoch, args.fr_module_type)




