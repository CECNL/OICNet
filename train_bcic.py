import os
import sys
root_dir = os.path.abspath('.')
sys.path.append(root_dir)
import os 
import numpy as np
import torch
import util as ut
import function as fn
import model as md
import argparse

data_path = root_dir + '/dataset/bcic_whole/'

window_size = 128

def main(x_train, alpha, epoch, ff, lr, blockSize, dev, contrast, org, init=None, record=False):
    n_component = 56
    n_ch = 56
    c_bias = False

    if init is None:
        init_point = torch.eye(n_ch).view(n_ch,1,n_ch,1).to(dev)
    else:
        init_point = np.load(init)
        init_point = torch.tensor(init_point).view(n_ch,1,n_ch,1).to(dev)

    ow_state = fn.ld_state(nChs=n_ch, blockSize=blockSize, lambda_now=ff)
    OICNet = md.OICNet(n_ch=n_ch, n_components=n_component, c_bias=c_bias, lr=lr, init=init_point, dev=dev)
    if record:
        inter_unmix, inter_mix = OICNet.fit(x_train=x_train, rls_whiten=ow_state, contrast=contrast, org=org,
                                            epoch=epoch, alpha=alpha, record=True)
        return inter_unmix, inter_mix
    else:
        unmix, mix = OICNet.fit(x_train=x_train, rls_whiten=ow_state, contrast=contrast, org=org,
                                epoch=epoch, alpha=alpha, record=False)
        return unmix, mix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparser for training OICNet')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate for Adam' )
    parser.add_argument('-a', '--alpha', type=float, default=0.994, help='The coefficient of negentropy')
    parser.add_argument('-e', '--epoch', type=int, default=5, help='Number of epoch for fine-tuning')
    parser.add_argument('-s', '--save', type=int, default=0, help='Save OICNet')
    parser.add_argument('-p', '--save_path', type=str, default=None, help='Specify path to save the model' )
    parser.add_argument('-g', type=str, default='g1', help='Contrast function')
    parser.add_argument('--org', type=str, default='rec', help='Orthogonal constraint in loss function')
    parser.add_argument('--bs', type=int, default=8, help='Blocksize for whitening')
    parser.add_argument('--ff', type=float, default=1e-3, help='Forgetting factor used in RLS whitening')
    parser.add_argument('--subj', type=str, default='S02', help='Specify the subject of training data')
    parser.add_argument('--session', type=str, default='1', help='Specify the session of training data')
    parser.add_argument('--init', type=str, default=None, help='Path to initial weights for OICNet')
    parser.add_argument('--filename', type=str, default=None, help='Postfix for filename')
    parser.add_argument('--verbose', type=int, default=0, help='Show system message')
    parser.add_argument('--record', type=int, default=0, help='Record the model after each fine-tuning')
    parser.add_argument('--dev', type=str, default=0, help='Device used for training')
    args = parser.parse_args()

    savepath = args.save_path
    init = args.init
    
    if args.filename is not None:
        postfix = args.filename
    else:
        postfix = ''
    
    if args.dev == -1:
        dev = torch.device('cpu')
    else:
        dev = torch.device('cuda:{}'.format(args.dev)) if torch.cuda.is_available() else torch.device('cpu')

    if savepath is not None:
        savepath = os.path.abspath(savepath)
    if init is not None:
        init = os.path.abspath(init)

    if args.verbose:
        print('Filename postfix: ', postfix)
        print('Save: Yes.') if args.save else print('Save: No')
        print('Record: Yes.') if args.record else print('Record: No')
        print('Subject: ', args.subj)
        print('Session: ', args.session)
        print('Loss function: {} + {}'.format(args.g, args.org))
        print('Alpha: ', args.alpha)
        print('Epoch: ', args.epoch)
        print('Learning Rate: ', args.lr)
        print('Block Size: ', args.bs)
        print('Device: ', dev)
        print('Initial Weight: Identity (Default).') if init is None else print('Initial Weight: ', init)

    x_train = np.load(data_path + '/preprocessed/npy/Data_' + args.subj + '_Sess0'+ str(args.session) +'.npy')
    x_train = ut.time_window(x_train, window_size)

    if args.record:
        inter_unmix, inter_mix = main(x_train, alpha=args.alpha, epoch=args.epoch, lr=args.lr, contrast=args.g, org=args.org, 
                                      ff=args.ff, blockSize=args.bs,
                                      dev=dev, init=init, record=True)
        if args.save:
            np.save(savepath + '/iu_' + postfix + '.npy', inter_unmix)
            np.save(savepath + '/im_' + postfix + '.npy', inter_mix)
    else:
        unmix, mix = main(x_train, alpha=args.alpha, epoch=args.epoch, lr=args.lr, contrast=args.g, org=args.org, 
                          ff=args.ff, blockSize=args.bs,
                          dev=dev, init=init, record=False)
        if args.save:
            np.save(savepath + '/unmix_' + postfix + '.npy', unmix)
            np.save(savepath + '/mix_' + postfix + '.npy', mix)

    if args.verbose:
        print('Finish training OICNet.')