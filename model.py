import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import function as fn
import torch.optim as optim
import time
from numpy.linalg import pinv

"""
SCCNet
"""
class lossRecr():
    def __init__(self):
        self.n_train_loss = 0 
        self.n_valid_loss = 0
        self.train_loss = [] # shape[epoch, n_loss]
        self.valid_loss = []

    def record_loss(self, loss_list, loss_type='train'):
        if loss_type == 'train':
            if self.n_train_loss == 0:
                self.n_train_loss = len(loss_list)
            self.train_loss.append(loss_list)
        elif loss_type == 'valid':
            if self.n_valid_loss == 0:
                self.n_valid_loss = len(loss_list)
            self.valid_loss.append(loss_list)
    
    def loss_hist(self, loss_type='train'):
        if loss_type == 'train':
            return np.array(self.train_loss).transpose()
        elif loss_type == 'valid':
            return np.array(self.valid_loss).transpose()

class SCCNet_SF(nn.Module):
    def __init__(self, n_components, n_ch, c_bias=True):
        super(SCCNet_SF, self).__init__()
        self.conv1 = nn.Conv2d(1, n_components, (n_ch, 1), bias=c_bias)
        self.Bn1 = nn.BatchNorm2d(n_components)
        self.n_train_loss = 0 
        self.n_valid_loss = 0
        self.n_comp = n_components
        self.n_ch = n_ch
        self.train_loss = [] # shape[epoch, n_loss]
        self.valid_loss = []

    def forward(self, x):
        x = self.conv1(x)
        bnx = self.Bn1(x)
        return x, bnx
    
    def record_loss(self, loss_list, loss_type='train'):
        if loss_type == 'train':
            if self.n_train_loss == 0:
                self.n_train_loss = len(loss_list)
            self.train_loss.append(loss_list)
        elif loss_type == 'valid':
            if self.n_valid_loss == 0:
                self.n_valid_loss = len(loss_list)
            self.valid_loss.append(loss_list)
    
    def loss_hist(self, loss_type='train'):
        if loss_type == 'train':
            return np.array(self.train_loss).transpose()
        elif loss_type == 'valid':
            return np.array(self.valid_loss).transpose()

class OICNet():
    def __init__(self, n_ch, n_components, c_bias=False, lr=5*1e-3, init=None, dev=None):
        """
        n_ch : Number of EEG channels
        n_components : Number of indepedent components
        c_bias : The bias of cnn kernel
        lr : Learning rate
        init : Initial weight
        dev : Device used in PyTorch framework
        """
        # Hyperparameters
        self.n_ch = n_ch
        self.n_components = n_components
        self.c_bias = c_bias
        self.lr = lr
        if dev == None:
            self.dev = torch.device("cpu")
        else:
            self.dev = dev
        self.unmix = SCCNet_SF(n_components=self.n_components, n_ch=self.n_ch, c_bias=self.c_bias).double().to(self.dev)
        if init is not None:
            self.unmix.conv1.weight.data.copy_(init)
        # Optimizer
        self.optimizer = optim.Adam(list(self.unmix.parameters()), lr=self.lr)
        # Records
        self.inter_white = None
        self.lossRecr = lossRecr()
        self.elapsed_times = []

    def fit(self, x_train, rls_whiten, epoch, alpha=0.5, sigmoid=torch.tanh, n_sub=0,
            contrast='g1', org='rec', exp='mean', square=True, record=False, verbose=False):

        beta = 1 - alpha

        if record:
            self.inter_white = np.zeros(shape=(len(x_train)+1, self.n_ch, self.n_ch))
            inter_unmix = np.zeros(shape=(len(x_train)+1, self.n_components, self.n_ch))
            inter_mix = np.zeros(shape=(len(x_train)+1, self.n_ch, self.n_components))
            self.inter_white[0] = rls_whiten.icasphere
            inter_unmix[0] = self.unmix.state_dict()['conv1.weight'].cpu().numpy().reshape(self.n_components, self.n_ch)
            inter_mix[0] = pinv(inter_unmix[0])

        for seg in range(len(x_train)):
            rec_neg = 0
            rec_org = 0
            rec_tot = 0
            
            # Online Whitening
            blockData = x_train[seg]
            fn.dynamicWhiten(blockData, rls_whiten)
            
            # Whiten Data
            blockData = rls_whiten.icasphere@blockData
            blockData = torch.tensor(blockData).unsqueeze(0).unsqueeze(0).double().to(self.dev)
            
            # Updating Unmixing matrix
            self.unmix.train()
            start = time.time()
            for e in range(epoch):
                comp, _ = self.unmix(blockData) # y = Wx
                loss_neg = fn.negentropy(comp, sigmoid=sigmoid, contrast=contrast, exp=exp, square=square, n_sub=n_sub) 

                if org == 'rec':
                    loss_org = fn.recons_penalty(data=blockData, model=self.unmix)
                elif org == 'mc':
                    loss_org = fn.mutual_coherence(model=self.unmix, dev=self.dev)
                elif org == 'srip':
                    loss_org = fn.l2_reg_ortho(model=self.unmix, dev=self.dev)
                else:
                    raise Exception('Unknown Orthogonal Criterion.')

                loss = alpha*loss_neg + beta*loss_org
                rec_neg += loss_neg.item()
                rec_org += loss_org.item()
                rec_tot += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.lossRecr.record_loss([rec_neg, rec_org, rec_tot])

            elased_time = time.time()-start
            self.elapsed_times.append(elased_time)
            if verbose:
                print("Training time for segment {} : {} sec".format(seg, elased_time))

            # Save intermedia result
            if record:
                self.unmix.eval()
                w = self.unmix.state_dict()['conv1.weight'].cpu().numpy().reshape(self.n_components, self.n_ch)
                inter_unmix[seg+1] = w @ rls_whiten.icasphere
                inter_mix[seg+1] = pinv(inter_unmix[seg+1])
                self.inter_white[seg+1] = rls_whiten.icasphere
      
        if record:
            return inter_unmix, inter_mix
        else:
            self.unmix.eval()
            w = self.unmix.state_dict()['conv1.weight'].cpu().numpy().reshape(self.n_components, self.n_ch)
            unmix = w @ rls_whiten.icasphere
            mix = pinv(unmix)
            return unmix, mix