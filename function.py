import sys
sys.path.append('.')
import torch
import numpy as np
import util as ut
from math import floor
from numpy import sqrt
from numpy.linalg import inv
from torch.nn.functional import normalize

"""
RLS Whitening
"""
class ld_state:
    def __init__(self, nChs, blockSize, lambda_now, icasphere=None, dev=None):
        self.blockSize = blockSize
        self.lambda_now = lambda_now
        # Initial Weights
        if icasphere is None:
            self.icasphere = np.eye(nChs)
        else:
            self.icasphere = icasphere
        self.icaweights = torch.eye(nChs, dtype=torch.double).to(dev)

def dynamicWhiten(blockdata, ld_state):
    nChs, nSams = blockdata.shape
    blockdata = centering(blockdata) # Re-mean
    
    numSegBlock = floor(nSams/ld_state.blockSize)

    for bi in range(numSegBlock):
        dataRange = np.arange(floor(bi*nSams/numSegBlock), min(nSams, floor((bi+1)*nSams/numSegBlock)))
        xWhite = ld_state.icasphere @ blockdata[:,dataRange]
        ld_avg = 1 - ld_state.lambda_now
        ld_state.ld_avg = ld_avg
        QWhite = ld_avg/(1-ld_avg) + np.trace(xWhite.T @ xWhite) / len(dataRange)
        ld_state.icasphere = 1/ld_avg * (ld_state.icasphere - xWhite @ xWhite.T / len(dataRange) / QWhite @ ld_state.icasphere)
    return ld_state

def centering(data:np.array):
    if data.ndim == 3 : data = ut.concat(data)
    row, col = data.shape
    if col > 1:
        data = data - np.mean(data, axis=1).reshape(row,1)
        return data
    else:
        return data


def negentropy(components, sigmoid=None, contrast='g1', exp='mean', square=True, n_sub=0):
    """
    components : Output of OICNet
    sigmoid : Activation function for computing the negentropy
    contrast : Option of contrast function
    exp : Definition of expectation
    square : Take the square of negentropy or not
    n_sub : Number of subgaussian components
    """
    components = components.squeeze(2).squeeze(0) #Shape: (n_ch, n_time)
    if sigmoid is not None:
        components = sigmoid(components)

    # Contrast function
    if contrast == 'g1':
        components = 0.5*torch.log(torch.cosh(2*components))
    elif contrast == 'g2':
        components = -torch.exp(-0.5*(components**2))
    elif contrast == 'g3':
        components = 0.25*(components**4)
    else:
        raise Exception('Unknown Contrast Function.')
    
    sub_ic = components[0:n_sub]
    super_ic = components[n_sub:]
    
    if exp == 'mean':
        sub_loss = torch.mean(sub_ic)
        super_loss = torch.mean(super_ic)
    else: 
        sub_loss = torch.sum(sub_ic)
        super_loss = torch.sum(super_ic)
        
    if square:
        super_loss = super_loss**2
        sub_loss = sub_loss**2
    
    if n_sub == 0 : sub_loss = 0
    if n_sub == len(components) : super_loss = 0
    
    loss = super_loss - sub_loss
    
    return loss

def recons_penalty(data, model, sigmoid=None):
    for param in model.parameters():
        if param.ndimension() < 2:
            continue
        else:
            # each row of w corresponds to a kernel
            cols = param[0].numel()
            w = param.view(-1, cols)

            if sigmoid is not None:
                recons = sigmoid(w@data)
                recons = sigmoid(w.T@recons)
            else:
                recons = w.T@w@data

            loss = torch.linalg.norm(recons-data, dim=2)
            loss = torch.square(loss)
            loss = torch.mean(loss, dim=2).squeeze(1)
            loss = torch.mean(loss)
    return loss

def mutual_coherence(model, dev):
    """
    Refer to https://github.com/VITA-Group/Orthogonality-in-CNNs
    """
    for param in model.parameters():
        if param.ndimension() < 2:
            continue
        else:
            # each row of w corresponds to a kernel
            cols = param[0].numel() # the length of flatten kernel
            rows = param.shape[0] # the number of kernel
            w = param.view(-1, cols) 
            wt = torch.transpose(w,0,1)
            wwt = torch.matmul(w, wt)
            ident= torch.eye(rows, requires_grad=True).to(dev)
            loss_ortho = torch.norm(wwt-ident, p=float('inf'))
    return loss_ortho

# SRIP
def l2_reg_ortho(model, dev):
    """
    Refer to https://github.com/VITA-Group/Orthogonality-in-CNNs
    """
    l2_reg = None
    for param in model.parameters():
        if param.ndimension() < 2:
            continue
        else:   
            cols = param[0].numel()
            rows = param.shape[0]
            w1 = param.view(-1,cols)
            wt = torch.transpose(w1,0,1)
            m  = torch.matmul(w1, wt)
            ident = torch.eye(rows, rows).to(dev)
            
            w_tmp = (m - ident)
            height = w_tmp.size(0)
            u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
            v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
            u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
            sigma = torch.dot(u, torch.matmul(w_tmp, v))

            if l2_reg is None:
                l2_reg = (sigma)**2
            else:
                l2_reg = l2_reg + (sigma)**2
    return l2_reg

"""
Other
"""

def zca_whiten(x):
    split = False
    if x.ndim == 4:
        x = np.squeeze(x, 1)
    if x.ndim != 2:
        n_batch, n_ch, n_time = x.shape
        x_cat = np.zeros((n_ch, n_batch*n_time))
        for i in range(n_batch):
            x_cat[:, i*n_time:(i+1)*n_time] = x[i]
        x = x_cat
        split = True

    tol = 1e-7
    x = centering(x)
    cov = np.cov(x)
    d, E = np.linalg.eigh(cov)
    validIdx = np.where(d > tol)[0]
    d = d[validIdx]
    E = E[:, validIdx]
    D = np.diag(d)
    sphmat = E@inv(sqrt(D))@E.T
    dsphmat = E@sqrt(D)@E.T
    x_pw = sphmat @ x

    if split:
        x_pw_split = np.zeros((n_batch, n_ch, n_time))
        for i in range(n_batch):
            x_pw_split[i] = x_pw[:, i*n_time:(i+1)*n_time]
        return x_pw_split, sphmat, dsphmat
    else:
        return x_pw, sphmat, dsphmat

def bssrho(c):
    m = c.shape[0]
    cabs = np.abs(c)**2
    rowms = cabs.max(axis=1)
    colms = cabs.max(axis=0)
    rowsum = cabs.sum(axis=1)
    colsum = cabs.sum(axis=0)
    rowterm = rowms/(rowsum+1e-8)
    colterm = colms/(colsum+1e-8)
    rho = (1/(m-1))*(m-0.5*(rowterm.sum()+colterm.sum()))
    return rho

