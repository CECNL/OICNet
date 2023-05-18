import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from math import floor, ceil
from isctest.computeSimilarities import computeSimilarities
from matplotlib.gridspec import GridSpec

"""
Data process
"""

def concat(x):
    """
    Concatenate 3-dimensional data to 2-dimensional continuous data.
    """
    n_trial, n_ch, n_time = x.shape
    raw = np.zeros(shape=(n_ch, n_trial*n_time))
    for i in range(n_trial):
        raw[:, n_time*i : n_time*(i+1)] = x[i]
    return raw

def time_window(x, window_size):
    n_ch, n_samples = x.shape
    n_window = floor(n_samples/window_size)
    windowed = np.zeros(shape=(n_window, n_ch, window_size))
    for i in range(n_window):
        windowed[i] = x[:, i*window_size:(i+1)*window_size]
    return windowed


"""
Plotting 
"""
def plot_loss(lossRecr, title_list, figsize=None, xlabel='Epochs'):
    loss_curve = lossRecr.loss_hist()
    n_rows = len(loss_curve)
    if n_rows == 1:
        if figsize is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title_list[0])
        ax.set_xlabel(xlabel)
        ax.plot(loss_curve[0])
        fig.tight_layout()
    else:
        if figsize is None:
            fig, ax = plt.subplots(nrows=len(loss_curve))
        else:
            fig, ax = plt.subplots(nrows=len(loss_curve), figsize=figsize)
        for i in range(len(loss_curve)):
            ax[i].set_title(title_list[i])
            ax[i].set_xlabel(xlabel)
            ax[i].plot(loss_curve[i])
        fig.tight_layout()

def plot_topoplot(mix, info, n_cols=5, title='topoplot', row_major=False, sort=False, sphere=None, figsize=(24,36)):
    n_comp = mix.shape[1]
    if not row_major:
        mix = mix.T
    
    if sort:
        mix, sort_idx = sortTopo(mix)

    # Plot
    n_rows = int(ceil(n_comp/n_cols))
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, gridspec_kw=dict(top=0.9),
                           sharex=True, sharey=True)

    for row in range(n_rows):
        for col in range(n_cols):
            n = row * n_cols + col
            if n >= n_comp : break
            if n_rows == 1 : ax_comp = ax[col]
            else : ax_comp = ax[row, col]
            mne.viz.plot_topomap(mix[n], info, axes=ax_comp, show=False, sphere=sphere)
            ax_comp.set_title(str(n), fontweight='bold')
    fig.suptitle(title, fontsize=16, fontweight='bold')

def plot_matched_mix(kernel1, kernel2, comp1, comp2, super_title, title_list, unimatch, n_match, info
                     ,figsize=(30,42)):
    fig=plt.figure(figsize=figsize)
    fig.suptitle(super_title, fontsize=30)
    gs=GridSpec(n_match, 8)
    
    for i in range(n_match):
        ax1=fig.add_subplot(gs[i,0]) 
        ax2=fig.add_subplot(gs[i,1:4])
        ax3=fig.add_subplot(gs[i,4])
        ax4=fig.add_subplot(gs[i,5:8])

        idx_k1 = unimatch[i][0]
        idx_k2 = unimatch[i][1]
        abs_rho = np.abs(np.corrcoef(comp1[idx_k1], comp2[idx_k2])[0,1])
        
        ax1.set_ylabel(str(idx_k1), rotation=0, fontsize=20)
        mne.viz.plot_topomap(kernel1[:, idx_k1], info, axes=ax1, show=False)
        ax2.plot(comp1[idx_k1])
        ax2.axhline(y = 0, color = 'black')
        
        ax3.set_ylabel(str(idx_k2),  rotation=0, fontsize=20)
        mne.viz.plot_topomap(kernel2[:, idx_k2], info, axes=ax3, show=False)
        ax4.plot(comp2[idx_k2])
        ax4.axhline(y = 0, color = 'black')
        if i == 0:
            ax1.set_title(title_list[0])
            ax2.set_title(title_list[1] + '\n' + 'corr: ' + str(round(abs_rho,2)), fontsize=15)
            ax3.set_title(title_list[2])
            ax4.set_title(title_list[3])
        else:
            ax2.set_title('corr:' + str(round(abs_rho,2)), fontsize=15)

def plot_ica_comp(mix, comp, super_title, info, srate, base=0, n_comp=5, fft=False,
                  downsample=1, comp_range=None, time_idx=None, figsize=(30,42)):
    fig=plt.figure(figsize=figsize)
    fig.suptitle(super_title, fontsize=30, y=1.01)
    _, n_sample = comp.shape
    gs=GridSpec(n_comp, 4)
    
    if comp_range is None:
        comp_range = np.arange(comp.shape[1]) 
        
    for i in range(n_comp):
        idx = base + i
        ax1=fig.add_subplot(gs[i,0]) 
        if fft:
            ax2=fig.add_subplot(gs[i,1:3])
            ax3=fig.add_subplot(gs[i,3])
        else:
            ax2=fig.add_subplot(gs[i,1:4])
        
        # Topoplot
        ax1.set_ylabel(str(idx), rotation=0, fontsize=20)
        mne.viz.plot_topomap(mix[:, idx], info, axes=ax1, show=False)
        
        # Components
        if time_idx is None:
            ax2.plot(comp[idx][comp_range][::downsample])
        else:
            ax2.plot(time_idx[::downsample], comp[idx][comp_range][::downsample])
            ax2.set_xlabel('Time(second)', fontsize=10)
        
        # Freq. spectrum
        if fft and i == 0:
            xf = rfftfreq(n_sample, 1 / srate)
            xf_idx_les = np.where(xf <= 50)[0]
        if fft:
            yf = rfft(comp[idx])
            ax3.plot(xf[xf_idx_les][::20], np.abs(yf)[xf_idx_les][::20])
            ax3.set_xlabel('Frequency', fontsize=10)
            ax3.set_ylabel('Magnitude', fontsize=10)
    
    fig.tight_layout()

"""
Match
"""

def compare_mix_kernel(mix1, mix2, return_idx=False, verbose=True, n_iter=1, nparr=False):
    """
    Compare two ICA mix matries.
    If return_idx, the return value unimatch would be a (n_match, 2) matrix.
    For each match, the first axis is the index for a component of mix1 and vice versa.
    """
    n_ch, n_comp = mix1.shape
    compare_set = np.zeros((2, n_ch, n_comp))
    compare_set[0] = mix1
    compare_set[1] = mix2
    simitensor = computeSimilarities(compare_set, testingComponents=False, verbose=False, onlyMax=False)
    simimatrix = simitensor[0,1]
    unimatch = matchComponent(simimatrix, n_iter=n_iter, nparr=nparr)

    if verbose:
        print("{} Unique Matches.".format(len(unimatch)))

    if return_idx:
        return unimatch

def matchComponent(simi_matrix, n_iter=2, nparr=False):
    # simi_matrix : row method 1, column method 2
    n_dim = simi_matrix.shape[0]
    m1_not_match = np.arange(0, n_dim, dtype=int)
    m2_not_match = np.arange(0, n_dim, dtype=int)
    matched_idx = [] # A list stores sets of index in the form of (idx_m1, idx_m2) , e.g., [(0, 3), (4, 5), ...]
    simi_matrix_c = simi_matrix.copy() # copy of simi. matrix
    for _ in range(n_iter):
        m1_max_arg = simi_matrix_c.argmax(axis=1)
        m2_max_arg = simi_matrix_c.argmax(axis=0)
        # Match making
        for idx in range(len(m1_max_arg)):
            if m2_max_arg[m1_max_arg[idx]] == idx: # Unique Match Criterion
                m1_idx = m1_not_match[idx]
                m2_idx = m2_not_match[m1_max_arg[idx]]
                matched_idx.append([m1_idx, m2_idx])
        
        # Update not_match for consistency
        for idx in range(len(matched_idx)):
            m1_not_match = np.delete(m1_not_match, np.where(m1_not_match == matched_idx[idx][0])[0])
            m2_not_match = np.delete(m2_not_match, np.where(m2_not_match ==matched_idx[idx][1])[0])
        simi_matrix_c = simi_matrix[m1_not_match][:, m2_not_match]

    if nparr:
        return np.array(matched_idx)
    else:
        return matched_idx

def findCommonComponent(match_index, n_comp, nparr=False):
    n_methods = len(match_index)
    vote = np.zeros(shape=n_comp)
    common = []
    for i in range(n_methods):
        for idx in match_index[i]:
            vote[idx] += 1
            if vote[idx] == n_methods:
                common.append(idx)
    if nparr:
        return np.array(sorted(common))
    else:
        return sorted(common)

def findMatchIdx(match_list, target, target_dim=1):
    """
    Find the index that correspond to target indice
    """
    target_idx = []

    for i in range(len(target)):
        idx = np.where(match_list[:,target_dim] == target[i])[0]
        target_idx.append(idx)
    match_dim = np.abs(target_dim-1)
    target_idx = np.array(target_idx)
    match_idx = match_list[:,match_dim][target_idx]
    return match_idx.reshape(-1)

def traceCorr(inter_unmix, gt_unmix, data, match_list):
    """
    Index of inter_unmix in match_list : 0
    Index of gt_unmix in match_list : 1
    """
    n_time, _, n_comp = inter_unmix.shape
    corrcurve = np.zeros(shape=(n_comp, n_time))
    comp_gt = gt_unmix@data
    for time_idx in range(n_time):
        unmix = inter_unmix[time_idx]
        comp = unmix@data
        for match_idx in range(len(match_list)):
            trace_idx = match_list[match_idx][0]
            gt_idx = match_list[match_idx][1]
            corrcurve[trace_idx, time_idx] = np.abs(np.corrcoef(comp[trace_idx], comp_gt[gt_idx])[0,1])
    return corrcurve


def init_iter_dict(iter_dict, keys):
    for key in keys:
        iter_dict[key] = 1

def sufficientMatches(im_dict, time_idx, gt_mix, gt_var, n_iter_dict, threshold=0.8):
    n_comp = gt_mix.shape[1]
    while True:
        match_dict = dict()
        gt_index = []
        method_list = list(n_iter_dict.keys())
        for method in method_list:
            mix = im_dict[method][time_idx]
            matches = compare_mix_kernel(mix, gt_mix, return_idx=True, nparr=True, verbose=False, n_iter=n_iter_dict[method])
            match_dict[method] = matches
            gt_index.append(matches[:,1])
        cc = findCommonComponent(match_index=gt_index, n_comp=n_comp, nparr=True)
        
        if len(cc) != 0:
            ev = (gt_var[cc]/gt_var.sum()).sum()
        else:
            ev = 0
        # increase the iterations on min. match method
        if ev <= threshold:
            min_method = ''
            min_match = np.inf
            for i in range(len(gt_index)):
                if len(gt_index[i]) < min_match:
                    min_method = list(match_dict.keys())[i]
                    min_match = len(gt_index[i])
            n_iter_dict[min_method] = n_iter_dict[min_method] + 1
        else:
            return match_dict, cc

"""
other
"""

def sortTopo(mix, axis=0):
    sort_idx = np.argsort(np.sum(mix**2, axis=axis))[::-1]
    mix = mix[:, sort_idx]
    return mix, sort_idx

def disableTick(ax, axis='both'):
    if axis == 'both':
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
    elif axis == 'x':
        ax.xaxis.set_ticks([])
    elif axis == 'y':
        ax.yaxis.set_ticks([])

def disableTickLabels(ax, axis='both'):
    if axis == 'both':
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
    elif axis == 'x':
        ax.xaxis.set_ticklabels([])
    elif axis == 'y':
        ax.yaxis.set_ticklabels([])

def disableFrame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

def setCompAxSpine(ax, xmin, xmax):
    ax.spines['bottom'].set_bounds((xmin, xmax))
    ax.spines['top'].set_bounds((xmin, xmax))
    ax.spines['right'].set_visible(False)