import numpy as np

def uppertriangindices(pcadim, subjects):
    mask = np.zeros((subjects, subjects, pcadim))
    for subj1 in range(subjects):
        for subj2 in range(subj1-1):
            mask[subj1, subj2, :] = np.ones(pcadim)
    indices = np.where(mask!=0)
    return indices