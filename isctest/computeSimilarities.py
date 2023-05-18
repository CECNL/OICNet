from math import sin
from operator import sub
import numpy as np

"""
Subroutine of ISCTEST, version 2.02
Computes the similarity tensor and outputs its maximal elements
as well as information on second-order moments of the tensor

Inputs: spatialPattTens : The spatial patterns,
        testingComponents, verbose : boolean variables
        All directly from input to isctest.py

Outputs: The maximal similarities (for each component) in simitensor
         Information on which components reached the maximum in maxtensor
         Second-order moments in simimom2,
            used in fitting the empirical distributions.
"""

def computeSimilarities(spatialPattTens, testingComponents, verbose, onlyMax=True):

    # Read basic dimensions
    subjects = spatialPattTens.shape[0]
    pcadim = spatialPattTens.shape[1]
    datadim = spatialPattTens.shape[2]

    # Check for any problems in the data
    is_zerovec = True in ~np.any(spatialPattTens, axis=0)
    is_nan = True in np.isnan(spatialPattTens)
    if is_zerovec:
        print("WARNING: There seem to be zero vectors in the input!")
    if is_nan:
        print("WARNING: There seem to be NaN vectors in the input, ISCTEST is likely to crash!")

    """
    REMOVE MEANS IF TESTING INDEPENDENT COMPONENTS
    """
    if testingComponents:
        for k in range(subjects):
            for i in range(pcadim):
                spatialPattTens[k,i,:] = spatialPattTens[k,i,:] - np.mean(spatialPattTens[k,i,:])

    """
    Compute stabilized inverse of covariance matrix if testing mixing matrix
    """

    if not testingComponents: # Only used in testing mixing matrix
        if verbose:
            print('\nComputing stabilized inverse of covariance matrix:')
            print('  Concatenating data')

        wholedata = np.zeros((datadim, subjects*pcadim))
        for k in range(subjects):
            wholedata[:,k*pcadim:(k+1)*pcadim] = spatialPattTens[k]

        if verbose:
            print(" Calling Numpy SVD function on matrix of size {} x {}"\
                .format(wholedata.shape[0], wholedata.shape[1]))
        Uorig, sig, Vorig = np.linalg.svd(wholedata)
        # Find the pcadim largest singular values
        order = np.argsort(sig)[::-1]
        seldim = order[0:pcadim]
        # Order singular values
        singd = sig[seldim]
        # Check if matrix very badly conditioned
        # and increase small singular values if so
        maxcondition = 10
        conditionnumber = singd[0]/singd[pcadim-1]
        if conditionnumber > maxcondition:
            changesigns = np.sum(singd < singd[0]/maxcondition)
            singd = np.maximum(singd, singd[0]/maxcondition)
            if verbose:
                if conditionnumber < 2*maxcondition:
                    print("'  Global covariance matrix moderately badly conditioned:")
                else:
                    print("  WARNING: Global covariance matrix quite badly conditioned!")
                print("    Original condition number {}".format(conditionnumber))
                print("    Increased {} smallest singular values".format(changesigns))

        # Compute the matrices U and D
        U = Uorig[:, seldim]
        D = np.diag(1./np.square(singd))*datadim*pcadim
        if verbose:
            print("  Stabilized inverse computed:")
            print(" condition number is {}".format(singd[0]/singd[pcadim-1]))

        # Now, stabilized inverse of covariance is given by U*D*U'
        # Instead of computing it here,
        # we plug these matrices directly to formulae below to optimize computation
        # because the matrix U*D*U' can be very large but has very low rank.

        """
        COMPUTE SIMILARITIES AND STORE MAX INTER-SUBJECT SIMILARITIES
        """
        if verbose:
            print("Computing similarities and storing maximal ones: ")
        # Create first tensors in which similarities stored
        # For each component, we store only the maximum similarities to other subjects
        similarities = np.zeros((subjects, subjects, pcadim, pcadim))
        simitensor = np.zeros((subjects, subjects, pcadim)) # values of max similarities
        maxtensor = np.zeros((subjects, subjects, pcadim)) # which components reached the max
        # for empirical test, compute 2nd moments of similarity distribution
        simimom2 = 0
        for subj1 in range(subjects):
            if verbose: print('.', end='')
            M1 = spatialPattTens[subj1]
            if testingComponents: # if testing components, normalize simply
                M1 = M1 / (np.sqrt(np.ones((datadim, 1))@np.diag(M1.T@M1).reshape(1,datadim)+1e-100))
            else: # if testing mixing matrix, normalize by weighted norm
                M1 = M1 / (np.sqrt(np.ones((datadim, 1))@np.diag(M1.T@U@D@U.T@M1).reshape(1,datadim)+1e-100))

            for subj2 in range(subjects):
                M2 = spatialPattTens[subj2]
                if testingComponents: # if testing components, normalize simply
                    M2 = M2 / (np.sqrt(np.ones((datadim, 1))@np.diag(M2.T@M2).reshape(1,datadim)+1e-100))
                else: # if testing mixing matrix, normalize by weighted norm
                    M2 = M2 / (np.sqrt(np.ones((datadim, 1))@np.diag(M2.T@U@D@U.T@M2).reshape(1,datadim))+1e-100)
                if testingComponents:
                    similarities[subj1, subj2] = np.abs(M1.T@M2)
                else:
                    similarities[subj1, subj2] = np.abs(M1.T@U@D@U.T@M2)
                if subj1 != subj2:
                    maxsimis = np.max(similarities[subj1, subj2], axis=1) # row major
                    maxindices = np.argmax(similarities[subj1, subj2], axis=1)
                    simitensor[subj1, subj2] = maxsimis 
                    maxtensor[subj1, subj2] = maxindices # maxindices[subj2_idx] = subj1_idx
                    simimom2 = simimom2 + np.sum(np.square(similarities[subj1, subj2]))

    # Compute moments used only in determining empirical thresholds
    simimom2 = simimom2/(subjects*(subjects-1)*pcadim**2)

    # Due to numerical inaccuracies, similarities can be slightly larger than 1.
    # This needs to be corrected to avoid error in betainc function.

    simitensor=np.minimum(simitensor, 1)

    if verbose: print("Done.")

    if onlyMax:
        return simitensor, maxtensor, simimom2
    else:
        return similarities