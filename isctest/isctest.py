import numpy as np
from computeSimilarities import computeSimilarities
from uppertriangindices import uppertriangindices
from scipy.special import betainc

"""
Python-Rewritten version of Matlab code verison 2.2, Feb 2013 by Aapo Hyvarinen, University of Helsinki

See http://www.cs.helsinki.fi/u/ahyvarin/code/isctest/ for more information

"""

"""
SET PARAMETERS TO DEFAULT VALUES
"""


def isctest(spatialPattTens, alphaFP, alphaFD, *args):
    # STRATEGY FOR LINKING (see below for options)
    strategy = 'SL'
    # USE SIMES PROCEDURE OR NEW DEFINITION OF FDR?
    simes = 0
    # CORRECT ERROR RATES FOR MULTIPLE TESTING, OR ASSUME THEY ARE ALREADY CORRECTED (if you want to compute them yourself for some reason)
    corrected = 0
    # Do we show diagnostics on screen? (0/1/2)
    verbose = 1
    # The user has to tell what to test (components/mixing) to avoid misunderstanding
    testingComponents = np.nan

    n_args = len(args)
    assert n_args > 0
    for option in args:
        # ARE INPUTS INDEPENDENT COMPONENTS OR COLUMNS OF MIXING MATRIX
        # i.e. basic choice of what to test, and the ensuing testing approach (analytic or empirical)
        if option == "mixing":
            testingComponents = 0
            empirical = 0
        elif option == "component":
            testingComponents = 1
            empirical = 1
        # You may also want to set the testing approach independently here:
        elif option == "analytic":
            empirical = 0
        elif option == "empirical":
            empirical = 1
        # Screen output options:
        elif option == "silent": # print no output on screen
            verbose = 0
        elif option == "verbose": # print moderate output on screen
            verbose = 1
        elif option == "extraverbose": # print some extra diagnostics
            verbose = 2
        # Linkage strategies
        elif option == "CL":
            strategy = "CL" # complete-linkage
        elif option == "ML":
            strategy = "ML" # median-linkage (original)
        elif option == "SL":
            strategy = "SL" # single-linkage
        # Statistical options
        elif option == "simes":
            simes = 1 # use Simes procedure for FDR computation (not recommended)
        elif option == "customfdr":
            simes = 0 # use our new method for FDR computation
        elif option == "corrected":
            corrected = 1 # if you want to compute corrected alphas yourself, choose this
        else:
            raise Exception("No such option: " + option)

    # Read basic dimensions
    subjects = spatialPattTens.shape[0]
    pcadim = spatialPattTens.shape[1]
    datadim = spatialPattTens.shape[2]
    complexvalued = np.isreal(spatialPattTens)
    complexvalued = True in complexvalued

    # Initial output to user
    if verbose:
        print('\n*** ISCTest: ICA testing algorithm ***\n')
        print('Input parameters:')
        print('  Number of subjects: ', subjects)
        print('  Number of vectors/components per subject:', pcadim)
        print('  Data vector dimension:', datadim)
        print('  False positive rate: ', alphaFP)
        print('  False discovery rate: ', alphaFD)
        if complexvalued:
            print('  Data is complex-valued,')
            print(' distributions adjusted accordingly\n')

    """
    Compute similarities
    """
    # Compute maximal parts of similarity tensor, and store information on which
    # connections were maximal, as well as 2nd order moments of the tensor
    simitensor, maxtensor, simimom2 = computeSimilarities(spatialPattTens, testingComponents, verbose)

    """
    Do clustering in deflation mode
    """
    if verbose : print("*** Launching clustering computations ***")

    # In analytical method, needs two iterations
    # but in empirical method, only one is need
    if empirical:
        iterations = 1
    else:
        iterations =2

    for iteration in range(iterations):
        if verbose:
            if empirical:
                print("Starting iteration, forming cluster ")
            else:
                print("Starting iteration {}, forming cluster ".format(iteration))

        # Initialize the tensor which gives the set of pvalues which have not been deflated away
        deflationtensor = np.zeros((subjects, subjects, pcadim))
        # Initialize output variables
        clustering=[]
        clusterorder=[]
        linkpvalues=[]
        linksimilarities=[]

        """
        MAIN LOOP FOR CLUSTERING (EACH TIME FINDING A NEW CLUSTER)
        STARTS HERE
        """

        # Boolean variable which tell when to stop
        nomoreclusters = 0
        # Initialize iteration counter
        clusterindex = 1

        while not nomoreclusters:
            if verbose : print(".", clusterindex)
            """
            COMPUTE EFFECTIVE DIMENSIONS
            """
            if empirical:
                """
                General case of empirical thresholds:
                Estimate effective dimension
                based on the empirical distribution of similarities
                Using method of moments estimator for beta (which gives effdim)
                """
                if not complexvalued:
                    effdimest = 1/simimom2
                else:
                    # This case (testing complex-valued independent components) has not been
                    # properly tested  because it is unlikely to occur in practice
                    effdimest = 1/simimom2-0.5
                effdim = effdimest*np.ones((subjects,subjects))
            else:
                # case of analytically computed thresholds
                if iteration == 1:
                    if clusterindex == 1:
                        effdim=pcadim*np.ones((subjects,subjects))
                    else:
                        # if not first cluster, compute new dimension parameters
                        # from the results of current clustering
                        # first compute effdim, starting from pcadim and reducing it
                        neweffdim = pcadim*np.ones((subjects,subjects))
                        for c in range(clustering.shape[0]):
                            clustersize = np.sum(clustering[c,:]>0)
                            for s1 in range(subjects):
                                for s2 in range(subjects):
                                    # for each pair of component in the same cluster
                                    # reduce new effdim of that similarity by 1
                                    # for numerical stability retain minimum of 2
                                    if clustering[c,s1]>0 and clustering[c,s2]>0:
                                        neweffdim[s1,s2] = np.maximum(2, neweffdim[s1,s2]-1)
                                        neweffdim[s2,s1] = np.maximum(2, neweffdim[s2,s1]-1)
                        # Update effdim
                        effdim = neweffdim
            """
            COMPUTE P-VALUES USING BETA DISTRIBUTION
            """
            # Compute p-values based on cdf of beta distribution
            # We raise the p-values to pcadim-th power because
            # the simitensor contains the maximal similarities for each component
            pvalues = np.ones(simitensor.shape)
            for subj1 in range(subjects):
                for subj2 in range(subjects):
                    if not complexvalued:
                        # Basic formula for real-valued data
                        pvalues[subj1, subj1, :] = 1 - np.square(betainc(0.5, (effdim[subj1, subj2]-1)/2, simitensor[subj1, subj2:]))

            """
            DEFLATION: REMOVE THOSE CONNECTIONS WHICH SHOULD NOT BE USED
            """
            # First create matrix which contains all the p-values,
            # even those deflated away (needed for FDR computation by Simes)
            pvalues_nodefl = pvalues
            # And now do deflation: Set to one those pvalues which have been deflated away
            pvalues = np.maximum(pvalues_nodefl, deflationtensor)

            """
            COMPUTE THRESHOLDS FOR P-VALUES CORRECTED FOR MULTIPLE TESTING
            """

            if corrected: # if the user input corrected values, just use them
                alphacorr_FP = alphaFP
                alphacorr_FD = alphaFD

            else: # otherwise do correction here
                # Compute Bonferroni-corrected FPR threshold (forming a new cluster)
                alphacorr_FP = alphaFP / (subjects*(subjects-1)*pcadim/2)

                # Compute corrected FDR threshold
                if simes: # Simes procedure (well-known FDR method)
                    # Note: here we use all p-values even those deflated away
                    # (ignoring only duplicates due to symmetry by using uppertriangindices)
                    pvalues_FDR = np.sort(pvalues_nodefl[uppertriangindices(pcadim, subjects)])
                    alphacorr_FD_idx = np.where(pvalues_FDR <= \
                                                (np.arange(pvalues_FDR.shape[1])+1)/pvalues_FDR.shape[1]*alphaFD)
                    alphacorr_FD = pvalues_FDR[alphacorr_FD_idx[0, -1], alphacorr_FD_idx[1, -1]]
                    if alphacorr_FD.size == 0: alphacorr_FD = 0
                else: # Simple alternative to simes procedure, in 2013 paper
                    alphacorr_FD=alphaFD/(subjects-2)

            # print some simple diagnostics on screen, if extraverbose
            if verbose == 2 :
                if not empirical:
                    print("  Average estimated effective dimension: {%6.2f}".format(np.mean(np.mean(effdim, axis=0))))
                    print("  Corrected alpha value for FDR: {}".format(alphacorr_FD))
                    print("  Corrected alpha value for FPR: {}".format(alphacorr_FP))
                FDpvalues = np.sum(pvalues < alphacorr_FD)
                FPpvalues = np.sum(pvalues < alphacorr_FP)
                print("  Number of similarities left above FDR threshold: {} {%2.6f} ".format(FDpvalues, FDpvalues/subjects/(subjects-1)/pcadim^2*2*100))
                print("  Number of similarities left above FPR threshold: {} {%2.6f} ".format(FPpvalues, FPpvalues/subjects/(subjects-1)/pcadim^2*2*100))

            """
            Main clustering Part(for creating one cluster)
            """
            # FIND INITIAL PAIR OF COMPONENTS TO FORM A CLUSTER

            # Find maximum single correlation to start clustering with
            pval = np.min(pvalues)
            minind = np.argmin(pvalues)

            # Abort if max similarity does not reach threshold given by FPR
            # (Or if, exceptionally, pval is equal to one which means zero similarity)
            if pval > alphacorr_FP or pval == 1 :
                nomoreclusters = 1

            else:
                # we have cluster, continue adding more components to it
                # First find subject and component number of the minimizing pvalue
                subjind1, subjind2, compind1 = np.unravel_index(minind, pvalues.shape)
                compind2 = maxtensor[subjind1, subjind2, compind1]
                # add these initial two to lists of components in this cluster.
                # a) create list of subjects in cluster
                clustersubjs=[subjind1,subjind2]
                # b) create list of components in cluster
                clustercomps=[compind1,compind2]
                # c) create list of linking pvalues
                clusterpvalues=[pval,pval]
                # d) create list of linking similarites
                clustersimis=[1,1]*simitensor[subjind1,subjind2,compind1]
                clustersimis = [ele*simitensor[subjind1,subjind2,compind1] for ele in clustersimis]

                # ADD NEW COMPONENTS TO CURRENT CLUSTER

                # Loop to find all components sufficiently connected to those already found
                if subjects > 2:
                    nomoreclusters = 0 # boolean which tells if all components have been found
                else :
                    nomoreclusters = 1 # if there are just two subjects in the data, no need to even start this adding of new components

                while not nomoreclusters:
                    # Create matrix with similarities of components in remaining subject
                    # to those subjects in present cluster
                    # First, get indices of subjects not in cluster
                    remainsubjs = np.ones(subjects)
                    remainsubjs[clustersubjs] = np.zeros(len(clustersubjs))
                    remainsubjsinds = np.argwhere(remainsubjs)

                    pvalues_selected = np.zeros(len(clustersubjs), np.sum(remainsubjs))
                    maxtensor_selected = np.zeros(len(clustersubjs), np.sum(remainsubjs))

                    # For each component in cluster, create row with similarities
                    # to all the candidate components (each column is one subject) which could be added
                    for i in range(len(clustersubjs)):
                        pvalues_selected[i] = pvalues[remainsubjsinds, clustersubjs[i], clustercomps[i]]
                        maxtensor_selected[i] = maxtensor[remainsubjsinds, clustersubjs[i], clustercomps[i]]

                    # USE DIFFERENT LINKAGE STRATEGIES TO FIND NEW CANDIDATE
                    # Get aggregate pvalues from the cluster to candidate components
                    if strategy == "SL": # single-linkage
                        outpvals = np.min(pvalues_selected, axis=0)
                        outmininds = np.argmin(pvalues_selected , axis=0)

                    elif strategy == "CL": # complete-linkage
                        # This is a bit tricky because we have to first determine
                        # if the links to the same subject are to the same component
                        samesimilarity = np.all(maxtensor_selected == np.ones((len(maxtensor_selected), 1))@maxtensor_selected[0].reshape(1,-1))
                        # Set p-value to ridiculously large if the links not the same subject
                        pvalues_selected_qualified = pvalues_selected + 100*(1-np.ones((pvalues_selected.shape[0],1))@samesimilarity.reshape(-1,1))
                        # And finally find the max p-values, i.e. weakest links from the cluster
                        outpvals = np.max(pvalues_selected_qualified, axis=0)
                        outmininds = np.argmax(pvalues_selected_qualified, axis=0)

                    elif strategy == "ML": # median-linkage
                        # First make check like in CL
                        samesimilarity = np.all(maxtensor_selected == np.ones((len(maxtensor_selected), 1))@maxtensor_selected[0].reshape(1,-1))
                        pvalues_selected_qualified = pvalues_selected + 100*(1-np.ones((pvalues_selected.shape[0],1))@samesimilarity.reshape(-1,1))
                        # Sort each column
                        sortvals = np.sort(pvalues_selected_qualified, axis=0)
                        sortinds = np.argsort(pvalues_selected_qualified, axis=0)
                        # Compute indes of median. (If you want to be conservative, add 1 to the size here.)
                        medianrow = np.floor((pvalues_selected_qualified.shape[0]+1)/2)
                        outpvals = sortvals[medianrow]
                        outmininds = sortinds[medianrow]

                    # For any strategy, finally find minimizing pvalue, i.e. best component to add among the significant/allowed connections
                    pval = np.min(outpvals)
                    minind = np.argmin(outpvals)

                    # See if similarity fails to pass the FDR threshold
                    if pval > alphacorr_FD:
                        # If so, abort since cannot find any sufficiently connected components
                        nomoreclusters = 1
                    else:
                        # If sufficiently similar, add to cluster
                        # Find subject and component number of the minimizing pvalue
                        # First, subject/component inside cluster
                        # from which the minimizing link goes out
                        oldsubj = clustersubjs[outmininds[minind]]
                        oldcomp = clustercomps[outmininds[minind]]
                        # Second, subject/component outside cluster which is to be added
                        # (convert subject index according to ordering in remainsubjinds)
                        newsubj=remainsubjsinds[minind]
                        # (fetch component number from maxtensor)
                        newcomp=maxtensor[newsubj, oldsubj, oldcomp]

                        # Then add to current cluster
                        clustersubjs = [clustersubjs, newsubj]
                        clustercomps = [clustercomps, newcomp]

                        # Store also linking p-value and similarities
                        clusterpvalues = [clusterpvalues, pval]
                        clustersimis = [clustersimis, simitensor[newsubj, oldsubj, oldcomp]]

                        # if cluster size maximum, abort
                        if clustersubjs.shape[1] == subjects: nomorecomponents=1

            # At this point, a new cluster has been processed

            # If the cluster was not empty, store cluster in matrix and continue
            if not nomoreclusters:
                clustering[clusterindex] = np.zeros(subjects)
                clusterorder[clusterindex] = np.zeros(subjects)
                linkpvalues[clusterindex] = np.zeros(subjects)
                linksimilarities[clusterindex] = np.zeros(subjects)

                clustering[clusterindex, clustersubjs] = clustercomps
                clusterorder[clusterindex, clustersubjs] = np.arange(clustersubjs.shape[1])
                linkpvalues[clusterindex, clustersubjs] = clusterpvalues
                linksimilarities[clusterindex, clustersubjs] = clustersimis

                # Store also information on which vectors should be deflated, i.e. ignored
                for i in range(clustersubjs.shape[1]):
                    # Easy to do deflation in one direction
                    deflationtensor[:, clustersubjs[i], clustercomps[i]] = np.ones((subjects,1))
                    # But in the other direction more difficult
                    tmpmatrix = maxtensor[clustersubjs[i]]
                    incomingindex = np.argwhere[tmpmatrix==clustercomps[i]]
                    [comp, subj] = np.unravel_index(incomingindex, [pcadim, subjects])
                    for j in range(comp):
                        deflationtensor[clustersubjs[i],subj(j), comp[j]] = 1

                # increment cluster number (iteration counter)
                clusterindex = clusterindex + 1

        # (At this point, all clusters have been found in the current iteration)
        if verbose:
            print("Done.")
            if iteration == 1 and iteration == 2:
                print("Number of clusters found in initial iteration: {}".format(clustering.shape[0]))
                print("Number of vectors clustered in initial iteration: {} ({2.2f} of all vectors)".format(np.sum(clustering>0), np.sum(clustering>0)/subjects/pcadim*100))

    # END OF LOOP FOR ITERATING CLUSTERING TWO TIMES when using analytical method:

    # (At this point, iterations have been finished and final clustering found)
    # The rest is just outputting stuff on the terminal:

    if verbose:
        print("*** Clustering computation finished ***")
        print("Number of clusters found: {}".format(len(clustering)))
        compsclustered = np.sum(np.array(clustering)>0)
        print("Number of vectors clustered: {} ({2.2f} of all vectors)".format(compsclustered, compsclustered/subjects/pcadim*100))
        print("Average number of vectors per cluster: {3.2f}".format(compsclustered/clustering.shape[0]))
        print("Internal parameters: ")
        print(" Average estimated effective dimension: {6.2f}".format(np.mean(effdim)))
        print(" Corrected alpha value for FDR: {}".format(alphacorr_FD))
        print(" Smallest similarity considered significant by FDR: {0.4f}".format(np.min(linksimilarities[np.argwhere(linksimilarities)])))
        print(" Corrected alpha value for FPR: {}".format(alphacorr_FP))
        print(" Smallest similarity considered significant by FPR: {0.4f}".format(np.min(np.max(linkpvalues))))
        print(" Exiting testing algorithm succesfully.")

    return clustering, clusterorder, linkpvalues, linksimilarities, simitensor, maxtensor, pvalues

