import numpy as np
import pandas as pd


# IMPORTANT: this will use multithreading if possible
multithread = False
if multithread is True:
    from joblib import Parallel, delayed
    import multiprocessing

def best_thresh_on_feat_old(values,labels,weights=None):
    '''
    This is the older version of this function. I used pandas to do the sorting,
    but later realized it was 10x slower as a result. The new version much better
    (but harder to read)
    '''
    if weights is None: weights = np.ones(values.shape)/len(values) 
    df = pd.DataFrame({'values':values,'labels':labels,'weights':weights}).sort_values('values')
    Tplus, Tminus = (df['labels'] * df.weights).sum(), ((1-df['labels'])*df.weights).sum()
    df['Splus'] = (df['labels'] * df.weights).cumsum()
    df['Sminus'] = (df.weights * (1-df['labels'])).cumsum()
    df['leftperf'] = df.Splus + Tminus - df.Sminus
    df['rightperf'] = df.Sminus + Tplus - df.Splus
    leftmin, rightmin = df.leftperf.min(), df.rightperf.min()
    if leftmin < rightmin:
        thresh, sgn, perf = df.loc[df.leftperf.idxmin(),'values'], +1, leftmin
    else: 
        thresh, sgn, perf = df.loc[df.rightperf.idxmin(),'values'], -1, rightmin
    return thresh, sgn, perf


def best_thresh_on_feat(values,labels,weights=None):
    """finds the best threshold of a list of feature values to optimize 0/1 loss

    Parameters
    ----------
    values: the feature values as a numpy array
    labels: the 0/1 labels for these feature values
    weights: if we are in a weighted setting, includes weights on examples

    Returns
    -------
    thresh: the best threshold to use
    sgn: the sign of the optimal threshold (i.e. predict 1 when value > thresh or value < thresh)
    perf: the performance of this threshold indicator
    """
    if weights is None:
        weights = np.ones(values.shape)/len(values)
    temp = np.zeros((values.shape[0],7))
    temp[:,0], temp[:,1], temp[:,6] = values, labels, weights
    newind = np.argsort(temp[:,0])
    temp = temp[newind]
    temp[:,2] = (temp[:,1] * temp[:,6]).cumsum()
    temp[:,3] = ((1-temp[:,1]) * temp[:,6]).cumsum()
    Tplus, Tminus = temp[-1,2], temp[-1,3]
    temp[:,4] = temp[:,2] - temp[:,3] + Tminus
    temp[:,5] = temp[:,3] - temp[:,2] + Tplus
    argmin_l, argmin_r = np.argmin(temp[:,4]), np.argmin(temp[:,5])
    if temp[argmin_l,4] < temp[argmin_r,5]:
        thresh, sgn, perf = temp[argmin_l,0], +1, temp[argmin_l,4]
    else:
        thresh, sgn, perf = temp[argmin_r,0], -1, temp[argmin_r,5]
    return thresh, sgn, perf



def get_best_in_range(rng,featmtx,labels,weights):
    """This calculates the best threshold for a range fo the features in a matrix
    This method was designed primarily for the parallelization, to chop up the full matrix
    into sets of columns to do on different cores. 

    Parameters
    ----------
    rng: a subset of the column indices
    featmtx: each column is a possible decision stump, and we will search for a threshold
    labels: the labels for the rows (examples)
    weights: the weights for each row

    Returns
    -------
    bestparams: a tuple of three values, as returned by best_thresh_on_feat()
    bestperf: the actual performance of this threshold
    """
    bestperf, bestparams = np.inf, None
    for featind in rng:
        feats = featmtx[:,featind]
        thresh, sgn, perf = best_thresh_on_feat(feats,labels,weights)
        if perf < bestperf:
            bestperf, bestparams = perf, (featind, thresh, sgn)
        # if (featind % 100) == 99: print("Completed %.2f of search job" % (featind/featmtx.shape[1]))
    return bestparams, bestperf

def get_best_threshold_func(featmtx,labels,weights=None):
    """Here were multithread the search for the best threshold on a matrix of feature values.
    Uses joblib to parallelize the search. Chops up the columns of featmtx and distributes.

    Parameters
    ----------
    featmtx: feature matrix to search for threshold functions
    labels: labels for the rows (examples)
    weights: weights for the rows, if needed

    Returns
    -------
    bestparams: a tuple of three values, as returned by best_thresh_on_feat()
    bestperf: the actual performance of this threshold
    """
    if multithread is True:
        num_cores = multiprocessing.cpu_count()
        rnglen = int(featmtx.shape[1]//num_cores)
        ranges = [range(ind*rnglen, (ind+1)*rnglen) for ind in range(num_cores)]
        print(ranges)
        results = Parallel(n_jobs=num_cores)(delayed(get_best_in_range)(rng,featmtx,labels,weights) for rng in ranges)
        bestparams, bestperf = None, np.inf
        for params, perf in results:
            if perf < bestperf:
                bestperf, bestparams = perf, params
    else:
        bestparams, bestperf = get_best_in_range(range(featmtx.shape[1]),featmtx,labels,weights)
    print(bestparams, bestperf)
    return bestparams, bestperf
