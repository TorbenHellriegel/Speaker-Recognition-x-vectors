import pickle

import numpy as np
import torch
from speechbrain.processing.PLDA_LDA import *


def get_train_x_vec(train_xv, train_label, x_id_train):
    """
    Generate a stat object for the training x-vectors.

    Parameters
    ----------
    train_xv: ndarray
        The x-vector
        
    train_label: int
        The x-vectors label
        
    x_id_train: string
        The x-vectors unique id

    Returns
    -------
    xvectors_stat: obj
        The x-vector stat object
    """
    # Get number of train_utterances and their dimension
    N = train_xv.shape[0]
    print('N train utt:', N)

    # Define arrays neccessary for special stat object
    md = ['id'+str(train_label[i]) for i in range(N)]
    modelset = np.array(md, dtype="|O")
    sg = [str(x_id_train[i]) for i in range(N)]
    segset = np.array(sg, dtype="|O")
    s = np.array([None] * N)
    stat0 = np.array([[1.0]]* N)

    # Define special stat object
    xvectors_stat = StatObject_SB(modelset=modelset, segset=segset, start=s, stop=s, stat0=stat0, stat1=train_xv)
    return xvectors_stat

def setup_plda(mean=None, F=None, Sigma=None, rank_f=150, nb_iter=10, scaling_factor=1):
    plda = PLDA(mean=mean, F=F, Sigma=Sigma, rank_f=rank_f, nb_iter=nb_iter, scaling_factor=scaling_factor)
    return plda

def train_plda(plda, xvectors_stat):
    plda.plda(xvectors_stat)
    return plda

def get_x_vec_stat(xv, id):
    """
    Generate a stat object for the x-vectors.

    Parameters
    ----------
    xv: ndarray
        The x-vector
        
    id: int
        The x-vectors unique id

    Returns
    -------
    xv_stat: obj
        The x-vector stat object
    """
    # Get number of utterances and their dimension
    N = xv.shape[0]

    # Define arrays neccessary for special stat object
    sgs = [str(id[i]) for i in range(N)]
    sets = np.array(sgs, dtype="|O")
    s = np.array([None] * N)
    stat0 = np.array([[1.0]]* N)

    # Define special stat object
    xv_stat = StatObject_SB(modelset=sets, segset=sets, start=s, stop=s, stat0=stat0, stat1=xv)
    return xv_stat

def plda_scores(plda, en_stat, te_stat):
    # Define special object for plda scoring
    ndx = Ndx(models=en_stat.modelset, testsegs=te_stat.modelset)

    # PLDA Scoring
    fast_plda_scores = fast_PLDA_scoring(en_stat, te_stat, ndx, plda.mean, plda.F, plda.Sigma, p_known=0.0)
    return fast_plda_scores

def EER(positive_scores, negative_scores):
    """Computes the EER (and its threshold).
    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.
    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_eer, threshold = EER(positive_scores, negative_scores)
    >>> val_eer
    0.0
    Credit
    ------
    Code taken from : https://github.com/Hemanshu-Bhargav/august_speechbrain/blob/2933b2a5a83662e9c554ba15e94f4a9ad31527bc/speechbrain/utils/metric_stats.py#L455
    """

    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    # Computing False Rejection Rate (miss detection)
    positive_scores = torch.cat(
        len(thresholds) * [positive_scores.unsqueeze(0)]
    )
    pos_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    FRR = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[1]
    del positive_scores
    del pos_scores_threshold

    # Computing False Acceptance Rate (false alarm)
    negative_scores = torch.cat(
        len(thresholds) * [negative_scores.unsqueeze(0)]
    )
    neg_scores_threshold = negative_scores.transpose(0, 1) > thresholds
    FAR = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[1]
    del negative_scores
    del neg_scores_threshold

    # Finding the threshold for EER
    min_index = (FAR - FRR).abs().argmin()

    # It is possible that eer != fpr != fnr. We return (FAR  + FRR) / 2 as EER.
    EER = (FAR[min_index] + FRR[min_index]) / 2

    return float(EER), float(thresholds[min_index])

def minDCF(
    positive_scores, negative_scores, c_miss=1.0, c_fa=1.0, p_target=0.5
):
    """Computes the minDCF metric normally used to evaluate speaker verification
    systems. The min_DCF is the minimum of the following C_det function computed
    within the defined threshold range:
    C_det =  c_miss * p_miss * p_target + c_fa * p_fa * (1 -p_target)
    where p_miss is the missing probability and p_fa is the probability of having
    a false alarm.
    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.
    c_miss : float
         Cost assigned to a missing error (default 1.0).
    c_fa : float
        Cost assigned to a false alarm (default 1.0).
    p_target: float
        Prior probability of having a target (default 0.01).
    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_minDCF, threshold = minDCF(positive_scores, negative_scores)
    >>> val_minDCF
    0.0
    Credit
    ------
    code taken from : https://github.com/Hemanshu-Bhargav/august_speechbrain/blob/2933b2a5a83662e9c554ba15e94f4a9ad31527bc/speechbrain/utils/metric_stats.py#L509
    """

    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    # Computing False Rejection Rate (miss detection)
    positive_scores = torch.cat(
        len(thresholds) * [positive_scores.unsqueeze(0)]
    )
    pos_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    p_miss = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[1]
    del positive_scores
    del pos_scores_threshold

    # Computing False Acceptance Rate (false alarm)
    negative_scores = torch.cat(
        len(thresholds) * [negative_scores.unsqueeze(0)]
    )
    neg_scores_threshold = negative_scores.transpose(0, 1) > thresholds
    p_fa = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[1]
    del negative_scores
    del neg_scores_threshold

    c_det = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
    c_min, min_index = torch.min(c_det, dim=0)

    return float(c_min), float(thresholds[min_index])

def save_plda(plda, file_name):
    try:
        with open('plda/'+file_name+'.pickle', 'wb') as f:
            pickle.dump(plda, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print('Error during pickling plda: ', ex)

def load_plda(file_path_name):
    try:
        with open(file_path_name, 'rb') as f:
            return pickle.load(f)
    except Exception as ex:
        print('Error during pickling plda: ', ex)

def lda(x_vec_stat, reduced_dim=2):
    lda = LDA()
    new_train_obj = lda.do_lda(x_vec_stat, reduced_dim=reduced_dim)
    return new_train_obj
