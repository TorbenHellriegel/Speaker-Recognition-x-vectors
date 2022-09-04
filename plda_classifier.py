import pickle

import numpy as np
import sklearn
import torch
from sklearn.model_selection import StratifiedKFold
from speechbrain.processing.PLDA_LDA import *

def split_en_te(x_vec_test, x_label_test, mean_same_speaker=False): #TODO outdated delete later
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    enroll_index, test_index = [], []
    for eni, tei in skf.split(x_vec_test, x_label_test):
        enroll_index = eni
        test_index = tei
        
    enroll_xv = x_vec_test[enroll_index]
    enroll_label = x_label_test[enroll_index]
    test_xv = x_vec_test[test_index]
    test_label = x_label_test[test_index]

    match_index = []
    for i, (el, tl) in enumerate(zip(enroll_label, test_label)):
        if(el==tl):
            match_index.append(i)
        
    enroll_xv = enroll_xv[match_index]
    enroll_label = enroll_label[match_index]
    test_xv = test_xv[match_index]
    test_label = test_label[match_index]
    
    if(mean_same_speaker):
        en_xv, en_label = mean_same_speakers(enroll_xv, enroll_label)
        te_xv, te_label = mean_same_speakers(test_xv, test_label)
    else:
        en_xv = enroll_xv
        en_label = enroll_label
        te_xv = test_xv
        te_label = test_label

    return en_xv, en_label, te_xv, te_label

def split_en_te2(x_vec_test, x_label_test):
    en_xv = []
    en_label = []
    te_xv = []
    te_label = []

    #x_vec_test, x_label_test = sklearn.utils.shuffle(x_vec_test, x_label_test)###TODO comment out and in to compare different results in plda_test
    en_xv = x_vec_test
    en_label = x_label_test
    #x_vec_test, x_label_test = sklearn.utils.shuffle(x_vec_test, x_label_test)###TODO comment out and in to compare different results in plda_test
    te_xv = x_vec_test
    te_label = x_label_test

    return en_xv, en_label, te_xv, te_label

def mean_same_speakers(test_xv, test_label):
    te_xv = []
    te_label = []
    unique_label = np.unique(test_label)
    for label in unique_label:
        xv = []
        for x, l in zip(test_xv, test_label):
            if label == l:
                xv.append(x)
        te_xv.append(np.mean(xv, axis=0))
        te_label.append(label)
    te_xv = np.array(te_xv, dtype=np.float64)
    te_label = np.array(te_label, dtype=np.int32)
    return te_xv, te_label

def get_train_x_vec(train_xv, train_label, x_id_train):
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

def get_enroll_x_vec(en_xv, en_id):
    # Get number of train_utterances and their dimension
    en_N = en_xv.shape[0]
    print('N enroll utt:', en_N)

    # Define arrays neccessary for special stat object
    en_sgs = [str(en_id[i]) for i in range(en_N)]
    en_sets = np.array(en_sgs, dtype="|O")
    en_s = np.array([None] * en_N)
    en_stat0 = np.array([[1.0]]* en_N)

    # Define special stat object
    en_stat = StatObject_SB(modelset=en_sets, segset=en_sets, start=en_s, stop=en_s, stat0=en_stat0, stat1=en_xv)
    return en_stat

def get_test_x_vec(te_xv, te_id):
    # Get number of train_utterances and their dimension
    te_N = te_xv.shape[0]
    print('N test utt:', te_N)

    # Define arrays neccessary for special stat object
    te_sgs = [str(te_id[i]) for i in range(te_N)]
    te_sets = np.array(te_sgs, dtype="|O")
    te_s = np.array([None] * te_N)
    te_stat0 = np.array([[1.0]]* te_N)

    # Define special stat object
    te_stat = StatObject_SB(modelset=te_sets, segset=te_sets, start=te_s, stop=te_s, stat0=te_stat0, stat1=te_xv)
    return te_stat

def test_plda(plda, en_stat, te_stat):
    # Define special object for plda scoring
    ndx = Ndx(models=en_stat.modelset, testsegs=te_stat.modelset)

    # PLDA Scoring
    scores_plda = fast_PLDA_scoring(en_stat, te_stat, ndx, plda.mean, plda.F, plda.Sigma, p_known=0.0)
    return scores_plda

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
    code taken from : https://github.com/Hemanshu-Bhargav/august_speechbrain/blob/2933b2a5a83662e9c554ba15e94f4a9ad31527bc/speechbrain/utils/metric_stats.py#L455
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

# TODO def minDCF

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
