import numpy as np
from sklearn.model_selection import StratifiedKFold
from speechbrain.processing.PLDA_LDA import *


def split_en_te(x_vec_test, x_label_test, mean_same_speaker=False):
    skf = StratifiedKFold(n_splits=2)
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
        
    enroll_xv = x_vec_test[match_index]
    enroll_label = x_label_test[match_index]
    test_xv = x_vec_test[match_index]
    test_label = x_label_test[match_index]
    
    if(mean_same_speaker):
        en_xv, en_label = mean_same_speakers(enroll_xv, enroll_label)
        te_xv, te_label = mean_same_speakers(test_xv, test_label)
    else:
        en_xv = enroll_xv
        en_label = enroll_label
        te_xv = test_xv
        te_label = test_label

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

def get_train_x_vec(train_xv, train_label):
    # Get number of train_utterances and their dimension
    N = train_xv.shape[0]
    print('N train utt:', N)

    # Define arrays neccessary for special stat object
    md = ['md'+str(train_label[i]) for i in range(N)]
    modelset = np.array(md, dtype="|O")
    sg = ['sg'+str(i) for i in range(N)]
    segset = np.array(sg, dtype="|O")
    s = np.array([None] * N)
    stat0 = np.array([[1.0]]* N)

    # Define special stat object
    xvectors_stat = StatObject_SB(modelset=modelset, segset=segset, start=s, stop=s, stat0=stat0, stat1=train_xv)
    return xvectors_stat

def setup_plda(mean=None, F=None, Sigma=None, rank_f=150, nb_iter=1, scaling_factor=1):
    plda = PLDA(mean=mean, F=F, Sigma=Sigma, rank_f=rank_f, nb_iter=nb_iter, scaling_factor=scaling_factor)
    return plda

def train_plda(plda, xvectors_stat):
    # Training PLDA model: M ~ (mean, F, Sigma)
    plda.plda(xvectors_stat)
    return plda

def get_enroll_x_vec(en_xv):
    # Get number of train_utterances and their dimension
    en_N = en_xv.shape[0]
    print('N enroll utt:', en_N)

    # Define arrays neccessary for special stat object
    en_sgs = ['en'+str(i) for i in range(en_N)]
    en_sets = np.array(en_sgs, dtype="|O")
    en_s = np.array([None] * en_N)
    en_stat0 = np.array([[1.0]]* en_N)

    # Define special stat object
    en_stat = StatObject_SB(modelset=en_sets, segset=en_sets, start=en_s, stop=en_s, stat0=en_stat0, stat1=en_xv)
    return en_sets, en_stat

def get_test_x_vec(te_xv):
    # Get number of train_utterances and their dimension
    te_N = te_xv.shape[0]
    print('N test utt:', te_N)

    # Define arrays neccessary for special stat object
    te_sgs = ['te'+str(i) for i in range(te_N)]
    te_sets = np.array(te_sgs, dtype="|O")
    te_s = np.array([None] * te_N)
    te_stat0 = np.array([[1.0]]* te_N)

    # Define special stat object
    te_stat = StatObject_SB(modelset=te_sets, segset=te_sets, start=te_s, stop=te_s, stat0=te_stat0, stat1=te_xv)
    return te_sets, te_stat

def test_plda(plda, en_sets, en_stat, te_sets, te_stat):
    # Define special object for plda scoring
    ndx = Ndx(models=en_sets, testsegs=te_sets)

    # PLDA Scoring
    scores_plda = fast_PLDA_scoring(en_stat, te_stat, ndx, plda.mean, plda.F, plda.Sigma, p_known=0.0)
    return scores_plda
