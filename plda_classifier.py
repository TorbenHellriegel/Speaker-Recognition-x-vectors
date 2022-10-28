import pickle

import numpy as np
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
