import random

import numpy
from speechbrain.processing.PLDA_LDA import *

def get_train_x_vec():
    # Define number of train_utterances
    N = 100

    # Randomly generate train x_vectors
    train_xv = numpy.random.rand(N, dim)

    # Define arrays neccessary for special stat object
    md = ['md'+str(random.randrange(1,n_spkrs,1)) for i in range(N)]
    modelset = numpy.array(md, dtype="|O")
    sg = ['sg'+str(i) for i in range(N)]
    segset = numpy.array(sg, dtype="|O")
    s = numpy.array([None] * N)
    stat0 = numpy.array([[1.0]]* N)

    # Define special stat object
    xvectors_stat = StatObject_SB(modelset=modelset, segset=segset, start=s, stop=s, stat0=stat0, stat1=train_xv)
    return xvectors_stat

def train_plda_on_x_vec(xvectors_stat):
    # Training PLDA model: M ~ (mean, F, Sigma)
    plda = PLDA(rank_f=5)
    plda.plda(xvectors_stat)
    if(print_array): print ('plda.mean', plda.mean)
    if(print_shape): print ('plda.mean.shape', plda.mean.shape)
    if(print_array): print ('plda.F', plda.F)
    if(print_shape): print ('plda.F.shape', plda.F.shape)
    if(print_array): print ('plda.Sigma', plda.Sigma)
    if(print_shape): print ('plda.Sigma.shape', plda.Sigma.shape)
    return plda

def get_enroll_x_vec():
    # Define number of enrollment_utterances
    en_N = 20

    # Randomly generate enroll x_vectors
    en_xv = numpy.random.rand(en_N, dim)

    # Define arrays neccessary for special stat object
    en_sgs = ['en'+str(i) for i in range(en_N)]
    en_sets = numpy.array(en_sgs, dtype="|O")
    en_s = numpy.array([None] * en_N)
    en_stat0 = numpy.array([[1.0]]* en_N)

    # Define special stat object
    en_stat = StatObject_SB(modelset=en_sets, segset=en_sets, start=en_s, stop=en_s, stat0=en_stat0, stat1=en_xv)
    return en_sets, en_stat

def get_test_x_vec():
    # Randomly generate test x_vectors
    te_N = 30

    # Define arrays neccessary for special stat object
    te_xv = numpy.random.rand(te_N, dim)
    te_sgs = ['te'+str(i) for i in range(te_N)]
    te_sets = numpy.array(te_sgs, dtype="|O")
    te_s = numpy.array([None] * te_N)
    te_stat0 = numpy.array([[1.0]]* te_N)

    # Define special stat object
    te_stat = StatObject_SB(modelset=te_sets, segset=te_sets, start=te_s, stop=te_s, stat0=te_stat0, stat1=te_xv)
    return te_sets, te_stat

def test_plda(en_sets, en_stat, te_sets, te_stat):
    # Define special object for plda scoring
    ndx = Ndx(models=en_sets, testsegs=te_sets)

    # PLDA Scoring
    scores_plda = fast_PLDA_scoring(en_stat, te_stat, ndx, plda.mean, plda.F, plda.Sigma)
    if(print_array): print ('scores_plda.scoremat', scores_plda.scoremat)
    if(print_shape): print ('scores_plda.scoremat.shape', scores_plda.scoremat.shape)

if __name__ == "__main__":
    # Define x_vec_dimensions, num_of_speakers
    print_array = False
    print_shape = True
    dim= 10
    n_spkrs = 10
    xvectors_stat = get_train_x_vec()
    plda = train_plda_on_x_vec(xvectors_stat)
    en_sets, en_stat = get_enroll_x_vec()
    te_sets, te_stat = get_test_x_vec()
    test_plda(en_sets, en_stat, te_sets, te_stat)
    print('DONE')