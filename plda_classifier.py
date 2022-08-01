import numpy
from speechbrain.processing.PLDA_LDA import *


def get_train_x_vec(train_xv, train_label):
    # Get number of train_utterances and their dimension
    N = train_xv.shape[0]
    print('N train utt:', N)

    # Define arrays neccessary for special stat object
    md = ['md'+str(train_label[i]) for i in range(N)]
    modelset = numpy.array(md, dtype="|O")
    sg = ['sg'+str(i) for i in range(N)]
    segset = numpy.array(sg, dtype="|O")
    s = numpy.array([None] * N)
    stat0 = numpy.array([[1.0]]* N)

    # Define special stat object
    xvectors_stat = StatObject_SB(modelset=modelset, segset=segset, start=s, stop=s, stat0=stat0, stat1=train_xv)
    return xvectors_stat

def train_plda_on_x_vec(xvectors_stat, rank_f=5):
    # Training PLDA model: M ~ (mean, F, Sigma)
    plda = PLDA(rank_f=rank_f)
    plda.plda(xvectors_stat, output_file_name='plda/plda_model')
    return plda

def get_enroll_x_vec(en_xv, en_label):
    # Get number of train_utterances and their dimension
    en_N = en_xv.shape[0]
    print('N enroll utt:', en_N)

    # Define arrays neccessary for special stat object
    en_sgs = ['en'+str(en_label[i]) for i in range(en_N)] #TODO might need to du 2 seperate segsets
    en_sets = numpy.array(en_sgs, dtype="|O")
    en_s = numpy.array([None] * en_N)
    en_stat0 = numpy.array([[1.0]]* en_N)

    # Define special stat object
    en_stat = StatObject_SB(modelset=en_sets, segset=en_sets, start=en_s, stop=en_s, stat0=en_stat0, stat1=en_xv)
    return en_sets, en_stat

def get_test_x_vec(te_xv, te_label):
    # Get number of train_utterances and their dimension
    te_N = te_xv.shape[0]
    print('N test utt:', te_N)

    # Define arrays neccessary for special stat object
    te_sgs = ['te'+str(te_label[i]) for i in range(te_N)]
    te_sets = numpy.array(te_sgs, dtype="|O")
    te_s = numpy.array([None] * te_N)
    te_stat0 = numpy.array([[1.0]]* te_N)

    # Define special stat object
    te_stat = StatObject_SB(modelset=te_sets, segset=te_sets, start=te_s, stop=te_s, stat0=te_stat0, stat1=te_xv)
    return te_sets, te_stat

def test_plda(plda, en_sets, en_stat, te_sets, te_stat):
    # Define special object for plda scoring
    ndx = Ndx(models=en_sets, testsegs=te_sets)

    # PLDA Scoring
    scores_plda = fast_PLDA_scoring(en_stat, te_stat, ndx, plda.mean, plda.F, plda.Sigma)
    return scores_plda
