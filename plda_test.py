import random

import numpy as np

from main import split_en_te
from plda_classifier import *

# Extract the x-vectors
print('extracting x-vectors')
x_vec_train = np.array([[random.randrange(i*100,(i+1)*100,1)/10000 for j in range(15)] for i in range(0, 50)])
x_label_train = np.array([int(i/10) for i in range(0, 50)])
x_vec_test = np.array([[random.randrange(i*100,(i+1)*100,1)/10000 for j in range(15)] for i in range(50, 100)])
x_label_test = np.array([int(i/10) for i in range(50, 100)])

# Split testing data into enroll and test data
print('splitting testing data into enroll and test data')
en_xv, en_label, te_xv, te_label = split_en_te(x_vec_test, x_label_test)

# Generate x_vec stat objects
print('generating x_vec stat objects')
xvectors_stat = get_train_x_vec(x_vec_train, x_label_train)
en_sets, en_stat = get_enroll_x_vec(en_xv)
te_sets, te_stat = get_test_x_vec(te_xv)

# PLDA train test loop
num_train_iter = 5
plda = setup_plda(rank_f=10, nb_iter=1) #load old plda here
for i in range(num_train_iter):
    # Training plda
    print('training plda')#TODO check if the iter goeas over all iter every time
    plda = setup_plda(mean=plda.mean, F=plda.F, Sigma=plda.Sigma, rank_f=plda.rank_f, nb_iter=plda.nb_iter+1, scaling_factor=plda.scaling_factor)
    plda = train_plda(plda, xvectors_stat)

    # Testing plda
    print('testing plda')
    scores_plda = test_plda(plda, en_sets, en_stat, te_sets, te_stat)

    mask = np.array(np.diag(np.diag(np.ones(scores_plda.scoremat.shape, dtype=np.int32))), dtype=bool)
    scores = scores_plda.scoremat[mask]
    print('scores', scores)
    print('mean scores', np.mean(scores))

print('DONE')
