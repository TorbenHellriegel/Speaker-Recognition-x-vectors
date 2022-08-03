import random

import numpy as np

import plda_classifier as pc

# Extract the x-vectors
print('extracting x-vectors')
x_vec_train = np.array([[random.randrange(i*100,(i+1)*100,1)/10000 for j in range(15)] for i in range(0, 50)])
x_label_train = np.array([int(i/10) for i in range(0, 50)])
x_vec_test = np.array([[random.randrange(i*100,(i+1)*100,1)/10000 for j in range(15)] for i in range(50, 100)])
x_label_test = np.array([int(i/10) for i in range(50, 100)])

# Split testing data into enroll and test data
print('splitting testing data into enroll and test data')
en_xv, en_label, te_xv, te_label = pc.split_en_te(x_vec_test, x_label_test)

# Generate x_vec stat objects
print('generating x_vec stat objects')
xvectors_stat = pc.get_train_x_vec(x_vec_train, x_label_train)
en_sets, en_stat = pc.get_enroll_x_vec(en_xv)
te_sets, te_stat = pc.get_test_x_vec(te_xv)

# Training plda (or load pretrained plda)
print('training plda')
plda = pc.setup_plda(rank_f=10)
plda = pc.train_plda(plda, xvectors_stat)
#plda = pc.load_plda('plda/plda_v1.pickle')

# Testing plda
print('testing plda')
scores_plda = pc.test_plda(plda, en_sets, en_stat, te_sets, te_stat)
mask = np.array(np.diag(np.diag(np.ones(scores_plda.scoremat.shape, dtype=np.int32))), dtype=bool)
scores = scores_plda.scoremat[mask]
print('scores', scores)
print('mean score', np.mean(scores))

pc.save_plda(plda, 'plda_v1')

print('DONE')
