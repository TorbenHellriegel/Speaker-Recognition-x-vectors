import random

import numpy as np
import sklearn

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

pc.save_plda(plda, 'plda_test')

print('DONE NORMAL CODE########################################################################################################################################################################################################################')
print('DONE NORMAL CODE########################################################################################################################################################################################################################')
print('DONE NORMAL CODE########################################################################################################################################################################################################################')

if(False):
    print('plda.mean', '  (shape: ', plda.mean.shape, ')')
    print(plda.mean)
    print('plda.F', '  (shape: ', plda.F.shape, ')')
    print(plda.F)
    print('plda.Sigma', '  (shape: ', plda.Sigma.shape, ')')
    print(plda.Sigma)

if(True):
    print('Functional Tests########################################################################################################################################################################################################################')

    print('en_label', '  (shape: ', en_label.shape, ')')
    print(en_label)
    print('te_label', '  (shape: ', te_label.shape, ')')
    print(te_label)
    print('en_stat.stat1', '  (shape: ', en_stat.stat1.shape, ')')
    print(en_stat.stat1)
    print('te_stat.stat1', '  (shape: ', te_stat.stat1.shape, ')')
    print(te_stat.stat1)

    # print('scores_plda.scoremat', '  (shape: ', scores_plda.scoremat.shape, ')')
    # print(scores_plda.scoremat)
    print('scores_plda.scoremask', '  (shape: ', np.array(scores_plda.scoremask, dtype=np.int32).shape, ')')
    print(np.array(scores_plda.scoremask, dtype=np.int32))

    sm_scores = scores_plda.scoremat[scores_plda.scoremask]

    print('scoremask scores', sm_scores)
    print('scoremask min score', np.min(sm_scores))
    print('scoremask mean score', np.mean(sm_scores))
    print('scoremask max score', np.max(sm_scores))
    print('scoremask abs min score', np.min(np.abs(sm_scores)))
    print('scoremask abs mean score', np.mean(np.abs(sm_scores)))
    print('scoremask abs max score', np.max(np.abs(sm_scores)))

    print('my scores', scores)
    print('my min score', np.min(scores))
    print('my mean score', np.mean(scores))
    print('my max score', np.max(scores))
    print('my abs min score', np.min(np.abs(scores)))
    print('my abs mean score', np.mean(np.abs(scores)))
    print('my abs max score', np.max(np.abs(scores)))

if(False):
    print('Shuffle Tests########################################################################################################################################################################################################################')

    te_xv, te_label = sklearn.utils.shuffle(te_xv, te_label)
    te_sets, te_stat = pc.get_test_x_vec(te_xv)

    scores_plda = pc.test_plda(plda, en_sets, en_stat, te_sets, te_stat)
    mask = np.array(np.diag(np.diag(np.ones(scores_plda.scoremat.shape, dtype=np.int32))), dtype=bool)
    scores = scores_plda.scoremat[mask]

    print('en_label', '  (shape: ', en_label.shape, ')')
    print(en_label)
    print('te_label', '  (shape: ', te_label.shape, ')')
    print(te_label)
    print('en_stat.stat1', '  (shape: ', en_stat.stat1.shape, ')')
    print(en_stat.stat1)
    print('te_stat.stat1', '  (shape: ', te_stat.stat1.shape, ')')
    print(te_stat.stat1)

    print('scores_plda.scoremat', '  (shape: ', scores_plda.scoremat.shape, ')')
    print(scores_plda.scoremat)
    print('scores_plda.scoremask', '  (shape: ', np.array(scores_plda.scoremask, dtype=np.int32).shape, ')')
    print(np.array(scores_plda.scoremask, dtype=np.int32))

    sm_scores = scores_plda.scoremat[scores_plda.scoremask]

    print('scoremask scores', sm_scores)
    print('scoremask min score', np.min(sm_scores))
    print('scoremask mean score', np.mean(sm_scores))
    print('scoremask max score', np.max(sm_scores))
    print('scoremask abs min score', np.min(np.abs(sm_scores)))
    print('scoremask abs mean score', np.mean(np.abs(sm_scores)))
    print('scoremask abs max score', np.max(np.abs(sm_scores)))

    print('my scores', scores)
    print('my min score', np.min(scores))
    print('my mean score', np.mean(scores))
    print('my max score', np.max(scores))
    print('my abs min score', np.min(np.abs(scores)))
    print('my abs mean score', np.mean(np.abs(scores)))
    print('my abs max score', np.max(np.abs(scores)))

if(True):
    print('Plot result########################################################################################################################################################################################################################')
    
    from torch.utils.tensorboard import SummaryWriter
    
    writer = SummaryWriter(log_dir="testlogs/lightning_logs/image")

    img = np.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
    img[0] = np.array([scores_plda.scoremat])
    writer.add_image('scores', img, 0)
    
    img = np.zeros((3, scores_plda.scoremask.shape[0], scores_plda.scoremask.shape[1]))
    img[0] = np.array([scores_plda.scoremask])
    writer.add_image('mask', img, 0)
    
    img = np.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
    img[0] = np.array([scores_plda.scoremat*scores_plda.scoremask])
    writer.add_image('masked_scores', img, 0)

    writer.close()