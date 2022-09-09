import random

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from torch.utils.tensorboard import SummaryWriter

import plda_classifier as pc

# Extract the x-vectors
print('extracting x-vectors')
x_vec_train = np.array([[random.randrange(i*100,(i+1)*100,1)/10000 for j in range(15)] for i in range(0, 50)])
x_label_train = np.array([int(i/10) for i in range(0, 50)])
en_xv = np.array([[random.randrange(i*100,(i+1)*100,1)/10000 for j in range(15)] for i in range(50, 100)])
en_label = np.array([int(i/10) for i in range(50, 100)])
te_xv = np.array([[random.randrange(i*100,(i+1)*100,1)/10000 for j in range(15)] for i in range(50, 100)])
te_label = np.array([int(i/10) for i in range(50, 100)])

# x_vec_train = np.random.rand(50, 15)
# x_label_train = np.random.random_integers(0, 10, 50)
# en_xv = np.random.rand(50, 15)
# en_label = np.random.random_integers(0, 10, 50)
# te_xv = np.random.rand(50, 15)
# te_label = np.random.random_integers(0, 10, 50)

#en_xv, en_label = sklearn.utils.shuffle(en_xv, en_label)###TODO comment out and in to compare different results in plda_test
#te_xv, te_label = sklearn.utils.shuffle(te_xv, te_label)###TODO comment out and in to compare different results in plda_test

# Generate x_vec stat objects
print('generating x_vec stat objects')
tr_stat = pc.get_train_x_vec(x_vec_train, x_label_train, x_label_train)
en_stat = pc.get_enroll_x_vec(en_xv, ['id'+str(i) for i in range(en_xv.shape[0])])
te_stat = pc.get_test_x_vec(te_xv, ['id'+str(i) for i in range(te_xv.shape[0])])

# Training plda (or load pretrained plda)
print('training plda')
plda = pc.setup_plda(rank_f=10)
plda = pc.train_plda(plda, tr_stat)
pc.save_plda(plda, 'plda_test')
#plda = pc.load_plda('plda/plda_test.pickle')

# Testing plda
print('testing plda')
scores_plda = pc.plda_scores(plda, en_stat, te_stat)

# Dividing scores into positive and negative
positive_scores = []
negative_scores = []
for en in en_label:
    for te in te_label:
        if(en == te):
            positive_scores.append(1)
            negative_scores.append(0)
        else:
            positive_scores.append(0)
            negative_scores.append(1)
positive_scores_mask = np.array(positive_scores, dtype=bool)
negative_scores_mask = np.array(negative_scores, dtype=bool)
positive_scores_mask = np.reshape(positive_scores_mask, (len(en_label),len(te_label)))
negative_scores_mask = np.reshape(negative_scores_mask, (len(en_label),len(te_label)))
positive_scores = scores_plda.scoremat[positive_scores_mask]
negative_scores = scores_plda.scoremat[negative_scores_mask]

# Calculating EER
print('calculating EER')
eer, eer_th = pc.EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
print('EER: ', eer)
print('threshold: ', eer_th)

# Calculating minDCF
print('calculating minDCF')
min_dcf, min_dcf_th = pc.minDCF(torch.tensor(positive_scores), torch.tensor(negative_scores))
print('minDCF: ', min_dcf)
print('threshold: ', min_dcf_th)

print('DONE NORMAL CODE########################################################################################################################################################################################################################')

def get_scatter_plot_data(new_stat_obj):
    x = np.array(new_stat_obj.stat1[:, 0])
    y = np.array(new_stat_obj.stat1[:, 1])
    unique_speakers = list(new_stat_obj.modelset)
    c = []
    i = 0
    for m in new_stat_obj.modelset:
        i = unique_speakers.index(m)
        c.append(i/len(unique_speakers))
    c = np.array(c)
    return x, y, c

if(False):
    print('plda.mean', '  (shape: ', plda.mean.shape, ')')
    print(plda.mean)
    print('plda.F', '  (shape: ', plda.F.shape, ')')
    print(plda.F)
    print('plda.Sigma', '  (shape: ', plda.Sigma.shape, ')')
    print(plda.Sigma)

if(False):
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
    
    writer = SummaryWriter(log_dir="testlogs/lightning_logs/images/0")
    
    scoremat_norm = np.array(scores_plda.scoremat)
    scoremat_norm -= np.min(scoremat_norm)
    scoremat_norm /= np.max(scoremat_norm)

    print('generating images for tensorboard')
    img = np.zeros((3, scores_plda.scoremask.shape[0], scores_plda.scoremask.shape[1]))
    img[0] = np.array([scores_plda.scoremask])
    img[1] = np.array([scores_plda.scoremask])
    img[2] = np.array([scores_plda.scoremask])
    writer.add_image('score_mask', img, 0)

    img = np.zeros((3, scoremat_norm.shape[0], scoremat_norm.shape[1]))
    img[0] = np.array([scoremat_norm])
    img[1] = np.array([scoremat_norm])
    img[2] = np.array([scoremat_norm])
    writer.add_image('score_matrix', img, 0)

    img = np.zeros((3, scoremat_norm.shape[0], scoremat_norm.shape[1]))
    img[1] = np.array([positive_scores_mask])
    img[0] = np.array([negative_scores_mask])
    writer.add_image('ground_truth', img, 0)

    img = np.zeros((3, scoremat_norm.shape[0], scoremat_norm.shape[1]))
    img[1] = np.array([scoremat_norm*positive_scores_mask])
    img[0] = np.array([scoremat_norm*negative_scores_mask])
    writer.add_image('ground_truth_scores', img, 0)
    
    checked_values_map = positive_scores_mask + negative_scores_mask
    checked_values = checked_values_map * scores_plda.scoremat

    eer_prediction_positive = np.where(checked_values >= eer_th, 1, 0) * checked_values_map
    eer_prediction_negative = np.where(checked_values < eer_th, 1, 0) * checked_values_map
    min_dcf_prediction_positive = np.where(checked_values >= min_dcf_th, 1, 0) * checked_values_map
    min_dcf_prediction_negative = np.where(checked_values < min_dcf_th, 1, 0) * checked_values_map

    img = np.ones((3, scoremat_norm.shape[0], scoremat_norm.shape[1]*2+5))
    img[1,:,:checked_values.shape[1]] = eer_prediction_positive
    img[0,:,:checked_values.shape[1]] = eer_prediction_negative
    img[1,:,-checked_values.shape[1]:] = min_dcf_prediction_positive
    img[0,:,-checked_values.shape[1]:] = min_dcf_prediction_negative
    img[2,:,:checked_values.shape[1]] = 0
    img[2,:,-checked_values.shape[1]:] = 0
    writer.add_image('prediction_eer_min_dcf', img, 0)

    img = np.ones((3, scoremat_norm.shape[0], scoremat_norm.shape[1]*2+5))
    img[1,:,:checked_values.shape[1]] = eer_prediction_positive * scoremat_norm
    img[0,:,:checked_values.shape[1]] = eer_prediction_negative * scoremat_norm
    img[1,:,-checked_values.shape[1]:] = min_dcf_prediction_positive * scoremat_norm
    img[0,:,-checked_values.shape[1]:] = min_dcf_prediction_negative * scoremat_norm
    img[2,:,:checked_values.shape[1]] = 0
    img[2,:,-checked_values.shape[1]:] = 0
    writer.add_image('prediction_scores_eer_min_dcf', img, 0)

    img = np.ones((3, scoremat_norm.shape[0], scoremat_norm.shape[1]*2+5))
    img[1,:,:checked_values.shape[1]] = eer_prediction_positive * positive_scores_mask
    img[0,:,:checked_values.shape[1]] = eer_prediction_negative * negative_scores_mask
    img[1,:,-checked_values.shape[1]:] = min_dcf_prediction_positive * positive_scores_mask
    img[0,:,-checked_values.shape[1]:] = min_dcf_prediction_negative * negative_scores_mask
    img[2,:,:checked_values.shape[1]] = 0
    img[2,:,-checked_values.shape[1]:] = 0
    writer.add_image('correct_prediction_eer_min_dcf', img, 0)

    img = np.ones((3, scoremat_norm.shape[0], scoremat_norm.shape[1]*2+5))
    img[1,:,:checked_values.shape[1]] = eer_prediction_positive * negative_scores_mask
    img[0,:,:checked_values.shape[1]] = eer_prediction_negative * positive_scores_mask
    img[1,:,-checked_values.shape[1]:] = min_dcf_prediction_positive * negative_scores_mask
    img[0,:,-checked_values.shape[1]:] = min_dcf_prediction_negative * positive_scores_mask
    img[2,:,:checked_values.shape[1]] = 0
    img[2,:,-checked_values.shape[1]:] = 0
    writer.add_image('false_prediction_eer_min_dcf', img, 0)

    def get_scatter_plot_data(new_stat_obj):
        x = np.array(new_stat_obj.stat1[:, 0])
        y = np.array(new_stat_obj.stat1[:, 1])
        c = np.array(new_stat_obj.modelset, dtype=np.float64)
        c -= np.min(c)-1
        c /= np.max(c)
        return x, y, c

    x_sum = []
    y_sum = []
    c_sum = []

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(64, 48)

    en_stat = pc.get_enroll_x_vec(en_xv, en_label)
    new_stat_obj = pc.lda(en_stat)
    x, y, c = get_scatter_plot_data(new_stat_obj)
    x_sum.append(x)
    y_sum.append(y)
    c_sum.append(c)
    axs[0,0].scatter(x, y, c=c)
    axs[0,0].title.set_text('Enrollment Data')

    te_stat = pc.get_test_x_vec(te_xv, te_label)
    new_stat_obj = pc.lda(te_stat)
    x, y, c = get_scatter_plot_data(new_stat_obj)
    x_sum.append(x)
    y_sum.append(y)
    c_sum.append(c)
    axs[0,1].scatter(x, y, c=c)
    axs[0,1].title.set_text('Test Data')
    
    axs[1,0].scatter(x_sum, y_sum, c=c_sum)
    axs[1,0].title.set_text('Enrollment + Test Data')
    
    ente_xv = np.zeros((en_xv.shape[0]*2, en_xv.shape[1]))
    ente_xv[:en_xv.shape[0]] = en_xv
    ente_xv[-te_xv.shape[0]:] = te_xv
    ente_label = np.zeros(en_label.shape[0]*2)
    ente_label[:en_label.shape[0]] = en_label
    ente_label[-te_label.shape[0]:] = te_label

    ente_stat = pc.get_test_x_vec(ente_xv, ente_label)
    new_stat_obj = pc.lda(ente_stat)
    x, y, c = get_scatter_plot_data(new_stat_obj)
    x_sum.append(x)
    y_sum.append(y)
    c_sum.append(c)
    axs[1,1].scatter(x, y, c=c)
    axs[1,1].title.set_text('Enrollment + Test Data new evaluation')

    writer.add_figure('scatter_plot_before_training', plt.gcf())



    x_sum = []
    y_sum = []
    c_sum = []

    for i, (e, t)in enumerate(zip(en_xv, te_xv)):
        en_xv[i,:] = np.dot(plda.Sigma, e)
        te_xv[i,:] = np.dot(plda.Sigma, t)

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(64, 48)

    en_stat = pc.get_enroll_x_vec(en_xv, en_label)
    new_stat_obj = pc.lda(en_stat)
    x, y, c = get_scatter_plot_data(new_stat_obj)
    x_sum.append(x)
    y_sum.append(y)
    c_sum.append(c)
    axs[0,0].scatter(x, y, c=c)
    axs[0,0].title.set_text('Enrollment Data')

    te_stat = pc.get_test_x_vec(te_xv, te_label)
    new_stat_obj = pc.lda(te_stat)
    x, y, c = get_scatter_plot_data(new_stat_obj)
    x_sum.append(x)
    y_sum.append(y)
    c_sum.append(c)
    axs[0,1].scatter(x, y, c=c)
    axs[0,1].title.set_text('Test Data')
    
    axs[1,0].scatter(x_sum, y_sum, c=c_sum)
    axs[1,0].title.set_text('Enrollment + Test Data')
    
    ente_xv = np.zeros((en_xv.shape[0]*2, en_xv.shape[1]))
    ente_xv[:en_xv.shape[0]] = en_xv
    ente_xv[-te_xv.shape[0]:] = te_xv
    ente_label = np.zeros(en_label.shape[0]*2)
    ente_label[:en_label.shape[0]] = en_label
    ente_label[-te_label.shape[0]:] = te_label

    ente_stat = pc.get_test_x_vec(ente_xv, ente_label)
    new_stat_obj = pc.lda(ente_stat)
    x, y, c = get_scatter_plot_data(new_stat_obj)
    x_sum.append(x)
    y_sum.append(y)
    c_sum.append(c)
    axs[1,1].scatter(x, y, c=c)
    axs[1,1].title.set_text('Enrollment + Test Data new evaluation')

    writer.add_figure('scatter_plot_after_training', plt.gcf())

    writer.close()

'''
# for i, x in enumerate(scores_plda.scoremat):
#     print(en_label[i], x)
    
# print(en_label)
# print(te_label)

# en_xv, en_label = sklearn.utils.shuffle(en_xv, en_label)
# te_xv, te_label = sklearn.utils.shuffle(te_xv, te_label)

en_stat = pc.get_enroll_x_vec(en_xv, ['id'+str(i) for i in range(en_xv.shape[0])])
te_stat = pc.get_test_x_vec(te_xv, ['id'+str(i) for i in range(te_xv.shape[0])])

# Testing plda
print('testing plda')
scores_plda = pc.test_plda(plda, en_stat, te_stat) #TODO test if i can calculate the result myself by using the matrix in plda.F

# Dividing scores into positive and negative
positive_scores = []
negative_scores = []
for i, en in enumerate(en_label):
    for j, te in enumerate(te_label):
        if(scores_plda.scoremask[i,j]):
            if(en == te):
                positive_scores.append(1)
                negative_scores.append(0)
            else:
                positive_scores.append(0)
                negative_scores.append(1)
        else:
            positive_scores.append(0)
            negative_scores.append(0)

positive_scores_mask = np.array(positive_scores, dtype=bool)
negative_scores_mask = np.array(negative_scores, dtype=bool)
positive_scores_mask = np.reshape(positive_scores_mask, (len(en_label),len(te_label)))
negative_scores_mask = np.reshape(negative_scores_mask, (len(en_label),len(te_label)))
positive_scores = scores_plda.scoremat[positive_scores_mask]
negative_scores = scores_plda.scoremat[negative_scores_mask]

# Calculating EER
eer, th = pc.EER(torch.tensor(positive_scores), torch.tensor(negative_scores))

print('EER: ', eer)
print('threshold: ', th)

# TODO minDCF
#min_dcf, th = minDCF( torch.tensor(positive_scores), torch.tensor(negative_scores))

img = np.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
img[0] = np.array([scores_plda.scoremat])
writer.add_image('scores2', img, 0)

img = np.zeros((3, scores_plda.scoremask.shape[0], scores_plda.scoremask.shape[1]))
img[0] = np.array([scores_plda.scoremask])
writer.add_image('scoremask2', img, 0)
    
img = np.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
img[1] = np.array([scores_plda.scoremat*positive_scores_mask])
img[0] = np.array([scores_plda.scoremat*negative_scores_mask])
writer.add_image('ground_truth2', img, 0)

new_mask = np.zeros_like(scores_plda.scoremat)
new_mask_inv = np.zeros_like(scores_plda.scoremat)
for i, score in np.ndenumerate(scores_plda.scoremat):
    if(score >= th):
        new_mask[i] = 1
    else:
        new_mask_inv[i] = 1

img = np.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
img[1] = np.array([scores_plda.scoremat*new_mask])
img[0] = np.array([scores_plda.scoremat*new_mask_inv])
writer.add_image('prediction2', img, 0)

img = np.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
img[1] = np.array([positive_scores_mask*new_mask])
img[0] = np.array([negative_scores_mask*new_mask_inv])
writer.add_image('correct_prediction2', img, 0)

# for i, x in enumerate(scores_plda.scoremat):
#     print(en_label[i], x)
    
# print(en_label)
# print(te_label)'''

print('done')
