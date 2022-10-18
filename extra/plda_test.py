import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
from sklearn.manifold import TSNE
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
'''
x_vec_train = np.random.rand(50, 15)
x_label_train = np.random.random_integers(0, 10, 50)
en_xv = np.random.rand(50, 15)
en_label = np.random.random_integers(0, 10, 50)
te_xv = np.random.rand(50, 15)
te_label = np.random.random_integers(0, 10, 50)'''

# Generate x_vec stat objects
print('generating x_vec stat objects')
tr_stat = pc.get_train_x_vec(x_vec_train, x_label_train, x_label_train)
en_stat = pc.get_x_vec_stat(en_xv, ['id'+str(i) for i in range(en_xv.shape[0])])
te_stat = pc.get_x_vec_stat(te_xv, ['id'+str(i) for i in range(te_xv.shape[0])])

# Training plda (or load pretrained plda)
print('training plda')
plda = pc.setup_plda(rank_f=10)
plda = pc.train_plda(plda, tr_stat)

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
min_dcf, min_dcf_th = pc.minDCF(torch.tensor(positive_scores), torch.tensor(negative_scores), p_target=0.5)
print('minDCF: ', min_dcf)
print('threshold: ', min_dcf_th)

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

if(True):
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

    ll_positive = np.where(scores_plda.scoremat >= 0, 1, 0)
    ll_negative = np.where(scores_plda.scoremat < 0, 1, 0)
    print('score_matrix_log_likelihood')
    img = np.zeros((3, scoremat_norm.shape[0], scoremat_norm.shape[1]))
    img[0] = np.array([scoremat_norm]*ll_negative)
    img[1] = np.array([scoremat_norm]*ll_positive)
    writer.add_image('score_matrix_log_likelihood', img, 0)

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
    fig.set_size_inches(32, 24)

    en_stat = pc.get_x_vec_stat(en_xv, en_label)
    new_stat_obj = pc.lda(en_stat)
    x, y, c = get_scatter_plot_data(new_stat_obj)
    x_sum.append(x)
    y_sum.append(y)
    c_sum.append(c)
    axs[0,0].scatter(x, y, c=c)
    axs[0,0].title.set_text('Enrollment Data')

    te_stat = pc.get_x_vec_stat(te_xv, te_label)
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

    ente_stat = pc.get_x_vec_stat(ente_xv, ente_label)
    new_stat_obj = pc.lda(ente_stat)
    x, y, c = get_scatter_plot_data(new_stat_obj)
    x_sum.append(x)
    y_sum.append(y)
    c_sum.append(c)
    axs[1,1].scatter(x, y, c=c)
    axs[1,1].title.set_text('Enrollment + Test Data new evaluation')

    writer.add_figure('scatter_plot_before_training', plt.gcf())

    tsne = TSNE(2)
    tsne_result = tsne.fit_transform(ente_xv)

    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': ente_label})
    fig, ax = plt.subplots(1)
    fig.set_size_inches(16, 12)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette="deep", data=tsne_result_df, ax=ax, s=80)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    writer.add_figure('scatter_plot_TSNE_before_training', plt.gcf())

    pca = sklearn.decomposition.PCA(n_components=2)
    pca.fit(sklearn.preprocessing.StandardScaler().fit_transform(tr_stat.stat1))
    pca_result = pca.transform(sklearn.preprocessing.StandardScaler().fit_transform(ente_xv))

    pca_result_df = pd.DataFrame({'pca_1': pca_result[:,0], 'pca_2': pca_result[:,1], 'label': ente_label})
    fig, ax = plt.subplots(1)
    fig.set_size_inches(16, 12)
    sns.scatterplot(x='pca_1', y='pca_2', hue='label', palette="deep", data=pca_result_df, ax=ax, s=80)
    lim = (pca_result.min()-5, pca_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    writer.add_figure('scatter_plot_PCA_before_training', plt.gcf())



    A = np.linalg.inv(plda.Sigma)
    for i, (e, t)in enumerate(zip(en_xv, te_xv)):
        en_xv[i,:] = np.dot(A, (e-plda.mean))
        te_xv[i,:] = np.dot(A, (t-plda.mean))



    x_sum = []
    y_sum = []
    c_sum = []

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(32, 24)

    en_stat = pc.get_x_vec_stat(en_xv, en_label)
    new_stat_obj = pc.lda(en_stat)
    x, y, c = get_scatter_plot_data(new_stat_obj)
    x_sum.append(x)
    y_sum.append(y)
    c_sum.append(c)
    axs[0,0].scatter(x, y, c=c)
    axs[0,0].title.set_text('Enrollment Data')

    te_stat = pc.get_x_vec_stat(te_xv, te_label)
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

    ente_stat = pc.get_x_vec_stat(ente_xv, ente_label)
    new_stat_obj = pc.lda(ente_stat)
    x, y, c = get_scatter_plot_data(new_stat_obj)
    x_sum.append(x)
    y_sum.append(y)
    c_sum.append(c)
    axs[1,1].scatter(x, y, c=c)
    axs[1,1].title.set_text('Enrollment + Test Data new evaluation')

    writer.add_figure('scatter_plot_after_training', plt.gcf())

    tsne = TSNE(2)
    tsne_result = tsne.fit_transform(ente_xv)

    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': ente_label})
    fig, ax = plt.subplots(1)
    fig.set_size_inches(16, 12)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette="deep", data=tsne_result_df, ax=ax, s=80)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    writer.add_figure('scatter_plot_TSNE_after_training', plt.gcf())

    pca = sklearn.decomposition.PCA(n_components=2)
    pca.fit(sklearn.preprocessing.StandardScaler().fit_transform(tr_stat.stat1))
    pca_result = pca.transform(sklearn.preprocessing.StandardScaler().fit_transform(ente_xv))

    pca_result_df = pd.DataFrame({'pca_1': pca_result[:,0], 'pca_2': pca_result[:,1], 'label': ente_label})
    fig, ax = plt.subplots(1)
    fig.set_size_inches(16, 12)
    sns.scatterplot(x='pca_1', y='pca_2', hue='label', palette="deep", data=pca_result_df, ax=ax, s=80)
    lim = (pca_result.min()-5, pca_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    writer.add_figure('scatter_plot_PCA_after_training', plt.gcf())

    writer.close()

print('done')
