import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE

import plda_classifier as pc


class plda_score_stat_object():
    def __init__(self, x_vectors_test):
        self.x_vectors_test = x_vectors_test
        self.x_id_test = np.array(x_vectors_test.iloc[:, 1])
        self.x_vec_test = np.array([np.array(x_vec[1:-1].split(), dtype=np.float64) for x_vec in x_vectors_test.iloc[:, 3]])

        self.en_stat = pc.get_x_vec_stat(self.x_vec_test, self.x_id_test)
        self.te_stat = pc.get_x_vec_stat(self.x_vec_test, self.x_id_test)

        self.plda_scores = 0
        self.positive_scores = []
        self.negative_scores = []
        self.positive_scores_mask = []
        self.negative_scores_mask = []

        self.eer = 0
        self.eer_th = 0
        self.min_dcf = 0
        self.min_dcf_th = 0

        self.en_xv = []
        self.en_label = []
        self.te_xv = []
        self.te_label = []

    def test_plda(self, plda, veri_test_file_path):
        self.plda_scores = pc.plda_scores(plda, self.en_stat, self.te_stat)
        self.positive_scores_mask = np.zeros_like(self.plda_scores.scoremat)
        self.negative_scores_mask = np.zeros_like(self.plda_scores.scoremat)
        
        for pair in open(veri_test_file_path):
            is_match = bool(int(pair.split(" ")[0].rstrip().split(".")[0].strip()))
            enrol_id = pair.split(" ")[1].strip()
            test_id = pair.split(" ")[2].strip()

            i = int(np.where(self.plda_scores.modelset == enrol_id)[0][0])
            self.en_xv.append(np.array(self.x_vectors_test.loc[self.x_vectors_test['id'] == enrol_id, 'xvector'].item()[1:-1].split(), dtype=np.float64))
            self.en_label.append(int(enrol_id.split(".")[0].split("/")[0][2:]))

            j = int(np.where(self.plda_scores.segset == test_id)[0][0])
            self.te_xv.append(np.array(self.x_vectors_test.loc[self.x_vectors_test['id'] == test_id, 'xvector'].item()[1:-1].split(), dtype=np.float64))
            self.te_label.append(int(test_id.split(".")[0].split("/")[0][2:]))
            
            current_score = float(self.plda_scores.scoremat[i,j])
            if(is_match):
                self.positive_scores.append(current_score)
                self.positive_scores_mask[i,j] = 1
            else:
                self.negative_scores.append(current_score)
                self.negative_scores_mask[i,j] = 1
                    
        self.en_xv = np.array(self.en_xv)
        self.en_label = np.array(self.en_label)
        self.te_xv = np.array(self.te_xv)
        self.te_label = np.array(self.te_label)

    def calc_eer_mindcf(self):
        self.eer, self.eer_th = pc.EER(torch.tensor(self.positive_scores), torch.tensor(self.negative_scores))
        self.min_dcf, self.min_dcf_th = pc.minDCF(torch.tensor(self.positive_scores), torch.tensor(self.negative_scores))

    def plot_images(self, writer, plda):
        scoremat_norm = np.array(self.plda_scores.scoremat)
        scoremat_norm -= np.min(scoremat_norm)
        scoremat_norm /= np.max(scoremat_norm)

        print('generating images for tensorboard')
        img = np.zeros((3, self.plda_scores.scoremask.shape[0], self.plda_scores.scoremask.shape[1]))
        img[0] = np.array([self.plda_scores.scoremask])
        img[1] = np.array([self.plda_scores.scoremask])
        img[2] = np.array([self.plda_scores.scoremask])
        writer.add_image('score_mask', img, 0)

        img = np.zeros((3, scoremat_norm.shape[0], scoremat_norm.shape[1]))
        img[0] = np.array([scoremat_norm])
        img[1] = np.array([scoremat_norm])
        img[2] = np.array([scoremat_norm])
        writer.add_image('score_matrix', img, 0)

        img = np.zeros((3, scoremat_norm.shape[0], scoremat_norm.shape[1]))
        img[1] = np.array([self.positive_scores_mask])
        img[0] = np.array([self.negative_scores_mask])
        writer.add_image('ground_truth', img, 0)

        img = np.zeros((3, scoremat_norm.shape[0], scoremat_norm.shape[1]))
        img[1] = np.array([scoremat_norm*self.positive_scores_mask])
        img[0] = np.array([scoremat_norm*self.negative_scores_mask])
        writer.add_image('ground_truth_scores', img, 0)
        
        checked_values_map = self.positive_scores_mask + self.negative_scores_mask
        checked_values = checked_values_map * self.plda_scores.scoremat

        eer_prediction_positive = np.where(checked_values >= self.eer_th, 1, 0) * checked_values_map
        eer_prediction_negative = np.where(checked_values < self.eer_th, 1, 0) * checked_values_map
        min_dcf_prediction_positive = np.where(checked_values >= self.min_dcf_th, 1, 0) * checked_values_map
        min_dcf_prediction_negative = np.where(checked_values < self.min_dcf_th, 1, 0) * checked_values_map
    
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
        img[1,:,:checked_values.shape[1]] = eer_prediction_positive * self.positive_scores_mask
        img[0,:,:checked_values.shape[1]] = eer_prediction_negative * self.negative_scores_mask
        img[1,:,-checked_values.shape[1]:] = min_dcf_prediction_positive * self.positive_scores_mask
        img[0,:,-checked_values.shape[1]:] = min_dcf_prediction_negative * self.negative_scores_mask
        img[2,:,:checked_values.shape[1]] = 0
        img[2,:,-checked_values.shape[1]:] = 0
        writer.add_image('correct_prediction_eer_min_dcf', img, 0)
    
        img = np.ones((3, scoremat_norm.shape[0], scoremat_norm.shape[1]*2+5))
        img[1,:,:checked_values.shape[1]] = eer_prediction_positive * self.negative_scores_mask
        img[0,:,:checked_values.shape[1]] = eer_prediction_negative * self.positive_scores_mask
        img[1,:,-checked_values.shape[1]:] = min_dcf_prediction_positive * self.negative_scores_mask
        img[0,:,-checked_values.shape[1]:] = min_dcf_prediction_negative * self.positive_scores_mask
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

        en_stat = pc.get_x_vec_stat(self.en_xv, self.en_label)
        new_stat_obj = pc.lda(en_stat)
        x, y, c = get_scatter_plot_data(new_stat_obj)
        x_sum.append(x)
        y_sum.append(y)
        c_sum.append(c)
        axs[0,0].scatter(x, y, c=c)
        axs[0,0].title.set_text('Enrollment Data')

        te_stat = pc.get_x_vec_stat(self.te_xv, self.te_label)
        new_stat_obj = pc.lda(te_stat)
        x, y, c = get_scatter_plot_data(new_stat_obj)
        x_sum.append(x)
        y_sum.append(y)
        c_sum.append(c)
        axs[0,1].scatter(x, y, c=c)
        axs[0,1].title.set_text('Test Data')
        
        axs[1,0].scatter(x_sum, y_sum, c=c_sum)
        axs[1,0].title.set_text('Enrollment + Test Data')
    
        ente_xv = np.zeros((self.en_xv.shape[0]*2, self.en_xv.shape[1]))
        ente_xv[:self.en_xv.shape[0]] = self.en_xv
        ente_xv[-self.te_xv.shape[0]:] = self.te_xv
        ente_label = np.zeros(self.en_label.shape[0]*2)
        ente_label[:self.en_label.shape[0]] = self.en_label
        ente_label[-self.te_label.shape[0]:] = self.te_label

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
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=80)
        lim = (tsne_result.min()-5, tsne_result.max()+5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

        writer.add_figure('scatter_plot_TSNE_before_training', plt.gcf())



        A = np.linalg.inv(plda.Sigma)
        for i, (e, t)in enumerate(zip(self.en_xv, self.te_xv)):
            self.en_xv[i,:] = np.dot(A, (e-plda.mean))
            self.te_xv[i,:] = np.dot(A, (t-plda.mean))

        x_sum = []
        y_sum = []
        c_sum = []

        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(32, 24)

        en_stat = pc.get_x_vec_stat(self.en_xv, self.en_label)
        new_stat_obj = pc.lda(en_stat)
        x, y, c = get_scatter_plot_data(new_stat_obj)
        x_sum.append(x)
        y_sum.append(y)
        c_sum.append(c)
        axs[0,0].scatter(x, y, c=c)
        axs[0,0].title.set_text('Enrollment Data')

        te_stat = pc.get_x_vec_stat(self.te_xv, self.te_label)
        new_stat_obj = pc.lda(te_stat)
        x, y, c = get_scatter_plot_data(new_stat_obj)
        x_sum.append(x)
        y_sum.append(y)
        c_sum.append(c)
        axs[0,1].scatter(x, y, c=c)
        axs[0,1].title.set_text('Test Data')
        
        axs[1,0].scatter(x_sum, y_sum, c=c_sum)
        axs[1,0].title.set_text('Enrollment + Test Data')
    
        ente_xv = np.zeros((self.en_xv.shape[0]*2, self.en_xv.shape[1]))
        ente_xv[:self.en_xv.shape[0]] = self.en_xv
        ente_xv[-self.te_xv.shape[0]:] = self.te_xv
        ente_label = np.zeros(self.en_label.shape[0]*2)
        ente_label[:self.en_label.shape[0]] = self.en_label
        ente_label[-self.te_label.shape[0]:] = self.te_label

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
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=80)
        lim = (tsne_result.min()-5, tsne_result.max()+5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

        writer.add_figure('scatter_plot_TSNE_after_training', plt.gcf())
