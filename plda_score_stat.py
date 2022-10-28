import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
from sklearn.manifold import TSNE
from speechbrain.utils.metric_stats import EER, minDCF

import plda_classifier as pc


class plda_score_stat_object():
    def __init__(self, x_vectors_test):
        self.x_vectors_test = x_vectors_test
        self.x_id_test = np.array(self.x_vectors_test.iloc[:, 1])
        self.x_vec_test = np.array([np.array(x_vec[1:-1].split(), dtype=np.float64) for x_vec in self.x_vectors_test.iloc[:, 3]])

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

        self.checked_xvec = []
        self.checked_label = []

    def test_plda(self, plda, veri_test_file_path):
        """
        Tests the PLDA performance based on the VoxCeleb veri test files speaker pairings.

        Parameters
        ----------
        PLDA: obj
            The PLDA getting tested
            
        veri_test_file_path: string
            The path to the VoxCeleb veri test file

        Returns
        -------
        sample: tensor
            The MFCC of the desires sample
        
        label: string
            The label of the sample

        id: string
            The scource directory of the sample (unique for each seperate sample)
        """
        self.plda_scores = pc.plda_scores(plda, self.en_stat, self.te_stat)
        self.positive_scores_mask = np.zeros_like(self.plda_scores.scoremat)
        self.negative_scores_mask = np.zeros_like(self.plda_scores.scoremat)
        
        checked_list = []
        for pair in open(veri_test_file_path):
            is_match = bool(int(pair.split(" ")[0].rstrip().split(".")[0].strip()))
            enrol_id = pair.split(" ")[1].strip()
            test_id = pair.split(" ")[2].strip()

            i = int(np.where(self.plda_scores.modelset == enrol_id)[0][0])
            if(not enrol_id in checked_list):
                checked_list.append(enrol_id)
                self.checked_xvec.append(np.array(self.x_vectors_test.loc[self.x_vectors_test['id'] == enrol_id, 'xvector'].item()[1:-1].split(), dtype=np.float64))
                self.checked_label.append(int(enrol_id.split(".")[0].split("/")[0][2:]))
                
            j = int(np.where(self.plda_scores.segset == test_id)[0][0])
            if(not test_id in checked_list):
                checked_list.append(test_id)
                self.checked_xvec.append(np.array(self.x_vectors_test.loc[self.x_vectors_test['id'] == test_id, 'xvector'].item()[1:-1].split(), dtype=np.float64))
                self.checked_label.append(int(test_id.split(".")[0].split("/")[0][2:]))

            current_score = float(self.plda_scores.scoremat[i,j])
            if(is_match):
                self.positive_scores.append(current_score)
                self.positive_scores_mask[i,j] = 1
            else:
                self.negative_scores.append(current_score)
                self.negative_scores_mask[i,j] = 1
                    
        self.checked_xvec = np.array(self.checked_xvec)
        self.checked_label = np.array(self.checked_label)

    def calc_eer_mindcf(self):
        """
        Calculate the EER and minDCF.
        """
        self.eer, self.eer_th = EER(torch.tensor(self.positive_scores), torch.tensor(self.negative_scores))
        self.min_dcf, self.min_dcf_th = minDCF(torch.tensor(self.positive_scores), torch.tensor(self.negative_scores), p_target=0.5)

    def plot_images(self, writer):
        """
        Plot images for the given writer.

        Parameters
        ----------
        writer: the writer the images are plotted for
        """
        split_xvec = []
        split_label = []
        group_kfold = sklearn.model_selection.GroupKFold(n_splits=2)
        groups1234 = np.where(self.checked_label<10290, 0, 1)
        for g12, g34 in group_kfold.split(self.checked_xvec, self.checked_label, groups1234):
            x12, x34 = self.checked_xvec[g12], self.checked_xvec[g34]
            y12, y34 = self.checked_label[g12], self.checked_label[g34]
            groups12 = np.where(y12<10280, 0, 1)
            groups34 = np.where(y34<10300, 0, 1)
            for g1, g2 in group_kfold.split(x12, y12, groups12):
                split_xvec.append(x12[g1])
                split_xvec.append(x12[g2])
                split_label.append(y12[g1])
                split_label.append(y12[g2])
                break
            for g3, g4 in group_kfold.split(x34, y34, groups34):
                split_xvec.append(x34[g3])
                split_xvec.append(x34[g4])
                split_label.append(y34[g3])
                split_label.append(y34[g4])
                break
            break
        split_xvec = np.array(split_xvec)
        split_label = np.array(split_label)

        print('generating images for tensorboard')
        scoremat_norm = np.array(self.plda_scores.scoremat)
        scoremat_norm -= np.min(scoremat_norm)
        scoremat_norm /= np.max(scoremat_norm)

        print('score_matrix')
        img = np.zeros((3, scoremat_norm.shape[0], scoremat_norm.shape[1]))
        img[0] = np.array([scoremat_norm])
        img[1] = np.array([scoremat_norm])
        img[2] = np.array([scoremat_norm])
        writer.add_image('score_matrix', img, 0)

        print('ground_truth')
        img = np.zeros((3, scoremat_norm.shape[0], scoremat_norm.shape[1]))
        img[1] = np.array([self.positive_scores_mask])
        img[0] = np.array([self.negative_scores_mask])
        writer.add_image('ground_truth', img, 0)

        print('ground_truth_scores')
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
    
        print('prediction_eer_min_dcf')
        img = np.ones((3, scoremat_norm.shape[0], scoremat_norm.shape[1]*2+5))
        img[1,:,:checked_values.shape[1]] = eer_prediction_positive
        img[0,:,:checked_values.shape[1]] = eer_prediction_negative
        img[1,:,-checked_values.shape[1]:] = min_dcf_prediction_positive
        img[0,:,-checked_values.shape[1]:] = min_dcf_prediction_negative
        img[2,:,:checked_values.shape[1]] = 0
        img[2,:,-checked_values.shape[1]:] = 0
        writer.add_image('prediction_eer_min_dcf', img, 0)
    
        print('correct_prediction_eer_min_dcf')
        img = np.ones((3, scoremat_norm.shape[0], scoremat_norm.shape[1]*2+5))
        img[1,:,:checked_values.shape[1]] = eer_prediction_positive * self.positive_scores_mask
        img[0,:,:checked_values.shape[1]] = eer_prediction_negative * self.negative_scores_mask
        img[1,:,-checked_values.shape[1]:] = min_dcf_prediction_positive * self.positive_scores_mask
        img[0,:,-checked_values.shape[1]:] = min_dcf_prediction_negative * self.negative_scores_mask
        img[2,:,:checked_values.shape[1]] = 0
        img[2,:,-checked_values.shape[1]:] = 0
        writer.add_image('correct_prediction_eer_min_dcf', img, 0)
    
        print('false_prediction_eer_min_dcf')
        img = np.ones((3, scoremat_norm.shape[0], scoremat_norm.shape[1]*2+5))
        img[1,:,:checked_values.shape[1]] = eer_prediction_positive * self.negative_scores_mask
        img[0,:,:checked_values.shape[1]] = eer_prediction_negative * self.positive_scores_mask
        img[1,:,-checked_values.shape[1]:] = min_dcf_prediction_positive * self.negative_scores_mask
        img[0,:,-checked_values.shape[1]:] = min_dcf_prediction_negative * self.positive_scores_mask
        img[2,:,:checked_values.shape[1]] = 0
        img[2,:,-checked_values.shape[1]:] = 0
        writer.add_image('false_prediction_eer_min_dcf', img, 0)

        def generate_scatter_plot(x, y, label, plot_name):
            df = pd.DataFrame({'x': x, 'y': y, 'label': label})
            fig, ax = plt.subplots(1)
            fig.set_size_inches(16, 12)
            sns.scatterplot(x='x', y='y', hue='label', palette='bright', data=df, ax=ax, s=80) #use sns.color_palette("hls", 40) for 40 speakers
            limx = (x.min()-5, x.max()+5)
            limy = (y.min()-5, y.max()+5)
            ax.set_xlim(limx)
            ax.set_ylim(limy)
            ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            ax.title.set_text(plot_name)
        
        for i, (checked_xvec, checked_label) in enumerate(zip(split_xvec, split_label)):
            print('scatter_plot_LDA'+str(i+1))
            new_stat = pc.get_x_vec_stat(checked_xvec, checked_label)
            new_stat = pc.lda(new_stat)
            generate_scatter_plot(new_stat.stat1[:, 0], new_stat.stat1[:, 1], checked_label, 'scatter_plot_LDA'+str(i+1))
            writer.add_figure('scatter_plot_LDA'+str(i+1), plt.gcf())

            print('scatter_plot_PCA'+str(i+1))
            pca = sklearn.decomposition.PCA(n_components=2)
            pca_result = pca.fit_transform(sklearn.preprocessing.StandardScaler().fit_transform(checked_xvec))
            generate_scatter_plot(pca_result[:,0], pca_result[:,1], checked_label, 'scatter_plot_PCA'+str(i+1))
            writer.add_figure('scatter_plot_PCA'+str(i+1), plt.gcf())

            print('scatter_plot_TSNE'+str(i+1))
            tsne = TSNE(2)
            tsne_result = tsne.fit_transform(checked_xvec)
            generate_scatter_plot(tsne_result[:,0], tsne_result[:,1], checked_label, 'scatter_plot_TSNE'+str(i+1))
            writer.add_figure('scatter_plot_TSNE'+str(i+1), plt.gcf())
