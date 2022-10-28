import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
from sklearn.manifold import TSNE
from speechbrain.utils.metric_stats import EER, minDCF

import plda_classifier as pc

'''
from scipy.linalg import eigh
def calc_scatter_matrices(X, Y):
    """ See Equations (1) on p.532 of Ioffe 2006. """
    assert len(X.shape) == 2
    assert X.shape[0] == len(Y)

    unique_labels = np.unique(Y)
    labels = np.asarray(Y)

    m = X.mean(axis=0)
    N = X.shape[0]

    cov_ks = []
    m_ks = []
    n_ks = []

    for k in unique_labels:
        bool_idxs = labels == k
        X_k = X[bool_idxs]

        m_ks.append(X_k.mean(axis=0))
        n_ks.append(bool_idxs.sum())

        cov_ks.append(np.cov(X_k.T))

    n_ks = np.asarray(n_ks)
    m_ks = np.asarray(m_ks)
    m_ks_minus_m = m_ks - m
    S_b = np.matmul(m_ks_minus_m.T * (n_ks / N), m_ks_minus_m)

    S_w = np.asarray(cov_ks) * ((n_ks - 1) / N)[:, None, None]
    S_w = np.sum(S_w, axis=0)

    return S_b, S_w

def calc_m(X):
    """ See Fig. 2 on p.537 of Ioffe 2006. """
    assert len(X.shape) == 2
    return X.mean(axis=0)

def calc_W(S_b, S_w):
    """ See Fig. 2 on p.537 of Ioffe 2006. """
    eigenvalues, eigenvectors = eigh(S_b, S_w)
    return eigenvectors

def calc_Lambda_b(S_b, W):
    """ See Fig. 2 on p.537 of Ioffe 2006. """
    return np.matmul(np.matmul(W.T, S_b), W)

def calc_Lambda_b(S_b, W):
    """ See Fig. 2 on p.537 of Ioffe 2006. """
    return (W.T@ S_b)@ W

def calc_Lambda_w(S_w, W):
    """ See Fig. 2 on p.537 of Ioffe 2006. """
    return np.matmul(np.matmul(W.T, S_w), W)

def calc_Lambda_w(S_w, W):
    """ See Fig. 2 on p.537 of Ioffe 2006. """
    return (W.T@ S_w)@ W

def calc_n_avg(Y):
    """ This is the \"hack\" suggested in Fig 2 on p.537 of Ioffe 2006. """
    unique = np.unique(Y)
    return len(Y) / unique.shape[0]

def calc_A(n_avg, Lambda_w, W):
    """ See Fig. 2 on p.537 of Ioffe 2006. """
    Lambda_w_diagonal = Lambda_w.diagonal()  # Should be diagonal matrix.
    inv_W_T = np.linalg.inv(W.T)
    return inv_W_T * (n_avg / (n_avg - 1) * Lambda_w_diagonal) ** .5


def calc_Psi(Lambda_w, Lambda_b, n_avg):
    """ See Fig. 2 on p.537 of Ioffe 2006. """
    Lambda_w_diagonal = Lambda_w.diagonal()  # Should be diagonal matrix.
    Lambda_b_diagonal = Lambda_b.diagonal()  # Should be diagonal matrix.
    Psi = (n_avg - 1) / n_avg * Lambda_b_diagonal / Lambda_w_diagonal
    Psi -= 1 / n_avg
    Psi[Psi <= 0] = 0

    return np.diag(Psi)

def get_relevant_U_dims(Psi):
    """ See Fig. 2 on p.537 of Ioffe 2006. """
    relevant_dims = np.squeeze(np.argwhere(Psi.diagonal() != 0))
    if relevant_dims.shape == ():
        relevant_dims = relevant_dims.reshape(1,)
    return relevant_dims

def optimize_maximum_likelihood(X, labels):
    """ Performs the optimization in Fig. 2 of p.537 of Ioffe 2006.

    DESCRIPTION
     - The main model parameters are `m`, `A`, and `Psi`.
     - However, to improve the performance (speed and numerical stability)
        of the plda.Model object,
        inv_A and relevant_U_dims are also returned here.

    ADDITIONAL NOTES
     Be sure to test that np.cov(X.T) is full rank before running this.

     Recall that there are 4 \"spaces\":
      'D' (data) <---> 'X' (preprocessed) <---> 'U' (latent) <---> 'U_model'

    ARGUMENTS
     X  (numpy.ndarray), shape=(n_data, n_dimensions)
       - Data in statistics format, i.e. row-wise.

     labels  (list or numpy.ndarray), length=X.shape[0]
       - Labels for the data in `X`.
       - Must be sorted in the same order as `X`.

    RETURNS
     m  (numpy.ndarray), shape=X.shape[-1]
       - The mean of the row vectors in X.
       - This is the prior mean fitted via maximum likelihood.

     A  (numpy.ndarray), shape=(X.shape[-1], X.shape[-1])
       - Transformation from X space to the latent U space.

     Psi  (numpy.ndarray), shape=(X.shape[-1], X.shape[-1])
       - The covariance matrix of the prior distribution on
          the category means in U space.

     relevant_U_dims  (numpy.ndarray), shape=(len(np.unique(labels)) - 1,)
       - The \"effective\" latent dimensions,
          i.e. the ones that are actually used by the model.

     inv_A  (numpy.ndarray), shape=A.shape
       - The inverse of the matrix A.
       - Transformation from the latent U space to the X space.
    """
    assert len(X.shape) == 2
    assert X.shape[0] == len(labels)

    m = X.mean(axis=0)

    S_b, S_w = calc_scatter_matrices(X, labels)
    W = calc_W(S_b, S_w)

    Lambda_b = calc_Lambda_b(S_b, W)
    Lambda_w = calc_Lambda_w(S_w, W)
    n_avg = calc_n_avg(labels)

    A = calc_A(n_avg, Lambda_w, W)
    inv_A = np.linalg.inv(A)

    Psi = calc_Psi(Lambda_w, Lambda_b, n_avg)
    relevant_U_dims = get_relevant_U_dims(Psi)

    return m, A, Psi, relevant_U_dims, inv_A''' #TODO remove

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
        # self.checked_xvec_latent_space = [] #TODO remove

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

        # self.checked_xvec_latent_space = np.zeros_like(self.checked_xvec) #TODO remove
        # A_inv = np.linalg.pinv(plda.F) #TODO maybe sigma within wurzel
        # # for i, xvec in enumerate(self.checked_xvec):
        # #     x = np.array([(xvec-plda.mean)]).T
        # #     self.checked_xvec_latent_space[i,:] = A_inv @ x
        # self.checked_xvec_latent_space = (A_inv @ self.checked_xvec.T).T

    def calc_eer_mindcf(self):
        """
        Calculate the EER and minDCF.
        """
        self.eer, self.eer_th = EER(torch.tensor(self.positive_scores), torch.tensor(self.negative_scores))
        self.min_dcf, self.min_dcf_th = minDCF(torch.tensor(self.positive_scores), torch.tensor(self.negative_scores), p_target=0.5)

    def plot_images(self, writer, plda):#, train_xvec, train_label): #TODO remove plda
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
        split_label = np.array(split_label) #TODO utilize split in plotting
        '''#TODO remove later
        self.checked_xvec = split_xvec[0]
        self.checked_label = split_label[0]

        # F_inv = np.linalg.pinv(plda.F)
        # data_temp = (self.checked_xvec-plda.mean) @ F_inv.T
        
        m, A, Psi, relevant_U_dims, inv_A = optimize_maximum_likelihood(self.checked_xvec, self.checked_label) #TODO change this! plotting works but is wrong

        data_temp=np.matmul(self.checked_xvec - m, inv_A.T) #TODO calculate A from train data then transform before and after data with A and after data also with plda.F
        self.checked_xvec_latent_space=data_temp[..., relevant_U_dims]

        if(False):
            self.checked_xvec_latent_space = np.zeros_like(self.checked_xvec)
            A_inv = np.linalg.pinv(plda.F)
            self.checked_xvec_latent_space = (data_temp-plda.mean) @ A_inv.T
        #TODO remove later'''

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
            sns.scatterplot(x='x', y='y', hue='label', palette='bright', data=df, ax=ax, s=80)#sns.color_palette("hls", 40)
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

        # print('scatter_plot_LDA_after_training') #TODO remove
        # new_stat = pc.get_x_vec_stat(self.checked_xvec_latent_space, self.checked_label)
        # new_stat = pc.lda(new_stat)
        # generate_scatter_plot(new_stat.stat1[:, 0], new_stat.stat1[:, 1], self.checked_label, 'scatter_plot_LDA_after_training')
        # writer.add_figure('scatter_plot_LDA_after_training', plt.gcf())

        # print('scatter_plot_PCA_after_training')
        # pca = sklearn.decomposition.PCA(n_components=2)
        # pca_result = pca.fit_transform(sklearn.preprocessing.StandardScaler().fit_transform(self.checked_xvec_latent_space))
        # generate_scatter_plot(pca_result[:,0], pca_result[:,1], self.checked_label, 'scatter_plot_PCA_after_training')
        # writer.add_figure('scatter_plot_PCA_after_training', plt.gcf())

        # print('scatter_plot_TSNE_after_training')
        # tsne = TSNE(2)
        # tsne_result = tsne.fit_transform(self.checked_xvec_latent_space)
        # generate_scatter_plot(tsne_result[:,0], tsne_result[:,1], self.checked_label, 'scatter_plot_TSNE_after_training')
        # writer.add_figure('scatter_plot_TSNE_after_training', plt.gcf())
