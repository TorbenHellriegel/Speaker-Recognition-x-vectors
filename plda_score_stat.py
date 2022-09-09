import matplotlib as plt
import numpy as np
import torch

import plda_classifier as pc


class plda_score_stat_object():
    def __init__(self, x_vectors_test, plda_scores):
        self.x_vectors_test = self.x_vectors_test
        self.x_id_test = np.array(x_vectors_test.iloc[:, 1])
        self.x_vec_test = np.array([np.array(x_vec[1:-1].split(), dtype=np.float64) for x_vec in x_vectors_test.iloc[:, 3]])

        self.en_stat = pc.get_enroll_x_vec(self.x_vec_test, self.x_id_test)
        self.te_stat = pc.get_test_x_vec(self.x_vec_test, self.x_id_test)

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

            try:
                i = int(np.where(self.plda_scores.modelset == enrol_id)[0][0])
                self.en_xv.append(np.array(self.x_vectors_test.loc[self.x_vectors_test['id'] == enrol_id, 'xvector'].item()[1:-1].split(), dtype=np.float64))
                self.en_label.append(int(enrol_id.split(".")[0].split("/")[0][2:]))

                j = int(np.where(self.plda_scores.segset == test_id)[0][0])
                self.te_xv.append(np.array(self.x_vectors_test.loc[self.x_vectors_test['id'] == test_id, 'xvector'].item()[1:-1].split(), dtype=np.float64))
                self.te_label.append(int(test_id.split(".")[0].split("/")[0][2:]))

                total_matches += 1
            except:
                num_failed_matches += 1
            else:
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

    def plot_images(self, writer):

        self.plda_scores.scoremat -= np.min(self.plda_scores.scoremat)
        self.plda_scores.scoremat /= np.max(self.plda_scores.scoremat)

        print('generating images for tensorboard')
        img = np.zeros((3, self.plda_scores.scoremask.shape[0], self.plda_scores.scoremask.shape[1]))
        img[0] = np.array([self.plda_scores.scoremask])
        img[1] = np.array([self.plda_scores.scoremask])
        img[2] = np.array([self.plda_scores.scoremask])
        writer.add_image('score_mask', img, 0)

        img = np.zeros((3, self.plda_scores.scoremat.shape[0], self.plda_scores.scoremat.shape[1]))
        img[0] = np.array([self.plda_scores.scoremat])
        img[1] = np.array([self.plda_scores.scoremat])
        img[2] = np.array([self.plda_scores.scoremat])
        writer.add_image('score_matrix', img, 0)

        img = np.zeros((3, self.plda_scores.scoremat.shape[0], self.plda_scores.scoremat.shape[1]))
        img[1] = np.array([self.positive_scores_mask])
        img[0] = np.array([self.negative_scores_mask])
        writer.add_image('ground_truth', img, 0)

        img = np.zeros((3, self.plda_scores.scoremat.shape[0], self.plda_scores.scoremat.shape[1]))
        img[1] = np.array([self.plda_scores.scoremat*self.positive_scores_mask])
        img[0] = np.array([self.plda_scores.scoremat*self.negative_scores_mask])
        writer.add_image('ground_truth_scores', img, 0)

        # checked_values_map = self.positive_scores_mask + self.negative_scores_mask
        # positive_prediction_mask = np.zeros_like(self.plda_scores.scoremat)
        # negative_prediction_mask = np.zeros_like(self.plda_scores.scoremat)
        # for i, self in np.ndenumerate(self.plda_scores.scoremat):
        #     if(self >= eer_th):
        #         positive_prediction_mask[i] = 1
        #     else:
        #         negative_prediction_mask[i] = 1

        # checked_values_map = self.positive_scores_mask + self.negative_scores_mask
        # eer_mask = np.zeros_like(self.plda_scores.scoremat)
        # eer_mask_inv = np.zeros_like(self.plda_scores.scoremat)
        # min_dcf_mask = np.zeros_like(self.plda_scores.scoremat)
        # min_dcf_mask_inv = np.zeros_like(self.plda_scores.scoremat)
        # for i, self in np.ndenumerate(self.plda_scores.scoremat):
        #     if(self >= eer_th):
        #         eer_mask[i] = 1
        #     else:
        #         eer_mask_inv[i] = 1
        #     if(self >= min_dcf):
        #         min_dcf_mask[i] = 1
        #     else:
        #         min_dcf_mask_inv[i] = 1
        
        checked_values_map = self.positive_scores_mask + self.negative_scores_mask
        checked_values = checked_values_map * self.plda_scores.scoremat

        eer_prediction_positive = np.where(checked_values >= self.eer_th, 1, 0)
        eer_prediction_negative = np.where(checked_values < self.eer_th, 1, 0)
        min_dcf_prediction_positive = np.where(checked_values >= self.min_dcf_th, 1, 0)
        min_dcf_prediction_negative = np.where(checked_values < self.min_dcf_th, 1, 0)
        
        # img = np.zeros((3, self.plda_scores.scoremat.shape[0], self.plda_scores.scoremat.shape[1]))
        # img[1] = np.array([positive_prediction_mask*checked_values_map])
        # img[0] = np.array([negative_prediction_mask*checked_values_map])
        # writer.add_image('prediction', img, 0)
    
        img = np.ones((3, self.plda_scores.scoremat.shape[0], self.plda_scores.scoremat.shape[1]*2+5))
        img[1,:,:checked_values.shape[1]] = eer_prediction_positive * self.plda_scores.scoremat
        img[0,:,:checked_values.shape[1]] = eer_prediction_negative * self.plda_scores.scoremat
        img[1,:,-checked_values.shape[1]:] = min_dcf_prediction_positive * self.plda_scores.scoremat
        img[0,:,-checked_values.shape[1]:] = min_dcf_prediction_negative * self.plda_scores.scoremat
        img[2,:,:checked_values.shape[1]] = 0
        img[2,:,-checked_values.shape[1]:] = 0
        writer.add_image('prediction_scores_eer_min_dcf', img, 0)
        
        # img = np.zeros((3, self.plda_scores.scoremat.shape[0], self.plda_scores.scoremat.shape[1]))
        # img[1] = np.array([self.plda_scores.scoremat*positive_prediction_mask*checked_values_map])
        # img[0] = np.array([self.plda_scores.scoremat*negative_prediction_mask*checked_values_map])
        # writer.add_image('prediction_scores', img, 0)
    
        img = np.ones((3, self.plda_scores.scoremat.shape[0], self.plda_scores.scoremat.shape[1]*2+5))
        img[1,:,:checked_values.shape[1]] = eer_prediction_positive
        img[0,:,:checked_values.shape[1]] = eer_prediction_negative
        img[1,:,-checked_values.shape[1]:] = min_dcf_prediction_positive
        img[0,:,-checked_values.shape[1]:] = min_dcf_prediction_negative
        img[2,:,:checked_values.shape[1]] = 0
        img[2,:,-checked_values.shape[1]:] = 0
        writer.add_image('prediction_eer_min_dcf', img, 0)
        
        # img = np.zeros((3, self.plda_scores.scoremat.shape[0], self.plda_scores.scoremat.shape[1]))
        # img[1] = np.array([self.positive_scores_mask*positive_prediction_mask])
        # img[0] = np.array([self.negative_scores_mask*negative_prediction_mask])
        # writer.add_image('correct_prediction', img, 0)
    
        img = np.ones((3, self.plda_scores.scoremat.shape[0], self.plda_scores.scoremat.shape[1]*2+5))
        img[1,:,:checked_values.shape[1]] = eer_prediction_positive * self.positive_scores_mask
        img[0,:,:checked_values.shape[1]] = eer_prediction_negative * self.negative_scores_mask
        img[1,:,-checked_values.shape[1]:] = min_dcf_prediction_positive * self.positive_scores_mask
        img[0,:,-checked_values.shape[1]:] = min_dcf_prediction_negative * self.negative_scores_mask
        img[2,:,:checked_values.shape[1]] = 0
        img[2,:,-checked_values.shape[1]:] = 0
        writer.add_image('correct_prediction_eer_min_dcf', img, 0)
        
        # img = np.zeros((3, self.plda_scores.scoremat.shape[0], self.plda_scores.scoremat.shape[1]))
        # img[1] = np.array([self.positive_scores_mask*negative_prediction_mask])
        # img[0] = np.array([self.negative_scores_mask*positive_prediction_mask])
        # writer.add_image('false_prediction', img, 0)
    
        img = np.ones((3, self.plda_scores.scoremat.shape[0], self.plda_scores.scoremat.shape[1]*2+5))
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

        en_stat = pc.get_enroll_x_vec(self.en_xv, self.en_label)
        new_stat_obj = pc.lda(en_stat)
        x, y, c = get_scatter_plot_data(new_stat_obj)
        x_sum.append(x)
        y_sum.append(y)
        c_sum.append(c)
        axs[0, 0].scatter(x, y, c=c)

        te_stat = pc.get_test_x_vec(self.te_xv, self.te_label)
        new_stat_obj = pc.lda(te_stat)
        x, y, c = get_scatter_plot_data(new_stat_obj)
        x_sum.append(x)
        y_sum.append(y)
        c_sum.append(c)
        axs[0, 1].scatter(x, y, c=c)
        
        axs[1, 0].scatter(x_sum, y_sum, c=c_sum)

        c_sum = np.array(c_sum)
        c_sum[:int(len(c_sum)/2)] = 0.3
        c_sum[int(len(c_sum)/2):] = 0.7
        
        axs[1, 1].scatter(x_sum, y_sum, c=c_sum)

        writer.add_figure('scatter_plot', plt.gcf())
