from speechbrain.processing.PLDA_LDA import *

import random, numpy

dim, N = 10, 100
n_spkrs = 10
train_xv = numpy.array([[random.randrange(i*100,(i+1)*100,1)/10000 for j in range(dim)] for i in range(N)]) #numpy.random.rand(N, dim)
md = ['md'+str(int(i/10)) for i in range(N)] #['md'+str(random.randrange(1,n_spkrs,1)) for i in range(N)]
modelset = numpy.array(md, dtype="|O")
sg = ['sg'+str(i) for i in range(N)]
segset = numpy.array(sg, dtype="|O")
s = numpy.array([None] * N)
stat0 = numpy.array([[1.0]]* N)
xvectors_stat = StatObject_SB(modelset=modelset, segset=segset, start=s, stop=s, stat0=stat0, stat1=train_xv)

plda = PLDA(rank_f=5)
plda.plda(xvectors_stat)

en_N = 20
en_xv = numpy.array([[random.randrange(i*100,(i+1)*100,1)/10000 for j in range(dim)] for i in range(N)]) #numpy.random.rand(en_N, dim)
en_sgs = ['en'+str(i) for i in range(en_N)]
en_sets = numpy.array(en_sgs, dtype="|O")
en_s = numpy.array([None] * en_N)
en_stat0 = numpy.array([[1.0]]* en_N)
en_stat = StatObject_SB(modelset=en_sets, segset=en_sets, start=en_s, stop=en_s, stat0=en_stat0, stat1=en_xv)
te_N = 30
te_xv = numpy.array([[random.randrange(i*100,(i+1)*100,1)/10000 for j in range(dim)] for i in range(N)]) #numpy.random.rand(te_N, dim)
te_sgs = ['te'+str(i) for i in range(te_N)]
te_sets = numpy.array(te_sgs, dtype="|O")
te_s = numpy.array([None] * te_N)
te_stat0 = numpy.array([[1.0]]* te_N)
te_stat = StatObject_SB(modelset=te_sets, segset=te_sets, start=te_s, stop=te_s, stat0=te_stat0, stat1=te_xv)
ndx = Ndx(models=en_sets, testsegs=te_sets)

scores_plda = fast_PLDA_scoring(en_stat, te_stat, ndx, plda.mean, plda.F, plda.Sigma)
print (scores_plda.scoremat.shape)












import torch

def EER(positive_scores, negative_scores):
    """Computes the EER (and its threshold).
    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.
    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_eer, threshold = EER(positive_scores, negative_scores)
    >>> val_eer
    0.0
    """

    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    # Computing False Rejection Rate (miss detection)
    positive_scores = torch.cat(
        len(thresholds) * [positive_scores.unsqueeze(0)]
    )
    pos_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    FRR = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[1]
    del positive_scores
    del pos_scores_threshold

    # Computing False Acceptance Rate (false alarm)
    negative_scores = torch.cat(
        len(thresholds) * [negative_scores.unsqueeze(0)]
    )
    neg_scores_threshold = negative_scores.transpose(0, 1) > thresholds
    FAR = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[1]
    del negative_scores
    del neg_scores_threshold

    # Finding the threshold for EER
    min_index = (FAR - FRR).abs().argmin()

    # It is possible that eer != fpr != fnr. We return (FAR  + FRR) / 2 as EER.
    EER = (FAR[min_index] + FRR[min_index]) / 2

    return float(EER), float(thresholds[min_index])

positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.4])
negative_scores = torch.tensor([0.5, 0.3, 0.2, 0.1])
val_eer, threshold = EER(positive_scores, negative_scores)
print(val_eer)


# def verification_performance(scorpes_plda):
#     """Computes the Equal Error Rate give the PLDA scores"""

#     # Create ids, labels, and scoring list for EER evaluation
#     ids = []
#     labels = []
#     positive_scores = []
#     negative_scores = []
#     for line in open(veri_file_path):
#         lab = int(line.split(" ")[0].rstrip().split(".")[0].strip())
#         enrol_id = line.split(" ")[1].rstrip().split(".")[0].strip()
#         test_id = line.split(" ")[2].rstrip().split(".")[0].strip()

#         # Assuming enrol_id and test_id are unique
#         i = int(numpy.where(scores_plda.modelset == enrol_id)[0][0])
#         j = int(numpy.where(scores_plda.segset == test_id)[0][0])

#         s = float(scores_plda.scoremat[i, j])
#         labels.append(lab)
#         ids.append(enrol_id + "<>" + test_id)
#         if lab == 1:
#             positive_scores.append(s)
#         else:
#             negative_scores.append(s)

#     # Clean variable
#     del scores_plda

#     # Final EER computation
#     eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
#     min_dcf, th = minDCF(
#         torch.tensor(positive_scores), torch.tensor(negative_scores)
#     )
#     return eer, min_dcf



# eer, min_dcf = verification_performance(scores_plda)

# print(eer)
# print(min_dcf)











from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="testlogs/lightning_logs/images/5")

img = numpy.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
img[0] = numpy.array([scores_plda.scoremat])
writer.add_image('scores', img, 0)

img = numpy.zeros((3, scores_plda.scoremask.shape[0], scores_plda.scoremask.shape[1]))
img[0] = numpy.array([scores_plda.scoremask])
writer.add_image('mask', img, 0)

img = numpy.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
img[0] = numpy.array([scores_plda.scoremat*scores_plda.scoremask])
writer.add_image('masked_scores', img, 0)

writer.close()