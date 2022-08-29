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

#TODO download test files and example results and test with them how to get the pla classifier working