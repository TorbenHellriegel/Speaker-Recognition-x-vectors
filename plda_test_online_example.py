import os
import pickle
import numpy
from numpy import linalg as LA
from speechbrain.processing.PLDA_LDA import StatObject_SB  # noqa F401
from speechbrain.processing.PLDA_LDA import PLDA
from speechbrain.processing.PLDA_LDA import Ndx
from speechbrain.processing.PLDA_LDA import fast_PLDA_scoring


# Load params file
experiment_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = "../../pldatest/august_speechbrain-2933b2a5a83662e9c554ba15e94f4a9ad31527bc/samples/plda_xvect_samples"
data_folder = os.path.abspath(experiment_dir + data_folder)

# Xvectors stored as StatObject_SB
train_file = data_folder + "/train_stat_xvect.pkl"
enrol_file = data_folder + "/enrol_stat_xvect.pkl"
test_file = data_folder + "/test_stat_xvect.pkl"
scores_file = data_folder + "/expected_plda_scores.pkl"

# Load Train
with open(train_file, "rb") as input:
    train_obj = pickle.load(input)

# Load Enrol
with open(enrol_file, "rb") as input:
    enrol_obj = pickle.load(input)

# Load Test
with open(test_file, "rb") as input:
    test_obj = pickle.load(input)

print("Training PLDA...")
plda = PLDA()
plda.plda(train_obj)

# Preparing Ndx map
models = enrol_obj.modelset
testsegs = test_obj.modelset
ndx_obj = Ndx(models=models, testsegs=testsegs)

# PLDA scoring between enrol and test
scores_plda = fast_PLDA_scoring(
    enrol_obj, test_obj, ndx_obj, plda.mean, plda.F, plda.Sigma
)
print("PLDA score matrix: (Rows: Enrol, Columns: Test)")
print(scores_plda.scoremat)

with open(scores_file, "rb") as input:
    expected_score_matrix = pickle.load(input)

print("Expected scores:\n", expected_score_matrix)

# Ensuring the scores are proper (for integration test)
dif = numpy.subtract(expected_score_matrix, scores_plda.scoremat)
f_norm = LA.norm(dif, ord="fro")

# Integration test: Ensure we get same score matrix
print(f_norm < 0.1)



# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter(log_dir="testlogs/lightning_logs/images/6")

# img = numpy.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
# img[0] = numpy.array([scores_plda.scoremat])
# writer.add_image('scores', img, 0)

# img = numpy.zeros((3, expected_score_matrix.shape[0], expected_score_matrix.shape[1]))
# img[1] = numpy.array([expected_score_matrix])
# writer.add_image('scores expected', img, 0)

# img = numpy.zeros((3, scores_plda.scoremask.shape[0], scores_plda.scoremask.shape[1]))
# img[0] = numpy.array([scores_plda.scoremask])
# writer.add_image('mask', img, 0)

# img = numpy.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
# img[0] = numpy.array([scores_plda.scoremat*scores_plda.scoremask])
# writer.add_image('masked_scores', img, 0)

# img = numpy.zeros((3, expected_score_matrix.shape[0], expected_score_matrix.shape[1]))
# img[1] = numpy.array([expected_score_matrix*scores_plda.scoremask])
# writer.add_image('masked_scores expected', img, 0)



# img = numpy.zeros((3, train_obj.stat1.shape[0], train_obj.stat1.shape[1]))
# img[2] = numpy.array([train_obj.stat1])
# writer.add_image('masked_scores expected', img, 0)

# img = numpy.zeros((3, enrol_obj.stat1.shape[0], enrol_obj.stat1.shape[1]))
# img[2] = numpy.array([enrol_obj.stat1])
# writer.add_image('masked_scores expected', img, 0)

# img = numpy.zeros((3, test_obj.stat1.shape[0], test_obj.stat1.shape[1]))
# img[2] = numpy.array([test_obj.stat1])
# writer.add_image('masked_scores expected', img, 0)

# writer.close()