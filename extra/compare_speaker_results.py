import pandas as pd
import plda_classifier as pc
import numpy as np

xvec = pd.read_csv('x_vectors/x_vector_test_v1_1.csv')
xvec.columns = ['index', 'id', 'label', 'xvector']
plda = pc.load_plda('plda/plda_v1_1.pickle')

x_id_test = np.array(xvec.iloc[:, 1])
x_vec_test = np.array([np.array(x_vec[1:-1].split(), dtype=np.float64) for x_vec in xvec.iloc[:, 3]])

en_stat = pc.get_x_vec_stat(x_vec_test, x_id_test)
te_stat = pc.get_x_vec_stat(x_vec_test, x_id_test)

scores = pc.plda_scores(plda, en_stat, te_stat)
pos_scores_mask = np.zeros_like(scores.scoremat)
neg_scores_mask = np.zeros_like(scores.scoremat)
for i, m in enumerate(scores.modelset):
    for j, s in enumerate(scores.segset):
        if(m.split(".")[0].split("/")[0] == s.split(".")[0].split("/")[0]):
            pos_scores_mask[i,j] = 1
        else:
            neg_scores_mask[i,j] = 1

pos_scores_mask = np.array(pos_scores_mask, dtype=int)
neg_scores_mask = np.array(neg_scores_mask, dtype=int)

print('remove false samples:')
print('index: 4053 id:id10304/HTL8iLI75TY/00004.wav')
scores.scoremat[:,4053].fill(0)
scores.scoremat[4053,:].fill(0)

pos_scores = scores.scoremat*pos_scores_mask
neg_scores = scores.scoremat*neg_scores_mask

print('highest score: ', np.max(pos_scores))
print('lowest score: ', np.min(neg_scores))

print('highest false pos score: ', np.max(neg_scores))
print('lowest false neg score: ', np.min(pos_scores))

print('false pos')

for n in range(200):
    if(n%2 == 0):
        false_pos = np.max(neg_scores)
        #print(false_pos)
        index = np.where(neg_scores == false_pos)
        #print(index)
        speakers1 = []
        speakers2 = []
        for i,j in zip(index[0], index[1]):
            speakers1.append(scores.modelset[i])
            speakers2.append(scores.segset[j])
            neg_scores[i,j] = 0
        #print(speakers1)
        #print(speakers2)
        print(n/2,' id1:',speakers1,' id2:',speakers2,' index:(',index[0],',',index[1],') LLR Score:',false_pos)

print('false neg')

for n in range(200):
    if(n%2 == 0):
        miss = np.min(pos_scores)
        #print(miss)
        index = np.where(pos_scores == miss)
        #print(index)
        speakers1 = []
        speakers2 = []
        for i,j in zip(index[0], index[1]):
            speakers1.append(scores.modelset[i])
            speakers2.append(scores.segset[j])
            pos_scores[i,j] = 0
        #print(speakers1)
        #print(speakers2)
        print(n/2,' id1:',speakers1,' id2:',speakers2,' index:(',index[0],',',index[1],') LLR Score:',miss)

print('done')