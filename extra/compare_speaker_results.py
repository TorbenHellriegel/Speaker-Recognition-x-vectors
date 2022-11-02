import pandas as pd
import plda_classifier as pc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

# print('remove false samples:')
# print('index: 4053 id:id10304/HTL8iLI75TY/00004.wav')
# scores.scoremat[:,4053].fill(0)
# scores.scoremat[4053,:].fill(0)

pos_scores = scores.scoremat*pos_scores_mask
neg_scores = scores.scoremat*neg_scores_mask

if(True):
    plt.rcParams.update({'font.size': 20})

    pos_scores_mask = np.array(pos_scores_mask, dtype=bool)
    neg_scores_mask = np.array(neg_scores_mask, dtype=bool)

    pos_scores = scores.scoremat[pos_scores_mask]
    neg_scores = scores.scoremat[neg_scores_mask]

    print('All scores')
    print('highest score: ', np.max(scores.scoremat))
    print('lowest score: ', np.min(scores.scoremat))
    print('score mean: ', np.mean(scores.scoremat))
    print('score variance:', np.var(scores.scoremat))

    print('Positive scores')
    print('highest score: ', np.max(pos_scores))
    print('lowest score: ', np.min(pos_scores))
    print('score mean: ', np.mean(pos_scores))
    print('score variance:', np.var(pos_scores))

    print('Negative scores')
    print('highest score: ', np.max(neg_scores))
    print('lowest score: ', np.min(neg_scores))
    print('score mean: ', np.mean(neg_scores))
    print('score variance:', np.var(neg_scores))
    
    from pytorch_lightning import loggers as pl_loggers
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="testlogs/")
    writer = tb_logger.experiment
    
    sns.set_style("white")

    # Import data
    x1 = np.rint(scores.scoremat).flatten()
    x2 = np.rint(pos_scores)
    x3 = np.rint(neg_scores)

    # Plot
    # kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})

    plt.figure(figsize=(10,7))#figsize=(10,7), dpi= 80)
    sns.distplot(x1, color="blue", label="All Scores")#, **kwargs)
    plt.xlim(-78,88)
    plt.xlabel('Score')
    plt.title('All Scores')
    plt.legend()
    
    writer.add_figure('score_dist_all', plt.gcf())

    plt.figure(figsize=(10,7))#figsize=(10,7), dpi= 80)
    sns.distplot(x2, color="green", label="Positive Scores")#, **kwargs)
    sns.distplot(x3, color="red", label="Negative Scores")#, **kwargs)
    plt.xlim(-78,88)
    plt.xlabel('Score')
    plt.title('Positive and Negative Scores')
    plt.legend()
    
    writer.add_figure('score_dist_pos_neg', plt.gcf())

if(False):
    print('false pos')

    pairs = []
    num = 0
    num_pairs = []
    performance = []
    for n in range(200):
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
        if(n%2 == 0):
            print(n/2,' id1:',speakers1,' id2:',speakers2,' index:(',index[0],',',index[1],') LLR Score:',false_pos)
            performance.append(false_pos)
            pair = np.sort([int(speakers1[0].split('/')[0][2:]),int(speakers2[0].split('/')[0][2:])])
            pair = pair[0], pair[1]
            if(pair in pairs):
                num_pairs.append(pairs.index(pair)+1)
            else:
                pairs.append(pair)
                num += 1
                num_pairs.append(num)
    print(pairs)
    print(num_pairs)
    print(performance)


    print('false neg')

    for n in range(200):
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
        if(n%2 == 0):
            print(n/2,' id1:',speakers1,' id2:',speakers2,' index:(',index[0],',',index[1],') LLR Score:',miss)

print('done')