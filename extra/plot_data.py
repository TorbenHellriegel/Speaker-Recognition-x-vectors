import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if(True):
    tag = np.array(['train_step_acc', 'train_step_loss', 'val_step_acc', 'val_step_loss'])
    ver = np.array(['v1', 'v1_1', 'v1_2', 'v1_3', 'v1_4', 'v1_5'])

    train_acc_x_full = []
    train_loss_x_full = []
    val_acc_x_full = []
    val_loss_x_full = []
    train_acc_y_full = []
    train_loss_y_full = []
    val_acc_y_full = []
    val_loss_y_full = []

    def get_step(l):
        if(type(l)==int):
            return l
        if(len(l) == 0):
            return 0
        else:
            return get_step(l[-1])

    for v in ver:
        train_acc_name = 'data/plot data/run-x_vector_'+v+'-tag-'+tag[0]+'.csv'
        train_loss_name = 'data/plot data/run-x_vector_'+v+'-tag-'+tag[1]+'.csv'
        val_acc_name = 'data/plot data/run-x_vector_'+v+'-tag-'+tag[2]+'.csv'
        val_loss_name = 'data/plot data/run-x_vector_'+v+'-tag-'+tag[3]+'.csv'

        train_acc_csv = pd.read_csv(train_acc_name)
        train_loss_csv = pd.read_csv(train_loss_name)
        val_acc_csv = pd.read_csv(val_acc_name)
        val_loss_csv = pd.read_csv(val_loss_name)
        
        train_acc_x = list(train_acc_csv['Step'])
        train_loss_x = list(train_loss_csv['Step'])
        val_acc_x = list(val_acc_csv['Step'])
        val_loss_x = list(val_loss_csv['Step'])
        train_acc_y = list(train_acc_csv['Value'])
        train_loss_y = list(train_loss_csv['Value'])
        val_acc_y = list(val_acc_csv['Value'])
        val_loss_y = list(val_loss_csv['Value'])

        length = val_loss_x[-1]

        train_acc_x = [x for x in train_acc_x if x <= length]
        train_loss_x = [x for x in train_loss_x if x <= length]
        train_acc_y = train_acc_y[:len(train_acc_x)]
        train_loss_y = train_loss_y[:len(train_loss_x)]

        step = get_step(val_loss_x_full)
        
        train_acc_x_full.append([x+step for x in train_acc_x])
        train_loss_x_full.append([x+step for x in train_loss_x])
        val_acc_x_full.append([x+step for x in val_acc_x])
        val_loss_x_full.append([x+step for x in val_loss_x])
        train_acc_y_full.append(train_acc_y)
        train_loss_y_full.append(train_loss_y)
        val_acc_y_full.append(val_acc_y)
        val_loss_y_full.append(val_loss_y)

    train_acc_x_full = np.concatenate(train_acc_x_full)
    train_loss_x_full = np.concatenate(train_loss_x_full)
    val_acc_x_full = np.concatenate(val_acc_x_full)
    val_loss_x_full = np.concatenate(val_loss_x_full)
    train_acc_y_full = np.concatenate(train_acc_y_full)
    train_loss_y_full = np.concatenate(train_loss_y_full)
    val_acc_y_full = np.concatenate(val_acc_y_full)
    val_loss_y_full = np.concatenate(val_loss_y_full)

    plt.rcParams.update({'font.size': 38})

    plt.figure(figsize=(16,12), layout='constrained')
    plt.plot(train_acc_x_full/783, train_acc_y_full, label='train_acc')
    plt.plot(val_acc_x_full/783, val_acc_y_full, label='val_acc', linewidth=3)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title("Training and Validation Accuracy")
    plt.grid(True)
    plt.legend(fontsize=42)
    plt.savefig('train_val_acc.pdf', bbox_inches='tight')

    plt.figure(figsize=(16,12), layout='constrained')
    plt.plot(train_loss_x_full/783, train_loss_y_full, label='train_loss')
    plt.plot(val_loss_x_full/783, val_loss_y_full, label='val_loss', linewidth=3)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend(fontsize=42)
    plt.savefig('train_val_loss.pdf', bbox_inches='tight')

if(False):
    #Data
    place = np.array([i for i in range(1, 51)])
    y_pos = np.arange(len(place))
    performance = np.array([27.705821411993384,
                    27.06155240270146,
                    26.690404570158073,
                    26.231146952337706,
                    25.59046231986114,
                    25.27378797744494,
                    24.8576751162702,
                    24.79027244517839,
                    24.73023401071766,
                    24.663622717646078,
                    24.535161600485942,
                    24.422140889130464,
                    24.342783065046646,
                    24.21828079278385,
                    24.002277854864197,
                    23.96818361438727,
                    23.96233057030901,
                    23.954883035728823,
                    23.84017266295163,
                    23.660798188249274,
                    23.359508612408433,
                    23.312426081951003,
                    23.174175524203932,
                    22.761232500399352,
                    22.657207612631844,
                    22.6534289530776,
                    22.49677408028876,
                    22.445564078001908,
                    22.34360070491359,
                    22.194878920409334,
                    22.181158290046543,
                    22.15128047036191,
                    22.03093675914478,
                    21.941824456632524,
                    21.86719116325947,
                    21.817618245094458,
                    21.640004423565834,
                    21.621708864098153,
                    21.527026544623524,
                    21.49016777594293,
                    21.487020736334678,
                    21.484317949856667,
                    21.471152457337105,
                    21.396805794776,
                    21.245468190680022,
                    20.965597638893477,
                    20.910562573750248,
                    20.894739566950825,
                    20.87018132601225,
                    20.82643461458366])
    pairs = np.array([1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    3,
                    1,
                    2,
                    3,
                    1,
                    2,
                    2,
                    1,
                    2,
                    3,
                    2,
                    4,
                    2,
                    5,
                    1,
                    1,
                    2,
                    2,
                    5,
                    2,
                    2,
                    3,
                    6,
                    1,
                    7,
                    3,
                    8,
                    2,
                    1,
                    1,
                    2,
                    1,
                    5,
                    1,
                    3,
                    2,
                    8,
                    1,
                    9,
                    1,
                    1,
                    3,
                    1,
                    8])

    fig, ax = plt.subplots()

    colors = np.where(pairs==1, 'red', 0.8)
    colors = np.where(pairs==2, 'green', colors)
    colors = np.where(pairs==3, 'orange', colors)

    red_patch = mpatches.Patch(color='red', label='286 + 303')
    green_patch = mpatches.Patch(color='green', label='274 + 301')
    blue_patch = mpatches.Patch(color='orange', label='285 + 289')
    white_patch = mpatches.Patch(color='grey', label='other')
    ax.legend(handles=[red_patch,green_patch,blue_patch,white_patch])

    ax.barh(y_pos, performance, color=colors)
    ax.invert_yaxis()
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel('LLR Score')
    ax.set_title('False Positive Scores')

    plt.show()
    #plt.savefig('False_Positive_Scores.pdf', bbox_inches='tight')

    print('done')

if(False):
    labels = [(10286, 10303), (10274, 10301), (10285, 10289), (10279, 10290), (10300, 10301), (10285, 10288), (10302, 10304), (10286, 10307), (10270, 10288), (10270, 10308), (10303, 10307), (10297, 10298), (10280, 10282), (10276, 10305), (10297, 10305), (10298, 10305), (10276, 10290), (10293, 10304), (10281, 10282)]
    num_pairs = [1, 1, 1, 2, 1, 1, 3, 1, 2, 3, 1, 2, 2, 1, 2, 3, 2, 4, 2, 5, 1, 1, 2, 2, 5, 2, 2, 3, 6, 1, 7, 3, 8, 2, 1, 1, 2, 1, 5, 1, 3, 2, 8, 1, 9, 1, 1, 3, 1, 8, 3, 10, 1, 1, 1, 1, 2, 1, 3, 1, 8, 1, 3, 1, 2, 11, 1, 3, 3, 3, 2, 1, 1, 12, 2, 5, 2, 8, 2, 1, 2, 2, 13, 14, 1, 1, 15, 14, 1, 16, 17, 8, 2, 2, 18, 14, 8, 19, 8, 11]
    scores = [27.705821411993384, 27.06155240270146, 26.690404570158073, 26.231146952337706, 25.59046231986114, 25.27378797744494, 24.8576751162702, 24.79027244517839, 24.73023401071766, 24.663622717646078, 24.535161600485942, 24.422140889130464, 24.342783065046646, 24.21828079278385, 24.002277854864197, 23.96818361438727, 23.96233057030901, 23.954883035728823, 23.84017266295163, 23.660798188249274, 23.359508612408433, 23.312426081951003, 23.174175524203932, 22.761232500399352, 22.657207612631844, 22.6534289530776, 22.49677408028876, 22.445564078001908, 22.34360070491359, 22.194878920409334, 22.181158290046543, 22.15128047036191, 22.03093675914478, 21.941824456632524, 21.86719116325947, 21.817618245094458, 21.640004423565834, 21.621708864098153, 21.527026544623524, 21.49016777594293, 21.487020736334678, 21.484317949856667, 21.471152457337105, 21.396805794776, 21.245468190680022, 20.965597638893477, 20.910562573750248, 20.894739566950825, 20.87018132601225, 20.82643461458366, 20.788735206150285, 20.76114690686577, 20.698241491287572, 20.65961983608179, 20.618075793110158, 20.583410159116227, 20.58067519020448, 20.560699672982217, 20.5365368650693, 20.467099591458584, 20.423588418792527, 20.402707115089733, 20.361296727383337, 20.323646153525125, 20.290637614417932, 20.278865692080444, 20.22161909197351, 20.2025658277199, 20.143087377119684, 19.925186541350875, 19.87105594069614, 19.78780149043257, 19.730171593029638, 19.718436555512838, 19.707233487023263, 19.6727489121931, 19.635619016121765, 19.63168358951379, 19.569183228477122, 19.558240880798124, 19.522377232545104, 19.519172663904243, 19.49284576078555, 19.473560595984427, 19.416225271692667, 19.410416528615976, 19.403025170606504, 19.387404188240975, 19.37797667291035, 19.352779189071693, 19.34339615822546, 19.31628372748908, 19.306481668603254, 19.200575804631892, 19.173743561997394, 19.171195870935186, 19.161807532026558, 19.161446555929736, 19.15642626581945, 19.15296383468037]
    sizes = []
    for i, _ in enumerate(labels):
        sizes.append(num_pairs.count(i+1))
    
    labels = ['286 and 303', '274 and 301', '285 and 289', '286 and 307', '300 and 301', 'Other speaker pairs']
    sizes = [34,24,13,8,4,17]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title('False Positive Speaker Pairs')

    #plt.show()
    plt.savefig('False_Positive_Scores.pdf', bbox_inches='tight')

if(False):
    plt.rcParams.update({'font.size': 16})

    x = np.linspace(-10, 10, 100)
    f = 1 / (1 + np.exp(-x))

    plt.axhline(color="grey")
    plt.axvline(color="grey")
    plt.plot(x, f, linewidth=2, label=r"$f(x) = \frac{1}{1 + e^{-x}}$")
    plt.xlim(-10, 10)
    plt.xlabel("x")
    plt.ylabel('y')
    plt.legend(fontsize=20)
    plt.savefig('sigmoid.pdf', bbox_inches='tight')
    plt.close()

    x = np.linspace(-10, 10, 100)
    f = np.where(x>0, x, 0)

    plt.axhline(color="grey")
    plt.axvline(color="grey")
    plt.plot(x, f, linewidth=2, label=r"$f(x) = max(0,x)$")
    plt.xlim(-10, 10)
    plt.xlabel("x")
    plt.ylabel('y')
    plt.legend(fontsize=20)
    plt.savefig('relu.pdf', bbox_inches='tight')
    plt.close()
    
    x = np.linspace(-10, 10, 100)
    f = np.where(x>0, x, 0.1*x)

    plt.axhline(color="grey")
    plt.axvline(color="grey")
    plt.plot(x, f, linewidth=2, label=r"$f(x) = max(\alpha x,x)$")
    plt.xlim(-10, 10)
    plt.xlabel("x")
    plt.ylabel('y')
    plt.legend(fontsize=20)
    plt.savefig('leakyrelu.pdf', bbox_inches='tight')
    plt.close()

    print('done')

if(False):
    plt.rcParams.update({'font.size': 14})

    x = np.linspace(-1, 30, 1000)

    a = 0.1 * np.sin(x*5)
    b = 0.3 * np.sin(x*3)
    c = 0.5 * np.sin(x*15)

    d = a+b+c
    e = a-0.5*c

    f = -25*np.log10(x*0.01 + 0.05) - 10 + d
    g = 0.04*(x-20)**2 + 8 + e

    plt.figure(figsize=(10,5))
    plt.axhline(color="grey")
    plt.axvline(color="grey")
    plt.axvline(linewidth=2, x=19.8, color="green")
    plt.plot(x, f, linewidth=2, label='train_loss')
    plt.plot(x, g, linewidth=2, label='val_loss')
    plt.xlim(-1, 30)
    plt.xlabel("x")
    plt.ylabel('y')
    plt.legend(fontsize=16)
    plt.savefig('overfitting.pdf', bbox_inches='tight')
    plt.close()
    print('done')

if(False):
    from pytorch_lightning import loggers as pl_loggers
    import seaborn as sns
    import sklearn
    import plda_classifier as pc
    from sklearn.manifold import TSNE
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="testlogs/")
    writer = tb_logger.experiment
    plt.rcParams.update({'font.size': 28})

    def generate_scatter_plot(x, y, label, plot_name, small=False):
        df = pd.DataFrame({'x': x, 'y': y, 'label': label})
        fig, ax = plt.subplots(1, layout='constrained')
        fig.set_size_inches(15.5, 12)
        if(small):
            sns.scatterplot(x='x', y='y', hue='label', palette=sns.color_palette("hls", 40), data=df, ax=ax, s=80) #use sns.color_palette("hls", 40) for 40 speakers
        else:
            sns.scatterplot(x='x', y='y', hue='label', palette='bright', data=df, ax=ax, s=80) #use sns.color_palette("hls", 40) for 40 speakers
        limx = (x.min()-5, x.max()+5)
        limy = (y.min()-5, y.max()+5)
        ax.set_xlim(limx)
        ax.set_ylim(limy)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        if(small):
            ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.0, fontsize=13)
        else:
            ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.0)
        ax.title.set_text(plot_name)

    score = pc.load_plda('plda/plda_score_v1_5_l7relu_d50.pickle')
        
    split_xvec = []
    split_label = []
    group_kfold = sklearn.model_selection.GroupKFold(n_splits=2)
    groups1234 = np.where(score.checked_label<10290, 0, 1)
    for g12, g34 in group_kfold.split(score.checked_xvec, score.checked_label, groups1234):
        x12, x34 = score.checked_xvec[g12], score.checked_xvec[g34]
        y12, y34 = score.checked_label[g12], score.checked_label[g34]
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
        
    for i, (checked_xvec, checked_label) in enumerate(zip(split_xvec, split_label)):
        print('xvec_scatter_plot_LDA'+str(i+1))
        new_stat = pc.get_x_vec_stat(checked_xvec, checked_label)
        new_stat = pc.lda(new_stat)
        generate_scatter_plot(new_stat.stat1[:, 0], new_stat.stat1[:, 1], checked_label, 'xvec_scatter_plot_LDA'+str(i+1))
        writer.add_figure('xvec_scatter_plot_LDA'+str(i+1), plt.gcf())

        print('xvec_scatter_plot_PCA'+str(i+1))
        pca = sklearn.decomposition.PCA(n_components=2)
        pca_result = pca.fit_transform(sklearn.preprocessing.StandardScaler().fit_transform(checked_xvec))
        generate_scatter_plot(pca_result[:,0], pca_result[:,1], checked_label, 'xvec_scatter_plot_PCA'+str(i+1))
        writer.add_figure('xvec_scatter_plot_PCA'+str(i+1), plt.gcf())

        print('xvec_scatter_plot_TSNE'+str(i+1))
        tsne = TSNE(2)
        tsne_result = tsne.fit_transform(checked_xvec)
        generate_scatter_plot(tsne_result[:,0], tsne_result[:,1], checked_label, 'xvec_scatter_plot_TSNE'+str(i+1))
        writer.add_figure('xvec_scatter_plot_TSNE'+str(i+1), plt.gcf())
        
    print('xvec_scatter_plot_LDA')
    new_stat = pc.get_x_vec_stat(score.checked_xvec, score.checked_label)
    new_stat = pc.lda(new_stat)
    generate_scatter_plot(new_stat.stat1[:, 0], new_stat.stat1[:, 1], score.checked_label, 'xvec_scatter_plot_LDA', small=True)
    writer.add_figure('xvec_scatter_plot_LDA', plt.gcf())

    print('xvec_scatter_plot_PCA')
    pca = sklearn.decomposition.PCA(n_components=2)
    pca_result = pca.fit_transform(sklearn.preprocessing.StandardScaler().fit_transform(score.checked_xvec))
    generate_scatter_plot(pca_result[:,0], pca_result[:,1], score.checked_label, 'xvec_scatter_plot_PCA', small=True)
    writer.add_figure('xvec_scatter_plot_PCA', plt.gcf())

    print('xvec_scatter_plot_TSNE')
    tsne = TSNE(2)
    tsne_result = tsne.fit_transform(score.checked_xvec)
    generate_scatter_plot(tsne_result[:,0], tsne_result[:,1], score.checked_label, 'xvec_scatter_plot_TSNE', small=True)
    writer.add_figure('xvec_scatter_plot_TSNE', plt.gcf())

    score = pc.load_plda('plda/plda_score_ivec_v2_d200.pickle')
        
    split_xvec = []
    split_label = []
    group_kfold = sklearn.model_selection.GroupKFold(n_splits=2)
    groups1234 = np.where(score.checked_label<10290, 0, 1)
    for g12, g34 in group_kfold.split(score.checked_xvec, score.checked_label, groups1234):
        x12, x34 = score.checked_xvec[g12], score.checked_xvec[g34]
        y12, y34 = score.checked_label[g12], score.checked_label[g34]
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
        
    for i, (checked_xvec, checked_label) in enumerate(zip(split_xvec, split_label)):
        print('ivec_scatter_plot_LDA'+str(i+1))
        new_stat = pc.get_x_vec_stat(checked_xvec, checked_label)
        new_stat = pc.lda(new_stat)
        generate_scatter_plot(new_stat.stat1[:, 0], new_stat.stat1[:, 1], checked_label, 'ivec_scatter_plot_LDA'+str(i+1))
        writer.add_figure('ivec_scatter_plot_LDA'+str(i+1), plt.gcf())

        print('ivec_scatter_plot_PCA'+str(i+1))
        pca = sklearn.decomposition.PCA(n_components=2)
        pca_result = pca.fit_transform(sklearn.preprocessing.StandardScaler().fit_transform(checked_xvec))
        generate_scatter_plot(pca_result[:,0], pca_result[:,1], checked_label, 'ivec_scatter_plot_PCA'+str(i+1))
        writer.add_figure('ivec_scatter_plot_PCA'+str(i+1), plt.gcf())

        print('ivec_scatter_plot_TSNE'+str(i+1))
        tsne = TSNE(2)
        tsne_result = tsne.fit_transform(checked_xvec)
        generate_scatter_plot(tsne_result[:,0], tsne_result[:,1], checked_label, 'ivec_scatter_plot_TSNE'+str(i+1))
        writer.add_figure('ivec_scatter_plot_TSNE'+str(i+1), plt.gcf())
        
    print('ivec_scatter_plot_LDA')
    new_stat = pc.get_x_vec_stat(score.checked_xvec, score.checked_label)
    new_stat = pc.lda(new_stat)
    generate_scatter_plot(new_stat.stat1[:, 0], new_stat.stat1[:, 1], score.checked_label, 'ivec_scatter_plot_LDA', small=True)
    writer.add_figure('ivec_scatter_plot_LDA', plt.gcf())

    print('ivec_scatter_plot_PCA')
    pca = sklearn.decomposition.PCA(n_components=2)
    pca_result = pca.fit_transform(sklearn.preprocessing.StandardScaler().fit_transform(score.checked_xvec))
    generate_scatter_plot(pca_result[:,0], pca_result[:,1], score.checked_label, 'ivec_scatter_plot_PCA', small=True)
    writer.add_figure('ivec_scatter_plot_PCA', plt.gcf())

    print('ivec_scatter_plot_TSNE')
    tsne = TSNE(2)
    tsne_result = tsne.fit_transform(score.checked_xvec)
    generate_scatter_plot(tsne_result[:,0], tsne_result[:,1], score.checked_label, 'ivec_scatter_plot_TSNE', small=True)
    writer.add_figure('ivec_scatter_plot_TSNE', plt.gcf())