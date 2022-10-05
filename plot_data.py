from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if(False):
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

    plt.figure(figsize=(16,12), layout='constrained')
    plt.rcParams['font.size'] = '28'
    plt.plot(train_acc_x_full, train_acc_y_full, label='train_acc')
    plt.plot(val_acc_x_full, val_acc_y_full, label='val_acc', linewidth=3)
    plt.xlabel('steps')
    plt.ylabel('accuracy')
    plt.title("Training and Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig('train_val_acc.svg')

    plt.figure(figsize=(16,12), layout='constrained')
    plt.rcParams['font.size'] = '28'
    plt.plot(train_loss_x_full, train_loss_y_full, label='train_loss')
    plt.plot(val_loss_x_full, val_loss_y_full, label='val_loss', linewidth=3)
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig('train_val_loss.svg')

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

    colors = np.where(pairs==1, 'r', 'c')

    # y_pos = y_pos[:10]
    # performance = performance[:10]
    # colors = colors[:10]

    ax.barh(y_pos, performance, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel('LLR Score')
    ax.set_title('False Positive Scores')

    #plt.show()
    plt.savefig('False_Positive_Scores.svg')

    # fig, ax = plt.subplots()

    # colors = np.where(pairs==2, 'r', 'b')

    # ax.barh(y_pos, performance*-1, color=colors)
    # ax.invert_yaxis()
    # ax.set_xlabel('LLR Score')
    # ax.set_title('False Positive Scores')

    # plt.show()

    # fig, ax = plt.subplots()

    # colors = np.where(pairs==3, 'r', 'b')

    # ax.barh(y_pos, performance*-1, color=colors)
    # ax.invert_yaxis()
    # ax.set_xlabel('LLR Score')
    # ax.set_title('False Positive Scores')

    # plt.show()

    # fig, ax = plt.subplots()

    # colors = np.where(pairs==1, 'r', 'b')
    # colors = np.where(pairs==2, 'r', colors)
    # colors = np.where(pairs==3, 'r', colors)

    # ax.barh(y_pos, performance*-1, color=colors)
    # ax.invert_yaxis()
    # ax.set_xlabel('LLR Score')
    # ax.set_title('False Positive Scores')

    # plt.show()