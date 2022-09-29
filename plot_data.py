import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

plt.figure(figsize=(8,6), layout='constrained')
plt.plot(train_acc_x_full, train_acc_y_full, label='train_acc')
plt.plot(val_acc_x_full, val_acc_y_full, label='val_acc', linewidth=3)
plt.xlabel('steps')
plt.ylabel('accuracy')
plt.title("Training and Validation Accuracy")
plt.grid(True)
plt.legend()
plt.savefig('train_val_acc.svg')

plt.figure(figsize=(8,6), layout='constrained')
plt.plot(train_loss_x_full, train_loss_y_full, label='train_loss')
plt.plot(val_loss_x_full, val_loss_y_full, label='val_loss', linewidth=3)
plt.xlabel('steps')
plt.ylabel('loss')
plt.title("Training and Validation Loss")
plt.grid(True)
plt.legend()
plt.savefig('train_val_loss.svg')