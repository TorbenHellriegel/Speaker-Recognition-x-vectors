import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
import torchmetrics
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

import plda_classifier as pc
from config import Config
from dataset import Dataset
from tdnn import TdnnLayer


class XVectorModel(pl.LightningModule):
    def __init__(self, input_size=24, hidden_size=512, num_classes=1211,
                x_vector_size=512, x_vec_extract_layer=6,
                batch_size=512, learning_rate=0.001, batch_norm=True, dropout_p=0.0,
                augmentations_per_sample=2, data_folder_path='data'):
        super().__init__()

        self.time_context_layers = nn.Sequential(
            TdnnLayer(input_size=input_size, output_size=hidden_size, context=[-2, -1, 0, 1, 2], batch_norm=batch_norm, dropout_p=dropout_p),
            TdnnLayer(input_size=hidden_size, output_size=hidden_size, context=[-2, 0, 2], batch_norm=batch_norm, dropout_p=dropout_p),
            TdnnLayer(input_size=hidden_size, output_size=hidden_size, context=[-3, 0, 3], batch_norm=batch_norm, dropout_p=dropout_p),
            TdnnLayer(input_size=hidden_size, output_size=hidden_size, batch_norm=batch_norm, dropout_p=dropout_p),
            TdnnLayer(input_size=hidden_size, output_size=1500, batch_norm=batch_norm, dropout_p=dropout_p)
        )
        self.segment_layer6 = nn.Linear(3000, x_vector_size)
        self.segment_layer7 = nn.Linear(x_vector_size, x_vector_size)
        self.output = nn.Linear(x_vector_size, num_classes)

        self.x_vec_extract_layer = x_vec_extract_layer
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.dataset = Dataset(data_folder_path=data_folder_path, augmentations_per_sample=augmentations_per_sample)
        self.accuracy = torchmetrics.Accuracy()

        self.save_hyperparameters()

    # Satistic pooling layer
    def stat_pool(self, x): #TODO replace with nn.averagepool2d
        mean = torch.mean(x, 1)
        stand_dev = torch.std(x, 1)
        out = torch.cat((mean, stand_dev), 1)
        return out
        
    def forward(self, x):
        out = self.time_context_layers(x)

        out = self.stat_pool(out) #TODO replace with nn.averagepool2d

        out = F.relu(self.segment_layer6(out))
        out = F.relu(self.segment_layer7(out))
        
        out = self.output(out)
        return out

    def extract_x_vec(self, x):
        out = self.time_context_layers.forward(x)

        out = self.stat_pool(out)

        if(self.x_vec_extract_layer == 6):
            x_vec = self.segment_layer6.forward(out)
        elif(self.x_vec_extract_layer == 7):
            out = F.relu(self.segment_layer6.forward(out))
            x_vec = self.segment_layer7.forward(out)
        else:
            x_vec = self.segment_layer6.forward(out)
            
        return x_vec

    def training_step(self, batch, batch_index):
        samples, labels, id = batch
        outputs = self(samples.float())
        loss = F.cross_entropy(outputs, labels)
        return {'loss': loss, 'train_preds': outputs, 'train_labels': labels, 'train_id': id}

    def training_step_end(self, outputs):
        self.log('train_step_loss', outputs['loss'])
        accuracy = self.accuracy(outputs['train_preds'], outputs['train_labels'])
        self.log('train_step_acc', self.accuracy)
        return {'loss': outputs['loss'], 'acc': accuracy}

    def training_epoch_end(self, outputs):
        if(self.current_epoch == 0):
            sample = torch.rand((1, 299, 24))
            self.logger.experiment.add_graph(XVectorModel(), sample)

        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def validation_step(self, batch, batch_index):
        samples, labels, id = batch
        outputs = self(samples.float())
        loss = F.cross_entropy(outputs, labels)
        return {'loss': loss, 'val_preds': outputs, 'val_labels': labels, 'val_id': id}

    def validation_step_end(self, outputs):
        self.log('val_step_loss', outputs['loss'])
        accuracy = self.accuracy(outputs['val_preds'], outputs['val_labels'])
        self.log('val_step_acc', self.accuracy)
        return {'loss': outputs['loss'], 'acc': accuracy}

    def validation_epoch_end(self, outputs):
        #TODO figure out how to plot 2in1
        todo=0
    
    def test_step(self, batch, batch_index):
        samples, labels, id = batch
        x_vecs = self.extract_x_vec(samples.float())
        return [(x_vecs, labels, id)]

    def test_epoch_end(self, test_step_outputs):
        for batch_output in test_step_outputs:
            for x_vec, label, id in batch_output:
                for x, l, i in zip(x_vec, label, id):
                    x_vector.append((i, int(l.cpu().numpy()), np.array(x.cpu().numpy(), dtype=np.float64)))
        return test_step_outputs
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        self.dataset.load_data(train=True)
        train_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)
        return train_data_loader

    def val_dataloader(self):
        self.dataset.load_data(val=True)
        val_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        return val_data_loader

    def test_dataloader(self):
        if(extract_mode == 'train'):
            self.dataset.load_data(train=True, val=True)
            test_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        if(extract_mode == 'test'):
            self.dataset.load_data(test=True)
            test_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        return test_data_loader



if __name__ == "__main__":
    # Set which parts of the code to run
    train_x_vector_model = False
    extract_x_vectors = False
    train_plda = False
    test_plda = False

    # Define model and trainer
    print('setting up model and trainer parameters')
    config = Config(num_epochs=20, checkpoint_path='lightning_logs/x_vector_v1_3/checkpoints/last.ckpt') #adjust batch size, epoch, etc. here

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./")
    early_stopping_callback = EarlyStopping(monitor="val_step_loss", mode="min")
    checkpoint_callback = ModelCheckpoint(monitor='val_step_loss', save_top_k=10, save_last=True, verbose=True)

    if(config.checkpoint_path == 'none'):
        model = XVectorModel(input_size=config.input_size, hidden_size=config.hidden_size, num_classes=config.num_classes,
                            x_vector_size=config.x_vector_size, x_vec_extract_layer=config.x_vec_extract_layer,
                            batch_size=config.batch_size, learning_rate=config.learning_rate, batch_norm=config.batch_norm, dropout_p=config.dropout_p,
                            augmentations_per_sample=config.augmentations_per_sample, data_folder_path=config.data_folder_path)
    else:
        model = XVectorModel.load_from_checkpoint(config.checkpoint_path)
    model.dataset.init_samples_and_labels()

    trainer = pl.Trainer(callbacks=[early_stopping_callback, checkpoint_callback],
                        logger=tb_logger, log_every_n_steps=1,
                        accelerator='gpu', devices=[0],# strategy='ddp',
                        max_epochs=config.num_epochs)
                        #small test adjust options: fast_dev_run=True, limit_train_batches=0.0001, limit_val_batches=0.001, limit_test_batches=0.002



    # Train the x-vector model
    if(train_x_vector_model):
        print('training x-vector model')
        if(config.checkpoint_path == 'none'):
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=config.checkpoint_path)



    # Extract the x-vectors
    if(extract_x_vectors):
        print('extracting x-vectors')
        x_vector = []
        extract_mode = 'train'
        if(train_x_vector_model):
            trainer.test(model)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv('x_vectors/x_vector_train_v1.csv')
        elif(config.checkpoint_path != 'none'):
            trainer.test(model, ckpt_path=config.checkpoint_path)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv('x_vectors/x_vector_train_v1.csv')
        else:
            print('could not extract train x-vectors')

        x_vector = []
        extract_mode = 'test'
        if(train_x_vector_model):
            trainer.test(model)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv('x_vectors/x_vector_test_v2.csv')
        elif(config.checkpoint_path != 'none'):
            trainer.test(model, ckpt_path=config.checkpoint_path)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv('x_vectors/x_vector_test_v2.csv')
        else:
            print('could not extract test x-vectors')
    


    if(train_plda):
        # Extracting the x-vectors, labels and id from the csv
        x_vectors_train = pd.read_csv('x_vectors/x_vector_train_v1.csv')
        x_id_train = np.array(x_vectors_train.iloc[:, 1])
        x_label_train = np.array(x_vectors_train.iloc[:, 2], dtype=int)
        x_vec_train = np.array([np.array(x_vec[1:-1].split(), dtype=np.float64) for x_vec in x_vectors_train.iloc[:, 3]])

        # Generate x_vec stat objects
        print('generating x_vec stat objects')
        #x_vec_train, x_label_train, x_id_train = sklearn.utils.shuffle(x_vec_train, x_label_train, x_id_train)###TODO comment out and in to compare different results in plda_test
        tr_stat = pc.get_train_x_vec(x_vec_train, x_label_train, x_id_train)

        # Training plda (or load pretrained plda)
        print('training plda')
        plda = pc.setup_plda(rank_f=config.plda_rank_f)
        plda = pc.train_plda(plda, tr_stat)
        pc.save_plda(plda, 'plda_v3')



    if(test_plda):
        print('loading x_vector data')
        # Extracting the x-vectors, labels and id from the csv
        x_vectors_test = pd.read_csv('x_vectors/x_vector_test_v1_1.csv')
        x_id_test = np.array(x_vectors_test.iloc[:, 1])
        x_label_test = np.array(x_vectors_test.iloc[:, 2], dtype=int)
        x_vec_test = np.array([np.array(x_vec[1:-1].split(), dtype=np.float64) for x_vec in x_vectors_test.iloc[:, 3]])

        # Generate x_vec stat objects
        print('generating x_vec stat objects')
        #x_vec_test, x_id_test = sklearn.utils.shuffle(x_vec_test, x_id_test)###TODO comment out and in to compare different results in plda_test
        en_stat = pc.get_enroll_x_vec(x_vec_test, x_id_test)
        #x_vec_test, x_id_test = sklearn.utils.shuffle(x_vec_test, x_id_test)###TODO comment out and in to compare different results in plda_test
        te_stat = pc.get_test_x_vec(x_vec_test, x_id_test)

        # Testing plda
        print('testing plda')
        if(not train_plda):
            plda = pc.load_plda('plda/plda_v2.pickle')
        scores_plda = pc.test_plda(plda, en_stat, te_stat)

        positive_scores = []
        negative_scores = []
        positive_scores_mask = np.zeros_like(scores_plda.scoremat)
        negative_scores_mask = np.zeros_like(scores_plda.scoremat)
        
        num_failed_matches = 0
        total_matches = 0
        for pair in open(config.data_folder_path + '/VoxCeleb/veri_test2.txt'):
            is_match = bool(int(pair.split(" ")[0].rstrip().split(".")[0].strip()))
            enrol_id = pair.split(" ")[1].rstrip().split(".")[0].strip()
            test_id = pair.split(" ")[2].rstrip().split(".")[0].strip()

            try:
                i = int(np.where(scores_plda.modelset == enrol_id+'.wav')[0][0])
                j = int(np.where(scores_plda.segset == test_id+'.wav')[0][0])
                total_matches += 1
            except:
                num_failed_matches += 1
            else:
                score = float(scores_plda.scoremat[i,j])
                if(is_match):
                    positive_scores.append(score)
                    positive_scores_mask[i,j] = 1
                else:
                    negative_scores.append(score)
                    negative_scores_mask[i,j] = 1
        print('num_failed_matches', num_failed_matches, '/', num_failed_matches)

        # Calculating EER
        print('calculating EER')
        eer, eer_th = pc.EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
        print('EER: ', eer)
        print('threshold: ', eer_th)

        # Calculating minDCF
        print('calculating minDCF')
        min_dcf, min_dcf_th = pc.minDCF(torch.tensor(positive_scores), torch.tensor(negative_scores))
        print('minDCF: ', min_dcf)
        print('threshold: ', min_dcf_th)

        # Generating images for tensorboard
        print('generating images for tensorboard')
        img = np.zeros((3, scores_plda.scoremask.shape[0], scores_plda.scoremask.shape[1]))
        img[2] = np.array([scores_plda.scoremask])
        tb_logger.experiment.add_image('score_mask', img, 0)

        img = np.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
        img[2] = np.array([scores_plda.scoremat])
        tb_logger.experiment.add_image('score_matrix', img, 0)

        img = np.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
        img[1] = np.array([positive_scores_mask])
        img[0] = np.array([negative_scores_mask])
        tb_logger.experiment.add_image('ground_truth', img, 0)

        img = np.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
        img[1] = np.array([scores_plda.scoremat*positive_scores_mask])
        img[0] = np.array([scores_plda.scoremat*negative_scores_mask])
        tb_logger.experiment.add_image('ground_truth_scores', img, 0)

        checked_values_map = positive_scores_mask + negative_scores_mask
        positive_prediction_mask = np.zeros_like(scores_plda.scoremat)
        negative_prediction_mask = np.zeros_like(scores_plda.scoremat)
        for i, score in np.ndenumerate(scores_plda.scoremat):
            if(score >= eer_th):
                positive_prediction_mask[i] = 1
            else:
                negative_prediction_mask[i] = 1
        
        img = np.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
        img[1] = np.array([positive_prediction_mask*checked_values_map])
        img[0] = np.array([negative_prediction_mask*checked_values_map])
        tb_logger.experiment.add_image('prediction', img, 0)
        
        img = np.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
        img[1] = np.array([scores_plda.scoremat*positive_prediction_mask*checked_values_map])
        img[0] = np.array([scores_plda.scoremat*negative_prediction_mask*checked_values_map])
        tb_logger.experiment.add_image('prediction_scores', img, 0)
        
        img = np.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
        img[1] = np.array([positive_scores_mask*positive_prediction_mask])
        img[0] = np.array([negative_scores_mask*negative_prediction_mask])
        tb_logger.experiment.add_image('correct_prediction', img, 0)
        
        img = np.zeros((3, scores_plda.scoremat.shape[0], scores_plda.scoremat.shape[1]))
        img[1] = np.array([positive_scores_mask*negative_prediction_mask])
        img[0] = np.array([negative_scores_mask*positive_prediction_mask])
        tb_logger.experiment.add_image('false_prediction', img, 0)

    print('DONE')
'''
Notes:

screen commands reminder:
-------------------------
screen          start screen
screen -list    list screens
ctrl+a d        detach from current screen
screen -r       reatach to screen
ctrl+a c        create new window
ctrl+a "        show windows
exit            exit/kill window
ctrl+a A        rename window
ctrl+a H        create log file/toggle logging

my data used
153516 sample each 3 sec
460548 sec
7676 min
127 h

total data available
153516 sample average 8.4 sec
1265760 sec
21096 min
350 h
'''
