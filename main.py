import numpy as np
import pandas as pd
import pytorch_lightning as pl
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
from plda_score_stat import plda_score_stat_object
from tdnn_layer import TdnnLayer


class XVectorModel(pl.LightningModule):
    def __init__(self, input_size=24,
                hidden_size=512,
                num_classes=1211,
                x_vector_size=512,
                x_vec_extract_layer=6,
                batch_size=512,
                learning_rate=0.001,
                batch_norm=True,
                dropout_p=0.0,
                augmentations_per_sample=2,
                data_folder_path='data'):
        super().__init__()

        # Set up the TDNN structure including the time context of the TdnnLayer
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

    # The statistic pooling layer
    def stat_pool(self, x):
        mean = torch.mean(x, 1)
        stand_dev = torch.std(x, 1)
        out = torch.cat((mean, stand_dev), 1)
        return out
        
    # The standard forward pass through the neural network
    def forward(self, x):
        out = self.time_context_layers(x)

        out = self.stat_pool(out)

        out = F.relu(self.segment_layer6(out))
        out = F.relu(self.segment_layer7(out))
        
        out = self.output(out)
        return out

    # This method is used to generate the x-vectors for the PLDA classifier
    # It is the same as the usual forward method exept it stops passing the
    # input through the layers at the specified x_vec_extract_layer
    # Finally it returns the x-vectors instead of the usual output
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

    # Train the model
    def training_step(self, batch, batch_index):
        samples, labels, id = batch
        outputs = self(samples.float())
        loss = F.cross_entropy(outputs, labels)
        return {'loss': loss, 'train_preds': outputs, 'train_labels': labels, 'train_id': id}

    # Log training loss and accuracy with the logger
    def training_step_end(self, outputs):
        self.log('train_step_loss', outputs['loss'])
        accuracy = self.accuracy(outputs['train_preds'], outputs['train_labels'])
        self.log('train_step_acc', self.accuracy)
        return {'loss': outputs['loss'], 'acc': accuracy}

    # Create graph and histogram for the logger
    def training_epoch_end(self, outputs):
        if(self.current_epoch == 0):
            sample = torch.rand((1, 299, 24))
            self.logger.experiment.add_graph(XVectorModel(), sample)

        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    # Calculate loss of validation data to check if overfitting
    def validation_step(self, batch, batch_index):
        samples, labels, id = batch
        outputs = self(samples.float())
        loss = F.cross_entropy(outputs, labels)
        return {'loss': loss, 'val_preds': outputs, 'val_labels': labels, 'val_id': id}

    # Log validation loss and accuracy with the logger
    def validation_step_end(self, outputs):
        self.log('val_step_loss', outputs['loss'])
        accuracy = self.accuracy(outputs['val_preds'], outputs['val_labels'])
        self.log('val_step_acc', self.accuracy)
        return {'loss': outputs['loss'], 'acc': accuracy}
    
    # The test step here is NOT used as a test step!
    # Instead it is used to extract the x-vectors
    def test_step(self, batch, batch_index):
        samples, labels, id = batch
        x_vecs = self.extract_x_vec(samples.float())
        return [(x_vecs, labels, id)]

    # After all x-vectros are generated append them to the predefined list
    def test_epoch_end(self, test_step_outputs):
        for batch_output in test_step_outputs:
            for x_vec, label, id in batch_output:
                for x, l, i in zip(x_vec, label, id):
                    x_vector.append((i, int(l.cpu().numpy()), np.array(x.cpu().numpy(), dtype=np.float64)))
        return test_step_outputs
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # Load only the training data
    def train_dataloader(self):
        self.dataset.load_data(train=True)
        train_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)
        return train_data_loader

    # Load only the validation data
    def val_dataloader(self):
        self.dataset.load_data(val=True)
        val_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        return val_data_loader

    # Load either both training and validation or test data for extracting the x-vectors
    # In 'train' mode extract x-vectors for PLDA training, in 'test' mode for testing PLDA
    def test_dataloader(self):
        if(extract_mode == 'train'):
            self.dataset.load_data(train=True, val=True)
            test_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        if(extract_mode == 'test'):
            self.dataset.load_data(test=True)
            test_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        return test_data_loader



if __name__ == "__main__":

    # Adjust parameters of model, PLDA, training etc. here
    # Set your own data folder path here!
    # VoxCeleb MUSAN and RIR must be in the same data/ directory!
    # It is also possible to execute only select parts of the program by adjusting:
    # train_x_vector_model, extract_x_vectors, train_plda and test_plda
    # When running only later parts of the program a checkpoint_path MUST be given and
    # earlier parts of the programm must have been executed at least once
    print('setting up model and trainer parameters')
    config = Config(data_folder_path='../../../../../../../../../data/7hellrie',
                    checkpoint_path='lightning_logs/x_vector_v1_5/checkpoints/last.ckpt',
                    train_x_vector_model = False,
                    extract_x_vectors = False,
                    train_plda = False,
                    test_plda = False,
                    x_vec_extract_layer=6,
                    plda_rank_f=25)#TODO delete most of this

    # Define model and trainer
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="testlogs/")
    early_stopping_callback = EarlyStopping(monitor="val_step_loss", mode="min")
    checkpoint_callback = ModelCheckpoint(monitor='val_step_loss', save_top_k=10, save_last=True, verbose=True)

    if(config.checkpoint_path == 'none'):
        model = XVectorModel(input_size=config.input_size,
                            hidden_size=config.hidden_size,
                            num_classes=config.num_classes,
                            x_vector_size=config.x_vector_size,
                            x_vec_extract_layer=config.x_vec_extract_layer,
                            batch_size=config.batch_size,
                            learning_rate=config.learning_rate,
                            batch_norm=config.batch_norm,
                            dropout_p=config.dropout_p,
                            augmentations_per_sample=config.augmentations_per_sample,
                            data_folder_path=config.data_folder_path)
    else:
        model = XVectorModel.load_from_checkpoint(config.checkpoint_path)
    model.dataset.init_samples_and_labels()

    trainer = pl.Trainer(callbacks=[early_stopping_callback, checkpoint_callback],
                        logger=tb_logger,
                        log_every_n_steps=1,
                        #accelerator='cpu',#TODO delete
                        accelerator='gpu', devices=[0],
                        max_epochs=config.num_epochs)
                        #small test adjust options: fast_dev_run=True, limit_train_batches=0.0001, limit_val_batches=0.001, limit_test_batches=0.002



    # Train the x-vector model
    if(config.train_x_vector_model):
        print('training x-vector model')
        if(config.checkpoint_path == 'none'):
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=config.checkpoint_path)



    # Extract the x-vectors
    if(config.extract_x_vectors):
        print('extracting x-vectors')
        # Extract the x-vectors for trainng the PLDA classifier and save to csv
        x_vector = []
        extract_mode = 'train'
        if(config.train_x_vector_model):
            trainer.test(model)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv('x_vectors/x_vector_train_v1_5_l7relu.csv')#TODO set to default name
        elif(config.checkpoint_path != 'none'):
            trainer.test(model, ckpt_path=config.checkpoint_path)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv('x_vectors/x_vector_train_v1_5_l7relu.csv')#TODO set to default name
        else:
            print('could not extract train x-vectors')

        # Extract the x-vectors for testing the PLDA classifier and save to csv
        x_vector = []
        extract_mode = 'test'
        if(config.train_x_vector_model):
            trainer.test(model)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv('x_vectors/x_vector_test_v1_5_l7relu.csv')#TODO set to default name
        elif(config.checkpoint_path != 'none'):
            trainer.test(model, ckpt_path=config.checkpoint_path)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv('x_vectors/x_vector_test_v1_5_l7relu.csv')#TODO set to default name
        else:
            print('could not extract test x-vectors')
    


    if(config.train_plda):
        print('loading x_vector data')
        # Extract the x-vectors, labels and id from the csv
        x_vectors_train = pd.read_csv('x_vectors/i_vector_train_v2.csv')#TODO set to default name
        x_id_train = np.array(x_vectors_train.iloc[:, 1])
        x_label_train = np.array(x_vectors_train.iloc[:, 2], dtype=int)
        x_vec_train = np.array([np.array(x_vec[1:-1].split(), dtype=np.float64) for x_vec in x_vectors_train.iloc[:, 3]])

        # Generate x_vec stat objects
        print('generating x_vec stat objects')
        tr_stat = pc.get_train_x_vec(x_vec_train, x_label_train, x_id_train)

        # # Train plda #TODO change back to ony one
        # print('training plda')
        # plda = pc.setup_plda(rank_f=config.plda_rank_f, nb_iter=10)
        # plda = pc.train_plda(plda, tr_stat)
        # pc.save_plda(plda, 'plda_v1_5_l6_d25')#TODO set to default name
        
        # Train plda
        print('training plda')
        plda = pc.setup_plda(rank_f=50, nb_iter=10)
        plda = pc.train_plda(plda, tr_stat)
        pc.save_plda(plda, 'plda_ivec_v2_d50')
        # Train plda
        print('training plda')
        plda = pc.setup_plda(rank_f=100, nb_iter=10)
        plda = pc.train_plda(plda, tr_stat)
        pc.save_plda(plda, 'plda_ivec_v2_d100')
        # Train plda
        print('training plda')
        plda = pc.setup_plda(rank_f=150, nb_iter=10)
        plda = pc.train_plda(plda, tr_stat)
        pc.save_plda(plda, 'plda_ivec_v2_d150')
        # Train plda
        print('training plda')
        plda = pc.setup_plda(rank_f=200, nb_iter=10)
        plda = pc.train_plda(plda, tr_stat)
        pc.save_plda(plda, 'plda_ivec_v2_d200')



    if(config.test_plda):
        # Extract the x-vectors, labels and id from the csv
        print('loading x_vector data')
        x_vectors_test = pd.read_csv('x_vectors/i_vector_test_v2.csv')#TODO set to default name
        x_vectors_test.columns = ['index', 'id', 'label', 'xvector']
        score = plda_score_stat_object(x_vectors_test)

        # Test plda
        print('testing plda')
        if(not config.train_plda):
            plda = pc.load_plda('plda/plda_ivec_v2_d200.pickle')#TODO set to default name
        score.test_plda(plda, config.data_folder_path + '/VoxCeleb/veri_test2.txt')

        # Calculate EER and minDCF
        print('calculating EER and minDCF')
        score.calc_eer_mindcf()
        print('EER: ', score.eer, '   threshold: ', score.eer_th)
        print('minDCF: ', score.min_dcf, '   threshold: ', score.min_dcf_th)

        # Generate images for tensorboard
        score.plot_images(tb_logger.experiment)

        pc.save_plda(score, 'plda_score_ivec_v2_d200')#TODO set to default name



    if(False):
        # x_vectors_train = pd.read_csv('x_vectors/x_vector_train_v1.csv')
        # train_label = np.array(x_vectors_train.iloc[:, 2], dtype=int)
        # train_xvec = np.array([np.array(x_vec[1:-1].split(), dtype=np.float64) for x_vec in x_vectors_train.iloc[:, 3]])

        # plda = pc.load_plda('plda/plda_v1_5.pickle')
        # score = pc.load_plda('plda/plda_score_v1_5.pickle')
        # score.plot_images(tb_logger.experiment, plda)#, train_xvec, train_label)
        
        score = pc.load_plda('plda/plda_score_v1_5_l7relu_d50.pickle')
        print('calculating EER and minDCF')
        print('EER: ', score.eer, '   threshold: ', score.eer_th)
        print('minDCF: ', score.min_dcf, '   threshold: ', score.min_dcf_th)
        
        score = pc.load_plda('plda/plda_score_v1_5_l7relu_d100.pickle')
        print('calculating EER and minDCF')
        print('EER: ', score.eer, '   threshold: ', score.eer_th)
        print('minDCF: ', score.min_dcf, '   threshold: ', score.min_dcf_th)
        
        score = pc.load_plda('plda/plda_score_v1_5_l7relu_d150.pickle')
        print('calculating EER and minDCF')
        print('EER: ', score.eer, '   threshold: ', score.eer_th)
        print('minDCF: ', score.min_dcf, '   threshold: ', score.min_dcf_th)
        
        score = pc.load_plda('plda/plda_score_v1_5_l7relu_d200.pickle')
        print('calculating EER and minDCF')
        print('EER: ', score.eer, '   threshold: ', score.eer_th)
        print('minDCF: ', score.min_dcf, '   threshold: ', score.min_dcf_th)

    print('DONE')
'''
Notes: TODO remove

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
