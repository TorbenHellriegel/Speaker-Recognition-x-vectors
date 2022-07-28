import numpy as np
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

from config import Config
from dataset import Dataset
from tdnn import TdnnLayer


class XVectorModel(pl.LightningModule):
    def __init__(self, input_size=24, hidden_size=512, num_classes=1211,
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
        self.segment_layer6 = nn.Linear(3000, hidden_size)
        self.segment_layer7 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)

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

        x_vec = self.segment_layer6.forward(out)
        return x_vec

    def training_step(self, batch, batch_index):
        samples, labels = batch
        outputs = self(samples.float())
        loss = F.cross_entropy(outputs, labels)
        return {'loss': loss, 'train_preds': outputs, 'train_labels': labels}

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
        samples, labels = batch
        outputs = self(samples.float())
        loss = F.cross_entropy(outputs, labels)
        return {'loss': loss, 'val_preds': outputs, 'val_labels': labels}

    def validation_step_end(self, outputs):
        self.log('val_step_loss', outputs['loss'])
        accuracy = self.accuracy(outputs['val_preds'], outputs['val_labels'])
        self.log('val_step_acc', self.accuracy)
        return {'loss': outputs['loss'], 'acc': accuracy}

    def validation_epoch_end(self, outputs):
        #TODO figure out how to plot 2in1
        todo=0
    
    def test_step(self, batch, batch_index):
        samples, labels = batch
        x_vecs = self.extract_x_vec(samples.float())
        return [(x_vecs, labels)]

    def test_epoch_end(self, test_step_outputs):
        for batch_output in test_step_outputs:
            for x_vec, label in batch_output:
                for x, l in zip(x_vec, label):
                    x_vectors.append((x.cpu().numpy(), l.cpu().numpy()))
        return test_step_outputs
    
    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        self.dataset.load_data(train=True)
        train_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)
        return train_data_loader

    def val_dataloader(self):
        self.dataset.load_data(val=True)
        train_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        return train_data_loader

    def test_dataloader(self):
        if(extract_mode == 'train'):
            self.dataset.load_data(train=True, val=True)
            test_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        if(extract_mode == 'test'):
            self.dataset.load_data(test=True)
            test_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        return test_data_loader

if __name__ == "__main__":
    # Define model and trainer
    print('setting up model and trainer parameters')
    config = Config() #adjust batch size, epoch, etc. here

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./")
    checkpoint_callback = ModelCheckpoint(monitor='val_step_loss', save_last=True, verbose=True)

    model = XVectorModel(input_size=config.input_size, hidden_size=config.hidden_size, num_classes=config.num_classes,
                        batch_size=config.batch_size, learning_rate=config.learning_rate, batch_norm=config.batch_norm, dropout_p=config.dropout_p,
                        augmentations_per_sample=config.augmentations_per_sample, data_folder_path=config.data_folder_path)
    model.dataset.init_samples_and_labels()

    trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_step_loss", mode="min"), checkpoint_callback],
                        logger=tb_logger, log_every_n_steps=1,
                        strategy='ddp', accelerator='gpu', devices=2,
                        max_epochs=config.num_epochs)
                        #small test adjust options: fast_dev_run=True, limit_train_batches=0.001, limit_val_batches=0.01, limit_test_batches=0.01

    # Train the x-vector model
    trainer.fit(model)#, ckpt_path="logs/lightning_logs/version_0/checkpoints/epoch=0-step=436.ckpt")
    
    # Extract the x-vectors
    x_vectors = []
    extract_mode = 'train'
    trainer.test(model)
    x_vectors_train = np.array(x_vectors, dtype=np.ndarray)
    
    x_vectors = []
    extract_mode = 'test'
    trainer.test(model)
    x_vectors_test = np.array(x_vectors, dtype=np.ndarray)
    
    x_vec_train = np.array(x_vectors_train[:, 0])
    x_label_train = np.array(x_vectors_train[:, 1])
    x_vec_test = np.array(x_vectors_test[:, 0])
    x_label_test = np.array(x_vectors_test[:, 1])

    # Train PLDA classifier #TODO

    # Test PLDA classifier #TODO

    print('done')
'''
Notes:

Can run in background with this command. Also saves output in .out file:
nohup python main.py &> out/NAME.out &

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