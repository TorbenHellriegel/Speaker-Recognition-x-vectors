import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import plda
from config import Config
from dataset import Dataset
from tdnn import TdnnLayer
import matplotlib.pyplot as plt
import torchmetrics
import torch.utils.tensorboard
from pytorch_lightning import loggers as pl_loggers

class XVectorModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes, batch_size=16):
        super().__init__()

        self.time_context_layers = nn.Sequential(
            TdnnLayer(input_size=input_size, output_size=hidden_size, context=[-2, -1, 0, 1, 2]),
            TdnnLayer(input_size=hidden_size, output_size=hidden_size, context=[-2, 0, 2]),
            TdnnLayer(input_size=hidden_size, output_size=hidden_size, context=[-3, 0, 3]),
            TdnnLayer(input_size=hidden_size, output_size=hidden_size),
            TdnnLayer(input_size=hidden_size, output_size=1500)
        )
        self.segment_layer6 = nn.Linear(3000, hidden_size)
        self.segment_layer7 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)

        self.batch_size = batch_size
        self.dataset = Dataset(data_folder_path=config.data_folder_path)
        self.accuracy = torchmetrics.Accuracy()

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

    #TODO save data for the thesis
    # what data is important for the thesis?
    # loss for a graph showing decreasing loss
    # prdiction results from the plda
    # what else?

    def training_step(self, batch, batch_index):
        samples, labels = batch
        outputs = self(samples.float())
        loss = F.cross_entropy(outputs, labels)
        return {'loss': loss, 'preds': outputs, 'labels': labels}

    def training_step_end(self, outputs):
        self.log('train_step_loss', outputs['loss'])
        self.accuracy(outputs['preds'], outputs['labels'])
        self.log('train_step_acc', self.accuracy)
    
    def test_step(self, batch, batch_index):
        samples, labels = batch
        x_vecs = self.extract_x_vec(samples.float())
        return [(x_vecs, labels)]

    def test_epoch_end(self, test_step_outputs):
        for batch_output in test_step_outputs:
            for x_vec, label in batch_output:
                for x, l in zip(x_vec, label):
                    x_vector.append(x.cpu().numpy())
                    x_label.append(l.cpu().numpy())
        return test_step_outputs
    
    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def train_dataloader(self):
        self.dataset.load_train_data()
        train_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)
        return train_data_loader

    def test_dataloader(self):
        self.dataset.load_test_data()
        test_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        return test_data_loader

if __name__ == "__main__": #TODO figure out how to keep long process running in background
    config = Config(num_epochs=5, batch_size=16) #TODO adjust batch (16, 32) epoch etc.

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/") #TODO logger accuracy cossentropyloss (plda)

    # Define neural network
    model = XVectorModel(config.input_size, config.hidden_size, config.num_classes, config.batch_size)
    # Long train mode
    # trainer = pl.Trainer(strategy='ddp', accelerator='gpu', devices=2, max_epochs=config.num_epochs, #TODO adjust devices and also add strategy='ddp' and devices=2
    #                     logger=tb_logger, log_every_n_steps=1, #TODO adjust log_every_n_steps
    #                     fast_dev_run=False)#, limit_train_batches=0.001, limit_test_batches=0.001) #TODO set limit_batches to 1 or 0.01
    # Short test mode
    trainer = pl.Trainer(accelerator='cpu', max_epochs=1, #TODO adjust devices and also add strategy='ddp' and devices=2
                        logger=tb_logger, log_every_n_steps=1, #TODO adjust log_every_n_steps
                        fast_dev_run=False, limit_train_batches=0.001, limit_test_batches=0.001) #TODO set limit_batches to 1 or 0.01

    # Train the x-vector model
    trainer.fit(model) #TODO ckpt_path="logs/lightning_logs/version_XX/checkpoints/epoch=YY-step=ZZ.ckpt"

    # Extract the x-vectors
    x_vector = []
    x_label = []
    trainer.test(model) #TODO train PLDA classifier in test and do actual testing in prediction loop (DOESNT WORK)
    # x_vector = pd.DataFrame(x_vector)
    # x_vector.to_csv('x_vectors/x_vector.csv')
    divider = int(len(x_vector)*0.95)
    x_vector_train = np.array(x_vector[:divider])
    x_label_train = np.array(x_label[:divider])
    x_vector_test = np.array(x_vector[divider:])
    x_label_test = np.array(x_label[divider:])
    print('shape: ', x_vector_train.shape, ' dtype: ', x_vector_train.dtype)
    print('shape: ', x_label_train.shape, ' dtype: ', x_label_train.dtype)
    print('shape: ', x_vector_test.shape, ' dtype: ', x_vector_test.dtype)
    print('shape: ', x_label_test.shape, ' dtype: ', x_label_test.dtype)

    plda_classifier = plda.Classifier()
    plda_classifier.fit_model(x_vector_train, x_label_train)

    predictions, log_p_predictions = plda_classifier.predict(x_vector_test)
    print('Accuracy: {}'.format((x_label_test == predictions).mean()))

    '''# TODO try to plot this
    n_examples = 10
    fig, ax_arr = plt.subplots(1, n_examples, figsize=(20, 2))

    for x in range(n_examples):
        ax_arr[x].imshow(x_vector_test[x].reshape(16, 32), cmap='gray')
        ax_arr[x].set_xticks([])
        ax_arr[x].set_yticks([])
        title = 'Prediction: {}'
        xlabel = 'Truth: {}'
        ax_arr[x].set_title(title.format(predictions[x]))
        ax_arr[x].set_xlabel(xlabel.format(x_label_test[x]))
    plt.show()'''

# bei batch size 512:
# etwa 13-17 sec pro iteration
# etwa 871 iterationen
# insgesamt etwa 3-4 stunden

# nohup python main.py &> out/NAME.out &