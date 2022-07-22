import numpy as np
import plda
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
import torchmetrics
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

from config import Config
from dataset import Dataset
from tdnn import TdnnLayer


class XVectorModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
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
                    # x_vector.append(x.cpu().numpy())
                    # x_label.append(l.cpu().numpy())
                    x_vectors.append((x.cpu().numpy(), l.cpu().numpy()))
        return test_step_outputs
    
    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def train_dataloader(self):
        self.dataset.load_train_data()
        train_data_loader = DataLoader(dataset=self.dataset, batch_size=config.batch_size, num_workers=4, shuffle=True)
        return train_data_loader

    def test_dataloader(self):
        self.dataset.load_test_data()
        test_data_loader = DataLoader(dataset=self.dataset, batch_size=config.batch_size, num_workers=4, shuffle=False)
        return test_data_loader

if __name__ == "__main__":
    # Define parameters, model, logger and trainer
    config = Config() #adjust batch size, epoch, etc. here
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    model = XVectorModel(config.input_size, config.hidden_size, config.num_classes)
    trainer = pl.Trainer(strategy='ddp', accelerator='gpu', devices=2, max_epochs=config.num_epochs,
                        logger=tb_logger, log_every_n_steps=1) #small test adjust options: fast_dev_run=True, limit_train_batches=0.001, limit_test_batches=0.001

    # Train the x-vector model
    trainer.fit(model)#, ckpt_path="logs/lightning_logs/version_0/checkpoints/epoch=0-step=436.ckpt")

    # Extract the x-vectors
    x_vectors = []
    trainer.test(model)
    x_vectors = np.array(x_vectors)
    unique_labels = np.unique(x_vectors[:, 1])
    x_vectors_train = []
    x_vectors_test = []
    for label in unique_labels:
        samples = [[x, l] for i, x, l in enumerate(x_vectors) if x_vectors[i, 1] == label]
        x_vectors_train = x_vectors_train + samples[1:]
        x_vectors_test = x_vectors_test + samples[:1]
    x_vec_train = np.array(x_vectors_train[:, 0])
    x_label_train = np.array(x_vectors_train[:, 1])
    x_vec_test = np.array(x_vectors_test[:, 0])
    x_label_test = np.array(x_vectors_test[:, 1])

    # Train PLDA classifier
    plda_classifier = plda.Classifier()
    plda_classifier.fit_model(x_vec_train, x_label_train)

    # Test PLDA classifier
    predictions, log_p_predictions = plda_classifier.predict(x_vec_test)
    print('Accuracy: {}'.format((x_label_test == predictions).mean()))

# Can run in background with this command. Also saves output in .out file
# nohup python main.py &> out/NAME.out &
