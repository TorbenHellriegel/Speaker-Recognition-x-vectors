import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.output = nn.Linear(hidden_size, num_classes) #TODO use softmax? nn.CrossEntropyLoss() appearently already includes a softmax

        self.dataset = Dataset(data_folder_path=config.data_folder_path)

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
        
        out = self.output(out)  #TODO use softmax? nn.CrossEntropyLoss() appearently already includes a softmax
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
        return loss
    
    def test_step(self, batch, batch_index):
        samples, labels = batch
        x_vecs = self.extract_x_vec(samples.float())
        return [(x_vecs, labels)]

    def test_epoch_end(self, test_step_outputs):
        for batch_output in test_step_outputs:
            for x_vec, label in batch_output:
                for x, l in zip(x_vec, label):
                    x_vector.append((x.cpu().numpy(), l.cpu().numpy()))
        return test_step_outputs
    
    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def train_dataloader(self):# TODO maybe visualize select data samples for images for the thesis
        self.dataset.load_train_data()
        train_data_loader = DataLoader(dataset=self.dataset, batch_size=config.batch_size, num_workers=4, shuffle=True)
        return train_data_loader

    def test_dataloader(self):
        self.dataset.load_test_data()
        test_data_loader = DataLoader(dataset=self.dataset, batch_size=config.batch_size, num_workers=4, shuffle=False)
        return test_data_loader

if __name__ == "__main__": #TODO figure out how to keep long process running in background
    config = Config(batch_size=10, load_existing_model=True, num_epochs=5) #TODO adjust batch (16, 32) epoch etc.

    # Define neural network
    model = XVectorModel(config.input_size, config.hidden_size, config.num_classes) #TODO num classes of the training set or also the test set
    trainer = pl.Trainer(accelerator='ddp', devices=2, max_epochs=config.num_epochs, log_every_n_steps=10, fast_dev_run=False) #TODO adjust log_every_n_steps
    # Maybe load an existing pretrained model dictionary
    if(config.load_existing_model):
        model.load_state_dict(torch.load(config.model_path))
        model.eval()

    # Train the x-vector model
    trainer.fit(model)
    #TODO logger accuracy cossentropyloss (plda)
    #TODO model checkpoint

    # Extract the x-vectors
    x_vector = []
    trainer.test(model) #TODO train PLDA classifier in test and do actual testing in prediction loop
    x_vector = pd.DataFrame(x_vector)
    # x_vector.to_csv('x_vectors/x_vector.csv')

#153516 sample each 3 sec
#460548 sec
#7676 min
#127 stunden

#153516 sample average 8.4 sec
#1265760 sec
#21096 min
#350 stunden