import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataset import Dataset
from tdnn import TdnnLayer


class XVectorModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(XVectorModel, self).__init__()

        self.time_context_layers = nn.Sequential(
            TdnnLayer(input_size=input_size, output_size=hidden_size, context=[-2, -1, 0, 1, 2]),
            TdnnLayer(input_size=hidden_size, output_size=hidden_size, context=[-2, 0, 2]),
            TdnnLayer(input_size=hidden_size, output_size=hidden_size, context=[-3, 0, 3]),
            TdnnLayer(input_size=hidden_size, output_size=hidden_size),
            TdnnLayer(input_size=hidden_size, output_size=1500)
        )
        
        self.segment_layer6 = nn.Linear(3000, hidden_size)
        self.segment_layer7 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        
        self.output = nn.Linear(hidden_size, num_classes)

        self.softmax = nn.Softmax() #TODO use softmax? nn.CrossEntropyLoss() appearently already includes a softmax

        self.loss_function = nn.CrossEntropyLoss()

        self.x_vectors = []

    def stat_pool(self, x):
        # Satistic pooling layer
        mean = torch.mean(x, 1)
        stand_dev = torch.std(x, 1)
        out = torch.cat((mean, stand_dev), 1)
        return out
        
    def forward(self, x):
        out = self.time_context_layers(x)

        out = self.stat_pool(out)

        out = self.relu(self.segment_layer6(out))
        out = self.relu(self.segment_layer7(out))
        
        out = self.output(out)  #TODO use softmax? nn.CrossEntropyLoss() appearently already includes a softmax
        return out

    def extract_x_vec(self, x):
        out = self.time_context_layers.forward(x)

        out = self.stat_pool(out)

        x_vec = self.segment_layer6.forward(out)
        return x_vec

    def training_step(self, batch, batch_index): #TODO save extra data for the graphs in the thesis
        samples, labels = batch                     # what data is important for the thesis?
        outputs = self(samples.float())             # loss for a graph showing decreasing loss
        loss = self.loss_function(outputs, labels)  # prdiction results from the plda
        return loss                                 # what else?
    
    def test_step(self, batch, batch_index):
        samples, _ = batch
        x_vec = self.extract_x_vec(samples.float())
        return x_vec

    def test_epoch_end(self, test_step_outputs):
        for batch_output in test_step_outputs:
            for x_vec in batch_output:
                self.x_vectors.append(x_vec)
        return test_step_outputs
    
    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def train_dataloader(self):# TODO maybe visualize select data samples for images for the thesis
        train_dataset = Dataset()
        train_dataset.load_train_data()

        # Set up dataloader for easy access to shuffled data batches
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=4, shuffle=True, drop_last=True)
        return train_data_loader

    def test_dataloader(self):
        test_dataset = Dataset()
        test_dataset.load_test_data()

        # Set up dataloader for easy access to data batches
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=4, shuffle=False, drop_last=True)
        return test_data_loader

#TODO imlement PLDA clasifier
class PLDAModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PLDAModel, self).__init__()

if __name__ == "__main__":
    config = Config(batch_size=10)

    # Define neural network
    x_model = XVectorModel(config.input_size, config.hidden_size, config.num_classes)
    plda_model = PLDAModel(config.input_size, config.hidden_size, config.num_classes)

    # Maybe load an existing pretrained model dictionary
    if(config.load_existing_model):
        x_model.load_state_dict(torch.load(config.model_path))
        x_model.eval()
    #TODO same for plda
    # if(config.load_existing_model):
    #     x_model.load_state_dict(torch.load(config.model_path))
    #     x_model.eval()
    model = x_model.float()

    x_vec_trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=config.num_epochs, log_every_n_steps=1, fast_dev_run=False)
    x_vec_trainer.fit(model)
    x_vec_trainer.test(model)

    plda_trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=config.num_epochs, log_every_n_steps=1, fast_dev_run=False)
    plda_trainer.fit(model)
    plda_trainer.test(model)

    # Save the model dictionary
    torch.save(model.state_dict(), config.model_path)
