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
        
        self.plda_layer = nn.Linear(hidden_size, 150)

        self.dataset = Dataset()

    # Satistic pooling layer
    def stat_pool(self, x):
        mean = torch.mean(x, 1)
        stand_dev = torch.std(x, 1)
        out = torch.cat((mean, stand_dev), 1)
        return out
        
    def forward(self, x):
        if(mode == 'x_vector'):
            out = self.time_context_layers(x)

            out = self.stat_pool(out)

            out = F.relu(self.segment_layer6(out))
            out = F.relu(self.segment_layer7(out))
            
            out = self.output(out)  #TODO use softmax? nn.CrossEntropyLoss() appearently already includes a softmax
            
        elif(mode == 'plda_classifier'):
            out = self.output(x)
            
        return out

    def extract_x_vec(self, x):
        out = self.time_context_layers.forward(x)

        out = self.stat_pool(out)

        x_vec = self.segment_layer6.forward(out)
        return x_vec

    #TODO save extra data for the graphs in the thesis
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

        if(mode == 'x_vector'):
            x_vecs = self.extract_x_vec(samples.float())
        elif(mode == 'plda_classifier'):
            #TODO plda predictions
            x_vecs = samples

        return [(x_vecs, labels)]

    def test_epoch_end(self, test_step_outputs):
        if(mode == 'x_vector'):
            for batch_output in test_step_outputs:
                for x_vec, label in batch_output:
                    self.dataset.load_train_x_vec(x_vec, label)
            self.dataset.change_mode('plda_classifier')
        return test_step_outputs
    
    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def train_dataloader(self):# TODO maybe visualize select data samples for images for the thesis
        if(mode == 'x_vector'):
            self.dataset.load_train_data(data_folder_path=config.data_folder_path)
            train_data_loader = DataLoader(dataset=self.dataset, batch_size=config.batch_size, num_workers=4, shuffle=True, drop_last=True)
        elif(mode == 'plda_classifier'):
            train_data_loader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=True, drop_last=True)
        return train_data_loader

    def test_dataloader(self):
        if(mode == 'x_vector'):
            self.dataset.load_test_data(data_folder_path=config.data_folder_path)
            test_data_loader = DataLoader(dataset=self.dataset, batch_size=config.batch_size, num_workers=4, shuffle=False, drop_last=True)
        elif(mode == 'plda_classifier'):
            test_data_loader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, drop_last=True)
        return test_data_loader

if __name__ == "__main__":
    config = Config(batch_size=10, load_existing_model=False, num_epochs=1)

    # Define neural network
    model = XVectorModel(config.input_size, config.hidden_size, config.num_classes)

    # Maybe load an existing pretrained model dictionary
    if(config.load_existing_model):
        model.load_state_dict(torch.load(config.model_path))
        model.eval()

    mode = 'x_vector'
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=config.num_epochs, log_every_n_steps=1, fast_dev_run=False)
    trainer.fit(model)
    trainer.test(model)
    
    mode = 'plda_classifier'
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=config.num_epochs, log_every_n_steps=1, fast_dev_run=False)
    trainer.fit(model)
    trainer.test(model)

    # Save the model dictionary
    torch.save(model.state_dict(), config.model_path)