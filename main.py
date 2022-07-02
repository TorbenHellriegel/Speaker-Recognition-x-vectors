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

        self.x_vectors = []
        
        self.plda_layer = nn.Linear(hidden_size, 150)

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
            out = self.plda_layer(x)
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
        train_dataset.load_train_data(data_folder_path=config.data_folder_path)

        # Set up dataloader for easy access to shuffled data batches
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=4, shuffle=True, drop_last=True)
        return train_data_loader

    def test_dataloader(self):
        test_dataset = Dataset()
        test_dataset.load_test_data(data_folder_path=config.data_folder_path)

        # Set up dataloader for easy access to data batches
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=4, shuffle=False, drop_last=True)
        return test_data_loader

if __name__ == "__main__":
    config = Config(batch_size=10, load_existing_model=False)

    # Define neural network
    model = XVectorModel(config.input_size, config.hidden_size, config.num_classes)

    # Maybe load an existing pretrained model dictionary
    if(config.load_existing_model):
        model.load_state_dict(torch.load(config.model_path))
        model.eval()

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=config.num_epochs, log_every_n_steps=1, fast_dev_run=False)

    mode = 'x_vector'
    trainer.fit(model)
    trainer.test(model)
    
    mode = 'plda_classifier'
    trainer.fit(model)
    trainer.test(model)

    # Save the model dictionary
    torch.save(model.state_dict(), config.model_path)