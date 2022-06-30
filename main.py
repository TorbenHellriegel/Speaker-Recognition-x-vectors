import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Dataset
from tdnn import TdnnLayer


class Config:
    def __init__(self, batch_size=100, input_size=24, hidden_size=512, num_classes=10, learning_rate=0.001,
                num_epochs=5, model_path='./trained_models/x_vector_model.pth', load_existing_model=True):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.load_existing_model = load_existing_model


class NeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

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
        
    def forward(self, x):
        out = self.time_context_layers(x)

        # Satistic pooling layer
        mean = torch.mean(out, 1)
        stand_dev = torch.std(out, 1)
        out = torch.cat((mean, stand_dev), 1)

        x_vec = self.segment_layer6(out)
        out = self.relu(x_vec)
        out = self.segment_layer7(out)
        out = self.relu(out)
        
        out = self.output(out)  #TODO use softmax? nn.CrossEntropyLoss() appearently already includes a softmax
        return out, x_vec
    
    def training_step(self, batch, batch_index): #TODO somehow save extra data for the graphs in the thesis
        samples, labels = batch
        outputs, x_vec = self(samples.float()) #TODO imlement PLDA clasifier
        loss = self.loss_function(outputs, labels)
        return loss
    
    def validation_step(self, batch, batch_index): #TODO somehow save extra data for the graphs in the thesis
        samples, labels = batch
        outputs, x_vec = self(samples.float()) #TODO imlement PLDA clasifier
        val_loss = self.loss_function(outputs, labels)
        return val_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def train_dataloader(self):# TODO maybe visualize select data samples for images for the thesis
        train_dataset = Dataset()
        train_dataset.load_train_data()

        # Set up dataloader for easy access to shuffled data batches
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=4, shuffle=True, drop_last=True)
        return train_data_loader

    def val_dataloader(self):
        test_dataset = Dataset()
        test_dataset.load_test_data()

        # Set up dataloader for easy access to shuffled data batches
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=4, shuffle=False, drop_last=True)
        return test_data_loader


if __name__ == "__main__":
    config = Config()

    # Define neural network
    model = NeuralNet(config.input_size, config.hidden_size, config.num_classes)

    # Maybe load an existing pretrained model dictionary
    if(config.load_existing_model):
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
    model = model.float()

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=config.num_epochs, log_every_n_steps=1, fast_dev_run=False)
    trainer.fit(model)

    # Save the model dictionary
    torch.save(model.state_dict(), config.model_path)
