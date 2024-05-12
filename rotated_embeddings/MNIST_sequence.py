import torch
import os
from config import config
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from mnist_stroke import MNISTStroke
import torchmetrics
from architectures_GNN import MultiClassificationMLP, BinaryClassificationMLP


class SequenceDatamodule(pl.LightningDataModule):
    def __init__(self, batch_size=256):
        super(SequenceDatamodule, self).__init__()
        self.batch_size = batch_size
        # self.transform = torch.transforms.Normalize((0.5,), (0.5,))

    def setup(self, stage=None):
        train = MNISTStroke(
            os.path.join(config['DATAROOT'], 'sequence'), train=True
        )
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            train, 
            [50000, 10000], 
            generator=torch.Generator().manual_seed(42)
            )
        
        self.test_dataset = MNISTStroke(
            os.path.join(config['DATAROOT'], 'sequence'), train=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=config['N_CPUS'] - 1
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=len(self.val_dataset), 
            shuffle=False, 
            num_workers=config['N_CPUS'] - 1
            )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=len(self.test_dataset), 
            shuffle=False, 
            num_workers=config['N_CPUS'] - 1
            )
    
    def get_labels(self, dataloader):
        x, y = next(iter(dataloader))
        return (y < 5).float()


class LSTM_MNIST(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.input_size = model_config.get('input_size', 4)
        self.hidden_channels = model_config.get('hidden_channels', 32)
        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_channels, batch_first=True)
        self.MLP = BinaryClassificationMLP(dict(
            decoder_input=self.hidden_channels,
        #     n_classes=model_config.get('n_classes')
        ))
        self.lr = model_config.get('lr', 0.001)
        self.weight_decay = model_config.get('weight_decay', 5e-4)
        self.loss_function = torch.nn.BCELoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.MLP(output.sum(axis=1)).flatten()

    def embedding(self, batch):
        x, y_all = batch
        y = (y_all < 5).float()
        output, _ = self.lstm(x)
        return output.sum(axis=1)

    def _common_step(self, batch):
        x, y_all = batch
        y = (y_all < 5).float()
        logits = self(x)
        loss = self.loss_function(logits, y)
        return loss, y, logits

    def training_step(self, batch, batch_no):
        loss, y, logits = self._common_step(batch)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', self.accuracy(logits, y), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_no):
        loss, y, logits = self._common_step(batch)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_accuracy', self.accuracy(logits, y), on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
            )