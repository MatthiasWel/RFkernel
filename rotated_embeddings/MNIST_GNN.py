import torch_geometric
from config import config
import os
import networkx as nx
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from architectures_GNN import GeneralGNN
from torch_geometric.data.lightning import LightningDataset
from architectures_GNN import TwoLayerGIN, MultiClassificationMLP, AddPoolingGNN
import torch
import torchmetrics
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as PyGDataloader
from utils import timestamp, Apply

class MNISTSuperpixelsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=256):
        super(MNISTSuperpixelsDataModule, self).__init__()
        self.batch_size = batch_size
        
    def add_pos(self, pyg_data):
        pyg_data.x = torch.concat((pyg_data.x, pyg_data.pos), dim=1)
        return pyg_data

    def setup(self, stage: str):
        train = torch_geometric.datasets.MNISTSuperpixels(train=True, root=config['DATAROOT'])
        test = torch_geometric.datasets.MNISTSuperpixels(train=False, root=config['DATAROOT'])
        train, val = train_test_split(train, random_state=42, train_size=10/12)
        self.datamodule = StandardGNNDataModule(
            train=Apply(train, self.add_pos), 
            val=Apply(val, self.add_pos), 
            test=Apply(test, self.add_pos), 
            batch_size=self.batch_size
            )
        
    def train_dataloader(self):
        return self.datamodule.train_dataloader()
    
    def val_dataloader(self):
        return self.datamodule.val_dataloader()
    
    def test_dataloader(self):
        return self.datamodule.test_dataloader()
    
    def get_labels(self, dataloader):
        return (next(iter(dataloader)).y < 5).float()
    
class StandardGNNDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train,
            val,
            test, 
            batch_size
            ):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.batch_size = batch_size


    def train_dataloader(self):
        return PyGDataloader(
            self.train, 
            batch_size=self.batch_size, 
            num_workers=config['N_CPUS'] - 1,
            shuffle=True
            ) 
    def val_dataloader(self):
        return PyGDataloader(
            self.val, 
            batch_size=len(self.val), 
            num_workers=config['N_CPUS'] - 1,
            shuffle=False
            )
    def test_dataloader(self):
        return PyGDataloader(
            self.test, 
            batch_size=len(self.test), 
            num_workers=config['N_CPUS'] - 1,
            shuffle=False
            )

class LitGNN(pl.LightningModule):
    def __init__(self, model_config):
        super(LitGNN, self).__init__()
        self.save_hyperparameters()
        self.GNN = GeneralGNN(model_config)
        self.loss_function = torch.nn.BCELoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.lr = model_config.get('lr', 0.001)
        self.weight_decay = model_config.get('weight_decay', 5e-4)
    
    def forward(self, data):
        return self.GNN(data).flatten()
    
    def embedding(self, data):
        x = self.GNN.embedding(data)
        return x
    
    def _common_step(self, data):
        y_all = data.y
        y = (y_all < 5).float()
        logits = self.forward(data)
        loss = self.loss_function(logits, y)
        return loss, y, logits

    def training_step(self, data):
        loss, y, logits = self._common_step(data)
        self.log('train_loss', loss, on_epoch=True)
        self.log("train_accuracy", self.accuracy(logits, y), on_epoch=True)
        return loss

    def validation_step(self, data):
        loss, y, logits = self._common_step(data)
        self.log('val_loss', loss, on_epoch=True)
        self.log("val_accuracy", self.accuracy(logits, y), on_epoch=True)

    def predict(self, data):
        loss, y, logits = self._common_step(data)
        return {'loss':loss, 'y': y, 'logits': logits}

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
            )