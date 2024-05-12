from torchvision.models import resnet18
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from config import config
import torchmetrics
from architectures_GNN import BinaryClassificationMLP

import pytorch_lightning as pl
     
from PIL import Image 
from IPython.display import display
to_img = transforms.ToPILImage()


class ResNetMNIST(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.resnet = torch.nn.Sequential(
                torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                *list(resnet18(num_classes=10).children())[1:9]
            )
        self.classifier = BinaryClassificationMLP({'hidden_channels': 512, 'batch_norm': True})
        self.resnet[8] = torch.nn.Flatten()
        self.loss = torch.nn.BCELoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def forward(self, x):
        return self.classifier(self.resnet(x)).flatten()
    
    def embedding(self, batch):
        x, y_all = batch
        y = (y_all < 5).float()
        return self.resnet[:9](x)
    
    def _common_step(self, batch):
        x, y_all = batch
        y = (y_all < 5).float()
        logits = self(x)
        loss = self.loss(logits, y)
        return loss, y, logits

    def training_step(self, batch):
        loss, y, logits = self._common_step(batch)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', self.accuracy(logits, y), on_epoch=True)
        return loss

    def validation_step(self, batch):
        loss, y, logits = self._common_step(batch)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_accuracy', self.accuracy(logits, y), on_epoch=True)

    def freeze_conv_block(self):
        for index, param in enumerate(self.resnet.parameters()):
            if index == 0:
                continue 
            param.requires_grad = False 

        for name, params in self.resnet.named_parameters():
            if 'fc' in name:
                param.requires_grad = True

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.005)

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=256):
        super(MNISTDataModule, self).__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def setup(self, stage=None):
        train = MNIST(root=config['DATAROOT'], train=True,  download=True, transform=self.transform)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            train, 
            [50000, 10000], 
            generator=torch.Generator().manual_seed(42)
            )
        self.test_dataset =  MNIST(root=config['DATAROOT'], train=False, download=True, transform=self.transform)

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