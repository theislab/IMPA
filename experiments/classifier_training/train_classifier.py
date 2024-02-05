import argparse
from IMPA.dataset.data_loader import CellDataLoader
from IMPA.model import Discriminator
from IMPA.solver import IMPAmodule
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
import torch
from torch import optim
from torch import nn
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import yaml

class Classifier(LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, labels = batch["X"], batch["mol_one_hot"].argmax(1)
        outputs = self.model(X, None)
        loss = self.criterion(outputs, labels)
        return loss

    def val_step(self, batch, batch_idx):
        X, labels = batch["X"], batch["mol_one_hot"].argmax(1)
        outputs = self.model(X, None)
        loss = self.criterion(outputs, labels)
        acc = (torch.argmax(outputs, dim=1) == labels).sum().item() / len(labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.val_step(batch, batch_idx)
        return loss
        
    def test_step(self, batch, batch_idx):
        loss = self.val_step(batch, batch_idx)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def main(args):
    args = OmegaConf.create(args)
    dataloader = CellDataLoader(args)

    classifier_net = Discriminator(img_size=96,
                                    num_domains=dataloader.n_mol, 
                                    max_conv_dim=512, 
                                    in_channels=args.n_channels, 
                                    dim_in=64,
                                    multi_task=False)
    
    # Set up the checkpoint callback
    dest_path = Path(args.dest_path)
    checkpoint_callback = ModelCheckpoint(
        dirpath=dest_path,
        filename='best_model',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
    )

    classifier = Classifier(classifier_net, learning_rate=0.0001)

    # Set up the trainer with the checkpoint callback
    trainer = Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(classifier, dataloader.train_dataloader(), dataloader.val_dataloader())

    # Test the model
    trainer.test(classifier, dataloader.val_dataloader())

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="The input for training the batch classifier models")

    # Define command-line arguments
    parser.add_argument('--args_path', type=str, help='Description of argument1')
    # Add more arguments as needed

    args = parser.parse_args()
    config_path = args.args_path
    
    with open(config_path, 'r') as file:
        args = yaml.safe_load(file)

    main(args)
    