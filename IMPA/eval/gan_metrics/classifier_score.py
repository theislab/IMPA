import sys
sys.path.insert(0, '/home/icb/alessandro.palma/IMPA/imCPA/compert')
from model import *

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

class Classifier(nn.Module):
    """Discriminator network for the GAN model 
    """
    def __init__(self, img_size=96, max_conv_dim=512, in_channels=3, dim_in=64):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(in_channels, dim_in, 3, 1, 1)]

        repeat_num = math.ceil(np.log2(img_size)) - 2
        # for _ in range(repeat_num):
        #     dim_out = min(dim_in*2, max_conv_dim)
        #     blocks += [ResBlk(dim_in, dim_out, downsample=True)]
        #     dim_in = dim_out
        
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [nn.Conv2d(dim_in, dim_out, 3, stride=2)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2, inplace=True)]
        blocks += [nn.Conv2d(dim_out, dim_out, 2, 1, 0)]
        blocks += [nn.LeakyReLU(0.2, inplace=True)]
        # Final convolutional layer that points to the number of domains 
        blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]
        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        # Apply the network on X 
        out = self.conv(x)
        out = out.view(out.size(0), -1)  # (batch, 1) 
        return out.squeeze()


def classifier_score(true_images, fake_images, epochs=20, patience=20):
    # Initialize classifier
    clf = Classifier(img_size=96, 
                     max_conv_dim=512, 
                     in_channels=3, 
                     dim_in=64).to('cuda')
    print(clf)
    
    # Best score and optimizer equal to zero 
    best_score, steps_best = 0, 0
    # Optimizer and loss
    optimizer = torch.optim.Adam(params=clf.parameters(),
                                 lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # The labels for true and fake observations 
    y_true = torch.ones(true_images.shape[0])
    y_fake = torch.zeros(fake_images.shape[0])
    
    X = torch.cat([true_images, fake_images], dim=0)
    Y = torch.cat([y_true, y_fake])

    train_size = int(0.8*len(X))
    idx_perm = torch.randperm(len(X))
    train_idx = idx_perm[:train_size+1]
    test_idx = idx_perm[train_size+1:]
    
    X_train, y_train = X[train_idx], Y[train_idx]
    X_test, y_test = X[test_idx], Y[test_idx]
    

    train_dataloader = DataLoader(TensorDataset( X_train, y_train), shuffle=True, batch_size=16, drop_last=True)
    test_dataloader = DataLoader(TensorDataset(X_test, y_test), shuffle=True, batch_size=16)
    
    for _ in range(epochs):
        # Train
        clf.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            X = batch[0].to('cuda')
            y = batch[1].to('cuda')
            y_hat = clf(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
        
        # Test 
        clf.eval()
        y_hats = []
        ys = []
        for batch in test_dataloader:
            X = batch[0].to('cuda')
            y = batch[1].to('cuda')
            with torch.no_grad():
                y_hat = torch.round(torch.sigmoid(clf(X)))
            y_hat = y_hat.to('cpu').tolist()
            y_hats += y_hat if type(y_hat)==list else [y_hat]
            ys += y.to('cpu').tolist()

        score_test = accuracy_score(y_hats, ys)
        if score_test >= best_score:
            best_score = score_test
            steps_best = 0
            print(f'Accuracy: {best_score}')
        else:
            steps_best += 1
            
        if steps_best > patience:
            break 
