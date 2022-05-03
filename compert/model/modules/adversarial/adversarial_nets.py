import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


"""The latent discriminator network
"""

class LeakyReLUConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, norm='None'):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(out_channels, affine=False)]
        if norm == 'Batch':
            model += [nn.BatchNorm2d(out_channels, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class DiscriminatorNet(nn.Module):
  def __init__(self, init_fm, out_fm, depth, num_outputs):
    super(DiscriminatorNet, self).__init__()
    self.init_fm = init_fm
    self.out_fm = out_fm 
    self.depth = depth 
    self.num_outpus = num_outputs

    # First number of feature maps 
    in_fm = self.init_fm 

    model = []
    for i in range(depth):
        model += [LeakyReLUConv2d(in_fm, out_fm, kernel_size=3, stride=2, padding=1)]
        if i == 0:
            in_fm = out_fm

    # Final convolution to produce number of outputs on the feature map dimension
    model += [nn.Conv2d(in_fm, num_outputs, kernel_size=1, stride=1, padding=0)]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    out = self.model(x)
    out = out.view(out.size(0), out.size(1)) # BxCx1x1 --> BxC
    return out


"""The covariate encoding network 
"""

class LabelEncoder(nn.Module):
    def __init__(self, output_dim, input_fm, output_fm):
        super(LabelEncoder, self).__init__()

        self.output_dim = output_dim
        self.input_fm = input_fm
        self.output_fm = output_fm
        
        # Depth 
        depth = int(np.log2(self.output_dim//3))  # Requited for upsampling
        
        # Initial feature map setup
        in_fm = self.input_fm
        out_fm = self.output_fm

        # Initialize the modules 
        self.modules = [torch.nn.ConvTranspose2d(in_fm, out_fm, kernel_size = 3, stride = 2, padding=0),
                        torch.nn.ReLU()] # 1x1 --> 3x3 spatial dimension 

        # Conve2d transpose till the latent dimension 
        for i in range(depth):
            self.modules.append(torch.nn.ConvTranspose2d(out_fm, out_fm, kernel_size = 4, stride = 2, padding=1))
            self.modules.append(torch.nn.ReLU())
        self.transp = torch.nn.Sequential(*self.modules)

    def forward(self, z):
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        x = self.transp(z)
        return x


class LabelEncoderLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LabelEncoderLinear, self).__init__()
        self.input_fm = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim))

    def forward(self, z):
        out = self.mlp(z)
        return out


"""Disentanglement classifier for classification 
"""

class DisentanglementClassifier(nn.Module):
    def __init__(self, init_dim, init_fm, out_fm, num_outputs):
        super(DisentanglementClassifier, self).__init__()
        self.init_dim = init_dim  # Spatial dimension
        self.init_fm = init_fm  # Input feature maps
        self.out_fm = out_fm  # Output feature maps 
        self.num_outpus = num_outputs  # Number of classes for the classification 

        # First number of feature maps 
        in_fm = self.init_fm 

        model = []
        for i in range(2):
            model += [LeakyReLUConv2d(in_fm, out_fm, kernel_size=4, stride=1, padding=1, norm='Instance')]
            if i == 0:
                in_fm = out_fm
        
        flattened_dim = self.init_dim *  out_fm
        # Compile model 
        self.conv = nn.Sequential(*model)
        
        # Linear classification layer 
        self.linear = torch.nn.Linear(flattened_dim, self.num_outpus)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        return out


"""GAN discriminator for prestine vs true
"""

class GANDiscriminator(nn.Module):
    def __init__(self, init_dim, init_ch, init_fm, device='cuda'):
        super(GANDiscriminator, self).__init__()
        self.device = device
        self.init_dim = init_dim  # Spatial dimension
        self.init_ch = init_ch  # Input feature maps (3) 
        self.init_fm = init_fm  # The number of feature maps in the first layer 

        # Modifiable numbers of feature maps
        in_fm = self.init_ch
        out_fm = self.init_fm

        model = []
        for i in range(4):
            model += [LeakyReLUConv2d(in_fm, out_fm, kernel_size=4, stride=2, padding=1, norm='Instance')]
            in_fm = out_fm
            out_fm = in_fm *2

        # Last convolution with stride 1
        model += [nn.Conv2d(in_fm, out_fm, kernel_size=4, stride=1, padding=1, norm='Instance')]

        # Compile model 
        self.conv = nn.Sequential(*model)

        # Acivation function
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        return self.activation(out).mean(1).view(out.size(0))

    def discriminator_pass(self, X, X_hat, loss):
        self.train()
        data_real = X  # From batch
        data_fake = X_hat.detach()  # From the autoencoder
        label_real = torch.ones(data_real.shape[0]).to(self.device) 
        label_fake = torch.zeros(data_fake.shape[0].to(self.device))

        pred_real = self.forward(data_real).squeeze()
        pred_fake = self.forward(data_fake).squeeze()
        gan_loss_real = loss(pred_real, label_real)
        gan_loss_fake = loss(pred_fake, label_fake)
        return (gan_loss_real + gan_loss_fake)/2

    def generator_pass(self, X_hat, loss):
        self.eval()
        labels = torch.ones(X_hat.shape[0]).to(self.device)   # Define y_hat as true 
        pred = self.forward(X_hat)  # Prediction by the model on the generated images (single value per batch observation)

        gan_loss = loss(pred, labels)  # GAN loss function (cross entropy)
        return gan_loss

 
"""GAN discriminator for prestine vs true
"""


class GANClassifier(nn.Module):
    def __init__(self, init_dim, in_channels, init_fm, num_outputs_drug, num_outputs_moa=0, predict_moa=None):

        super(GANClassifier, self).__init__()
        self.init_dim = init_dim
        self.in_channels = in_channels  # Spatial dimension
        self.init_fm = init_fm  # Input feature maps
        self.num_outputs_drug = num_outputs_drug  # Number of classes for the classification 
        self.num_outputs_moa = num_outputs_moa
        self.predict_moa = predict_moa 

        # First number of feature maps 
        in_fm = self.in_channels
        out_fm = self.init_fm

        model = []
        for i in range(4):
            model += [LeakyReLUConv2d(in_fm, out_fm, kernel_size=4, stride=2, padding=1, norm='Instance')]
            in_fm = out_fm
            out_fm *= 2
        
        # Dimension of the latent 
        flattened_dim = (self.init_dim//16)**2  * in_fm
        
        # Compile model 
        self.conv = nn.Sequential(*model)
        
        # Linear classification layer. The network has two heads, one for the drug and one for the moa 
        self.linear_drug = torch.nn.Linear(flattened_dim, self.num_outputs_drug)
        if self.predict_moa:
            self.linear_moa = torch.nn.Linear(flattened_dim, self.num_outputs_moa)

    def forward(self, x):
        out = self.conv(x)
        # Output drug
        out_drug = self.linear_drug(out.view(out.shape[0], -1))
        # Output moa
        if self.predict_moa:
            out_moa = self.linear_moa(out.view(out.shape[0], -1))
        else:
            out_moa = None
        return out_drug, out_moa
    

    def discriminator_pass(self, X, loss, labels_drug, labels_moa=None):
        self.train()
        out_drug, out_moa = self.forward(X)
        # Given input image X, train the classifier on the true label 
        loss_drug = loss(out_drug, labels_drug)
        loss_moa = loss(out_moa, labels_moa) if self.predict_moa else 0
        return (loss_drug+loss_moa)/2

    
    def generator_pass(self, X_counterf, loss, swapped_idx_drugs, swapped_idx_moas):
        self.eval()
        # Swapped indices are meant to fool the classifier 
        pred_drug, pred_moa = self.forward(X_counterf)  # Prediction by the model on the generated images (single value per batch observation)

        gan_loss_drug = loss(pred_drug, swapped_idx_drugs)  # GAN loss function (cross entropy)
        gan_loss_moa = loss(pred_moa, swapped_idx_moas) if self.predict_moa else 0
        return (gan_loss_drug + gan_loss_moa)/2



if __name__ == '__main__':
    # x = torch.rand(3, 512, 6, 6)
    # dis = DiscriminatorNet(512, 256, 3, 4)
    # print(dis(x).shape)

    x = torch.rand(64, 3, 96, 96)
    x_hat = torch.rand(64, 3, 96, 96)
    loss = torch.nn.BCELoss()

    enc = GANDiscriminator(init_dim=96, init_ch=3, init_fm=64, device='cuda')
    print(enc.discriminator_pass(x, x_hat, loss))
    print(enc.generator_pass(x_hat, loss))    

    # enc = GANClassifier(init_dim=96, in_channels=3, init_fm=64, num_outputs=15)
    # print(enc(x).shape)
