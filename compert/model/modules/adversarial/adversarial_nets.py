from tkinter import S
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


"""
The latent discriminator network
"""

class LeakyReLUConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, norm='Instance'):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        # Normalization layer, can be instance or batch
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(out_channels, affine=True)]
        if norm == 'Batch':
            model += [nn.BatchNorm2d(out_channels)]
        # Leaky ReLU loss with default parameters
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

"""
The latent discriminator: tries to predict drugs on the latent space 
"""

class DiscriminatorNet(nn.Module):
  def __init__(self, init_fm, out_fm, depth, num_outputs, norm='Instance'):
    super(DiscriminatorNet, self).__init__()
    self.init_fm = init_fm
    self.out_fm = out_fm 
    self.depth = depth 
    self.num_outpus = num_outputs
    self.norm = norm

    # First number of feature maps 
    in_fm = self.init_fm 

    # Go as deep as necessary to transform the spatial dimension to 1 
    model = []
    for i in range(depth):
        model += [LeakyReLUConv2d(in_fm, out_fm, kernel_size=3, stride=2, padding=1, norm=self.norm if i < depth-1 else 'None')]
        if i == 0:
            in_fm = out_fm

    # Final convolution to produce number of outputs on the feature map dimension
    model += [nn.Conv2d(in_fm, num_outputs, kernel_size=1, stride=1, padding=0)]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    out = self.model(x)
    out = out.view(out.size(0), out.size(1)) # BxCx1x1 --> BxC
    return out


"""
Label encoder networks start from label embeddings and produce a high dimensional encoding that is used to condition the latent 
"""

class LabelEncoder(nn.Module):
    def __init__(self, output_dim, input_fm, output_fm):
        super(LabelEncoder, self).__init__()

        self.output_dim = output_dim  # Spatial dimension of the transposed embeddings
        self.input_fm = input_fm
        self.output_fm = output_fm  # Number of feature maps in the middle channels 
        
        # Depth required for upsampling 
        depth = int(np.log2(self.output_dim//3))  
        
        # Initial feature map setup
        in_fm = self.input_fm
        out_fm = self.output_fm

        # Initialize the modules 
        self.modules = [torch.nn.ConvTranspose2d(in_fm, out_fm, kernel_size = 3, stride = 2, padding=0),
                        torch.nn.LeakyReLU(inplace=True)] # 1x1 --> 3x3 spatial dimension 

        # Conv2d transpose till the latent dimension 
        for i in range(depth):
            self.modules.append(torch.nn.ConvTranspose2d(out_fm, out_fm, kernel_size = 4, stride = 2, padding=1))
            if i < depth-1:
                self.modules.append(torch.nn.LeakyReLU(inplace=True))

        self.transp = torch.nn.Sequential(*self.modules)

    def forward(self, z):
        # Since working with linear embeddings, we must reshape the to a unitary spatial dimension 
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        x = self.transp(z)
        return x


class LabelEncoderLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LabelEncoderLinear, self).__init__()
        self.input_fm = input_dim
        self.output_dim = output_dim 

        self.mlp = nn.Sequential(
            nn.Linear(self.input_fm, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dim, self.output_dim))

    def forward(self, z):
        out = self.mlp(z)
        return out


class LabelEncoderLinearMultiTask(nn.Module):
    def __init__(self, input_dim, output_dim, num_drugs):
        super(LabelEncoderLinearMultiTask, self).__init__()
        self.input_fm = input_dim
        self.output_dim = output_dim 
        self.num_drugs = num_drugs

        self.shared = nn.Sequential(
            nn.Linear(self.input_fm, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512))
    
        self.nets = nn.ModuleList()
        for _ in range(self.num_drugs):
            self.nets += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, self.output_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.nets:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


"""
Disentanglement classifier for classification 
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

        # Two layers of leaky relu convolutions
        model = []
        for i in range(2):
            model += [LeakyReLUConv2d(in_fm, out_fm, kernel_size=3, stride=2, padding=1, norm='Batch')]
            if i == 0:
                in_fm = out_fm
        
        flattened_dim = (np.around(self.init_dim/4)**2*out_fm).astype(int)
        # Compile model 
        self.conv = nn.Sequential(*model)
        
        # Linear classification layer 
        self.linear = torch.nn.Linear(flattened_dim, self.num_outpus)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        return self.linear(out)


"""
GAN discriminator for prestine vs true based on patchGAN 
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

        # Four layers of strided LeakyReLU
        model = []
        for i in range(5):
            model += [LeakyReLUConv2d(in_fm, out_fm, kernel_size=4, stride=2, padding=1, norm='none')]
            in_fm = out_fm
            out_fm = in_fm*2

        # Last convolution with stride 1
        model += [nn.Conv2d(in_fm, 1, kernel_size=4, stride=1, padding=1, bias=False)]

        # Compile model 
        self.conv = nn.Sequential(*model)

        # Acivation function
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)  # BxCx6x6
        out = out.view(out.shape[0], -1)
        return self.activation(out).mean(1).view(out.size(0))

    def discriminator_pass(self, X, X_hat, loss):
        self.train()
        # From batch
        data_real = X
        # From the autoencoder model 
        data_fake = X_hat.detach()  
        # Assign labels 
        label_real = torch.ones(data_real.shape[0]).to(self.device) 
        label_fake = torch.zeros(data_fake.shape[0]).to(self.device)
        
        # Predict on the real dataset vs the fake one 
        pred_real = self.forward(data_real)
        pred_fake = self.forward(data_fake)
        gan_loss_real = loss(pred_real, label_real)
        gan_loss_fake = loss(pred_fake, label_fake)
        # Average losses 
        return (gan_loss_real + gan_loss_fake)/2

    def generator_pass(self, X_hat, loss):
        # Put model in eval mode in case there are BN layers 
        self.eval()
        labels = torch.ones(X_hat.shape[0]).to(self.device)  
        pred = self.forward(X_hat)  # Prediction by the model on the generated images (single value per batch observation)
        
        # GAN loss function (cross entropy)
        gan_loss = loss(pred, labels)  
        return gan_loss

 
"""
GAN classifier for correctness of the cell
"""
class GANClassifier(nn.Module):
    def __init__(self, init_dim, in_channels, init_fm, num_outputs_drug):

        super(GANClassifier, self).__init__()
        self.init_dim = init_dim  # Starting spatial dimension 
        self.in_channels = in_channels  # Spatial dimension
        self.init_fm = init_fm  # Input feature maps
        self.num_outputs_drug = num_outputs_drug  # Number of classes for the classification 

        # First number of feature maps 
        in_fm = self.in_channels
        out_fm = self.init_fm

        model = []
        for i in range(5):
            model += [LeakyReLUConv2d(in_fm, out_fm, kernel_size=4, stride=2, padding=1, norm='none')]
            in_fm = out_fm
            out_fm *= 2
        
        # Dimension of the latent 
        model += [torch.nn.Conv2d(in_fm, self.num_outputs_drug, kernel_size=3, bias=False)]  # Division by 16 because we use 4 2x2 strided convolutions 
        
        # Compile model 
        self.conv = nn.Sequential(*model)

    def forward(self, x):
        out = self.conv(x).view(x.shape[0], -1)
        return out
    
    def discriminator_pass(self, X, loss, labels_drug):
        self.train()
        out_drug = self.forward(X)
        # Given input image X, train the classifier on the true label 
        loss_drug = loss(out_drug, labels_drug)  # Cross entropy 
        return loss_drug

    def generator_pass(self, X_counterf, loss, swapped_idx_drugs):
        self.eval()
        # Swapped indices are meant to fool the classifier 
        pred_drug = self.forward(X_counterf)  # Prediction by the model on the generated images (single value per batch observation)

        gan_loss_drug = loss(pred_drug, swapped_idx_drugs)  # GAN loss function (cross entropy)
        return gan_loss_drug

"""
Implement discriminator and classifier together 
"""

class DiscriminatorClassifier(nn.Module):
    def __init__(self, init_dim, in_channels, init_fm, num_outputs_drug, device):

        super(DiscriminatorClassifier, self).__init__()
        self.device = device
        self.init_dim = init_dim  # Starting spatial dimension 
        self.in_channels = in_channels  # Spatial dimension
        self.init_fm = init_fm  # Input feature maps
        self.num_outputs_drug = num_outputs_drug  # Number of classes for the classification 

        # First number of feature maps 
        in_fm = self.in_channels
        out_fm = self.init_fm

        model = []
        for i in range(5):
            model += [LeakyReLUConv2d(in_fm, out_fm, kernel_size=4, stride=2, padding=1, norm='none')]
            in_fm = out_fm
            out_fm *= 2
        
        # The main body of both classifier and discriminator 
        self.model = nn.Sequential(*model)

        # Discriminator and classifier heads 
        self.classif = torch.nn.Conv2d(in_fm, self.num_outputs_drug, kernel_size=3, bias=False)   
        self.discr = torch.nn.Conv2d(in_fm, 1, kernel_size=3, stride=1, bias=False)
        self.activation_discr = torch.nn.Sigmoid()

    def forward(self, x):
        # Output of the shared convolution layer 
        h = self.model(x)
        # Output of the classification layer 
        out_class = self.classif(h).view(h.shape[0], -1)
        # Output of Patch GAN
        out_patch = self.activation_discr(self.discr(h)).mean(1).view(h.size(0))
        return out_patch, out_class
    
    def discriminator_pass(self, X, X_hat, y_real, loss_discr, loss_classif):
        self.train()
        # Get the output of both patch and classifier 
        out_patch_fake, _ = self.forward(X_hat.detach())
        out_patch_real, out_classif_real = self.forward(X)
    
        # Discriminator loss
        label_real = torch.ones(X.shape[0]).to(self.device) 
        label_fake = torch.zeros(X_hat.shape[0]).to(self.device)
        loss_real = loss_discr(out_patch_real, label_real)
        loss_fake = loss_discr(out_patch_fake, label_fake)

        # Classifier loss
        loss_class = loss_classif(out_classif_real, y_real)  # Cross entropy 
    
        return (loss_real + loss_fake)/2 , loss_class

    def generator_pass(self, X_hat, y_fake, loss_discr, loss_classif):
        self.eval()
        
        # Swapped indices are meant to fool the classifier 
        out_patch_fake, out_classif_fake = self.forward(X_hat)

        # Classifier loss 
        loss_class = loss_classif(out_classif_fake, y_fake)

        # Discriminator loss
        labels_fake = torch.ones(X_hat.shape[0]).to(self.device)  
        loss_fake = loss_discr(out_patch_fake, labels_fake)
        
        return loss_fake, loss_class 

"""
Style encoder (from image to embedding)
"""
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class StyleEncoder(nn.Module):
    def __init__(self, in_channels=3, in_width=96, style_dim=64, num_drugs=2, max_conv_dim=512):
        super().__init__()
        dim_in = 64
        blocks = []
        blocks += [nn.Conv2d(in_channels, dim_in, 3, 1, 1)]
        repeat_num = int(np.log2(in_width)) - 1

        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [LeakyReLUConv2d(dim_in, dim_out, 4, 2, 1, norm='Instance')]
            dim_in = dim_out

        blocks += [nn.Conv2d(dim_out, dim_out, 3, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_drugs):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s
