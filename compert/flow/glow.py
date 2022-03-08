import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
from tqdm import tqdm
from metrics.metrics import *
from models.model import TemplateModel


def logabs(x):
    """
    Compute the absolute value of the logarithm of the input 
    """
    return torch.log(torch.abs(x))

class ActNorm(nn.Module):
    """
    ActNorm layer: it scales all its inputs by scale and offset parameters initialized to 1 and 0, respectively 
    """
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        # Scale and offset parameters
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))  # 1 x C x 1 x 1
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))  # 1 x C x 1 x 1

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))  # Creates the parameter "initialized" and saves it as non trainable
        self.logdet = logdet  # Apply log-determinant 

    def initialize(self, input):
        """
        Data-based initialization of the actnorm layer 
        """
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)  # C x B*W*H
            mean = (
                flatten.mean(1)  # Mean by channel 
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)  # 1 x C x 1 x 1 where C contains mean per channel 
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)  # Initialize mean channel value based on the data
            self.scale.data.copy_(1 / (std + 1e-6))  # Initialize inverse std based on the data

    def forward(self, input):
        """
        Application of the actnorm layer 
        """
        _, _, height, width = input.shape 

        # Data-based initialization if not performed yet
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        # Take log-absolute value of the scale parameter of the layer 
        log_abs = logabs(self.scale)
        
        # Log-determinant: dependent on the scale parameter
        logdet = height * width * torch.sum(log_abs)

        # Return actnorm transform of the input and (if required) the logdet as well 
        if self.logdet:
            return self.scale * (input + self.loc), logdet  # Plus because we computed the - mean 

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        """
        Inverse of the actnorm transform 
        """
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    """
    Invertible convolutional layer 
    """
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)  # Weight matrix C X C
        q, _ = torch.qr(weight)  # qr decomposition of the weight matrix to macke sure to get a random orthonormal matrix of weights 
        weight = q.unsqueeze(2).unsqueeze(3)  # C x C x 1 x 1
        self.weight = nn.Parameter(weight)  # Make the weights trainable 

    def forward(self, input):
        _, _, height, width = input.shape

        # Convolve the input with C x C weights such that each in channel is convolved by its own set of weights 
        out = F.conv2d(input, self.weight)  
        # Calculate the log determinant of the convolutional operation 
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()  # Compute sign and negative logarithm of the weight matrix
        )
        return out, logdet
     
    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)  # Invert the C x C matrix of weights 
        )


class InvConv2dLU(nn.Module):
    """
    Computationally efficient calculation of the convolutional layer 
    """
    def __init__(self, in_channel):
        super().__init__()

        # Initialize the weights at random 
        weight = np.random.randn(in_channel, in_channel)
        # Orthogonal matrix from QR decomposition 
        q, _ = la.qr(weight)
        # Perform LU decomposition of the orthogonal matrix Q
        w_p, w_l, w_u = la.lu(q.astype(np.float32))  # w_p: perm matrix, w_l and w_u: upper and lower triangular matrices 
        # Compose the various decomposition matrices
        w_s = np.diag(w_u)  # Take s as the diagonal of u
        w_u = np.triu(w_u, 1)  # Zero-out the main diagonal and the ones below
        u_mask = np.triu(np.ones_like(w_u), 1)  # Upper diagonal matrix of ones   
        l_mask = u_mask.T  # Lower diagonal mastrix of ones 

        # Convert to numpy 
        w_p = torch.from_numpy(np.array(w_p))
        w_l = torch.from_numpy(np.array(w_l))
        w_s = torch.from_numpy(np.array(w_s))
        w_u = torch.from_numpy(np.array(w_u))

        # Untrainable elements
        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(np.array(u_mask)))
        self.register_buffer("l_mask", torch.from_numpy(np.array(l_mask)))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))  # Diagonal matrix of ones 
        # Trainable elements
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        # Retrieve the weight matrix from the decomposition 
        weight = self.calc_weight()

        # Perform convolution and calculate the log determinant 
        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        """
        The inverse of the LU-based conv2d transformation
        """
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        # Convolutional layer with zero weights 
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))  # 1 x C x 1 x 1

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)  # Increase dimensionality of the convolution input by 2 in the last two dimensions 
        out = self.conv(out)  
        out = out * torch.exp(self.scale * 3)
        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        # Shallow convolutional network within affine coupling (last layer is the zero-initialized convolution)
        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        # Initialization of the layers 
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        # Chunk the input into two in the channel dimension 
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            # Apply and chunk the result of the convolution into two 
            log_s, t = self.net(in_a).chunk(2, 1)  # NB. In case of odd channels the first element takes an extra dimension 

            #s = torch.exp(log_s)  # Paper 
            s = torch.sigmoid(log_s + 2)  # New solution 
            
            #out_a = s * in_a + t
            out_b = (in_b + t) * s 

            # The log det here is not averaged 
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)  # B x C*W*H
            
        else:
            net_out = self.net(in_a)
            in_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet  # Reconcatenate on the channel dimension  


    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)
            
            # in_a = (out_a - t) / s
            in_b = out_b/ s - t  

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()

        # Initialize layers of a flow unit 
        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out) 

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        # Sum the log determinant of the three steps 
        return out, logdet

    def reverse(self, output):
        """
        Since all the functions are reversable, we can compute the output from the inputs 
        """
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input


def gaussian_log_p(x, mean, log_sd):
    """
    Basic Gaussian negative log likelihood  
    """
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    """
    Reparametrization     
    """
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()
        
        # The degree of increase in the number of squeezed channels 
        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split
        
        # Set the prior 
        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)

        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        # Split the height and the width 
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)  
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)  # Push the halved channels at the end 
        # Squeezing means aggregating half of the width and height dimensions into channel
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)  # Augment the number of channel multiplyimg by 4 

        # Accumulate the log determinant 
        logdet = 0  

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            # Divide the output into 2 along the channel dimension 
            out, z_new = out.chunk(2, 1)  # Divide by the channel dimension (B x C*2 x W x H) 
            mean, log_sd = self.prior(out).chunk(2, 1)  # Apply conv 2D and rechunk obtaining 2 tensors by (B x C*2 x W x H) 
            log_p = gaussian_log_p(z_new, mean, log_sd)  # Apply to chunks 
            log_p = log_p.view(b_size, -1).sum(1)  # Sum across dimensions other than batch size 

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out
        
        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        """
        From the output of the flow + random noise, go back to the images 
        """
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)  # Use the prior on output of the convolutions and chunk on the channel dim (B x C*2 x W x H)
                z = gaussian_sample(eps, mean, log_sd)  # Sample Z encoding (reparametrization trick)
                input = torch.cat([output, z], 1)  # Input is the concatenation of the output and Gaussian noise (B x C*4 x W x H)

            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        # Apply the flow in reverse 
        for flow in self.flows[::-1]:
            input = flow.reverse(input)
            
        b_size, n_channel, height, width = input.shape    

        # Reverse the squeezing transformation 
        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )  # Go back to original image dimension
        return unsqueezed


class Glow(TemplateModel):
    def __init__(
        self, in_channels, n_flow, n_block, affine=True, conv_lu=True,
        in_width=64, in_height=64, n_bits = 8, device = 'cuda'):
        super().__init__()

        # The number of bits is used to compute the loss
        self.n_bits = n_bits 
        self.n_bins = 2**n_bits
        self.n_pixels = in_width * in_height * in_channels
        self.n_block = n_block
        self.n_flow = n_flow
        self.device = device

        # Fix the rest of the variables
        self.in_width = in_width
        self.in_height = in_height 
        self.blocks = nn.ModuleList()
        self.in_channels = in_channels
        n_channel = self.in_channels
        
        # Append a series of blocks on top of each other
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2  # The number of channels is augmented by two each time due to the concatenation of z with output 
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))

        # z-shapes calculation
        self.z_shapes = self.calc_z_shapes()

        # Setup metrics 
        self.metrics = TrainingMetrics(self.in_height, self.in_width, self.in_channels, self.z_shapes, flow=True, device = self.device)

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        # You get a z dimension for each step 
        return log_p_sum, logdet, z_outs
    
    # Start from a list of z encodings 
    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)
        return input

    def calc_z_shapes(self):
        """
        Caluclate the shapes of the latents z for all levels 
        """
        # Shapes of z on all layers 
        z_shapes = []
        input_size = self.in_width 
        n_channel = self.in_channels

        for i in range(self.n_block - 1):
            # For each block, the image size decreases and the number of channels goes up
            input_size //= 2  
            n_channel *= 2
            z_shapes.append((n_channel, input_size, input_size))

        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))
        return z_shapes
    
    def sample(self, n_samples, temperature):
        """
        Sample z embeddings from normal distributions 
        """
        z_sample = []
        
        for z in self.z_shapes:
            z_new = torch.randn(n_samples, *z) * temperature 
            z_sample.append(z_new.to(self.device))
        
        with torch.no_grad():
            generated_img = self.reverse(z_sample)
        return generated_img
        
    
    def generate(self, input):
        """
        Given an input, it calls the Glow model in forward and backward directions 
        """
        # Forward encoding of the input 
        _, _, z_list = self.forward(input)
        # backward image decoding 
        output = self.reverse(z_list, reconstruct=True)
        return output 


    def update_model(self, train_loader, epoch, optimizer):
        """
        Compute a forward step and returns the losses 
        """
        # Total loss
        training_loss = 0
        self.metrics.reset()
        for batch in tqdm(train_loader): 
            # Collect batch 
            X_batch = batch['X'].to(self.device) # Load batch
            log_p, log_det, z_outs = self.forward(X_batch)

            # Compute the loss
            loss = -log(self.n_bins) * self.n_pixels
            log_det = log_det.mean()  # Average the log-determinant across batch
            loss = loss + log_det + log_p
            loss = (-loss / (log(2) * self.n_pixels)).mean()

            # Optimizer step  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        avg_loss = training_loss/len(train_loader)
        print(f'Mean loss after epoch {epoch}: {avg_loss}')
        self.metrics.print_metrics()
        return dict(loss=avg_loss), self.metrics.metrics


    def evaluate(self, loader, dataset, checkpoint_path='', fold = 'val'):
        """
        Validation loop 
        """
        if fold == 'test':
            # Testing phase 
            self.load_state_dict(torch.load(checkpoint_path))

        val_loss = 0
        # Zero out the metrics for the next step
        self.metrics.reset()
        
        for val_batch in loader:
            X_val_batch = val_batch['X'].to(self.device) # Load batch
            with torch.no_grad():
                log_p, log_det, z_outs = self.forward(X_val_batch)

            # Accumulate the validation loss 
            loss = -log(self.n_bins) * self.n_pixels
            log_det = log_det.mean()  # Average the log-determinant across batch
            loss = loss + log_det + log_p
            loss = (-loss / (log(2) * self.n_pixels)).mean()

            val_loss += loss

            # Update the image metrics
            with torch.no_grad():
                val_batch_reconstruct = self.reverse(z_outs, reconstruct=True)

        avg_validation_loss = val_loss/len(loader) 
        print(f'Average validation loss: {avg_validation_loss}')
        self.metrics.print_metrics(flow=True)
        return dict(loss=avg_validation_loss), self.metrics.metrics
