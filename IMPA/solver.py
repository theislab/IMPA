from munch import Munch
from os.path import join as ospj

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pytorch_lightning import LightningModule

from IMPA.checkpoint import CheckpointIO
from IMPA.eval.eval import evaluate
from IMPA.model import build_model
from IMPA.utils import he_init, print_network, swap_attributes, debug_image

class IMPAmodule(LightningModule):
    def __init__(self, args, dest_dir, datamodule):
        """Initialize IMPA module

        Args:
            args (dict): dictionary with hparams 
            dest_dir (pathlib.Path): directory where to save results 
            datamodule (CellDataLoader): data module class
        """
        super().__init__()
        self.args = args
        self.dest_dir = dest_dir
        self.automatic_optimization = False
        
        # Save hyperparameters for lightning/hydra
        self.save_hyperparameters(args)
        
        # Get the nets
        self.nets = build_model(args, datamodule.n_mol, self.device)
        self.loader_test = datamodule.val_dataloader()
        self.embedding_matrix = datamodule.embedding_matrix
        self.id2mol = datamodule.id2mol
        self.num_domains = datamodule.n_mol
        
        # Set modules as attributes of the solver class
        for name, module in self.nets.items():
            print_network(module, name)
            setattr(self, name, module)

        # Initialize checkpoints
        self._create_checkpoints()

        # Initialize nets 
        for name, network in self.named_children():
            print('Initializing %s...' % name)
            if name != 'embedding_matrix':
                network.apply(he_init)
        
        # Initial values of diversity loss
        self.initial_lambda_ds = self.args.lambda_ds
                
    def configure_optimizers(self):
        """Initialize optimizer

        Returns:
            dict: dictionary with optimizers for different neural networks
        """
        self.optims = {}
        for net in self.nets.keys():
            params = list(self.nets[net].parameters())
            if net == 'mapping_network' and self.args.trainable_emb:
                params += list(self.embedding_matrix.parameters())  
                
            # Define optimizers and LR scheduler here
            self.optims[net] = Adam(params=params,
                                lr=self.args.f_lr if net == 'mapping_network' else self.args.lr,
                                betas=[self.args.beta1, self.args.beta2],
                                weight_decay=self.args.weight_decay)
        return list(self.optims.values())
    
    def training_step(self, batch): 
        """Method for IMPA training across a pre-defined number of iterations. 
        """   
        generator_opt, style_encoder_opt, discriminator_opt, mapping_network_opt = self.optimizers()
        # Fetch the real and fake inputs and outputs 
        x_real, y_one_hot = batch['X'].to(self.device), batch['mol_one_hot']
        y_org = y_one_hot.argmax(1).long().to(self.device)
        y_trg = swap_attributes(y_one_hot, y_org, self.device).argmax(1).long().to(self.device)

        # Get the perturbation embedding for the target mol
        z_emb_trg = self.embedding_matrix(y_trg).to(self.device)
        z_emb_org = self.embedding_matrix(y_org).to(self.device)

        # Pick two random weight vectors 
        if self.args.stochastic:
            z_trg, z_trg2 = (torch.randn(x_real.shape[0], self.args.z_dimension).to(self.device), 
                                torch.randn(x_real.shape[0], self.args.z_dimension).to(self.device))
        else:
            z_trg, z_trg2 = None, None
        
        # Train the discriminator
        d_loss, d_losses_latent = self._compute_d_loss(
            x_real, y_org, y_trg, z_emb_trg=z_emb_trg, z_trg=z_trg)
        discriminator_opt.zero_grad()
        self.manual_backward(d_loss)
        discriminator_opt.step()
        
        # Train the generator
        g_loss, g_losses_latent = self._compute_g_loss(
            x_real, y_org, y_trg, z_emb_trg=z_emb_trg, z_emb_org=z_emb_org, z_trgs=[z_trg, z_trg2])
        style_encoder_opt.zero_grad()
        generator_opt.zero_grad()
        mapping_network_opt.zero_grad()   
        self.manual_backward(g_loss)
        style_encoder_opt.step()
        generator_opt.step()
        mapping_network_opt.step() 
    
        # Decay weight for diversity sensitive loss (moves towards 0) - Decrease sought amount of diversity 
        if self.args.lambda_ds > 0 and self.args.stochastic:
            self.args.lambda_ds -= (self.initial_lambda_ds / self.args.ds_iter)

        # Log the losses 
        all_losses = dict()
        for loss, prefix in zip([d_losses_latent, g_losses_latent],
                                ['D/latent_', 'G/latent_']):
            for key, value in loss.items():
                all_losses[prefix + key] = value
        all_losses['G/lambda_ds'] = self.args.lambda_ds
        self.log_dict(all_losses)
    
    def on_train_start(self):
        self.ckptios.append(CheckpointIO(ospj(self.dest_dir, self.args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims))
        
    def on_train_epoch_end(self):
        # Scores on the validation images 
        rmse_disentanglement_dict = evaluate(self.nets,
                                                self.loader_test,
                                                self.device,
                         zz                       self.dest_dir,
                                                self.args.embedding_folder,
                                                args=self.args,
                                                embedding_matrix=self.embedding_matrix)
        self.log_dict(rmse_disentanglement_dict)
        
        inputs_val = next(iter(self.loader_test)) 
        debug_image(self.nets, 
                    self.embedding_matrix, 
                    self.args, 
                    inputs=inputs_val, 
                    step=self.current_epoch+1, 
                    device=self.device, 
                    id2mol=self.id2mol,expli
                    dest_dir=self.dest_dir, 
                    num_domains=self.num_domains)

        # Save model checkpoints
        self._save_checkpoint(step=self.current_epoch+1)        
        
    
    def _compute_d_loss(self, x_real, y_org, y_trg, z_emb_trg, z_trg):
        """Compute the discriminator loss real batches

        Args:
            x_real (torch.tensor): batch of real data
            y_org (torch.tensor): domain labels of the real data 
            y_trg (torch.tensor): domain labels of the fake data 
            z_emb_trg (torch.tensor): embedding of the target perturbation 
            z_trg (torch.tensor): random noise vector 
        """

        # Gradient requirement for gradient penalty loss
        x_real.requires_grad_()
        out = self.nets.discriminator(x_real, y_org)
        # Discriminator assigns a 1 to the real 
        loss_real = self._adv_loss(out, 1)
        # Gradient-based regularization (penalize high gradient on the discriminator)
        loss_reg = self._r1_reg(out, x_real)

        # The discriminator does not train the mapping network and the generator, so they need no gradient 
        with torch.no_grad():
            if self.args.stochastic:
                z_emb_trg = torch.cat([z_emb_trg, z_trg], dim=1)

            # Single of the noisy embedding vector to a style vector 
            s_trg = self.nets.mapping_network(z_emb_trg)  
            
            # Generate the fake image
            _, x_fake = self.nets.generator(x_real, s_trg)

        # Discriminator trained to predict transformed image as fake in its domain 
        out = self.nets.discriminator(x_fake, y_trg)
        loss_fake = self._adv_loss(out, 0)
        loss = loss_real + loss_fake + self.args.lambda_reg * loss_reg 

        return loss, Munch(real=loss_real.item(),
                        fake=loss_fake.item(),
                        reg=loss_reg.item())

    def _compute_g_loss(self, x_real, y_org, y_trg, z_emb_trg, z_emb_org, z_trgs=None):
        """Compute the discriminator loss real batches

        Args:
            x_real (torch.tensor): real data batch
            y_org (torch.tensor): labels of real data batch
            y_trg (torch.tensor): labels of fake data batch
            z_emb_trg (torch.tensor): embedding vector for swapped labels
            z_trgs (torch.tensor, optional): pair of randomly drawn noise vectors. Defaults to None.
        """
        # Couple of random vectors for the difference-sensitivity loss
        z_trg, z_trg2 = z_trgs
        
        # Adversarial loss
        if z_trg != None:
            z_emb_trg1 = torch.cat([z_emb_trg, z_trg], dim=1)
        else: 
            z_emb_trg1 = z_emb_trg

        # Style vector with the first random component 
        s_trg1 = self.nets.mapping_network(z_emb_trg1)  

        # Generation of fake images 
        _, x_fake = self.nets.generator(x_real, s_trg1)
        # Try to deceive the discriminator 
        out = self.nets.discriminator(x_fake, y_trg)
        loss_adv = self._adv_loss(out, 1)

        # Encode the fake image and measure the distance from the encoded style
        if not self.args.single_style:
            s_pred = self.nets.style_encoder(x_fake, y_trg)
        else:
            s_pred = self.nets.style_encoder(x_fake)

        # Predict style back from image 
        loss_sty = torch.mean(torch.abs(s_pred - s_trg1))  

        # Diversity sensitive loss 
        if self.args.stochastic:
            z_emb_trg2 = torch.cat([z_emb_trg, z_trg2], dim=1)
            s_trg2 = self.nets.mapping_network(z_emb_trg2) 
            _, x_fake2 = self.nets.generator(x_real, s_trg2)
            x_fake2 = x_fake2.detach()
            loss_ds = torch.mean(torch.abs(x_fake - x_fake2))  # generate outputs as far as possible from each other 
        else:
            loss_ds = 0
            
        # Cycle-consistency loss
        if not self.args.single_style:
            s_org = self.nets.style_encoder(x_real, y_org)
        else:
            s_org = self.nets.style_encoder(x_real)

        _, x_rec = self.nets.generator(x_fake, s_org)
        loss_cyc = torch.mean(torch.abs(x_rec - x_real))  # Mean absolute error reconstructed versus real 

        loss = loss_adv + self.args.lambda_sty * loss_sty \
            - self.args.lambda_ds * loss_ds + self.args.lambda_cyc * loss_cyc

        return loss, Munch(adv=loss_adv.item(),
                        sty=loss_sty.item(),
                        ds=loss_ds.item() if self.args.stochastic else loss_ds,
                        cyc=loss_cyc.item())

    def _r1_reg(self, logits, x):
        """Gradient penalty loss on discriminator output

        Args:
            logits (torch.tensor): discriminator predicition
            x (torch.tensor): input data tensor

        Returns:
            torch.tensor: penalty loss
        """
        batch_size = x.size(0)
        # Compute the gradient penalty of the discriminator 
        grad = torch.autograd.grad(
            outputs=logits.sum(), inputs=x,
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad = grad.pow(2)
        assert(grad.size() == x.size())
        reg = 0.5 * grad.view(batch_size, -1).sum(1).mean(0)
        return reg
    
    def _adv_loss(self, logits, target):
        """Adversarial loss as binary cross-entropy

        Args:
            logits (torch.tensor): discriminator prediction 
            target (torch.tensor): label (0 or 1 depending on what network is trained)

        Returns:
            torch.tensor: evaluated binary cross-entropy loss
        """
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss

    def _load_checkpoint(self, step):
        """Load model checkpoints

        Args:
            step (int): step at which loading is performed
        """
        for ckptio in self.ckptios:
            ckptio.load(step)
    
    def _create_checkpoints(self):
        """Create the checkpoints objects regulating model weight loading and dumping
        """
        self.ckptios = [
            CheckpointIO(ospj(self.dest_dir, self.args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
            CheckpointIO(ospj(self.dest_dir, self.args.checkpoint_dir, '{:06d}_embeddings.ckpt'), **{'embedding_matrix':self.embedding_matrix})]
    
    def _save_checkpoint(self, step):
        """Save model checkpoints

        Args:
            step (int): step at which saving is performed
        """
        for ckptio in self.ckptios:
            ckptio.save(step)