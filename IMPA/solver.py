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
        # Save hyperparameters for lightning/hydra
        self.save_hyperparameters(args)
        
        # Initialize attributes
        self.args = args
        self.dest_dir = dest_dir
        self.automatic_optimization = False  # Required to train the generator and discriminator manually
        
        self.id2mol = datamodule.id2mol  # Numerical encoding to mol name
        self.num_domains = datamodule.n_mol  # Number of categories 
        self.loader_test = datamodule.val_dataloader()  # For evaluation during the training process
        self.embedding_matrix = datamodule.embedding_matrix  # Matrix of embeddings (either pre-trained or loaded) 
        self.n_mol = datamodule.n_mol
        latent_dim = datamodule.latent_dim
        
        if self.args.multimodal and self.args.use_condition_embeddings and not self.batch_correction:
            n_cat = len(self.args.modality_list)
            self.condition_embedding_matrix = torch.nn.Embedding(n_cat, 
                                                                 self.args.condition_embedding_dimension).to(self.device).to(torch.float32) 
            
        # Get the nets
        self.nets = build_model(args, 
                                datamodule.n_mol, 
                                self.device, 
                                multimodal=args.multimodal,
                                batch_correction=self.args.batch_correction, 
                                modality_list=args.modality_list,
                                latent_dim=latent_dim)
        
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
        print(self)
                
    def configure_optimizers(self):
        """Initialize optimizer

        Returns:
            dict: dictionary with optimizers for different neural networks
        """
        self.optims = {}
        for net in self.nets.keys():
            params = list(self.nets[net].parameters())
            if net == 'mapping_network':
                if self.args.trainable_emb:
                    params += list(self.embedding_matrix.parameters())  
                if self.args.use_condition_embeddings:
                    # Add condition embedding matrix
                    params += list(self.condition_embedding_matrix.parameters())
                
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
        if self.args.batch_correction:
            x_real_ctrl, y_org, y_mod = batch['X'].to(self.device), batch['mols'], batch['y_id']
            x_real_trt = None
            y_org = y_org.long().to(self.device)
            y_trg = swap_attributes(self.n_mol, y_org, self.device).long().to(self.device)
        else:
            x_real, y_trg = batch['X'], batch['mols']
            x_real_ctrl, x_real_trt = x_real
            x_real_ctrl, x_real_trt = x_real_ctrl.to(self.device), x_real_trt.to(self.device)
            y_trg = y_trg.long().to(self.device)            
            y_org = None 

        if self.args.multimodal and not self.args.batch_correction:
            x_real_trt, s_trg1, y_org, y_mod = self.encode_label(x_real_trt, y_org, y_mod, 3)
            _, s_trg2, _, _ = self.encode_label(x_real_trt, y_org, y_mod, 3)
        else:
            s_trg1 = self.encode_label(x_real_ctrl, y_trg, None, None)
            s_trg2 = self.encode_label(x_real_ctrl, y_trg, None, None)
            y_mod = None 
        
        # Train the discriminator
        if self.args.batch_correction:
            d_loss, d_losses_latent = self._compute_d_loss(
                x_real_ctrl, x_real_trt, y_org, y_mod, s_trg=s_trg1)
            discriminator_opt.zero_grad()
            self.manual_backward(d_loss)
            discriminator_opt.step()
        else:
            d_loss, d_losses_latent = self._compute_d_loss(
                x_real_ctrl, x_real_trt, y_trg, y_mod, s_trg=s_trg1)
            discriminator_opt.zero_grad()
            self.manual_backward(d_loss)
            discriminator_opt.step()
        
        if self.args.batch_correction:
            # Train the generator
            g_loss, g_losses_latent = self._compute_g_loss(
                x_real_ctrl, y_org, y_mod, s_trg1=s_trg1, s_trg2=s_trg2)
            style_encoder_opt.zero_grad()
            generator_opt.zero_grad()
            mapping_network_opt.zero_grad()   
            self.manual_backward(g_loss)
            style_encoder_opt.step()
            generator_opt.step()
            mapping_network_opt.step() 
        else:
            g_loss, g_losses_latent = self._compute_g_loss(
                x_real_ctrl, y_trg, y_mod, s_trg1=s_trg1, s_trg2=s_trg2)
            style_encoder_opt.zero_grad()
            generator_opt.zero_grad()
            mapping_network_opt.zero_grad()   
            self.manual_backward(g_loss)
            style_encoder_opt.step()
            generator_opt.step()
            mapping_network_opt.step() 
    
        # Decay weight for diversity sensitive loss (moves towards 0) - Decrease sought amount of diversity 
        if self.args.lambda_ds > 0:
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
        metrics_dict = evaluate(self.nets,
                                    self.loader_test,
                                    self.device,
                                    self.args,
                                    self.embedding_matrix,
                                    self.args.batch_correction, 
                                    self.n_mol)
        self.log_dict(metrics_dict)
        
        inputs_val = next(iter(self.loader_test)) 
        debug_image(self,
                    self.nets, 
                    self.embedding_matrix, 
                    inputs=inputs_val, 
                    step=self.current_epoch+1, 
                    device=self.device, 
                    dest_dir=self.dest_dir, 
                    num_domains=self.num_domains,
                    multimodal=self.args.multimodal, 
                    mod_list=self.args.modality_list)

        # Save model checkpoints
        self._save_checkpoint(step=self.current_epoch+1)        
    
    def _compute_d_loss(self, x_real_ctrl, x_real_trt, y_org, y_mod, s_trg):
        """Compute the discriminator loss real batches

        Args:
            x_real (torch.tensor): batch of real data
            y_org (torch.tensor): domain labels of the real data 
            y_trg (torch.tensor): domain labels of the fake data 
            z_emb_trg (torch.tensor): embedding of the target perturbation 
            z_trg (torch.tensor): random noise vector 
        """
        if not self.args.batch_correction:
            x_real_trt.requires_grad_()
            if self.args.multimodal:
                out = self.discriminate(x_real_trt, y_org, y_mod, 3)
            else:
                out = self.discriminate(x_real_trt, y_org, None, None)
        else:
            x_real_ctrl.requires_grad_()
            out = self.discriminate(x_real_ctrl, y_org, None, None)
        
        # Discriminator assigns a 1 to the real 
        loss_real = self._adv_loss(out, 1)
        # Gradient-based regularization (penalize high gradient on the discriminator)
        if not self.args.batch_correction:
            loss_reg = self._r1_reg(out, x_real_trt)
        else:
            loss_reg = self._r1_reg(out, x_real_ctrl)

        # The discriminator does not train the mapping network and the generator, so they need no gradient 
        with torch.no_grad():
            # Generate the fake image
            _, x_fake = self.nets.generator(x_real_ctrl, s_trg)

        # Discriminator trained to predict transformed image as fake in its domain 
        if not self.args.batch_correction:
            if self.args.multimodal:
                out = self.discriminate(x_fake, y_org, y_mod, 3)
            else:
                out = self.discriminate(x_fake, y_org, None, None)
        else:
            out = self.discriminate(x_fake, y_org, None, None)
            
        loss_fake = self._adv_loss(out, 0)
        loss = loss_real + loss_fake + self.args.lambda_reg * loss_reg 

        return loss, Munch(real=loss_real.item(),
                        fake=loss_fake.item(),
                        reg=loss_reg.item())

    def _compute_g_loss(self, x_real_ctrl, y_trg, y_mod, s_trg1, s_trg2):
        """Compute the discriminator loss real batches

        Args:
            x_real (torch.tensor): real data batch
            y_org (torch.tensor): labels of real data batch
            y_trg (torch.tensor): labels of fake data batch
            z_emb_trg (torch.tensor): embedding vector for swapped labels
            z_trgs (torch.tensor, optional): pair of randomly drawn noise vectors. Defaults to None.
        """
        _, x_fake = self.nets.generator(x_real_ctrl, s_trg1)
        
        # Try to deceive the discriminator 
        if self.args.multimodal and not self.args.batch_correction:
            out = self.discriminate(x_fake, y_trg, y_mod, 3)
        else:
            out = self.discriminate(x_fake, y_trg, None, None)
            
        loss_adv = self._adv_loss(out, 1)

        # Encode the fake image and measure the distance from the encoded style
        if not self.args.single_style:
            s_pred = self.nets.style_encoder(x_fake, y_trg)
        else:
            s_pred = self.nets.style_encoder(x_fake)

        # Predict style back from image 
        loss_sty = torch.mean(torch.abs(s_pred - s_trg1))  

        # Diversity sensitive loss 
        _, x_fake2 = self.nets.generator(x_real_ctrl, s_trg2)
        x_fake2 = x_fake2.detach()
        loss_ds = torch.mean(torch.abs(x_fake - x_fake2))  # generate outputs as far as possible from each other 
            
        # Cycle-consistency loss
        if not self.args.single_style:
            s_org = self.nets.style_encoder(x_real_ctrl, y_trg)
        else:
            s_org = self.nets.style_encoder(x_real_ctrl)

        _, x_rec = self.nets.generator(x_fake, s_org)
        loss_cyc = torch.mean(torch.abs(x_rec - x_real_ctrl))  # Mean absolute error reconstructed versus real 

        loss = loss_adv + self.args.lambda_sty * loss_sty \
            - self.args.lambda_ds * loss_ds + self.args.lambda_cyc * loss_cyc

        return loss, Munch(adv=loss_adv.item(),
                            sty=loss_sty.item(),
                            ds=loss_ds.item(),
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
        if self.args.use_condition_embeddings:
            self.ckptios = [
                CheckpointIO(ospj(self.dest_dir, self.args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(self.dest_dir, self.args.checkpoint_dir, '{:06d}_embeddings.ckpt'), **{'embedding_matrix':self.embedding_matrix}),
                CheckpointIO(ospj(self.dest_dir, self.args.checkpoint_dir, '{:06d}_condition_embeddings.ckpt'), **{'condition_embeddings':self.condition_embedding_matrix}) 
                ]
        else:
            self.ckptios = [
                CheckpointIO(ospj(self.dest_dir, self.args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(self.dest_dir, self.args.checkpoint_dir, '{:06d}_embeddings.ckpt'), **{'embedding_matrix':self.embedding_matrix})
                ]
    
    def _save_checkpoint(self, step):
        """Save model checkpoints

        Args:
            step (int): step at which saving is performed
        """
        for ckptio in self.ckptios:
            ckptio.save(step)
            
    def encode_label(self, X, y_trg, y_mod, n_mod, y_org=None):    
        # For each unique modality
        if self.args.multimodal and not self.args.batch_correction:
            s_trg = []  
            X_trg = []
            y_org_trg = []
            y_mod_trg = []
            for i in range(n_mod):
                y_mod_i = y_mod == i 
                if not y_mod_i.any().item():
                    continue
                
                # Append cell images based on the mol
                X_trg.append(X[y_mod_i])
                
                y_mol_mod = y_trg[y_mod_i]
                y_org_trg.append(y_mol_mod)
                
                y_mod_batch = y_mod[y_mod_i]
                y_mod_trg.append(y_mod_batch)
                del y_mod_batch
                
                # Molecule indices for a certain mode
                z_embeddings_mol_mod = self.embedding_matrix[i](y_mol_mod)
                
                if self.args.use_condition_embeddings:
                    cond_embeddings_tensor = self.condition_embedding_matrix(y_mod_batch)
            
                # Draw random vector and collect embedding
                z_trg = torch.randn(z_embeddings_mol_mod.shape[0], self.args.z_dimension).cuda()
                if self.args.use_condition_embeddings:
                    z_embeddings_mol_mod = torch.cat([z_embeddings_mol_mod, cond_embeddings_tensor, z_trg], dim=1)
                else:
                    z_embeddings_mol_mod = torch.cat([z_embeddings_mol_mod, z_trg], dim=1)
                                        
                s_trg.append(self.nets.mapping_network(z_embeddings_mol_mod, y_mol_mod, i))
            
            X_trg = torch.cat(X_trg, dim=0)
            s_trg = torch.cat(s_trg, dim=0)
            y_org_trg = torch.cat(y_org_trg, dim=0)
            y_mod_trg = torch.cat(y_mod_trg, dim=0)
            return X_trg, s_trg, y_org_trg, y_mod_trg 
        
        else:
            z_emb_trg = self.embedding_matrix(y_trg).to(self.device)
            z_trg = torch.randn(X.shape[0], self.args.z_dimension).cuda()
            z_emb_trg = torch.cat([z_emb_trg, z_trg], dim=1)
            s_trg = self.nets.mapping_network(z_emb_trg, y_trg, None)
            return s_trg
            
    def discriminate(self, X, y_mol, y_mod, n_mod):
        if self.args.multimodal and not self.args.batch_correction:
            pred = []
            for i in range(n_mod):
                y_mod_i = y_mod == i
                if not y_mod_i.any().item():
                    continue
                X_mod = X[y_mod_i]
                y_mol_mod = y_mol[y_mod_i]
                pred_mod = self.nets.discriminator(X_mod, y_mol_mod, i)
                pred.append(pred_mod)
            return torch.cat(pred, dim=0)
        else:
            return self.nets.discriminator(X, y_mol, None)
                