import os
from os.path import join as ospj
from re import A
import time
import datetime
from munch import Munch
from .data_loader import BBBC021Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

from .model import build_model
from .checkpoint import CheckpointIO
from .utils import *  

sys.path.insert(0, '../..')
# from compert.metrics.eval import * 
from metrics.eval import * 

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Pass arguments in as a dictionary "args"
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize datasets
        self.init_dataset()
        args['num_domains'] = self.n_seen_drugs

        # If the RDKit embeddings are not encoded their dimension is also the dimension of the style
        if not self.args.encode_rdkit:
            self.args['style_dim'] = self.args.latent_dim

        print(self.args)

        # Create directory 
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        if self.args.resume_iter==0:
            self.dest_dir = ospj(self.args.experiment_directory, timestamp+'_'+str(self.args[self.args['naming_key']]))
        else:
            self.dest_dir = self.args.resume_dir

        # Get the nets 
        self.nets = build_model(args)
        
        # Set modules as attributes of the class
        for name, module in self.nets.items():
            print_network(module, name)
            setattr(self, name, module)
        
        # Initialize optimizers and checkpoint path 
        self.optims = Munch()
        for net in self.nets.keys():
            self.optims[net] = torch.optim.Adam(
                params=self.nets[net].parameters(),
                lr=args.f_lr if net == 'mapping_network' else args.lr,  # The mapping network has a different learning rate 
                betas=[args.beta1, args.beta2],
                weight_decay=args.weight_decay)

        self.ckptios = [
            CheckpointIO(ospj(self.dest_dir, args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
            CheckpointIO(ospj(self.dest_dir, args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
   
        # Put the model on GPU 
        self.to(self.device)
        for name, network in self.named_children():
            print('Initializing %s...' % name)
            network.apply(he_init) 


    def init_dataset(self):
        """Initialize dataset and data loaders
        """
        # Prepare the data
        print('Lodading the data...') 
        self.training_set, self.test_set = self.create_torch_datasets()
        
        # Create data loaders 
        self.loader_train = torch.utils.data.DataLoader(self.training_set, batch_size=self.args.batch_size, shuffle=True, 
                                                    num_workers=self.args.num_workers, drop_last=True)  # Drop last batch for a better estimate of the accuracy 

        self.loader_test = torch.utils.data.DataLoader(self.test_set, batch_size=self.args.val_batch_size, shuffle=True, 
                                                    num_workers=self.args.num_workers, drop_last=False)

        # The id of the dmso to exclude from the prediction 
        self.drug2moa = self.training_set.couples_drug_moa 
        print('Successfully loaded the data')
    
    
    def create_torch_datasets(self):
        """
        Create dataset compatible with the pytorch training loop 
        """
        dataset = BBBC021Dataset(self.args.image_path, self.args.data_index_path, self.args.drug_embeddings_path, device=self.device, augment_train=self.args.augment_train, 
                                                normalize=self.args.normalize) 
        
        # Channel dimension
        self.dim = 3  

        # Drug embeddings 
        self.embedding_matrix = dataset.embedding_matrix  # RDKit embedding matrix

        # Number of drugs and number of modes of action 
        self.n_seen_drugs = dataset.num_drugs
        self.num_moa = dataset.num_moa

        # Collect ids 
        self.drug2idx = dataset.drugs2idx
        self.moa2idx = dataset.moa2idx
        self.id2drug = {val:key for key,val in self.drug2idx.items()}
        self.id2moa = {val:key for key,val in self.moa2idx.items()}        

        # Collect training and test set 
        training_set, test_set = dataset.fold_datasets.values()  

        # Free cell painting dataset memory
        del dataset
        return training_set, test_set


    def train(self):        
        # Create directories to save partial and definitive results 
        os.makedirs(self.dest_dir, exist_ok=True)
        os.makedirs(ospj(self.dest_dir, self.args.sample_dir), exist_ok=True)
        os.makedirs(ospj(self.dest_dir, self.args.checkpoint_dir), exist_ok=True)
        os.makedirs(ospj(self.dest_dir, self.args.basal_vs_real_folder), exist_ok=True)
        os.makedirs(ospj(self.dest_dir, self.args.embedding_folder), exist_ok=True)

        # The history of the training process (useful to keep track of the metrics and tor return)
        self.history = {"train":{'epoch':[]}, "val":{'epoch':[]}, "test":{'epoch':[]}}

        args = self.args  # Hparams
        nets = self.nets  # Neural networks 
        optims = self.optims  # Optimizers

        # Fixed batch used for the evaluation on the validation set 
        inputs_val = next(iter(self.loader_test))  

        # Resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # Remember the initial value of diversity-sensitivity weight and decay it 
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            
            # Fetch images and labels
            inputs = next(iter(self.loader_train))

            # Fetch the real and fake inputs and outputs 
            x_real, y_one_hot = inputs['X'].to('cuda'), inputs['mol_one_hot'].to('cuda')
            y_org = y_one_hot.argmax(1).long().to('cuda')
            y_trg = swap_attributes(y_one_hot, y_org, self.device).argmax(1).long().to('cuda')
            # Get standardized RDKit embedding for the target drug
            z_emb_trg = self.embedding_matrix(y_trg).to('cuda')  


            z_trg, z_trg2 = torch.randn(x_real.shape[0], args.z_dimension).to('cuda'), torch.randn(x_real.shape[0], args.z_dimension).to('cuda')

            # train the discriminator
            d_loss, d_losses_latent = self.compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_emb_trg=z_emb_trg, z_trg=z_trg)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator
            g_loss, g_losses_latent = self.compute_g_loss(
                nets, args, x_real, y_org, y_trg, z_emb_trg=z_emb_trg, z_trgs=[z_trg, z_trg2])
            self._reset_grad()
            g_loss.backward()
            optims.style_encoder.step()
            optims.generator.step()
            if self.args.encode_rdkit:
                optims.mapping_network.step()
        
            # decay weight for diversity sensitive loss (moves towards 0) - Decrease sought amount of diversity 
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, g_losses_latent],
                                        ['D/latent_', 'G/latent_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                debug_image(nets, 
                            self.embedding_matrix, 
                            args, 
                            inputs=inputs_val, 
                            step=i+1, 
                            device=self.device, 
                            id2drug=self.id2drug,
                            dest_dir=self.dest_dir)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)
                # Save losses in the history 
                self.save_history(i, all_losses, 'train')

            # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                rmse_disentanglement_dict = calculate_rmse_and_disentanglement_score(nets,
                                                                                    self.loader_test,
                                                                                    self.device,
                                                                                    self.dest_dir,
                                                                                    args.embedding_folder,
                                                                                    args=args,
                                                                                    step=i+1,
                                                                                    embedding_matrix=self.embedding_matrix)

                # Save metrics to history 
                self.save_history(i, rmse_disentanglement_dict, 'test')
                # Print metrics 
                print_metrics(rmse_disentanglement_dict, i+1)

        results = self.format_seml_results(self.history)
        return results 
    
    
    def format_seml_results(self, history):
        """Format results for seml 

        Args:
            history (_dict_): dictionary containing the history of the model's statisistics
        """
        results = {}
        for fold in history:
            for stat in history[fold]:
                key = f'{fold}_{stat}'
                results[key] = history[fold][stat]
        return results


    def save_history(self, epoch, losses, fold):
        """Save partial model results in the history dictionary (model attribute) 
        Args:
            epoch (int): the current epoch 
            losses (dict): dictionary containing the partial losses of the model 
            metrics (dict): dictionary containing the partial metrics of the model 
            fold (str): train or valid
        """
        self.history[fold]["epoch"].append(epoch)
        # Append the losses to the right fold dictionary 
        for loss in losses:
            if loss not in self.history[fold]:
                self.history[fold][loss] = [losses[loss]]
            else:
                self.history[fold][loss].append(losses[loss])


    def compute_d_loss(self, nets, args, x_real, y_org, y_trg, z_emb_trg, z_trg):
        # With real images
        x_real.requires_grad_()
        out = nets.discriminator(x_real, y_org)
        # Discriminator assigns a 1 to the real 
        loss_real = self.adv_loss(out, 1)
        # Gradient-based regularization (penalize high gradient on the discriminator)
        loss_reg = self.r1_reg(out, x_real)

        # The discriminator does not train the mapping network and the generator, so they need no gradient 
        with torch.no_grad():

            # If stochastic, concatenate embeddings with noise
            if self.args.stochastic:
                z_emb_trg = torch.cat([z_emb_trg, z_trg], dim=1)

            # Single of the noisy embedding vector to a style vector 
            s_trg = nets.mapping_network(z_emb_trg) if self.args.encode_rdkit else z_emb_trg  
            
            # Generate the fake image
            _, x_fake = nets.generator(x_real, s_trg)

        out = nets.discriminator(x_fake, y_trg)
        loss_fake = self.adv_loss(out, 0)

        loss = loss_real + loss_fake + args.lambda_reg * loss_reg 
        return loss, Munch(real=loss_real.item(),
                        fake=loss_fake.item(),
                        reg=loss_reg.item())


    def compute_g_loss(self, nets, args, x_real, y_org, y_trg, z_emb_trg, z_trgs=None):
        # Couple of random vectors for the difference-sensitivity loss
        z_trg, z_trg2 = z_trgs

        # Adversarial loss
        if self.args.stochastic:
            z_emb_trg1 = torch.cat([z_emb_trg, z_trg], dim=1)
        else:
            z_emb_trg1 = z_emb_trg

        # Single of the noisy embedding vector to a style vector 
        s_trg1 = nets.mapping_network(z_emb_trg1) if self.args.encode_rdkit else z_emb_trg1 

        # Generator for fake images 
        _, x_fake = nets.generator(x_real, s_trg1)
        # Try to deceive the generator such that it is persuaded that the generated image is from the target domain
        out = nets.discriminator(x_fake, y_trg)
        # Adversarial loss setting the output of the network to 1 
        loss_adv = self.adv_loss(out, 1)

        # Encode the fake image and measure the distance from the encoded style
        if not args.single_style:
            s_pred = nets.style_encoder(x_fake, y_trg)
        else:
            s_pred = nets.style_encoder(x_fake)
        
        loss_sty = torch.mean(torch.abs(s_pred - s_trg1))  # Predict style back from image 

        # Diversity sensitive loss - only if stochastic 
        if args.stochastic:
            z_emb_trg2 = torch.cat([z_emb_trg, z_trg2], dim=1)
            s_trg2 = nets.mapping_network(z_emb_trg2) if self.args.encode_rdkit else z_emb_trg2 
            _, x_fake2 = nets.generator(x_real, s_trg2)
            x_fake2 = x_fake2.detach()
            loss_ds = torch.mean(torch.abs(x_fake - x_fake2))  # Generate outputs as far as possible from each other 
        else:
            loss_ds = torch.tensor(0)

        # Cycle-consistency loss
        if not args.single_style:
            s_org = nets.style_encoder(x_real, y_org)
        else:
            s_org = nets.style_encoder(x_real)

        _, x_rec = nets.generator(x_fake, s_org)
        loss_cyc = torch.mean(torch.abs(x_rec - x_real))  # Mean absolute error reconstructed versus real 

        loss = loss_adv + args.lambda_sty * loss_sty \
            - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc

        return loss, Munch(adv=loss_adv.item(),
                        sty=loss_sty.item(),
                        ds=loss_ds.item(),
                        cyc=loss_cyc.item(),
                        rmse=torch.sqrt(torch.mean((x_rec.detach()-x_real)**2)).item())


    def adv_loss(self, logits, target):
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss


    def r1_reg(self, d_out, x_in):
        # zero-centered gradient penalty for real images
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg
        
    def _early_stopping(self, score):
        cond = (score < self.best_score)
        if cond:
            self.best_score = score
            self.patience_trials = 0 
        else:
            self.patience_trials += 1
        return cond, self.patience_trials > self.args.patience 


    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)


    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)


    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()
