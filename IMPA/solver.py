import datetime
import os
import time
from os.path import join as ospj

from munch import Munch
from torch.utils.data import WeightedRandomSampler

import torch
import torch.nn as nn
import torch.nn.functional as F
from IMPA.checkpoint import CheckpointIO
from IMPA.dataset.data_loader import CellDataset
from IMPA.eval.eval import evaluate
from IMPA.model import build_model
from munch import Munch
from torch.utils.data import WeightedRandomSampler
from IMPA.utils import he_init, print_network, swap_attributes, print_metrics, print_checkpoint, debug_image


class Solver(nn.Module):
    """Solver class embedding attributes and methods for training the model. 
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize datasets
        self.init_dataset()
        args['num_domains'] = self.n_mol 
        
        # Log the parameters 
        print('Solver loaded with parameters: \n', self.args)

        # Create directories 
        self._create_dirs()

        # Get the nets 
        self.nets = build_model(args)
        
        # Set modules as attributes of the solver class
        for name, module in self.nets.items():
            print_network(module, name)
            setattr(self, name, module)
        
        # Initialize optimizers and checkpoint path 
        self.optims = Munch()
        for net in self.nets.keys():
            # Optimize the embeddings with the mapping network 
            if net == 'mapping_network' and self.args.trainable_emb:
                params = list(self.nets[net].parameters()) + list(self.embedding_matrix.parameters())
            else:
                params = self.nets[net].parameters()

            self.optims[net] = torch.optim.Adam(
                params=params,
                lr=args.f_lr if net == 'mapping_network' else args.lr,  # The mapping network has a different learning rate 
                betas=[args.beta1, args.beta2],
                weight_decay=args.weight_decay)

        # Initialize checkpoints
        self._create_checkpoints()

        # Perform network initialization 
        self.to(self.device)
        for name, network in self.named_children():
            print('Initializing %s...' % name)
            if name != 'embedding_matrix':
                network.apply(he_init) 


    def init_dataset(self):
        """Initialize dataset and data loaders
        """
        # Prepare the data
        print('Lodading the data...') 
        self.training_set, self.test_set = self.create_torch_datasets()
        
        # Create data loaders 
        if self.args.balanced:
            # Balanced sampler
            sampler = WeightedRandomSampler(torch.tensor(self.training_set.weights), len(self.training_set.weights), replacement=False)
            self.loader_train = torch.utils.data.DataLoader(self.training_set, batch_size=self.args.batch_size, sampler=sampler, 
                                            num_workers=self.args.num_workers, drop_last=True)   
        else:
            self.loader_train = torch.utils.data.DataLoader(self.training_set, batch_size=self.args.batch_size, shuffle=True, 
                                                        num_workers=self.args.num_workers, drop_last=True)  

        self.loader_test = torch.utils.data.DataLoader(self.test_set, batch_size=self.args.val_batch_size, shuffle=True, 
                                                    num_workers=self.args.num_workers, drop_last=False)

        self.mol2y = self.training_set.couples_mol_y 
        print('Successfully loaded the data')
    
    
    def create_torch_datasets(self):
        """Create dataset compatible with the pytorch training loop 
        """
        dataset = CellDataset(self.args, device=self.device) 
        
        # Channel dimension
        self.dim = self.args['n_channels']

        # Integrate embeddings as class attribute
        self.embedding_matrix = dataset.embedding_matrix  

        # Number of mols and annotations (the latter can be mode of actions/genes...)
        self.n_mol = dataset.n_mol
        print(f'Training with {self.n_mol} mols')
        self.num_y = dataset.n_y

        # Collect ids 
        self.mol2id = dataset.mol2id
        self.y2id = dataset.y2id
        self.id2mol = {val:key for key,val in self.mol2id.items()}
        self.id2y = {val:key for key,val in self.y2id.items()}        

        # Collect training and test set 
        training_set, test_set = dataset.fold_datasets.values()  

        # Free cell painting dataset memory
        del dataset
        return training_set, test_set


    def train(self):     
        """Method for IMPA training across a pre-defined number of iterations. 
        """
        # The history of the training process
        self.history = {"train":{'epoch':[]}, "test":{'epoch':[]}}

        # Fixed batch used for the evaluation on the validation set 
        inputs_val = next(iter(self.loader_test)) 
        
        # Remember the initial value of diversity-sensitivity weight and decay it 
        initial_lambda_ds = self.args.lambda_ds

        # Resume training if necessary
        if self.args.resume_iter > 0:
            self._load_checkpoint(self.args.resume_iter)
            # Initialize decayed lambda
            self.args.lambda_ds = self.args.lambda_ds - \
                            (initial_lambda_ds / self.args.ds_iter)*self.args.resume_iter

        print('Start training...')
        start_time = time.time()
        for i in range(self.args.resume_iter, self.args.total_iters):
            
            # Fetch images and labels as iteration batch
            inputs = next(iter(self.loader_train))

            # Fetch the real and fake inputs and outputs 
            x_real, y_one_hot = inputs['X'].to(self.device), inputs['mol_one_hot']
            y_org = y_one_hot.argmax(1).long().to(self.device)
            y_trg = swap_attributes(y_one_hot, y_org, self.device).argmax(1).long().to(self.device)

            # Get the perturbation embedding for the target mol
            z_emb_trg = self.embedding_matrix(y_trg).to(self.device)
            z_emb_org = self.embedding_matrix(y_org).to(self.device)

            # Pick two random weight vectors 
            if self.args.stochastic:
                z_trg, z_trg2 = torch.randn(x_real.shape[0], self.args.z_dimension).to(self.device), torch.randn(x_real.shape[0], 
                                                                                                             self.args.z_dimension).to(self.device)
            else:
                z_trg, z_trg2 = None, None
            
            # Train the discriminator
            d_loss, d_losses_latent = self._compute_d_loss(
                x_real, y_org, y_trg, z_emb_trg=z_emb_trg, z_trg=z_trg)
            self._reset_grad()
            d_loss.backward()
            self.optims.discriminator.step()
            
            # Train the generator
            g_loss, g_losses_latent = self._compute_g_loss(
                x_real, y_org, y_trg, z_emb_trg=z_emb_trg, z_emb_org=z_emb_org, z_trgs=[z_trg, z_trg2])
            self._reset_grad()
            g_loss.backward()
            self.optims.style_encoder.step()
            self.optims.generator.step()
            self.optims.mapping_network.step()
        
            # Decay weight for diversity sensitive loss (moves towards 0) - Decrease sought amount of diversity 
            if self.args.lambda_ds > 0 and self.args.stochastic:
                self.args.lambda_ds -= (initial_lambda_ds / self.args.ds_iter)

            # Format the losses
            all_losses = dict()
            for loss, prefix in zip([d_losses_latent, g_losses_latent],
                                    ['D/latent_', 'G/latent_']):
                for key, value in loss.items():
                    all_losses[prefix + key] = value
            all_losses['G/lambda_ds'] = self.args.lambda_ds
            
            # Log time and losses
            if (i+1) % self.args.print_every == 0:
                print_checkpoint(i, start_time, 
                        self.args.total_iters, 
                        all_losses)

            # Generate images for debugging
            if (i+1) % self.args.sample_every == 0:
                debug_image(self.nets, 
                            self.embedding_matrix, 
                            self.args, 
                            inputs=inputs_val, 
                            step=i+1, 
                            device=self.device, 
                            id2mol=self.id2mol,
                            dest_dir=self.dest_dir)

            # Save model checkpoints
            if (i+1) % self.args.save_every == 0:
                self._save_checkpoint(step=i+1)
                # Save losses in the history 
                self._save_history(i, all_losses, 'train')

            # Compute evaluation metrics 
            if (i+1) % self.args.eval_every == 0:
                rmse_disentanglement_dict = evaluate(self.nets,
                                                        self.loader_test,
                                                        self.device,
                                                        self.dest_dir,
                                                        self.args.embedding_folder,
                                                        args=self.args,
                                                        embedding_matrix=self.embedding_matrix)

                # Save metrics to history 
                self._save_history(i, rmse_disentanglement_dict, 'test')
                # Print metrics 
                print_metrics(rmse_disentanglement_dict, i+1)
                

        # Format the history results to proper storage into Mongo DB
        results = self._format_seml_results(self.history)
        return results 
    
    
    def _format_seml_results(self, history):
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


    def _save_checkpoint(self, step):
        """Save model checkpoints

        Args:
            step (int): step at which saving is performed
        """
        for ckptio in self.ckptios:
            ckptio.save(step)


    def _load_checkpoint(self, step):
        """Load model checkpoints

        Args:
            step (int): step at which loading is performed
        """
        for ckptio in self.ckptios:
            ckptio.load(step)


    def _reset_grad(self):
        """Reset the gradient of all optimizers
        """
        for optim in self.optims.values():
            optim.zero_grad()


    def _create_dirs(self):
        """Create the directories and sub-directories for training
        """
        # Directory is named based on time stamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        # Setup the key(s) naming the folder (passed as hyperparameter)
        if type(self.args['naming_key']) == list: 
            keys = [str(self.args[key]) for key in self.args['naming_key']]
            keys_str = '_'.join(keys)
        else:
            keys_str = str(self.args[self.args['naming_key']])

        # Set the directory for the results based on whether training is from begginning or resumed
        if self.args.resume_iter==0:
            self.dest_dir = ospj(self.args.experiment_directory, timestamp+'_'+keys_str)
        else:
            self.dest_dir = self.args.resume_dir+'_'+keys_str

        # Create sub-directories to save partial and definitive results 
        os.makedirs(self.dest_dir, exist_ok=True)
        os.makedirs(ospj(self.dest_dir, self.args.sample_dir), exist_ok=True)
        os.makedirs(ospj(self.dest_dir, self.args.checkpoint_dir), exist_ok=True)
        os.makedirs(ospj(self.dest_dir, self.args.basal_vs_real_folder), exist_ok=True)
        os.makedirs(ospj(self.dest_dir, self.args.embedding_folder), exist_ok=True)


    def _create_checkpoints(self):
        """Create the checkpoints objects regulating model weight loading and dumping
        """
        self.ckptios = [
            CheckpointIO(ospj(self.dest_dir, self.args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
            CheckpointIO(ospj(self.dest_dir, self.args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims),
            CheckpointIO(ospj(self.dest_dir, self.args.checkpoint_dir, '{:06d}_embeddings.ckpt'), **{'embedding_matrix':self.embedding_matrix})]

    
    def _save_history(self, epoch, losses, fold):
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
                