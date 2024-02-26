import numpy as np 
import torch
import sys
import ot
import pandas as pd
from tqdm import tqdm
from nets import *
import pickle as pkl 
import seml 
import git
from pathlib import Path

from IMPA.model import Discriminator 

from util_functions import classifier_score
from IMPA.eval.gan_metrics.fid import *
from IMPA.eval.gan_metrics.density_and_coverage import compute_d_c


def gather_data_by_pert(dataset):
    """Gather images by pertubation
    """
    # Add empty perturbation entries 
    pert_list = list(dataset.mol2id.keys())
    pert_dict = dataset.id2mol
    # Collect the images
    images = {}
    for pert_key in pert_list:
        images[pert_key] = []
    if "DMSO" not in pert_list:
        images["DMSO"] = []

    # Loop across loaders 
    for loader in [dataset.train_dataloader(), dataset.val_dataloader()]:
        for batch in loader:
            images["DMSO"].append(batch["X"][0])
            # Get the drugs in the batch
            drugs_batch = batch["mol_one_hot"].argmax(1).numpy()
            drugs_batch_unique = np.unique(drugs_batch)
            for drug_id in drugs_batch_unique:
                mask = batch["mol_one_hot"].argmax(1)==drug_id
                images[pert_dict[drug_id]].append(batch["X"][1][mask])

    images = {key: torch.cat(val, dim=0) for key, val in images.items()}
    return images 


def append_scores(score, score_type, bs_fold, mol_name, score_dict, model_name):
    """Update existing dictionary with the computed scores 

    Args:
        score (float): the score to record 
        score_type (str): the name of the score to record
        bs_fold (int): the number of the bootstrap fold
        mol_name (str): name of the perturbation
        score_dict (dict): dictionary with the scores 
        model_name (str): name of the mode

    Returns:
        dict: score dictionary updated with the newly computed scores 
    """
    if type(score)==torch.Tensor:
        score=score.item()
    score_dict['mol'] += [mol_name]
    score_dict['score'] += [score]
    score_dict['score_type'] += [score_type]
    score_dict['run'] += [bs_fold]
    score_dict['model'] += [model_name]
    return score_dict


def perform_transformation(solver, 
                        model_name, 
                        X, 
                        y_one_hot,
                        y, 
                        device, 
                        score_type='FID'):
    """Given input images, we use the model to perform transformations and predict on the data

    Args:
        solver (Solver): solver object 
        model_name (str): name of the model to use  
        X (torch.Tensor): the images submitted to transformation  
        y_one_hot (torch.tensor): one-hot encoded binary tensor for conditioning
        y (torch.Tensor): 1D tensor with condition labels included 
        device (str): device used for the transformation

    Returns:
        torch.Tensor: transformed images
    """
    if model_name == 'IMPA' or model_name == 'starGANv2':
        if score_type!='Accuracy':
            # z = torch.randn(X.shape[0], solver.args.z_dimension).to(device)
            z = torch.randn(X.shape[0], 100, solver.args.z_dimension).mean(1).to(device)
        else:
            z = torch.randn(X.shape[0], 100, solver.args.z_dimension).to(device).quantile(0.75, dim=1)
                        
        # Embedding of the labels from RDKit
        if model_name == "IMPA":
            z_emb = solver.embedding_matrix(y).view(X.shape[0], -1)
        
            if solver.args.stochastic:
                z_emb = torch.cat([z_emb, z], dim=1)
            
            s_trg = solver.nets.mapping_network(z_emb, y=None)
        
        else:
            z_emb = z
            s_trg = solver.nets.mapping_network(z_emb, y=y)
        

        # Generate the image 
        _, x_fake = solver.nets.generator(X, s_trg)
    

    elif model_name=='DRIT++': 
        z = torch.randn(X.shape[0], 8).to(device)
        z_c = solver.model.enc_c.forward(X)
        x_fake = solver.model.gen(z_c, z, y_one_hot)

    elif model_name=='starGANv1':
        x_fake = solver.G(X, y_one_hot)

    # DMIT
    else:
        z = torch.randn(X.shape[0], solver.args.n_style).to(device)
        z_c = solver.model.enc_content(X)
        z_code = torch.cat([y_one_hot, z],  dim=1)
        x_fake = solver.model.dec(z_c,z_code)

    return x_fake


def transform_controls(solver, 
                       mol2id,
                       data, 
                       mol, 
                       model_name, 
                       device,
                       n_mols,
                       score_type='FID'):
    fake_domain = []
    for X in data:
        # Get batch example
        X = X.unsqueeze(0)
        
        # Transform drugs from the domain to the domain of interest 
        if model_name != 'baseline':
            mol_id = mol2id[mol]
            
            # Create an array of target drug labels 
            y_trg = mol_id * torch.ones(X.shape[0]).to(device).long()
            y_trg_one_hot = torch.nn.functional.one_hot(y_trg, num_classes=n_mols).float()
            with torch.no_grad():
                X_fake = perform_transformation(solver, 
                                                model_name, 
                                                X, 
                                                y_trg_one_hot,
                                                y_trg, 
                                                device, 
                                                score_type=score_type)
            fake_domain.append(X_fake)
        else: 
            fake_domain.append(X)  # Keep original DMSOs as baseline
    return torch.cat(fake_domain, dim=0)


def compute_all_scores(solver, 
                        dataset,
                        save_path,
                        ckpt_path=None, 
                        model_name='IMPA',
                        dataset_name='bbbc021'):
    
    """Compute all scores and save them as a data frame

    Args:
        solver (Solver): solver object 
        save_path (str): path to save images to 
        model_name (str, optional): Model used to draw the results. Defaults to 'IMPA'

    Returns:
        pandas.DataFrame: data frame containing the results 
    """
    
    # Seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Seed for reproducibility
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mol2id = dataset.mol2id
    n_mols = len(mol2id)
    
    # Collect mol images
    X_mols = gather_data_by_pert(dataset)
    
    final_scores = {'score':[],
                    'score_type':[],
                    'run':[],
                    'mol': [], 
                    'model':[]}

    # The index of the control first depending on the dataset 
    if dataset_name == 'bbbc021':
        control_id = 'DMSO'
        channels_fid = [0,1,2]
        # Initialize the classifier for classifier score
        classifier = Discriminator(img_size=96, 
                          num_domains=13, 
                          max_conv_dim=512, 
                          in_channels=3, 
                          dim_in=64, 
                          multi_task=False).to(device)
        classifier.load_state_dict(torch.load(ckpt_path))
        classifier.eval()
        
    elif dataset_name == 'bbbc025':
        control_id = '0'
        channels_fid = [1,3,4]
    else:
        control_id = 'UNTREATED'
        channels_fid = [0,1,5]
  
    n_bootstrap = 3
    for i in range(n_bootstrap):
        # Do three bootstrap replicates 
        idx = np.random.choice(np.arange(len(X_mols[control_id])), len(X_mols[control_id]), replace=True )
        
        # Control indexes
        X_control = X_mols[control_id][idx].to(device).float().contiguous()
        
        # Iterate through the drugs 
        for mol in tqdm(X_mols):
            if mol in solver.args.ood_set:
                continue

            print(f'Evaluate on molecule {mol}')

            # The true domain we'll compare against
            true_domain = X_mols[mol].to(device).float().contiguous()

            # Skip controls 
            if mol==control_id:
                continue 
            
            fake_domain=transform_controls(solver, 
                                            dataset.mol2id,
                                            X_control, 
                                            mol, 
                                            model_name, 
                                            device,
                                            n_mols,
                                            score_type='FID')            

            # Create the data loaders 
            fake_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(fake_domain), batch_size=20)
            true_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(true_domain), batch_size=20)
            
            # Flatten datasets (needed for some of the metrics)
            fake_dataset_flat = fake_domain.view(fake_domain.shape[0], -1)
            true_dataset_flat = true_domain.view(true_domain.shape[0], -1)

            # FID
            fid = cal_fid(fake_dataset, true_dataset, 2048, True, channels_fid)
            print(f'FID score {fid}')
            final_scores=append_scores(fid/100, 'FID', i, mol, final_scores, model_name)

            # Density and coverage 
            K = 5   # Smaller value for bbbc025 and recursion datasets because each class is smaller 
            d_and_c = compute_d_c(true_dataset_flat.cpu().numpy(), fake_dataset_flat.cpu().numpy(), K)
            final_scores=append_scores(d_and_c['density'], 'Density', i, mol, final_scores, model_name)
            final_scores=append_scores(d_and_c['coverage'], 'Coverage', i, mol, final_scores, model_name)
            print(f'D&C score {d_and_c}')
            
            # classifier score - take the 0.75 quantile of IMPA 
            if  dataset_name == 'bbbc021':
                fake_domain= transform_controls(solver, 
                                                dataset.mol2id,
                                                X_control, 
                                                mol, 
                                                model_name, 
                                                device,
                                                n_mols,
                                                score_type='Accuracy')
                fake_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(fake_domain), batch_size=20)
                
                score = classifier_score(classifier, fake_dataset, mol2id[mol])
                final_scores=append_scores(1-score, '1-Accuracy', i, mol, final_scores, model_name)
                print(f'1-Accuracy {score}')
                
    score_df = pd.DataFrame(final_scores)    
    score_df.to_pickle(save_path)
    return score_df
