import numpy as np 
import torch
import pandas as pd
from tqdm import tqdm
from nets import *

# Stay because constant for alternative methods too
from IMPA.model import Discriminator 
from util_functions import classifier_score
from IMPA.eval.gan_metrics.fid import *
from IMPA.eval.gan_metrics.density_and_coverage import compute_d_c
from IMPA.dataset.data_utils import CustomTransform


def gather_data_by_pert(dataset, ood_set=None):
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
    if ood_set != None:
        images = {key: val for key, val in images.items() if key in ood_set+["DMSO"]}
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

def transform_controls(solver, 
                       mol2id,
                       data_content,
                       data_style, 
                       mol, 
                       model_name, 
                       device,
                       n_mols,
                       score_type='FID'):
    fake_domain = []
    
    batch_size=64
    # Calculate the number of batches needed
    num_batches = (len(data_content) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        # Get the start and end indices for the current batch
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(data_content))
        
        # Extract the current batch
        data_content_batch = data_content[start_idx:end_idx].to(device)
        data_style_batch = data_style[start_idx:end_idx].to(device)

        with torch.no_grad():
            # Transform the batch
            X_fake, _, _, _, _ = solver["network"](data_content_batch, 
                                                   data_style_batch, 
                                                   solver["embedding"])
            fake_domain.append(X_fake.cpu())
    
    # Concatenate all the batches into a single tensor
    return torch.cat(fake_domain, dim=0)



def compute_all_scores(solver, 
                        dataset,
                        save_path,
                        ckpt_path=None, 
                        model_name='IMPA',
                        dataset_name='bbbc021', 
                        ood_set=None):
    
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
    mol2id = dataset.mol2id # Mol to index
    mol2y = dataset.mol2y # Mol to MoA
    n_mols = len(mol2id)
    transform = CustomTransform(augment=False, normalize=True)
    
    # COLLECT DATA LIKE DMSO TO EMBEDDING 
    X_mols = gather_data_by_pert(dataset, ood_set)
    
    final_scores = {'score':[],
                    'score_type':[],
                    'run':[],
                    'mol': [], 
                    'model':[]}

    # The index of the control first depending on the dataset 
    control_id = 'DMSO'
    channels_fid = [0,1,2]
    # Initialize the classifier for classifier score
    classifier = Discriminator(img_size=96, 
                        num_domains=6, 
                        max_conv_dim=512, 
                        in_channels=3, 
                        dim_in=64, 
                        multi_task=False).to("cuda")
    
    classifier.load_state_dict(torch.load(ckpt_path))
    classifier.eval()
  
    n_bootstrap = 3
    for i in range(n_bootstrap):
        # Do three bootstrap replicates 
        idx = np.random.choice(np.arange(len(X_mols[control_id])), len(X_mols[control_id]), replace=True )
        
        # Control indexes
        X_control = X_mols[control_id][idx].to(device).float().contiguous()
        
        # Iterate through the drugs 
        for mol in tqdm(X_mols):
            if ood_set!=None and mol in solver.args.ood_set:
                continue

            print(f'Evaluate on molecule {mol}')

            # The true domain we'll compare against
            true_domain = X_mols[mol].to(device).float().contiguous()
        
            # Difference in numbers of images 
            delta = X_control.shape[0] - true_domain.shape[0] 
            if delta > 0:
                idx_resamp = np.random.choice(range(len(true_domain)), delta, replace=True, p=None)
                true_domain = torch.cat([true_domain, true_domain[idx_resamp]], dim=0)

            # Skip controls 
            if mol==control_id:
                continue 
            
            # TO MODIFY: use transformer 
            fake_domain=transform_controls(solver, 
                                            dataset.mol2id,
                                            X_control, 
                                            true_domain[:X_control.shape[0]],
                                            mol, 
                                            model_name, 
                                            device,
                                            n_mols,
                                            score_type='FID')    
            
            true_domain = (true_domain * 2)-1
            fake_domain = (fake_domain * 2)-1
            true_domain = true_domain.cuda()
            fake_domain = fake_domain.cuda()
            
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
            
            score = classifier_score(classifier, fake_dataset, mol2y[mol2id[mol]])
            final_scores=append_scores(score, 'Accuracy', i, mol, final_scores, model_name)
            print(f'Accuracy {score}')
                
    score_df = pd.DataFrame(final_scores)    
    score_df.to_pickle(save_path)
    return score_df
