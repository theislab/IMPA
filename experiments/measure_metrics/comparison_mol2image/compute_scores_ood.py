import numpy as np 
import torch
import sys
import pandas as pd
from tqdm import tqdm
import pickle as pkl 
from run_info_unseen import RUN_INFO
from IMPA.solver import IMPAmodule
from IMPA.dataset.data_loader import CellDataLoader
from omegaconf import OmegaConf
import yaml
from pathlib import Path
import torch.nn.functional as F

sys.path.append('/home/icb/alessandro.palma/environment/IMPA/IMPA/IMPA/eval/gan_metrics')
sys.path.insert(0, '/home/icb/alessandro.palma/environment/IMPA/IMPA/IMPA')

from fid import *
from density_and_coverage import compute_d_c


sys.path.insert(0, '/lustre/groups/ml01/workspace/alessandro.palma/imCPA_official/experiments/general_experiments/1.benchmark_scores')
sys.path.insert(0, '/lustre/groups/ml01/workspace/alessandro.palma/imCPA_official/experiments/general_experiments/5.interpretability')
from compute_scores import *
from util_functions import *
from util_functions import CustomTransform

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

#################### INITIALIZATION FUNCTIONS ####################

def initialize_impa(config_params, ckpt_dir, ckpt_epoch, dataloader):
    solver = IMPAmodule(config_params, ckpt_dir, dataloader)
    solver._load_checkpoint(200)
    return solver

def initialize_mol2image(args_path, checkpoint_dict):
    # import mol2image 
    sys.path.append('/home/icb/alessandro.palma/environment/IMPA/mol2image/')
    from solver_mol2image import Solver as solver_mol2image
    with open(args_path, 'r') as file:
        # Load YAML data using safe_load() from the file
        args_mol2image = OmegaConf.create(yaml.safe_load(file))
    args_mol2image['modules'] = checkpoint_dict
    solver = solver_mol2image(args_mol2image)
    solver._load_checkpoint(solver.net, i=0)
    return solver

#################### COLLECT DMSOS ####################

def collect_dmsos(solver, mol2id):
    """Collect DMSOs into a single tensor 
    """
    dmsos = []

    for batch in solver.loader_test:
        X_dmso = batch['X'][0]
        dmsos.append(X_dmso)

    dmsos = torch.cat(dmsos, dim=0)
    return dmsos

#################### COLLECT OOD OBSERVATIONS ####################

def gather_ood(ood_list, transform, data_index_path, img_path):
    # Read the data frame with metadata
    data_index = pd.read_csv(data_index_path)
    # Gather files per image
    imgs_ood = {}
    # Gather the cell images 
    for ood in ood_list:
        imgs_ood[ood] = []
        filenames_ood = np.array(data_index.loc[data_index.CPD_NAME==ood].SAMPLE_KEY)
        for sample_id in filenames_ood:
            sample_id_split = sample_id.split("_")
            img_path_to_read = img_path / sample_id_split[0] / sample_id_split[1] / ('_'.join(sample_id_split[2:])+".npy")
            img = np.load(img_path_to_read)
            imgs_ood[ood].append(transform(torch.tensor(img).float().permute(2,0,1)).unsqueeze(0))
    
    imgs_ood = {key: torch.cat(value, dim=0) for key, value in imgs_ood.items()}
    return imgs_ood


def transform_controls_ood(model_type,
                       solver, 
                       X_control, 
                       z_ood, 
                       device, 
                       score_type='FID', 
                       n_samples = None):
    """
    Transform controls 
    """
    fake_domain = []
    if model_type == 'IMPA':
        untransformed_domain = []
        # Forward pass on all the DMSOs of the batch
        for X in X_control:
            # Get batch example
            X = X.unsqueeze(0)
            
            if score_type != 'Accuracy':
                z = torch.randn(X.shape[0], solver.args.z_dimension).to(device) 
            else:
                # z = torch.randn(X.shape[0], 100, solver.args.z_dimension).to(device).quantile(0.75, 1)
                z = torch.randn(X.shape[0], 100, solver.args.z_dimension).to(device).mean(1)
                
            z_emb = torch.cat([z_ood, z], dim=1)
            # collect the mapping of the label by the model 
            s_trg = solver.nets.mapping_network(z_emb)

            # Generate the image 
            _, X_fake = solver.nets.generator(X, s_trg)
                
            fake_domain.append(X_fake)
            untransformed_domain.append(X)
        untransformed_domain = torch.cat(untransformed_domain, dim=0)
    
    elif model_type == 'mol2image':
        untransformed_domain = None
        z_sample = []
        for z,t in zip(solver.args.z_shapes, solver.args.temp):
            z_new = torch.randn(2000, *z) * t
            z_sample.append(z_new.cuda() if solver.args.use_gpu else z_new)
        
        for i in tqdm(range(0, len(z_sample[0]), 128)):
            z = [z_sample[k][i:i+128] for k in range(len(z_sample))]
            X_fake = solver.net.reverse(z, z_ood*z[0].shape[0]).data
            fake_domain.append(X_fake)
            
    # Fix the generated images into a tensor 
    fake_domain = torch.cat(fake_domain, dim=0)
    
    return fake_domain,untransformed_domain


def compute_scores_ood(data_index_path,
                        ood_embeddings,
                        save_path, 
                        image_path, 
                        ckpt_path,
                        mol2annot,
                        mol2smile,
                        model_name='IMPA',
                        dataset_name='bbbc021'):
    """
    General interface for score computation 
    """
    ood_set = [
        "taxol", 
        "ALLN", 
        "bryostatin", 
        "simvastatin", 
        "MG-132", 
        "methotrexate", 
        "colchicine", 
        "cytochalasin B", 
        "AZ258", 
        "cisplatin"
        ]
    
    # Seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize device and the transform
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = CustomTransform(augment=False, normalize=True)
    initialized = False
    
    # Store final scores 
    final_scores = {'score':[],
                    'score_type':[],
                    'run':[],
                    'mol': [], 
                    'model':[]}
    
    for model_name in RUN_INFO: 
        # INITIALIZE MODEL AND COLLECT DMSOS
        if model_name == 'IMPA':
            with open(RUN_INFO[model_name][0], 'r') as file:
                # Load YAML data using safe_load() from the file
                args_impa = OmegaConf.create(yaml.safe_load(file))
            dataloader = CellDataLoader(args_impa)
            solver = initialize_impa(args_impa, 
                                     RUN_INFO[model_name][1], 
                                     RUN_INFO[model_name][2], 
                                     dataloader)
            dmso_imgs = collect_dmsos(solver, dataloader.mol2id)
            
        elif model_name == 'mol2image':
            solver = initialize_mol2image(RUN_INFO[model_name][0], 
                                     RUN_INFO[model_name][1])
            
        if not initialized:
            # GATHER OOD DRUGS 
            X_ood = gather_ood(ood_set, transform, data_index_path, image_path)

            # INITIALIZE CLASSIFIER (OF MOA)
            classifier = Discriminator(img_size=128, 
                                num_domains=13, 
                                max_conv_dim=512, 
                                in_channels=3, 
                                dim_in=64, 
                                multi_task=False).to(device)  
            classifier.load_state_dict(torch.load(ckpt_path))
            classifier.eval()
            initialized = True 
        
        for i in range(1):
            # Bootstrap sample if model is IMPA 
            if model_name == 'IMPA':
                # Do three bootstrap replicates 
                idx = np.random.choice(np.arange(dmso_imgs.shape[0]), dmso_imgs.shape[0], replace=True)
                # Control indexes
                X_control = dmso_imgs[idx].to(device).float().contiguous() 
            
            else:
                X_control = None 
                
            # Iterate through the drugs 
            for mol in tqdm(X_ood):
                # COLLECT EMBEDDINGS 
                if model_name == 'IMPA':
                    z_ood = ood_embeddings.loc[mol]
                    z_ood = torch.tensor(z_ood).unsqueeze(0).cuda().to(torch.float32)
                    
                elif model_name == 'mol2image':
                    z_ood = [mol2smile[mol]]
                
                print(f'Evaluate on molecule {mol}')

                # COLLECT OOD REAL IMAGE 
                true_domain = X_ood[mol].to(device).float().contiguous()[:3000]
                
                # TRANSFORM CONTROLS 
                with torch.no_grad():
                    fake_domain, _ = transform_controls_ood(model_name,
                        solver, 
                        X_control, 
                        z_ood, 
                        device, 
                        score_type='FID', 
                        n_samples = None if model_name=='IMPA' else true_domain.shape[0])
                    
                    fake_domain = F.interpolate(fake_domain, size=(128, 128), mode='bilinear', align_corners=False)
        
                    # Create the data loaders 
                    fake_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(fake_domain), batch_size=20)
                    true_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(true_domain), batch_size=20)
                    
                    # Flatten datasets (needed for some of the metrics)
                    fake_dataset_flat = fake_domain.view(fake_domain.shape[0], -1)
                    true_dataset_flat = true_domain.view(true_domain.shape[0], -1)

                    # FID
                    fid = cal_fid(fake_dataset, true_dataset, 2048, True, [0,1,2])
                    print(f'FID score {fid}')
                    final_scores=append_scores(fid/100, 'FID', i, mol, final_scores, model_name)
                    

                    # Density and coverage 
                    K = 10 if dataset_name == 'bbbc021' else 5  # Smaller value for bbbc025 and recursion datasets because each class is smaller 
                    d_and_c = compute_d_c(true_dataset_flat.cpu().numpy(), fake_dataset_flat.cpu().numpy(), K)
                    final_scores=append_scores(1/d_and_c['density'], 'Density', i, mol, final_scores, model_name)
                    final_scores=append_scores(1/d_and_c['coverage'], 'Coverage', i, mol, final_scores, model_name)
                    print(f'D&C score {d_and_c}')
                    
                    # classifier score - take the 0.75 quantile of IMPA 
                    if model_name == 'IMPA':
                        fake_domain, _ = transform_controls_ood(model_name,
                                                                    solver, 
                                                                    X_control, 
                                                                    z_ood, 
                                                                    device, 
                                                                    score_type='Accuracy', 
                                                                    n_samples = None if model_name=='IMPA' else true_domain.shape[0])
                        

                    fake_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(fake_domain), batch_size=20)                    
                    class_score_fake = classifier_score(classifier, fake_dataset, dataloader.y2id[mol2annot[mol]])
                    final_scores=append_scores(class_score_fake, '1-Accuracy', i, mol, final_scores, model_name)
                    print(f'1-Accuracy {class_score_fake}')
                                
    score_df = pd.DataFrame(final_scores)    
    score_df.to_pickle(save_path)
    return score_df


if __name__=='__main__':
    # Data index
    data_index_path = '/lustre/groups/ml01/workspace/alessandro.palma/imCPA_official/data/bbbc021_unannotated/processed/bbbc021_unannotated_large/metadata/bbbc021_unannotated_large.csv'
    data_index = pd.read_csv(data_index_path)
    mol2annot = {mol:y for mol, y in zip(data_index.CPD_NAME, data_index.ANNOT)}  # mol to moa
    mol2smile = {mol:y for mol, y in zip(data_index.CPD_NAME, data_index.SMILES)}  # mol to moa

    # Drug embeddings 
    ood_embeddings = pd.read_csv('/home/icb/alessandro.palma/environment/IMPA/IMPA/embeddings/csv/emb_fp_all.csv', index_col=0)
    # Classifier path
    ckpt_path = '/lustre/groups/ml01/workspace/alessandro.palma/imCPA_official/experiments/general_experiments/7.train_classifier/checkpoints/larger_fov/checkpoints_all_drugs.ckpt'
    # Image path 
    image_path = Path('/lustre/groups/ml01/workspace/alessandro.palma/imCPA_official/data/bbbc021_unannotated/processed/bbbc021_unannotated_large/')
    # Metrics path 
    save_path = '/home/icb/alessandro.palma/environment/IMPA/IMPA/experiments/measure_metrics/comparison_mol2image/results/eval_metrics_ckpt.ckpt'
    
    compute_scores_ood(data_index_path, 
                        ood_embeddings,
                        save_path,
                        image_path, 
                        ckpt_path,
                        mol2annot, 
                        mol2smile, 
                        model_name='IMPA',
                        dataset_name='bbbc021')
