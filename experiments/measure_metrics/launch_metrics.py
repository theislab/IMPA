from IMPA.dataset.data_loader import CellDataLoader
from IMPA.solver import IMPAmodule
import argparse
from omegaconf import OmegaConf
import yaml

import sys
sys.path.insert(0, "/home/icb/alessandro.palma/environment/IMPA/IMPA/experiments/measure_metrics")
from compute_metrics import *

def main(yaml_path,
         ckpt_dir, 
         ckpt_epoch, 
         dataset_name,
         model_name, 
         classifier_ckpt_path, 
         result_path, 
         leaveout):
    
    # Read the yaml file 
    with open(yaml_path, 'r') as file:
        # Load YAML data using safe_load() from the file
        config_params = yaml.safe_load(file)
    
    # Config 
    args = OmegaConf.create(config_params)
    dataloader = CellDataLoader(args)
    
    solver = IMPAmodule(args, ckpt_dir, dataloader)
    solver._load_checkpoint(ckpt_epoch)
    
    leaveout = False
    if leaveout:
        ood_set = config_params["ood_set"]
        config_params["ood_set"] = None
        args = OmegaConf.create(config_params)
        dataloader = CellDataLoader(args)
        solver.embedding_matrix = dataloader.embedding_matrix
    else:
        ood_set = None
    print(ood_set)
    compute_all_scores(solver=solver, 
                    dataset=dataloader,
                    save_path=result_path,
                    ckpt_path=classifier_ckpt_path, 
                    model_name=model_name,
                    dataset_name=dataset_name,
                    ood_set=ood_set)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure evaluation metrics for a given yaml config")
    parser.add_argument('--yaml_path')
    parser.add_argument('--ckpt_dir')
    parser.add_argument('--ckpt_epoch')
    parser.add_argument('--dataset_name')
    parser.add_argument('--model_name')
    parser.add_argument('--classifier_ckpt_path')
    parser.add_argument('--result_path')
    parser.add_argument('--leaveout')
    
    
    args = parser.parse_args()
    yaml_path = args.yaml_path
    ckpt_dir = args.ckpt_dir
    ckpt_epoch = eval(args.ckpt_epoch)
    dataset_name = args.dataset_name
    model_name = args.model_name
    classifier_ckpt_path = args.classifier_ckpt_path
    result_path = args.result_path
    leaveout = args.leaveout
    

    main(yaml_path,
         ckpt_dir, 
         ckpt_epoch, 
         dataset_name,
         model_name, 
         classifier_ckpt_path, 
         result_path,
         leaveout)