import sys
sys.path.insert(0, "/home/icb/alessandro.palma/environment/IMPA/PhenDiff")
sys.path.insert(0, "/home/icb/alessandro.palma/environment/IMPA/IMPA/experiments/measure_metrics")

from IMPA.dataset.data_loader import CellDataLoader
import argparse
from omegaconf import OmegaConf
from diffusers import (
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
import yaml
from src.pipeline_conditional_ddim import ConditionalDDIMPipeline
from src.dataset_loading_impa import *

from compute_metrics_phendff import *

def main(yaml_path,
         ckpt_dir, 
         ckpt_epoch, 
         dataset_name,
         model_name, 
         classifier_ckpt_path, 
         result_path, 
         leaveout):
    
    # Configuration 
    with open(yaml_path, 'r') as file:
        config_params = yaml.safe_load(file)
    
    # Data loader can be IMPA's
    args = OmegaConf.create(config_params)
    dataloader = CellDataLoader(args)
    
    # Noising steps 
    nb_noising_iter = 100
    
    # Pipeline path
    DDIM_pipeline = ConditionalDDIMPipeline.from_pretrained(ckpt_dir)
    DDIM_denoiser = DDIM_pipeline.unet.to("cuda").eval()
    DDIM_noise_scheduler = DDIM_pipeline.scheduler  
    DDIM_noise_scheduler.set_timesteps(nb_noising_iter)
    DDIM_inv_scheduler = DDIMInverseScheduler.from_config(
        DDIM_noise_scheduler.config,
    )
    DDIM_inv_scheduler.set_timesteps(nb_noising_iter)
    
    solver = {"pipeline": DDIM_pipeline,
              "denoiser": DDIM_denoiser,
              "noise_scheduler": DDIM_noise_scheduler,
              "inverse_scheduler": DDIM_inv_scheduler,
              "noising_steps": nb_noising_iter}
    
    compute_all_scores(solver=solver, 
                        dataset=dataloader,
                        save_path=result_path,
                        ckpt_path=classifier_ckpt_path, 
                        model_name=model_name,
                        dataset_name=dataset_name,
                        ood_set=None)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure evaluation metrics for a given yaml config")
    parser.add_argument('--yaml_path')
    parser.add_argument('--ckpt_dir')
    parser.add_argument('--ckpt_epoch')
    parser.add_argument('--dataset_name')
    parser.add_argument('--model_name')
    parser.add_argument('--classifier_ckpt_path')
    parser.add_argument('--result_path')
    parser.add_argument('--leaveout', default=False)
    
    
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