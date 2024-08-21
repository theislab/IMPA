import sys
from pathlib import Path
sys.path.insert(0, "/home/icb/alessandro.palma/environment/IMPA/StyTR-2")

from IMPA.dataset.data_loader import CellDataLoader
import yaml
import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data

from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import models.transformer as transformer
import models.StyTR  as StyTR 
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from dataloader_impa import CellDataLoader
from omegaconf import OmegaConf
from compute_metrics_stytr import *

def main(yaml_path,
         ckpt_dir, 
         ckpt_epoch, 
         dataset_name,
         model_name, 
         classifier_ckpt_path, 
         result_path, 
         leaveout):
    
    # Transform the checkpoint path string into a Path object 
    ckpt_dir = Path(ckpt_dir)
    
    # Configuration 
    with open(yaml_path, 'r') as file:
        config_params = yaml.safe_load(file)
    
    # Data loader can be IMPA's
    args = OmegaConf.create(config_params)
    dataloader = CellDataLoader(args)
    
    # Initialize encoder
    vgg = StyTR.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])
    
    # Initialize decoder and batch embedder 
    decoder = StyTR.decoder
    embedding = StyTR.PatchEmbed()
    
    # Initialize decoder and batch embed
    Trans = transformer.Transformer()
    with torch.no_grad():
        network = StyTR.StyTrans(vgg,
                                 decoder,
                                 embedding, 
                                 Trans,
                                 args)
        
    decoder_weights = torch.load(ckpt_dir / f"decoder_iter_{ckpt_epoch}.pth")
    trans_weights = torch.load(ckpt_dir / f"transformer_iter_{ckpt_epoch}.pth")
    embedding_weights = torch.load(ckpt_dir / f"embedding_iter_{ckpt_epoch}.pth")

    decoder.load_state_dict(decoder_weights)
    Trans.load_state_dict(trans_weights)
    embedding.load_state_dict(embedding_weights)

    solver = {"network": network.cuda(),
                "decoder": decoder.cuda(),
                "Trans": Trans.cuda(),
                "embedding": embedding.cuda()}
    
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