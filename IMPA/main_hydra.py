import sys
import traceback
import os
from os.path import join as ospj
import warnings
import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

import uuid
from IMPA.dataset.data_loader import CellDataLoader
from IMPA.solver import IMPAmodule

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Filter out torch warnings 
warnings.filterwarnings(
    "ignore",
    "There is a wandb run already in progress",
    module="pytorch_lightning.loggers.wandb",
)

def create_dirs(args):
    """Create the directories and sub-directories for training
    """
    # date and time to name run 
    unique_id = str(uuid.uuid4())
    
    # Directory is named based on time stamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    # Setup the key(s) naming the folder (passed as hyperparameter)
    task_name = args.task_name

    # Set the directory for the results based on whether training is from begginning or resumed
    if args.resume_iter==0:
        dest_dir = ospj(args.experiment_directory, timestamp+'_'+unique_id+'_'+task_name)
    else:
        dest_dir = args.resume_dir+'_'+task_name

    # Create sub-directories to save partial and definitive results 
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(ospj(dest_dir, args.sample_dir), exist_ok=True)
    os.makedirs(ospj(dest_dir, args.checkpoint_dir), exist_ok=True)
    os.makedirs(ospj(dest_dir, args.embedding_folder), exist_ok=True)   
    return dest_dir

@hydra.main(config_path="../config_hydra", config_name="train", version_base=None)
def main(config: DictConfig):
    # Extract config dict 
    args = config.config
    
    # Initialize folders
    dest_dir = create_dirs(args)
        
    # Initialize datamodule
    datamodule = CellDataLoader(args)
        
    # Initialize the solver 
    solver = IMPAmodule(args, dest_dir, datamodule)
        
    # Initialize callbacks 
    model_ckpt_callbacks = ModelCheckpoint(dirpath=Path(dest_dir) / "hydra_checkpoints", 
                                            filename=args.filename,
                                            monitor=args.monitor,
                                            mode=args.mode,
                                            save_last=args.save_last)

    # Initialize logger 
    logger = WandbLogger(save_dir=dest_dir, 
                            offline=args.offline,
                            project=args.project,
                            log_model=args.log_model) 
                
    # Initialize the lightning trainer 
    trainer = Trainer(callbacks=model_ckpt_callbacks, 
                        default_root_dir=dest_dir,
                        logger=logger, 
                        max_epochs=args.total_epochs,
                        accelerator=args.accelerator,
                        log_every_n_steps=args.log_every_n_steps)
            
    # Fit the model 
    trainer.fit(model=solver, 
                train_dataloaders=datamodule.train_dataloader(),
                val_dataloaders=datamodule.val_dataloader())

if __name__=="__main__":
    try:
        main()
    except:
        traceback.print_exc(file=sys.stderr)
        raise
    