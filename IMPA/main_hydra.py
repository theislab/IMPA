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
    """Create the directories and sub-directories for storing training results.

    Args:
        args: A configuration object with details about experiment directories, 
              task name, resumption iteration, sample and checkpoint directories.

    Returns:
        dest_dir (str): The path to the directory where results will be saved.
    """
    unique_id = str(uuid.uuid4())  # Generate a unique ID for the run.
    timestamp = datetime.datetime.now().strftime("%Y%m%d")  # Timestamp for directory name.

    task_name = args.task_name  # Name the folder based on the task being performed. From config.

    if args.resume_iter == 0:
        # If training is from scratch, create a new directory.
        dest_dir = ospj(args.experiment_directory, f"{timestamp}_{unique_id}_{task_name}")
    else:
        # If resuming, use the existing directory with a modified name.
        dest_dir = f"{args.resume_dir}_{task_name}"

    # Create main and sub-directories for saving samples and checkpoints.
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(ospj(dest_dir, args.sample_dir), exist_ok=True)
    os.makedirs(ospj(dest_dir, args.checkpoint_dir), exist_ok=True)
    
    return dest_dir  # Return the destination directory path.


@hydra.main(config_path="../config_hydra", config_name="train", version_base=None)
def main(config: DictConfig):
    """Main function to initialize and run the training pipeline using Hydra configuration.

    Args:
        config (DictConfig): The configuration dictionary, loaded by Hydra.

    Steps:
        1. Extract arguments from the config.
        2. Create necessary directories for saving outputs.
        3. Initialize the data loader, solver, and PyTorch Lightning Trainer.
        4. Set up logging and model checkpointing.
        5. Train the model with the specified settings.
    """
    args = config.config  # Extract arguments from the configuration.

    # Step 1: Create directories for storing results and checkpoints.
    dest_dir = create_dirs(args)
        
    # Step 2: Initialize the data module (data loading, preprocessing).
    datamodule = CellDataLoader(args)
        
    # Step 3: Initialize the solver (the model and associated logic).
    solver = IMPAmodule(args, dest_dir, datamodule)
        
    # Step 4: Set up model checkpointing to save model state at intervals.
    model_ckpt_callbacks = ModelCheckpoint(
        dirpath=Path(dest_dir) / "hydra_checkpoints",  # Save checkpoints here.
        filename=args.filename,  # Filename template for checkpoints.
        monitor=args.monitor,  # Metric to monitor for saving the best model.
        mode=args.mode,  # Mode of comparison (e.g., 'min' or 'max').
        save_last=args.save_last  # Save the last model checkpoint.
    )

    # Step 5: Initialize logging using Weights and Biases (WandB).
    logger = WandbLogger(
        save_dir=dest_dir,  # Directory to save logs.
        offline=args.offline,  # Whether to log offline.
        project=args.project,  # Name of the project.
        log_model=args.log_model  # Whether to log the model.
    )
                
    # Step 6: Initialize PyTorch Lightning Trainer to manage the training loop.
    trainer = Trainer(
        callbacks=model_ckpt_callbacks,  # Pass the checkpoint callback.
        default_root_dir=dest_dir,  # Directory to save outputs.
        logger=logger,  # Pass the WandB logger.
        max_epochs=args.total_epochs,  # Number of epochs to train.
        accelerator=args.accelerator,  # Specify hardware (e.g., 'gpu').
        log_every_n_steps=args.log_every_n_steps  # Frequency of logging.
    )
            
    # Step 7: Start the training process.
    trainer.fit(
        model=solver,  # The model/solver to train.
        train_dataloaders=datamodule.train_dataloader(),  # Train data loader.
        val_dataloaders=datamodule.val_dataloader()  # Validation data loader.
    )
    
if __name__ == "__main__":
    try:
        main()  # Run the main function.
    except:
        traceback.print_exc(file=sys.stderr)  # Print the full traceback if an error occurs.
        raise  # Re-raise the exception.
    