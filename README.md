# IMPA

# Image Perturbation Autoencoder (IMPA)

**Image Perturbation Autoencoder** (IMPA) is a model designed for style transfer on images of cells treated with various perturbations. IMPA is a Generative Adversarial Network (GAN) based on an autoencoder model. Images of control cells are encoded into a high-dimensional latent space and decoded conditioned on a perturbation representation. The decoder's output is used to deceive a discriminator model into classifying the decoded cells as truly coming from the target perturbation. In this way, our approach allows us to predict how a control cell would look had it been perturbed by a given treatment. 

The perturbation can be designed as:

- **Perturbation embeddings**: Representations capturing compound-specific physiochemical properties, such as drug characteristics.  
- **Trainable embeddings**: Learned representations optimized in tandem with the model during training.

Beyond style transfer, IMPA is also effective for batch correction. By training the model to harmonize cell images from different experimental sources, it can standardize data into a unified batch for downstream analysis.

<p align="center">
  <img src="https://github.com/theislab/IMPA/blob/main/docs/IMPA.png" width="700" height="300">
</p>

## Install repository 
To run the model, clone this repository and create the environment via:

```
conda env create -f environment.yml
```

Navigate to the repository and install the Python package. 

```
pip install -e .
```

## Codebase description 
All files related to the model are stored in the  `IMPA` folder. 

* `checkpoint.py`: implements the util class for saving and loading checkpoints.
* `main_hydra.py`: calls the `Solver` class and implements training supported by [hydra](https://hydra.cc/docs/1.3/intro/).
* `model.py`: implements the neural network modules and initialization function.
* `solver.py`: contains the `Solver` class implementing the model setup, data loading and training loop.
* `utils.py`: contains helper functions.
* `eval/eval.py`: contains the evaluation script used during training by the `Solver` class.
* `dataset/data_loader.py`: implements `torch` dataset and data loader wrappers around the image data.
* `dataset/data_utils.py`: implements utilfunctions for the data loader.

## Set up

Setting up the repository requires creating a project folder. 

```
cd IMPA
mkdir project_folder
```

One can also create the project folder elsewhere, e.g. a directory with larger storage capacity, and create a symlink from the IMPA directory to the chosen project folder location:

```
cd IMPA
ln -s path/to/storage/folder/ project_folder
```

Subsequently, download the data (and the model checkpoints) [here](https://zenodo.org/record/8307629). Unzip and move the dataset folder to `IMPA/project folder`. The checkpoints should be at `IMPA/checkpoints`.
 
## Train the models

We trained the model using a combination of [hydra](https://hydra.cc/docs/1.3/intro/) and [pytorch lightning](https://lightning.ai/docs/pytorch/stable/):
* hydra: used to handle configurations and hyperparameter optimization.
* pytorch lightning: used to set up the data loading, model training and logging.  

We configure model training and hyperparameter tuning in line with the requirements for the SLURM job manager. If SLURM is not present on the user's system, the scripts for launching training can be adapted to a normal bash session by removing the slurm syntax in the provided `sbatch` scripts.

We provide training scripts for perturbation prediction and batch correction in the `script` folder. To run training on a selected task, launch the following command:

```
conda activate IMPA
sbatch training_config.sbatch
```

And the script will be submitted automatically. 

To train the model with the provided yaml files, adapt the `.yaml` files to the experimental setup (*i.e.* add path strings referencing the used directories).

## Tutorials 
Tutorials are available in the `IMPA/tutorials` folder as notebooks.

## Dataset and checkpoints
Model checkpoints and pre-processed data are made available [here](https://zenodo.org/record/8307629).
