# IMPA

# Image Perturbation Autoencoder (IMPA)

**Image Perturbation Autoencoder** (IMPA) is a model designed for style transfer on images of cells subjected to various perturbations. IMPA is a Generative Adversarial Network (GAN) based on an autoencoder model. Images of control cells are encoded into a high-dimensional latent space and decoded conditioned on a perturbation representation. The decoder's output is used to deceive a discriminator model into classifying the decoded cells as truly coming from the target perturbation. In this way, our approach allows us to predict how a control cell would look had it been perturbed by a given treatment. 

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

* `utils.py`: contains helper functions
* `solver.py`: contains the `Solver` class implementing the model setup, data loading and training loop. 
* `model.py`: implements the neural network modules and initialization function.
* `main.py`: calls the `Solver` class and implements training supported by `seml` and `sacred`.
* `checkpoint.py`: implements the util class for handling saving and loading checkpoints.
* `eval/eval.py`: contains the evaluation script used during training by the `Solver` class.
* `data/data_loader.py`: implements `torch` dataset and data loader wrappers around the image data.

## Train the models

We trained the models using the [seml](https://github.com/TUM-DAML/seml) framework. Configurations can be found in the `training_config` folder. IMPA can be trained both with and without the support of `seml`. This is possible via two independent main files:  
* `main.py`: train with `seml` on the `slurm` scheduling system 
* `main_not_seml.py`: train without `seml` on the `slurm` scheduling system via sbatch files

Scripts to run the code without `seml` can be found in the `scripts` folder. In a terminal, enter:
```
sbatch training_config.yaml 
```
And the script will be submitted automatically. The logs of the run will be saved in the `training_config/logs` folder. 

For other scheduling systems the user may be required to apply minor modifications to the `main.py` file to accept custom configuration files. For training with `seml` we redirect the user to the [official page](https://github.com/TUM-DAML/seml) of the package.



To train the model with the provided yaml files, adapt the `.yaml` files to the experimental setup (*i.e.* add  path strings referencing the used directories).


## Dataset and checkpoints
Datasets are available at:
* BBBC021 https://bbbc.broadinstitute.org/BBBC021
* BBBC025 https://bbbc.broadinstitute.org/BBBC025
* RxRx1 https://www.kaggle.com/c/recursion-cellular-image-classification/overview/resources  
  
Model checkpoints and pre-processed data are made available [here](https://zenodo.org/record/8307629).
