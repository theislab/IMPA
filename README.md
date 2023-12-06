# IMPA

**IMage Perturbation Autoencoder** (IMPA) is a computer vision model performing style transfer on images of cells undergoing perturbation. IMPA learns a perturbation space and uses it to perform style transfer by conditioning the latent space of an autoencoder. Through this process, the model is used to translate a cell image into what it would look like had it been treated with a certain perturbation. 

The perturbation space can be expressed as:
* A prior on the perturbation space (e.g. drug embeddings reflecting compound-specific physiochemical characteristics)
* Trainable embeddings optimized alongside the model 

If trained on a meaniningful prior perturbation space, IMPA learns to map unseen drugs in proximity of drugs used for training. When proximity also involves functional similarity, IMPA is able to predict the effect of unseen drugs on control cells. Moreover, distances between the style encodings of different perturbations are correlated with distances in the phenotypic space. As a result, IMPA can be used to fastly inspect active compounds based on the comparison of the style vectors learned for different perturbations. 

<p align="center">
  <img src="https://github.com/theislab/IMPA/blob/main/docs/IMPA.png" width="700" height="400">
</p>

## Install repository 
To run the model, clone this repository and create the environment via:

```
conda env create environment.yml
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
