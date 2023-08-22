# IMPA

**IMage Perturbation Autoencoder** (IMPA) is a computer vision model performing style transfer on images of cells undergoing perturbation. IMPA learns a perturbation space and uses it to perform style transfer by conditioning the latent space of an autoencoder. Through this process, the model is used to translate a cell image into what it would look like had it been treated with a certain perturbation. 

The perturbation space can be expressed as:
* A prior on the perturbation space (e.g. drug embeddings reflecting compound-specific physiochemical characteristics)
* Trainable embeddings optimized alongside the model 

If trained on a meaniningful prior perturbation space, IMPA learns to map unseen drugs in proximity of drugs used for training. When proximity also involves functional similarity, IMPA is able to predict the effect of unseen drugs on control cells. Moreover, distances between the style encodings of different perturbations are correlated with distances in the phenotypic space. As a result, IMPA can be used to fastly inspect active compounds based on the comparison of the style vectors learned for different perturbations. 

<p align="center">
  <img src="https://github.com/theislab/imCPA/blob/add_readme_and_package/docs/IMPA.png" width="700" height="500">
</p>

## Install repository 
To run the model, clone this repository and create the environment via 

```
conda env create environment.yml
```

Navigate to the repository and install the Python package. 

```
pip install -e .
```

## Create a project folder
The data and the experiment files should be contained in a `project_folder` located somewhere with enough storage. The project folder should be connected to the IMPA repository via a symlink through the following command. 

```
ln -s /path/to/your/folder/ project_folder
```

In the `project_folder`, create a `data` and a `results` subfolders.  

```
cd project_folder
mkdir datasets
mkdir results
```

In these subfolders, the data and the model checkpoints will be dumped.

## Codebase description 
All files related to the model are stored in the  `IMPA` folder. 

* `utils.py`: contains helper functions.
* `solver.py`: contains the `Solver` class implementing the model setup, data loading and training loop. 
* `model.py`: implements the neural network modules and initialization function.
* `main.py`: calls the `Solver` class and implements training supported by `seml` and `sacred`.
* `checkpoint.py`: implements the helper class for handling saving and loading checkpoints.
* `eval/eval.py`: contains the evaluation script used during training by the `Solver` class.
* `data/data_loader.py`: implements `torch` dataset and data loader wrappers around the image data.

## Train the models

We trained the models using the [seml](https://github.com/TUM-DAML/seml) framework. Configurations can be found in the `training_config` folder. If the `Slurm` scheduler is not implemented on the server, the `main.py` script can be re-written to run independently. To train the model with the provided yaml files, adapt the `.yaml` files to the experimental setup (*i.e.* add  path strings referencing the used directories).


## Dataset and checkpoints
Datasets are available at:
* BBBC021 https://bbbc.broadinstitute.org/BBBC021
* BBBC025 https://bbbc.broadinstitute.org/BBBC025
* RxRx1 https://www.kaggle.com/c/recursion-cellular-image-classification/overview/resources  
  
Pre-processed datasets will be available soon. Model checkpoints are made available [here](https://1drv.ms/f/s!AqLF-jPbzBG0sDkKwm0kjLtXAWi4?e=HdGC4h).
