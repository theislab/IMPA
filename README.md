# IMPA

**IMage Perturbation Autoencoder** (IMPA) is a computer vision model performing style transfer on images of cells undergoing perturbation. IMPA learns a perturbation space and uses it to perform style transfer by conditioning the latent space of an autoencoder. Through this process, the model is used to translate a cell image into what it would look like had it been treated with a certain perturbation. 

The perturbation space can be expressed as:
* A prior on the perturbation space (e.g. drug embeddings reflecting compound-specific physiochemical characteristics)
* Trainable embeddings optimized alongside the model 

If trained on a meaniningful prior perturbation space, IMPA learns to map unseen drugs in proximity of drugs used for training. When proximity also involves functional similarity, IMPA is able to predict the effect of unseen drugs on control cells. Moreover, distances between the style encodings of different perturbations are correlated with distances in the phenotypic space. As a result, IMPA can be used to fastly inspect active compounds based on the comparison of the style vectors learned for different perturbations. 

<p align="center">
  <img src="https://github.com/theislab/imCPA/blob/add_readme_and_package/docs/IMPA.png">
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
