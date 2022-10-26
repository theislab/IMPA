# IMPA

**IMage Perturbation Autoencoder** (IMPA) is a computer vision model performing style transfer on images of cells undergoing perturbation. IMPA learns a perturbation space and uses it to perform style transfer by conditioning the latent space of an autoencoder. Through this process, the model is used to translate a cell image into what it would look like had it been treated with a certain perturbation. 

The perturbation space can be expressed as:
* A prior on the perturbation space (e.g. drug embeddings reflecting compound-specific physiochemical characteristics)
* Trainable embeddings optimized alongside the model 


