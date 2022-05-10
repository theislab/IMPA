from tqdm import tqdm
import sklearn
from sklearn.metrics import silhouette_score, f1_score
from sklearn import model_selection
import numpy as np
import torch
from torch import nn
import warnings
from torch.nn import functional as F
# from .model.modules.discriminator.discriminator_net import *
from model.modules.adversarial.adversarial_nets import *
import os 
import pickle as pkl

def training_evaluation(model,  
                        dataset_loader, 
                        adversarial, 
                        metrics,
                        losses, 
                        device, 
                        end=False, 
                        variational=True, 
                        ds_name=None, 
                        drug2moa=None, 
                        save_path = None):

    """Evaluation loop on the validation set to compute evaluation metrics of the model 

    Args:
        model (nn.Model): The torch model being trained
        dataset_loader (torch.utils.data.DataLoader): Data loader of the split of interest
        adversarial (bool): Whether adversarial training is performed or not
        metrics (Metrics): Metrics object for computing qulity scores  
        device (str): `cuda` or `cpu`
        end (bool, optional): If the evaluation is performed at the end of the training loop. Defaults to False.
        variational (bool, optional): Whether a VAE is used over an AE. Defaults to True.

    Returns:
        tuple: validation losses and quality metrics dictionaries 
    """
    # Reset the metrics and the losses 
    metrics.reset()  
    losses.reset()

    if adversarial:
        rmse_basal_full = 0  # The difference between a decoded image with and without addition of the drug and moa
        conterfactual_rmse = 0 # The average difference between the image and the counterfacted version of it 

    # STORE PREDICTIONS AND GROUND TRUTH LABELS 
    y_true_ds_drugs = []  
    y_hat_ds_drugs = []

    # STORE THE LATENTS FOR LATER ANALYSIS
    z_basal_ds = []  
    z_ds = []  

    # LOOP OVER SINGLE OBSERVATIONS 
    for observation in tqdm(dataset_loader):

        # COLLECT DATA FROM BATCH 
        X = observation['X'].to(device)  # X matrix
        y_adv_drugs = observation['mol_one_hot'].to(device)  # Adversary drug prediction  
        y_true_ds_drugs.append(torch.argmax(y_adv_drugs, dim=1).item())  # Record the labels 

        # PREDICTIONS  
        if not adversarial:
            with torch.no_grad():
                res = model.autoencoder.forward_ae(X, y_drug=None, mode='eval')
                out, _, _, z_basal, ae_loss = res.values()

            # Record the obtained loss functions 
            z_basal_ds.append(z_basal)

        else:
            # Label id for encoding
            drug_id = y_adv_drugs.argmax(1).to(device)
            
            if model.encoded_covariates:
                # Get the embedded drug and mode of action 
                z_drug = model.encode_cov_labels(drug_id)
            
            else:
                z_drug = y_adv_drugs
                        
            # Autoencoder step 
            with torch.no_grad():
                out, out_basal, z, z_basal, ae_loss = model.autoencoder.forward_ae(X, y_drug=z_drug, mode='eval').values()
            # Save non-redundacy latent basal and latent 
            rmse_basal_full += metrics.compute_batch_rmse(out, out_basal).item()  # RMSE between out and out_basal 
            
            # Append the z for score later on 
            z_ds.append(z)
            z_basal_ds.append(z_basal)

            # Latent adversary step and prediction append 
            y_hat_drug = model.adversary_drugs(z_basal)
            y_hat_ds_drugs.append(torch.argmax(y_hat_drug, dim=1).item())

            # Update the counterfactual score 
            conterfactual_rmse += counterfactual_score(X, z_basal, model, y_adv_drugs, drug_id)

        # Update the autoencoder losses
        losses.update_losses(ae_loss)

        # Update RMSE on valid set
        metrics.update_rmse(X, out)


    # Compute classification report adversary out of the for loop 
    if X.shape[0]>1:
        # Update metric for drug prediction
        y_true_ds_drugs = torch.cat(y_true_ds_drugs, dim=0).to('cpu').numpy()
        y_hat_ds_drugs = torch.cat(y_hat_drug, dim=0).to('cpu').numpy()
    
    # Check classification report on the labels
    if adversarial:
        metrics.compute_classification_report(y_true_ds_drugs, y_hat_ds_drugs, '_drug')
    
    # Average losses 
    losses.average_losses(len(dataset_loader))    
    losses.print_losses()

    # Update the bit per dimension metric
    recon_loss = losses.loss_dict['recon_loss'] if variational else losses.loss_dict['total_loss']

    metrics.metrics['rmse'] = metrics.metrics['rmse']/len(dataset_loader)
    # Add the adversarial scores to the metrics 
    if adversarial:
        metrics.metrics['rmse_basal_full'] = rmse_basal_full/len(dataset_loader)
        metrics.metrics['conterfactual_rmse'] = conterfactual_rmse/len(dataset_loader)


    # COMPUTE THE DISENTANGLEMENT SCORES for drugs that are not DMSO
    y_true_ds_drugs = np.array(y_true_ds_drugs)

    z_basal_ds = torch.cat(z_basal_ds, dim=0)
    disentanglement_score_basal_drug = compute_disentanglement_score(z_basal_ds, y_true_ds_drugs)  # Evaluate on the non-controls
    metrics.metrics["disentanglement_score_basal_drug"] = disentanglement_score_basal_drug  

    if adversarial:
        z_ds = torch.cat(z_ds, dim=0)
        disentanglement_score_z_drug = compute_disentanglement_score(z_ds, y_true_ds_drugs)
        metrics.metrics["disentanglement_score_z_drug"] = disentanglement_score_z_drug
        metrics.metrics["difference_disentanglement_drug"] = disentanglement_score_z_drug - disentanglement_score_basal_drug


    del z_basal_ds
    del z_ds
        
    # Print metrics 
    metrics.print_metrics()
    return losses.loss_dict, metrics.metrics


def compute_disentanglement_score(Z, y, return_misclass_report=False):
    """Train a classifier that evaluates the disentanglement of the latents space from the information
    on the drug

    Args:
        Z (torch.tensor): Latent representation of the validation set under the model 
        y (torch.tensor): The label assigned to each observation

    Returns:
        dict: dictionary with the results
    """
    print('Training discriminator network on drug latent space')
    
    # Fetch unique molecules and their counts  
    unique_classes = np.unique(y)  # Given a class vector y, reduce it to unique definitions 
    label_to_idx = {labels: idx for idx, labels in enumerate(unique_classes)}

    # Filter the inputs and the labels
    X = Z
    y = np.array(y)

    # Split into training and test set 
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.10, stratify=y)  # Make the dataset balanced 

    y_train_tensor = torch.tensor(
        [label_to_idx[label] for label in y_train], dtype=torch.long, device="cuda")
    y_test_tensor = torch.tensor(
        [label_to_idx[label] for label in y_test], dtype=torch.long, device="cpu")

    # Create loader and dataset
    dataset = torch.utils.data.TensorDataset(X_train, y_train_tensor)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    # initialize nwtwork and training hyperparameters 
    net = DisentanglementClassifier(X_train.shape[2], X_train.shape[1], 32, len(unique_classes)).to('cuda')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Train the small network 
    for epoch in tqdm(range(100)):
        for X, y in data_loader:
            pred = net(X.to('cuda'))
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    test_pred = net(X_test.to('cuda'))

    if return_misclass_report:
        return sklearn.metrics.classification_report(y_test_tensor.numpy(), test_pred.argmax(1).to('cpu').numpy())
    else:
        return f1_score(y_test_tensor.numpy(), test_pred.argmax(1).to('cpu').numpy(), average="weighted")


def counterfactual_score(X, z_basal, model, y_adv_drug,  drug_id):
    max_drug = y_adv_drug.shape[1]  # Single integer representing maximum drug id

    drug_ids = [i for i in range(max_drug) if i!=drug_id]  # Drug ids for counterfactual (different from the data point)

    with torch.no_grad():
        if model.encoded_covariates:
            # Encode drugs and moas torch
            drug_emb = model.drug_embeddings(torch.tensor(drug_ids).to('cuda'))  # Get the embeddings of all the drugs 
            z_drug = model.drug_embedding_encoder(drug_emb)  # One for each drug 
        
            res, _ = model.autoencoder.decoder(z_basal.repeat(drug_emb.shape[0],1,1,1), z_drug)  # Must repeat z on the batch dimension

        else:
            # One hot encoded drugs and moas
            y_adv_drug = torch.eye(max_drug)

            y_adv_drug = torch.cat((y_adv_drug[:drug_id], y_adv_drug[drug_id+1:]))

            res, _ = model.autoencoder.decoder(z_basal.repeat(y_adv_drug.shape[0],1,1,1), y_adv_drug.to('cuda'))
        
        # Decode the counterfactual vector
        rmse = torch.sqrt(torch.mean((X.repeat(res.shape[0], 1, 1, 1) - res)**2))
    
    return rmse.item()


if __name__ == '__main__':
    Z = torch.rand(20, 256, 12, 12)
    y = torch.randint(low=0, high=3, size=(20,) )
    print(compute_disentanglement_score(Z, y, return_misclass_report=False))
