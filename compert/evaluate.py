from tkinter import E
from zlib import Z_HUFFMAN_ONLY
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

def training_evaluation(model,  
                        dataset_loader, 
                        adversarial, 
                        metrics,
                        losses, 
                        dmso_id,
                        device, 
                        end=False, 
                        variational=True, 
                        ood=False, 
                        predict_moa=False, 
                        ds_name=None, 
                        drug2moa=None):

    """Evaluation loop on the validation set to compute evaluation metrics of the model 

    Args:
        model (nn.Model): The torch model being trained
        dataset_loader (torch.utils.data.DataLoader): Data loader of the split of interest
        adversarial (bool): Whether adversarial training is performed or not
        metrics (Metrics): Metrics object for computing qulity scores  
        device (str): `cuda` or `cpu`
        end (bool, optional): If the evaluation is performed at the end of the training loop. Defaults to False.
        variational (bool, optional): Whether a VAE is used over an AE. Defaults to True.
        ood (bool, optional): Whether evaluation is performed on the ood dataset. Defaults to False.
        predict_moa (bool, optional): whether the mode of action should be predicted or not 

    Returns:
        tuple: validation losses and quality metrics dictionaries 
    """
    if (ds_name=='cellpainting' and model.hparams['concat_one_hot'] and ood):
        warnings.warn("OOD + one-hot unsupported on Cell Painting")
        return {}, {} 
    
    # Reset the metrics and the losses 
    metrics.reset()  
    losses.reset()

    if adversarial:
        rmse_basal_full = 0  # The difference between a decoded image with and without addition of the drug
        conterfactual_rmse = 0 # The average difference between the image and the counterfacted version of it 

    # STORE PREDICTIONS AND GROUND TRUTH LABELS 
    y_true_ds_drugs = []  
    y_hat_ds_drugs = []
    if predict_moa:
        y_true_ds_moa = []
        y_hat_ds_moa = []

    # STORE THE LATENTS FOR LATER ANALYSIS
    z_basal_ds = []  
    z_ds = []  

    # LOOP OVER SINGLE OBSERVATIONS 
    for observation in tqdm(dataset_loader):

        # COLLECT DATA FROM BATCH 
        X = observation['X'].to(device)  # X matrix
        y_adv_drugs = observation['mol_one_hot'].to(device)  # Adversary drug prediction  
        y_true_ds_drugs.append(torch.argmax(y_adv_drugs, dim=1).item())  # Record the labels 

        if predict_moa:
            y_adv_moa = observation['moa_one_hot'].to(device)
            y_true_ds_moa.append(torch.argmax(y_adv_moa, dim=1).item())
        else: 
            y_adv_moa =  None 


        # PREDICTIONS  
        if not adversarial:
            with torch.no_grad():
                res = model.forward_ae(X, y_drug=None, y_moa=None, mode='eval')
                out, z_basal, ae_loss = res.values()

            # Record the obtained loss functions 
            losses.update_losses(ae_loss)
            z_basal_ds.append(z_basal)

        else:
            # ENCODE THE LABELS IF NECESSARY 
            # Label encoder 
            if model.encoded_covariates:
                drug_id = y_adv_drugs.argmax(1).to(device)
                if predict_moa:
                    moa_id = y_adv_moa.argmax(1).to(device)
                else: 
                    moa_id = None
                # Get the embedded drug and mode of action 
                z_drug, z_moa = model.encode_cov_labels(drug_id, moa_id)
            
            else:
                z_drug, z_moa = y_adv_drugs, y_adv_moa
            
            # Autoencoder step 
            out, out_basal, z, z_basal, ae_loss = model.forward_ae(X, y_drug=drug_id, y_moa=moa_id, mode='eval')
            rmse_basal_full += metrics.compute_batch_rmse(out, out_basal).item()  # RMSE between out and out_basal 
            
            # Append the z for score later on 
            z_ds.append(z)
            z_basal_ds.append(z_basal)

            # Latent adversary step and prediction append 
            y_hat_drug = model.adversary_drugs(z_basal)
            y_hat_moa = model.adversary_moa(z_basal)
            y_hat_ds_drugs.append(torch.argmax(y_hat_drug, dim=1).item())
            if predict_moa:
                y_hat_ds_moa.append(torch.argmax(y_hat_moa, dim=1).item())

            # Update the counterfactual score 
            conterfactual_rmse += counterfactual_score(X, z_basal, model, y_adv_drugs, y_adv_moa, drug_id, moa_id, drug2moa)
            # Update the autoencoder losses
            losses.update_losses(ae_loss)

        # Update RMSE on valid set
        metrics.update_rmse(X, out)

    # Compute classification report adversary
    if X.shape[0]>1:
        # Update metric for drug prediction
        y_true_ds_drugs = torch.cat(y_true_ds_drugs, dim=0).to('cpu').numpy()
        y_hat_ds_drugs = torch.cat(y_hat_drug, dim=0).to('cpu').numpy()
        if predict_moa:
            y_true_ds_moa = torch.cat(y_true_ds_moa, dim=0).to('cpu').numpy()
            y_hat_ds_moa = torch.cat(y_hat_ds_moa, dim=0).to('cpu').numpy()
    
    # Check classification report on the labels
    metrics.compute_classification_report(y_true_ds_drugs, y_hat_ds_drugs, '_drug')
    # Update the metric for the moa prediction, if applicable
    if predict_moa: 
        metrics.compute_classification_report(y_true_ds_moa, y_hat_ds_moa, '_moa')
    
    # Average losses 
    losses.average_losses()
    losses.print_losses()

    # Update the bit per dimension metric
    recon_loss = losses.loss_dict['recon_loss'] if variational else losses.loss_dict['total_loss']
    metrics.update_bpd(recon_loss*X.shape[1]*X.shape[2]*X.shape[3]) 

    metrics.average_losses()
    # Add the adversarial scores to the metrics 
    if adversarial:
        metrics.metrics['rmse_basal_full'] = rmse_basal_full/len(dataset_loader)
        metrics.metrics['conterfactual_rmse'] = conterfactual_rmse/len(dataset_loader)


    # COMPUTE THE DISENTANGLEMENT SCORES for drugs that are not DMSO
    y_true_ds_drugs = np.array(y_true_ds_drugs)
    idx_not_dmso = np.where(y_true_ds_drugs!=dmso_id)

    z_basal_ds = torch.cat(z_basal_ds, dim=0)
    disentanglement_score_basal_drug = compute_disentanglement_score(z_basal_ds[idx_not_dmso], y_true_ds_drugs[idx_not_dmso])  # Evaluate on the non-controls
    metrics.metrics["disentanglement_score_basal_drug"] = disentanglement_score_basal_drug  

    if adversarial:
        z_ds = torch.cat(z_ds, dim=0)
        disentanglement_score_z_drug = compute_disentanglement_score(z_ds[idx_not_dmso], y_true_ds_drugs[idx_not_dmso])
        metrics.metrics["disentanglement_score_z_drug"] = disentanglement_score_z_drug
        metrics.metrics["difference_disentanglement_drug"] = disentanglement_score_z_drug - disentanglement_score_basal_drug

    if predict_moa:
        y_true_ds_moa = np.array(y_true_ds_moa)
        disentanglement_score_basal_moa = compute_disentanglement_score(z_basal_ds[idx_not_dmso], y_true_ds_moa[idx_not_dmso])
        metrics.metrics["disentanglement_score_basal_moa"] = disentanglement_score_basal_moa
        if adversarial:
            disentanglement_score_z_moa= compute_disentanglement_score(z_ds[idx_not_dmso], y_true_ds_moa[idx_not_dmso]) 
            metrics.metrics["disentanglement_score_z_moa"] = disentanglement_score_z_moa
            metrics.metrics["difference_disentanglement_moa"] = disentanglement_score_z_moa - disentanglement_score_basal_moa

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
    unique_classes, freqs = np.unique(y, return_counts=True)  # Given a class vector y, reduce it to unique definitions 
    label_to_idx = {labels: idx for idx, labels in enumerate(unique_classes)}
    # Bind each class to the number of occurrences in the test set 
    class2freq = {key:value for key, value in zip(unique_classes, freqs)}
    # Keep only the molecules with more than 3 instances
    class_to_keep = [key for key in class2freq if class2freq[key]>3]

    # Single out the indexes to keep 
    idx_to_keep = np.array([i for i in np.arange(len(y)) if y[i] in class_to_keep])

    # Filter the inputs and the labels
    X = Z[idx_to_keep]
    y = np.array(y)[idx_to_keep]

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
    net = DisentanglementClassifier(X_train.shape[2], X_train.shape[1], 256, len(unique_classes)).to('cuda')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

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


def counterfactual_score(X, z_basal, model, y_adv_drug, y_adv_moa, drug_id, moa_id, drug2moa):
    max_drug = np.max(list(drug2moa.keys()))+1  # Single integer representing maximum drug id
    max_moa = np.max(list(drug2moa.values()))+1
    drug_ids = [i for i in range(max_drug) if i!=drug_id]  # Drug ids for counterfactual (different from the data point)
    moas = [drug2moa[i] for i in drug_ids]  # Moa ids assicuated to the drugs 

    with torch.no_grad():
        if model.hparams["decoding_style"] == 'sum' or (model.hparams["decoding_style"] == 'concat' and not model.hparams["concatenate_one_hot"]):
            # Encode drugs and moas torch
            drug_emb = model.drug_embeddings(torch.tensor(drug_ids).to('cuda'))
            moa_emb = model.moa_embeddings(torch.tensor(moas).to('cuda'))

            z_drug = model.drug_embedding_encoder(drug_emb)
            z_moa = model.moa_embedding_encoder(moa_emb)
        
            if model.hparams["decoding_style"] == 'sum': 
                z_counter = z_basal + z_drug + z_moa  # Broadcast automatocally
                res = model.decoder(z_counter, None, None)
            else:
                res = model.decoder(z_basal.repeat(drug_emb.shape[0],1,1,1), z_drug, z_moa)  # Must repeat z on the batch dimension

        elif model.hparams["decoding_style"] == 'concat':
            # One hot encoded drugs and moas
            y_adv_drug = torch.eye(max_drug)
            y_adv_moa = torch.zeros(y_adv_drug.shape[0], max_moa)

            y_adv_drug = torch.cat((y_adv_drug[:drug_id], y_adv_drug[drug_id+1:]))
            y_adv_moa = torch.cat((y_adv_moa[:moa_id], y_adv_moa[moa_id+1:]))
            y_adv_moa[torch.arange(len(y_adv_moa)), moas] = 1

            res = model.decoder(z_basal.repeat(y_adv_drug.shape[0],1,1,1), y_adv_drug.to('cuda'), y_adv_moa.to('cuda'))
        
        # Decode the counterfactual vector
        # res = model.decoder(z_basal, z_drug_broadcast, z_moa_broadcast)
        rmse = torch.sqrt(torch.mean((X.repeat(res.shape[0], 1, 1, 1) - res)**2))
    
    return rmse.item()


if __name__ == '__main__':
    Z = torch.rand(20, 256, 12, 12)
    y = torch.randint(low=0, high=3, size=(20,) )
    print(compute_disentanglement_score(Z, y, return_misclass_report=False))
