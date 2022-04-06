from tqdm import tqdm
import sklearn
from sklearn.metrics import silhouette_score, f1_score
from sklearn import model_selection
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def training_evaluation(model,  
                        dataset_loader, 
                        adversarial, 
                        metrics, 
                        dmso_id,
                        device, end=False, variational=True, ood=False, predict_moa=False):
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

    Returns:
        tuple: validation losses and quality metrics dictionaries 
    """
    # Final dictionary containing all the losses
    losses = {}

    # Initialize the null metrics
    val_loss = 0  # Total validation loss
    if variational:
        val_recon_loss = 0  # Total reconstruction loss
        val_kl_loss = 0  # Total Kullback-Leibler loss
    if adversarial:
        rmse_basal_full = 0  # The difference between a decoded image with and without addition of the drug
    
    # Zero out the metrics for the next step
    metrics.reset()  

    # Settings for ground truth prediction
    y_true_ds_drugs = []
    y_hat_ds_drugs = []
    if predict_moa:
        y_true_ds_moa = []
        y_hat_ds_moa = []
    
    # If we are at the last iteration of a CPA-like model we also store z_basal predictions for disentanglement metrics
    if end:
        z_basal_ds = []  # Will contain the basal latent representation (no drug effect)
        z_ds = []  # Will contain the total drug representation (with added drug effect)

    for observation in tqdm(dataset_loader):
        # Load observation X
        X = observation['X'].to(device)

        # If not number of cells prediction, store the whole drug label array      
        y_adv_drugs = observation['mol_one_hot'].to(device)
        # Store labels
        y_true_ds_drugs.append(torch.argmax(y_adv_drugs, dim=1).item())
        if predict_moa:
            y_adv_moa = observation['moa_one_hot'].to(device)
            y_true_ds_moa.append(torch.argmax(y_adv_moa, dim=1).item())

        else: 
            # If we are evaluating the ood fold, we use drug ids and not the one-hot encoded molecules
            y_adv_drugs = observation['smile_id'].to(device)
            y_true_ds_drugs.append(y_adv_drugs.item())
        
        if not adversarial:
            res = model.evaluate(X)
            out, z_basal, ae_loss = res.values()
        else:
            # Get evaluation results
            drug_id = observation["smile_id"].to(device)
            if predict_moa:
                moa_id = observation["moa_id"].to(device)
            else: 
                moa_id = None 

            res = model.evaluate(X, y_adv_drugs, y_adv_moa, drug_id=drug_id, moa_id=moa_id)
            out, out_basal, z_basal, z, y_hat_drug, y_hat_moa, ae_loss, _, _ = res.values()

            rmse_basal_full += metrics.compute_batch_rmse(out, out_basal).item()
            # Collect the predicted labels 
            if not ood or predict_moa:
                y_hat_ds_drugs.append(torch.argmax(y_hat_drug, dim=1).item())
                if predict_moa:
                    y_hat_ds_moa.append(torch.argmax(y_hat_moa, dim=1).item())
            
        # Only at the end of training we store the latent vectors for analysis 
        if end:
            if adversarial:
                z_ds.append(z)
            z_basal_ds.append(z_basal)
            

        # Update the losses and the metrics 
        val_loss += ae_loss['total_loss'].item()
        if variational:
            val_recon_loss += ae_loss['reconstruction_loss'].item()
            val_kl_loss += ae_loss['KLD'].item()
        
        # Update RMSE on valid set
        metrics.update_rmse(X, out)

    if not ood and adversarial:
        if X.shape[0]>1:
            # Update metric for drug prediction
            y_true_ds_drugs = torch.cat(y_true_ds_drugs, dim=0).to('cpu').numpy()
            y_hat_ds_drug = torch.cat(y_hat_drug, dim=0).to('cpu').numpy()
            if predict_moa:
                y_true_ds_moa = torch.cat(y_true_ds_moa, dim=0).to('cpu').numpy()
                y_hat_ds_moa = torch.cat(y_hat_ds_moa, dim=0).to('cpu').numpy()
        
        # Check classification report on the labels
        metrics.compute_classification_report(y_true_ds_drugs, y_hat_ds_drugs, '_drug')
        
        # Update the metric for the moa prediction, if applicable
        if predict_moa: 
            metrics.compute_classification_report(y_true_ds_moa, y_hat_ds_moa, '_moa')
                    
    # Print loss results 
    losses["loss"] = val_loss/len(dataset_loader)
    if variational:
        losses["avg_validation_recon_loss"] = val_recon_loss/len(dataset_loader)
        metrics.update_bpd(losses["avg_validation_recon_loss"])
        losses["avg_validation_kld_loss"] = val_kl_loss/len(dataset_loader)
    else:
        metrics.update_bpd(losses["loss"])

    # Update the rmse and rmse_basal_full metric 
    metrics.metrics['rmse'] /= len(dataset_loader)
    if adversarial:
        metrics.metrics['rmse_basal_full'] = rmse_basal_full/len(dataset_loader)

    # Disentanglement and clustering evaluated only at the end
    if end:
        # Exclude the DMSO from the predictions 
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
                metrics.metrics["difference_disentanglement_drug"] = disentanglement_score_z_moa - disentanglement_score_basal_moa

        z_basal_ds = z_basal_ds.to('cpu').numpy()
        if adversarial:
            z_ds = z_ds.to('cpu').numpy()

        if not ood or predict_moa:
            # Silhouette score before and after drug (and moa) additions
            silhouette_score_basal_drugs = compute_silhouette_coefficient(z_basal_ds, y_true_ds_drugs)
            metrics.metrics["silhouette_score_basal_drugs"] = silhouette_score_basal_drugs
            if adversarial:
                silhouette_score_z_drugs = compute_silhouette_coefficient(z_ds, y_true_ds_drugs)
                metrics.metrics["silhouette_score_z_drugs"] = silhouette_score_z_drugs
            if predict_moa:
                silhouette_score_basal_moa = compute_silhouette_coefficient(z_basal_ds, y_true_ds_moa)
                metrics.metrics["silhouette_score_basal_moa"] = silhouette_score_basal_moa
                if adversarial:
                    silhouette_score_z_moa = compute_silhouette_coefficient(z_ds, y_true_ds_moa)
                    metrics.metrics["silhouette_score_z_moa"] = silhouette_score_z_moa

    print(f'Average validation loss: {losses["loss"]}')
    if variational:
        print(f'Average validation reconstruction loss: {losses["avg_validation_recon_loss"]}')
        print(f'Average kld reconstruction loss: {losses["avg_validation_kld_loss"]}')

    metrics.print_metrics()
    return losses, metrics.metrics


def compute_disentanglement_score(Z, drug, return_misclass_report=False):
    """Train a classifier that evaluates the disentanglement of the latents space from the information
    on the drug

    Args:
        Z (torch.tensor): Latent representation of the validation set under the model 
        y (torch.tensor): The label assigned to each observation

    Returns:
        dict: dictionary with the results
    """
    print('Training discriminator network on drug latent space')

    # Normalize the latent descriptors 
    mean = Z.mean(dim=0, keepdim=True)
    stddev = Z.std(0, unbiased=False, keepdim=True)
    normalized_basal = (Z - mean) / stddev

    # Fetch unique molecules and their counts  
    unique_classes, freqs = np.unique(drug, return_counts=True)  # Given a class vector y, reduce it to unique definitions 
    label_to_idx = {labels: idx for idx, labels in enumerate(unique_classes)}
    # Bind each class to the number of occurrences in the test set 
    class2freq = {key:value for key, value in zip(unique_classes, freqs)}
    # Keep only the molecules with more than 3 instances
    class_to_keep = [key for key in class2freq if class2freq[key]>3]

    # Single out the indexes to keep 
    idx_to_keep = np.array([i for i in np.arange(len(drug)) if drug[i] in class_to_keep])

    # Filter the inputs and the labels
    X = normalized_basal[idx_to_keep]
    y = np.array(drug)[idx_to_keep]

    # Split into training and test set 
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    y_train_tensor = torch.tensor(
        [label_to_idx[label] for label in y_train], dtype=torch.long, device="cuda")
    y_test_tensor = torch.tensor(
        [label_to_idx[label] for label in y_test], dtype=torch.long, device="cpu")

    # Create loader and dataset
    dataset = torch.utils.data.TensorDataset(X_train, y_train_tensor)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    # initialize nwtwork and training hyperparameters 
    net = torch.nn.Linear(X_train.shape[1], len(unique_classes)).to('cuda')
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


def compute_silhouette_coefficient(Z, y):
    """Compute the silhouette score of the dataset given the labels y

    Args:
        Z (torch.tensor): A matrix of latent oservations
        y (torch.tensor): The labels of the matrix
    """
    return silhouette_score(Z, y, metric='euclidean')
    