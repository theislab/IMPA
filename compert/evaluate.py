import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm
import numpy as np
from compert.model.modules import MLP
from sklearn.metrics import silhouette_score
from compert.model.template_model import TemplateModel
from metrics.metrics import *

def training_evaluation(model,  
                        dataset_loader, 
                        adversarial, 
                        metrics, 
                        binary_task, 
                        device, end=False):
    """
    Given a dataset to perform validation on, aggregate the metrics on it and return them 
    """
    losses = {}
    # Initialize the null metrics
    val_loss = 0  # Total validation loss
    val_recon_loss = 0  # Total reconstruction loss
    val_kl_loss = 0  # Total Kullback-Leibler loss
    
    # Zero out the metrics for the next step
    metrics.reset()  

    # Classification vectors 
    y_true_ds = []
    y_hat_ds = []
    if end:
        z_basal_ds = []
        z_ds = []

    for observation in tqdm(dataset_loader):
        # Load observation X
        X = observation['X'].to(device)
        # Select the right task 
        if binary_task:
            y_adv = observation['state'].item()
            y_true_ds.append(y_adv)
        else:
            y_adv = observation['mol_one_hot'].to(device)
            # Store labels
            y_true_ds.append(torch.argmax(y_adv, dim=1).item())

        if not adversarial:
            res = model.evaluate(X)
            out, z_basal, z, ae_loss, recon_loss, kld = res.values()

        else:
            # Get evaluation results
            drug_id = observation["smile_id"].to(device)
            res = model.evaluate(X, y_adv=y_adv, drug_id=drug_id)
            out, z_basal, z, y_hat, ae_loss, recon_loss, kld = res.values()
            
            # Collect the labels 
            y_hat_ds.append(torch.argmax(y_hat, dim=1).item())
            metrics.compute_classification_report(y_true_ds, y_hat_ds)

        z_basal_ds.append(z_basal)
        z_ds.append(z)

        val_loss += ae_loss
        val_recon_loss += recon_loss
        val_kl_loss += kld
        
        # Perform optimizer step depending on the iteration
        metrics.update_rmse(X, out)

        
    # Print loss results
    losses["loss"] = val_loss/len(dataset_loader)
    losses["avg_validation_recon_loss"] = val_recon_loss/len(dataset_loader)
    metrics.update_bpd(losses["avg_validation_recon_loss"])
    losses["avg_validation_kld_loss"] = val_kl_loss/len(dataset_loader)

    if end:
        z_ds = torch.cat(z_ds, dim=0)
        z_basal_ds = torch.cat(z_basal_ds, dim=0)

        # Disentanglement score before and after drug addition 
        disentanglement_score_basal = compute_disentanglement_score(z_basal_ds, y_true_ds, device, binary_task)
        disentanglement_score_z = compute_disentanglement_score(z_ds, y_true_ds, device, binary_task)
        metrics.metrics["disentanglement_score_basal"] = disentanglement_score_basal
        metrics.metrics["disentanglement_score_z"] = disentanglement_score_z
        metrics.metrics["difference_disentanglement"] = disentanglement_score_basal - disentanglement_score_z

        z_ds = z_ds.to('cpu').numpy()
        z_basal_ds = z_basal_ds.to('cpu').numpy()

        # Silhouette score before and after drug addition 
        silhouette_score_basal = compute_silhouette_coefficient(z_basal_ds, y_true_ds)
        silhouette_score_z = compute_silhouette_coefficient(z_ds, y_true_ds)
        metrics.metrics["silhouette_score_basal"] = silhouette_score_basal
        metrics.metrics["silhouette_score_z"] = silhouette_score_z
        metrics.metrics["difference_silhouette"] = silhouette_score_z - silhouette_score_basal
        
    print(f'Average validation loss: {losses["loss"]}')
    print(f'Average validation reconstruction loss: {losses["avg_validation_recon_loss"]}')
    print(f'Average kld reconstruction loss: {losses["avg_validation_kld_loss"]}')

    metrics.print_metrics()
    return losses, metrics.metrics


def compute_disentanglement_score(Z, y, device, binary_task):
    """Train a classifier that evaluates the disentanglement of the latents space from the information
    on the drug

    Args:
        Z (torch.tensor): Latent representation of the validation set under the model 
        y (torch.tensor): The label assigned to each observation
        binary_task (bool): Whether the task is to predict the drug or the case/control condition  

    Returns:
        dict: dictionary with the results
    """
    print('Training discriminator network on drug latent space')
    # Normalize the latent descriptors 
    mean = Z.mean(dim=0, keepdim=True)
    stddev = Z.std(0, unbiased=False, keepdim=True)
    normalized_basal = (Z - mean) / stddev

    if not binary_task:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Collect labels for a classifier
    unique_labels = set(y)
    label_to_idx = {labels: idx for idx, labels in enumerate(unique_labels)}
    labels_tensor = torch.tensor(
        [label_to_idx[label] for label in y], dtype=torch.float if binary_task else torch.long, device="cuda"
    )
    assert normalized_basal.size(0) == len(labels_tensor), f'Z of length {normalized_basal.size(0)}, y of length {len(labels_tensor)}'

    dataset = torch.utils.data.TensorDataset(normalized_basal, labels_tensor)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    # 2 non-linear layers of size <input_dimension>
    # followed by a linear layer.
    disentanglement_classifier = MLP(
        [normalized_basal.size(1)]
        + [normalized_basal.size(1) for _ in range(2)]
        + [len(unique_labels) if not binary_task else 1]
    ).to(device)
    optimizer = torch.optim.Adam(disentanglement_classifier.parameters(), lr=1e-2)

    for epoch in tqdm(range(400)):
        for X, y in data_loader:
            pred = disentanglement_classifier(X)
            if not binary_task:
                loss = criterion(pred, y)
            else:
                loss = criterion(pred.squeeze(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print('Training discriminator network on drug latent space')

    with torch.no_grad():
        pred = disentanglement_classifier(normalized_basal).argmax(dim=1)
        acc = torch.sum(pred == labels_tensor) / len(labels_tensor)
    return acc.item()

def compute_silhouette_coefficient(Z, y):
    """Compute the silhouette score of the dataset given the labels y

    Args:
        Z (torch.tensor): A matrix of latent oservations
        y (torch.tensor): The labels of the matrix
    """
    return silhouette_score(Z, y, metric='euclidean')