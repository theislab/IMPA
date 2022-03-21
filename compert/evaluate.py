from tqdm import tqdm
from sklearn.metrics import silhouette_score
import torch
from torch import nn
from torch.nn import functional as F

# MLP for the discriminator
from .model.modules import MLP


def training_evaluation(model,  
                        dataset_loader, 
                        adversarial, 
                        metrics, 
                        predict_n_cells, 
                        device, end=False, variational=True, ood=False):
    """Evaluation loop on the validation set to compute evaluation metrics of the model 

    Args:
        model (nn.Model): The torch model being trained
        dataset_loader (torch.utils.data.DataLoader): Data loader of the split of interest
        adversarial (bool): Whether adversarial training is performed or not
        metrics (Metrics): Metrics object for computing qulity scores  
        predict_n_cells (bool): Whether the task is to predict the drug or the number of cells
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

    # Classification vectors 
    y_true_ds = []
    y_hat_ds = []
    
    # If we are at the last iteration of a CPA-like model we also store z_basal predictions 
    if end:
        z_basal_ds = []  # Will contain the basal latent representation (no drug effect)
        z_ds = []  # Will contain the total drug representation (with added drug effect)

    for observation in tqdm(dataset_loader):
        # Load observation X
        X = observation['X'].to(device)
        # Select the right task 
        if predict_n_cells:
            # If number of cells prediction, then we store as labels 1.0 or 0.0 determining activity versus inactivity 
            y_adv = observation['state'].item()
            y_true_ds.append(y_adv)
        else:
            # If not number of cells prediction, store the whole drug label array 
            if not ood:
                y_adv = observation['mol_one_hot'].to(device)
                # Store labels
                y_true_ds.append(torch.argmax(y_adv, dim=1).item())
            else: 
                # If we are evaluating the ood fold, we use drug ids and not the one-hot encoded molecules
                y_adv = observation['smile_id'].to(device)
                y_true_ds.append(y_adv.item())
        

        if not adversarial:
            res = model.evaluate(X)
            out, z, ae_loss = res.values()
        else:
            # Get evaluation results
            drug_id = observation["smile_id"].to(device)
            res = model.evaluate(X, drug_id=drug_id)
            out, out_basal, z_basal, z, y_hat, ae_loss, _ = res.values()
            rmse_basal_full += metrics.compute_batch_rmse(out, out_basal)
            # Collect the labels 
            if not ood:
                y_hat_ds.append(torch.argmax(y_hat, dim=1).item())
            
            # Only at the end of training we store the latent vectors for analysis 
            if end:
                z_basal_ds.append(z_basal)
                z_ds.append(z)
        
        val_loss += ae_loss['total_loss'].item()
        if variational:
            val_recon_loss += ae_loss['reconstruction_loss'].item()
            val_kl_loss += ae_loss['KLD'].item()
        
        # Perform optimizer step depending on the iteration
        metrics.update_rmse(X, out)

    if not ood and adversarial:
        if X.shape[0]>1:
            y_true_ds = torch.cat(y_true_ds, dim=0).to('cpu').numpy()
            y_hat_ds = torch.cat(y_hat_ds, dim=0).to('cpu').numpy()
        metrics.compute_classification_report(y_true_ds, y_hat_ds)
        
    # Print loss results
    losses["loss"] = val_loss/len(dataset_loader)
    if variational:
        losses["avg_validation_recon_loss"] = val_recon_loss/len(dataset_loader)
        metrics.update_bpd(losses["avg_validation_recon_loss"])
        losses["avg_validation_kld_loss"] = val_kl_loss/len(dataset_loader)
    else:
        metrics.update_bpd(losses["loss"])

    metrics.metrics['rmse'] /= len(dataset_loader)
    if adversarial:
        metrics.metrics['rmse_basal_full'] = rmse_basal_full/len(dataset_loader)

    # Disentanglement and clustering evaluated only at the end
    if end:
        z_ds = torch.cat(z_ds, dim=0)
        z_basal_ds = torch.cat(z_basal_ds, dim=0)
        
        # Disentanglement score before and after drug addition 
        disentanglement_score_basal = compute_disentanglement_score(z_basal_ds, y_true_ds, device, predict_n_cells, linear=True)
        disentanglement_score_z = compute_disentanglement_score(z_ds, y_true_ds, device, predict_n_cells, linear=True)
        metrics.metrics["disentanglement_score_basal"] = disentanglement_score_basal
        metrics.metrics["disentanglement_score_z"] = disentanglement_score_z
        metrics.metrics["difference_disentanglement"] = disentanglement_score_basal - disentanglement_score_z

        z_ds = z_ds.to('cpu').numpy()
        z_basal_ds = z_basal_ds.to('cpu').numpy()

        if not (predict_n_cells and ood):
            # Silhouette score before and after drug addition 
            silhouette_score_basal = compute_silhouette_coefficient(z_basal_ds, y_true_ds)
            silhouette_score_z = compute_silhouette_coefficient(z_ds, y_true_ds)
            metrics.metrics["silhouette_score_basal"] = silhouette_score_basal
            metrics.metrics["silhouette_score_z"] = silhouette_score_z
            metrics.metrics["difference_silhouette"] = silhouette_score_z - silhouette_score_basal
        
    print(f'Average validation loss: {losses["loss"]}')
    if variational:
        print(f'Average validation reconstruction loss: {losses["avg_validation_recon_loss"]}')
        print(f'Average kld reconstruction loss: {losses["avg_validation_kld_loss"]}')

    metrics.print_metrics()
    return losses, metrics.metrics


def compute_disentanglement_score(Z, y, device, predict_n_cells, linear=True):
    """Train a classifier that evaluates the disentanglement of the latents space from the information
    on the drug

    Args:
        Z (torch.tensor): Latent representation of the validation set under the model 
        y (torch.tensor): The label assigned to each observation
        predict_n_cells (bool): Whether the task is to predict the drug or the number of cells

    Returns:
        dict: dictionary with the results
    """
    print('Training discriminator network on drug latent space')
    # Normalize the latent descriptors 
    mean = Z.mean(dim=0, keepdim=True)
    stddev = Z.std(0, unbiased=False, keepdim=True)
    normalized_basal = (Z - mean) / stddev

    # Collect labels for a classifier
    unique_labels = set(y)
    label_to_idx = {labels: idx for idx, labels in enumerate(unique_labels)}
    labels_tensor = torch.tensor(
        [label_to_idx[label] for label in y], dtype=torch.float if predict_n_cells else torch.long, device="cuda"
    )
    assert normalized_basal.size(0) == len(labels_tensor), f'Z of length {normalized_basal.size(0)}, y of length {len(labels_tensor)}'
    
    if not predict_n_cells:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(normalized_basal, labels_tensor)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    # 2 non-linear layers of size <input_dimension>
    # followed by a linear layer.
    classifier_depth = 2 if not linear else 0
    disentanglement_classifier = MLP(
        [normalized_basal.size(1)]
        + [normalized_basal.size(1) for _ in range(classifier_depth)]
        + [len(unique_labels) if not predict_n_cells else 1]
    ).to(device)
    optimizer = torch.optim.Adam(disentanglement_classifier.parameters(), lr=1e-2)

    for epoch in tqdm(range(100)):
        for X, y in data_loader:
            pred = disentanglement_classifier(X)
            if not predict_n_cells:
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
    