from tqdm import tqdm
import sklearn
from sklearn.metrics import f1_score
from sklearn import model_selection
import numpy as np
import torch
import sys

import sys
sys.path.insert(0, '../..')
# from compert.core.model import DisentanglementClassifier
from core.model import DisentanglementClassifier

def compute_disentanglement_score(X, y, return_misclass_report=False):
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
    y = np.array(y)

    # Split into training and test set 
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.10, stratify=y)  # Make the dataset balanced 

    y_train_tensor = torch.tensor(
        [label_to_idx[label] for label in y_train], dtype=torch.long, device="cuda")
    y_test_tensor = torch.tensor(
        [label_to_idx[label] for label in y_test], dtype=torch.long, device="cpu")

    # Create loader and dataset
    dataset = torch.utils.data.TensorDataset(X_train, y_train_tensor)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # initialize network and training hyperparameters 
    net = DisentanglementClassifier(X_train.shape[2], X_train.shape[1], 256, len(unique_classes)).to('cuda')
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
        