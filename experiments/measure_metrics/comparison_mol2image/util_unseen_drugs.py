import torch
import numpy as np

def t2i(tensor):
    """Transform a tensor to image

    Args:
        tensor (torch.Tensor): Tensor to transform to image

    Returns:
        torch.Tensor: Denormalized tensor for better image plotting 
    """
    tensor = tensor.detach().cpu().permute(1,2,0).numpy()
    return (tensor+1.)/2.  

def perform_unseen_transformations(solver, 
                                   emb_rdkit, 
                                   DMSOs, 
                                   append_orig=True):
    """
    Args:
        solver (Solver): The model solver
        emb_rdkit (pd.DataFrame): The data frame with the molecular embeddings
        DMSOs (torch.Tensor): Tensor with the DMSO images
        append_orig (bool, optional): Whether to save pre-. Defaults to True.

    Returns:
        tuple: tuple with original images and their transformations
    """
    
    ood_set = solver.args.ood_set
    # Create a dictionary supporting the transformations
    transf_ood = {}
    orig_dmso = []

    for ood in ood_set:
        transf_ood[ood] = []

    append_orig = True
    for ood_name in ood_set:
        print(f'Transforming to {ood_name}')
        z_emb = emb_rdkit.loc[ood_name]
        z_emb = torch.tensor(z_emb).unsqueeze(0).to(solver.device).to(torch.float32)
        for X in DMSOs[0]:
            X = X.unsqueeze(0).to(solver.device)
            z = torch.randn(1, 100, solver.args.z_dimension).mean(1).to(solver.device)
            z_emb_tot = torch.cat([z_emb, z], dim=1)

            # Perform transformations 
            s_trg = solver.nets.mapping_network(z_emb_tot)

            # Generations
            _, X_fake = solver.nets.generator(X, s_trg)

            # Transform to image 
            X_fake = t2i(X_fake.squeeze(0))
            X_true = t2i(X.squeeze(0))

            transf_ood[ood_name].append(X_fake[np.newaxis,:,:,:])
            if append_orig:
                orig_dmso.append(X_true[np.newaxis,:,:,:])

        # Concatenation 
        if append_orig:
            orig_dmso = np.concatenate(orig_dmso, axis=0)
        transf_ood[ood_name] = np.concatenate(transf_ood[ood_name], axis=0)

        # Append the originals only once
        append_orig=False
    return transf_ood, orig_dmso 