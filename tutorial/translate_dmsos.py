import sys 
import numpy as np
sys.path.insert(0, '/lustre/groups/ml01/workspace/alessandro.palma/imCPA_official/experiments/general_experiments/1.benchmark_scores')
from compute_scores import perform_transformation
from run_info import RUN_INFO
from initialize_nets import *
from torchvision.utils import save_image

path = '/lustre/groups/ml01/workspace/alessandro.palma/imCPA_official/experiments/general_experiments/3.image_sample/transformed_results'

def transform_test_dmsos(solver, model_name):
    for batch in solver.loader_test:
        dmso_id = solver.mol2id['DMSO']
        X = batch['X'].to(solver.device)
        y = batch['mol_one_hot'].argmax(1).to(solver.device)
        filenames = np.array(batch['file_names'])
        idx_dmso = [i for i in range(len(y)) if y[i].item()==dmso_id]
        if idx_dmso == []:
            continue
        X = X[idx_dmso]
        filenames = filenames[idx_dmso]
        result = [X.cpu()]
        for i in range(6):
            if i == dmso_id:
                continue
            # Setup the target transformation 
            y_trg = i*torch.ones_like(y).to(solver.device)[idx_dmso]
            y_trg_one_hot = torch.nn.functional.one_hot(y_trg, num_classes=6).to(solver.device)
            # Perform transf
            X_fake = perform_transformation(solver, 
                        model_name, 
                        X, 
                        y_trg_one_hot,
                        y_trg, 
                        solver.device).cpu()
            result.append(X_fake.cpu())
        
        results = torch.cat(result, dim=3)
        for j, obs in enumerate(results):
            save_image((obs.unsqueeze(0)+1.)/2., ospj(path, model_name, filenames[j]+'.jpg'))

for model in RUN_INFO['bbbc021']:
    if model == 'baseline':
        continue
    
    args_path = RUN_INFO['bbbc021'][model][0]

    # IMPA 
    if model == 'IMPA':
        ckpt_path = RUN_INFO['bbbc021'][model][1]
        best_iter = RUN_INFO['bbbc021'][model][2] 
        solver = initialize_impa(args_path, 
                                best_iter, 
                                ckpt_path)
        
    elif model == 'starGANv1':
        best_iter = RUN_INFO['bbbc021'][model][1]
        
        solver = initalize_stargan(args_path,
                                best_iter)
    
    elif model == 'DRIT++':
        ckpt_path = RUN_INFO['bbbc021'][model][1]
        solver = initialize_drit(args_path,
                                ckpt_path)

    elif model == 'DMIT':
        ckpt_path = RUN_INFO['bbbc021'][model][1]
        solver = initialize_dmit(args_path,
                                ckpt_path)
    
    transform_test_dmsos(solver, model)
