import sys 
sys.path.insert(0, "home/icb/alessandro.palma/")  # this depends on the notebook depth and must be adapted per notebook
import numpy as np
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import pandas as pd 

# Name of the dataset 
dataset_name = 'cellpainting_100'
data_dir = '/home/icb/alessandro.palma/data/metadata_processed'

# Get the training and ood splits (containing the whole set of drugs in the dataset)
training_set = pd.read_csv(os.path.join(data_dir, 'datasplit-train.csv'))
ood_set = pd.read_csv(os.path.join(data_dir, 'datasplit-ood.csv'))



data_file = pd.read_csv(EMBEDDING_DIR / f'{dataset_name}.smiles')
smiles_list = smiles_df['smiles'].values