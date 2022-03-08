import pandas as pd
import numpy as np
import cv2
import zipfile
import os
import matplotlib.pyplot as plt
import tqdm
from PIL import Image

data_dir = '/storage/groups/ml01/workspace/alessandro.palma/dmso'

# Fetch DMSO files 
dmso_ims = pd.read_csv('/home/icb/alessandro.palma/data/metadata/dmso/dmso_ims.csv')[['0','1']]
dmso_ims.columns = ['plate', 'well']

# Get unique plates
plates = np.unique(dmso_ims['plate'].to_numpy())
# Fix names of the stains 
stains = {'Hoechst':{}, 'ERSyto':{}, 'ERSytoBleed':{}, 'Ph_golgi':{}, 'Mito':{}}

# For the first plates (the one I have images for)
for plate in plates[:16]:  
    # Take the wells with dmso for a certain plate 
    df_plate = dmso_ims[dmso_ims['plate']==plate]
    plate_wells = np.array(df_plate['well'])
    # Foar all the stains
    for stain in stains:
        stains[stain][plate] = {}
        # Get the zip files with the plate and the stain  
        zip_path = os.path.join(data_dir, f"{plate}-{stain}.zip")
        with zipfile.ZipFile(zip_path, "r") as f:
            # Loop over all the files in the zip
            for name in f.namelist():
                if stain != 'Ph_golgi':
                    if len(name.split('_'))>1 and name.split('_')[1] in plate_wells:
                        well_name = name.split('_')[1]
                        view_name = name.split('_')[2]
                        # Derive a path name
                        if well_name not in stains[stain][plate]:
                            stains[stain][plate][well_name] = {}
                        
                        data = f.open(name)
                        stains[stain][plate][well_name][view_name] = np.array(Image.open(data))
                else:
                    if len(name.split('_'))>2 and name.split('_')[2] in plate_wells:
                        well_name = name.split('_')[2]
                        view_name = name.split('_')[3]
                        # Derive a path name
                        if well_name not in stains['Ph_golgi'][plate]:
                            stains['Ph_golgi'][plate][well_name] = {}

                        data = f.open(name)
                        stains['Ph_golgi'][plate][well_name][view_name] = np.array(Image.open(data))


# Save the results 
import pickle as pkl
with open('stains.pkl', 'wb') as f:
    pkl.dump(stains, f)


