# Import libraries
import os
import random
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define global variables
user = "M"
if user == "R":
    PROJECT_DIR = "drive/MyDrive/BC_TTIC_Project/"
elif user == "M":
    PROJECT_DIR = "C:/Users/Mario/Google Drive/Documents/University of Chicago/Master/Classes/1st Year/TTIC 31220/BC_TTIC_Project/"      
DATA_DIR = PROJECT_DIR + "Data/BreaKHis_v1/histology_slides/breast/"

def GetAllData():
    data = pd.DataFrame()
    tumor_classes = ["benign", "malignant"]
    for tumor_class in tumor_classes:
        tumor_types = os.listdir(f"{DATA_DIR}{tumor_class}/SOB/")
        tumor_types.remove("desktop.ini")
        for tumor_type in tumor_types:
            year_slide_ids = os.listdir(f"{DATA_DIR}{tumor_class}/SOB/{tumor_type}/")
            year_slide_ids.remove("desktop.ini")
            for year_slide_id in year_slide_ids:
                magnifications = os.listdir(f"{DATA_DIR}{tumor_class}/SOB/{tumor_type}/{year_slide_id}/")
                magnifications.remove("desktop.ini")
                for magnification in magnifications:
                    filenames = os.listdir(f"{DATA_DIR}{tumor_class}/SOB/{tumor_type}/{year_slide_id}/{magnification}/")
                    filenames.remove("desktop.ini")
                    for filename in filenames:
                        filepath = f"{DATA_DIR}{tumor_class}/SOB/{tumor_type}/{year_slide_id}/{magnification}/{filename}"
                        row = {"TumorClass"     : tumor_class, 
                               "TumorType"      : tumor_type, 
                               "YearAndSlideID" : year_slide_id, 
                               "Magnification"  : magnification, 
                               "Filename"       : filename, 
                               "Filepath"       : filepath}
                        data = data.append(row, ignore_index = True)
    return data

def GetData(data, n = 1000, patch_size = 64, mag = "40X", train_frac = 2/3, valid_frac = 1/6):
    '''
    Inputs:
        data          : (pd.DataFrame) Dataframe of all data in format produced by GetAllData()
        n             : (int) number of patches returned
        patch_size    : (int) size of sampled patches, i.e. 32, 64, 128
        magnification : (str) one of "40X", "100X", "200X", "400X"
        train_frac    : (float) fraction of patches to be returned as training set
        valid_frac    : (float) fraction of patches to be returned as validate set
    Outputs:
        train      : (np.array of shape int(train_frac*n) x 3*patch_size**2 + 1) train_frac fraction of patches
        valid      : (np.array of shape int(valid_frac*n) x 3*patch_size**2 + 1) valid_frac fraction of patches
        test       : (np.array of shape int((1 - train_frac - valid_frac)*n) x 3*patch_size**2 + 1) remaining patches
        (Note)     : All arrays returned contain the data label (1 : malignant, 0 : benign) in the last column.
    '''
    
    # Select data with given magnification level and shuffle
    data = data[data["Magnification"] == mag].sample(frac = 1).reset_index(drop = True)
    
    df   = np.zeros((n, 3*patch_size**2 + 1))
    m    = len(data)
    for i in tqdm(range(n)):
        # Loop through all available images as many times as needed to obtain n patches.
        # At every iteration sample a random patch from current image.  
        idx   = i%m
        image = imageio.imread(data["Filepath"][idx])
        label = 1 if data["TumorClass"][idx] == "malignant" else 0
        
        # Sample a random patch from it:
        y_start = random.randint(0, image.shape[0] - patch_size)
        x_start = random.randint(0, image.shape[1] - patch_size)
        patch   = image[y_start : y_start + patch_size, 
                        x_start : x_start + patch_size, :]
        
        # Flatten and add to dataset:
        flat_patch = patch.flatten()
        df[i, :-1] = flat_patch
        df[i, -1]  = label
    
    train, valid, test = np.split(df, [int(train_frac*n), int((train_frac + valid_frac)*n) + 1, ])

    return train.astype(int), valid.astype(int), test.astype(int)

def DisplayPatch(flat_patch):
    '''
        Inputs:
            patch : (np.array of shape 1 x 3*patch_size**2)
    '''
    patch_size = int((len(flat_patch)/3)**(0.5))
    patch = flat_patch.reshape(patch_size, patch_size, 3)
    plt.imshow(patch)
    plt.axis("off")
    
    return None

# Generate all the data:
all_data = GetAllData()
# Double the number of benign observations to reach 48% benign to 52% malignant
all_data = all_data.append(all_data[all_data["TumorClass"] == "benign"]) 

patch_sizes = [32, 64, 128]
mags        = ["40X", "100X", "200X", "400X"]
for patch_size in patch_sizes:
    for mag in mags:
        print(f"Using patch size {patch_size} and magnification {mag}.")
        train, valid, test = GetData(all_data, n = 30000, patch_size = patch_size, mag = mag)
        with open(f"../PythonData/{patch_size}_{mag}.npy", "wb") as f:
            np.save(f, train)
            np.save(f, valid)
            np.save(f, test)
print("Done!")

'''
# Sample code to retrieve data:
with open(f"../PythonData/128_40X.npy", "rb") as f:
    train = np.load(f)
    valid = np.load(f)
    test  = np.load(f)
    
    train_x = train[:, :-1]
    train_y = train[:, -1]
    valid_x = valid[:, :-1]
    valid_y = valid[:, -1]
    test_x  = test[:, :-1]
    test_y  = test[:, -1]
'''