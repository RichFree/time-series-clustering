# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# from pyts.image import GramianAngularField
from pyts.image import GramianAngularField, MarkovTransitionField
from tqdm import tqdm
from PIL import Image

# %%
def normalize_global(df: pd.DataFrame, global_min: float, global_max: float):
    df_normalize = ((df - global_min) / (global_max - global_min))
    return df_normalize



def list_directories(path: str):
    entries = os.listdir(path)
    directories = [ entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
    return directories


def check_files(dataname: str, root_path: str, directories: list[str]):
    assert(dataname in directories)
    selected_data = dataname
    print("selected data: ", selected_data)
    train_datapath = root_path + f'/{selected_data}/{selected_data}_TRAIN.tsv'
    test_datapath = root_path + f'/{selected_data}/{selected_data}_TEST.tsv'
    assert(os.path.exists(train_datapath))
    assert(os.path.exists(test_datapath))


# helper for setup_dirs()
def safe_make_dir(new_dir_path: str):
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
        print(f"Directory '{new_dir_path}' created")
    else:
        print(f"Directory '{new_dir_path}' already exists")

def setup_dirs(dataname: str, unique_labels: list):
    export_path = 'data_segments'
    new_dir_path = export_path + f'/{dataname}'

    for type in ['train', 'test']:
        for cls in unique_labels:
            dir_path = new_dir_path + f'/{type}/{cls}'
            safe_make_dir(dir_path)

def import_data(root_path: str, dataname: str, type: str):
    datapath = root_path + f'/{dataname}/{dataname}_{type.upper()}.tsv'
    df = pd.read_csv(datapath, sep='\t', header=None).T
    labels = df.loc[0,:].to_list()
    labels = [int(value) for value in labels]
    df= df.loc[1:,:]
    return df, labels


# def save_combined_image(gaf_image, mtf_image, file_path):
#     # Stack GAF and MTF images along the channel dimension
#     combined_image = np.stack([gaf_image, mtf_image], axis=2)
#     # Convert to uint8 format (if necessary)
#     combined_image = (combined_image * 255).astype(np.uint8)
#     # Convert numpy array to Image
#     combined_image_pil = Image.fromarray(combined_image)
#     # Save the image
#     combined_image_pil.save(file_path)


def generate_gaf_for_train(df: pd.DataFrame, labels: list[int], type: str, dataname: str):
    # normalize by whole dataset
    global_min = np.min(df)
    global_max = np.max(df)

    num_data = len(df.columns)
    for img_num in tqdm(range(0, num_data)):
        # normalize per time series
        # global_min = np.min(df[img_num])
        # global_max = np.max(df[img_num])

        data = normalize_global(df[img_num], global_min, global_max)
        data = data.to_numpy().astype(np.float32)

        if (type == "test"):
            # get path attributes
            label = labels[img_num]
            dir_path =  f"data_segments/{dataname}" + f'/test/{label}' 
            assert(os.path.exists(dir_path))
            file_name = f'img_{type}_{img_num}'
            file_path = dir_path + f'/{file_name}' 
            np.savez_compressed(file_path, data=data)

        if (type == "train"):
            label = labels[img_num]
            dir_path =  f"data_segments/{dataname}" + f'/train/{label}'
            assert(os.path.exists(dir_path))
            file_name = f'img_{type}_{img_num}'
            file_path = dir_path + f'/{file_name}' 
            np.savez_compressed(file_path, data=data)

# %%
# arguments
root_path = './UCRArchive_2018'


data_list = [
    'Beef',
    'DistalPhalanxOutlineAgeGroup',
    'ECG200',
    'ECGFiveDays',
    'Meat',
    'MoteStrain',
    'OSULeaf',
    'Plane',
    'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxTW']

for dataname in data_list:

    df, labels = import_data(root_path, dataname, type)
    image_size = len(df)
    unique_labels = list(set(labels))

    # get directories
    directories = list_directories(root_path)
    # check that source data files exist
    check_files(dataname, root_path, directories)
    # create new directories based on labels
    setup_dirs(dataname, unique_labels)

    for type in ["test", "train"]:
        df, labels = import_data(root_path, dataname, type)
        generate_gaf_for_train(df, labels, type, dataname)

