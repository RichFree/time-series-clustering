# %%
# check
import glob
import random
import os
import numpy as np
import matplotlib.pyplot as plt

def get_random_file(root_dir):
    # Search for all files in the directory and subdirectories
    file_list = glob.glob(os.path.join(root_dir, '**', '*'), recursive=True)
    # Filter out directories from the list
    file_list = [f for f in file_list if os.path.isfile(f)]
    # If there are no files found, return None or raise an exception
    if not file_list:
        raise FileNotFoundError("No files found in the specified directory")
    # Select and return a random file path
    return random.choice(file_list)

# Example usage
root_directory = "data_segments/Beef/train"
random_file = get_random_file(root_directory)

data = np.load(random_file)['data']
len(data)
plt.plot(data)
# %%
