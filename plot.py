# %%
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import glob, os
from pyts.image import GramianAngularField, MarkovTransitionField
import plotly.graph_objs as go

# import utils
from model.utils import get_random_file
from model.vicreg import SelfSupervisedMethod
from model.model_params import VICRegParams

from attr import evolve
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, normalized_mutual_info_score
from model.dataload import normalize_params

import warnings
import warnings
warnings.filterwarnings("ignore")

# %%
# arguments
# set your data root folder
data_root = "data/data_segments/"
# SELECT variable refers to which dataset to plot
# possible value: 0-9
SELECT = 0

# data parameters
data_params = list()

# 0 Beef
data_params.append({
    'batch_size':30,
    'num_clusters': 5,
    'train_path': data_root + "Beef/train",
    'test_path' : data_root + "Beef/test",
    'checkpoint': 'checkpoint_beef'
    })
# 1 dist.phal.outl.agegroup
data_params.append({
    'batch_size':139,
    'num_clusters': 3,
    'train_path' : data_root + "DistalPhalanxOutlineAgeGroup/train",
    'test_path' :  data_root + "DistalPhalanxOutlineAgeGroup/test",
    'checkpoint': 'checkpoint_dist_agegroup'
    })
# 2 ECG200
data_params.append({
    'batch_size':100,
    'num_clusters': 2,
    'train_path' : data_root + "ECG200/train",
    'test_path' :  data_root + "ECG200/test",
    'checkpoint': 'checkpoint_ecg200'
    })
# 3 ECGFiveDays
data_params.append({
    'batch_size':23,
    'num_clusters': 2,
    'train_path' : data_root + "ECGFiveDays/train",
    'test_path' : data_root + "ECGFiveDays/test",
    'checkpoint': 'checkpoint_ecg5days'
    })
# 4 Meat
data_params.append({
    'batch_size':60,
    'num_clusters': 3,
    'train_path' : data_root + "Meat/train",
    'test_path' : data_root + "Meat/test",
    'checkpoint': 'checkpoint_meat'
    })
# 5 mote strain
data_params.append({
    'batch_size': 20,
    'num_clusters': 2,
    'train_path' : data_root + "MoteStrain/train",
    'test_path' : data_root + "MoteStrain/test",
    'checkpoint': 'checkpoint_motestrain'
    })
# 6 osuleaf
data_params.append({
    'resize': 428, 
    'batch_size': 64, # 200
    'num_clusters': 6,
    'train_path' : data_root + "OSULeaf/train",
    'test_path' : data_root + "OSULeaf/test",
    'checkpoint': 'checkpoint_osuleaf'
    })
# 7 plane
data_params.append({
    'batch_size': 105,
    'num_clusters': 7,
    'train_path' : data_root + "Plane/train",
    'test_path' : data_root + "Plane/test",
    'checkpoint': 'checkpoint_plane'
    })
# 8 proximal_agegroup
data_params.append({
    'batch_size': 205,
    'num_clusters': 3,
    'train_path' : data_root + "ProximalPhalanxOutlineAgeGroup/train",
    'test_path' : data_root + "ProximalPhalanxOutlineAgeGroup/test",
    'checkpoint': 'checkpoint_prox_agegroup'
    })
# 9 proximal_tw
data_params.append({
    'batch_size': 100, # 400
    'num_clusters': 6,
    'train_path' : data_root + "ProximalPhalanxTW/train",
    'test_path' : data_root + "ProximalPhalanxTW/test",
    'checkpoint': 'checkpoint_prox_tw'
    })

# %%
def list_directories(path):
    entries = os.listdir(path)
    directories = [ entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
    return directories

def normalize(data):
    data_normalize = ((data - data.min()) / (data.max() - data.min()))
    return data_normalize



def load_image(filepath, gaf_function, mtf_function):
    data = np.load(filepath)['data'].astype(np.float32)
    data = data.reshape((1,-1))
    gaf_image = normalize(gaf_function.transform(data)[0])
    # mtf_image = gaf_image
    mtf_image = normalize(mtf_function.transform(data)[0])
    image = torch.from_numpy(np.stack([gaf_image, mtf_image], axis=0).astype(np.float32))
    # image = (np.stack([gaf_image, mtf_image], axis=0) * 255).astype(np.uint8)
    # image = torch.from_numpy((image/255.0).astype(np.float32))

    return image



def inference(method, classes, path, transform, gaf_function, mtf_function):
    batch_size = 32
    image_tensors = []
    result = []
    labels = []

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    device = torch.device('cuda')
    method.model.to(device)
    method.projection_model.to(device)


    for key in classes:
        image_dir = path + '/'  + key 
        for img_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, img_name)
            # image = Image.open(image_path)
            # image = image.convert('RGB')
            image = load_image(image_path, gaf_function, mtf_function)

            # Preprocess the image
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            image_tensors.append(input_tensor)

            # perform batching
            if len(image_tensors) == batch_size:
                batch_tensor = torch.cat(image_tensors).to(device)
                # Use the pre-trained model to extract features
                with torch.no_grad():
                    emb = method.model(batch_tensor)
                    projection = method.projection_model(emb)
                    # projection = method.model(input_tensor)
                result.extend(projection.cpu())
                # reset back to 0
                image_tensors = []


            labels.append(int(key))

    if len(image_tensors) > 0:
        batch_tensor = torch.cat(image_tensors).to(device)
        # Use the pre-trained model to extract features
        with torch.no_grad():
            emb = method.model(batch_tensor)
            projection = method.projection_model(emb)
            # projection = method.model(input_tensor)
        result.extend(projection.cpu())

    return result, labels


# %%

run_num = 0
config = evolve(VICRegParams(), 
                encoder_arch = "resnet18") # resnet18, resnet34, resnet50

method = SelfSupervisedMethod
# Initialize your ResNet model
checkpoint = data_params[SELECT]['checkpoint']
path = f'checkpoints/{checkpoint}/last_{run_num}.ckpt'
method = method.load_from_checkpoint(path)
# Set the model to evaluation mode
method.eval()

# prepare functions
train_path = data_params[SELECT]['train_path']
random_file = get_random_file(train_path)
img_size = len(np.load(random_file)['data'])
gaf_function = GramianAngularField(image_size=img_size, 
                                method="difference",
                                sample_range=(0,1))
mtf_function = MarkovTransitionField(image_size=img_size,
                                    n_bins = 5)




# Define transform
path = data_params[SELECT]['test_path']
normalize_means, normalize_stds = normalize_params(path).calculate_mean_std()
transform = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize(mean=normalize_means, std=normalize_stds),
])


# get all the classes
classes = list_directories(path)

result, labels = inference(method, classes, path, transform, gaf_function, mtf_function)
torch.cuda.empty_cache()


# %%
# cluster analysis

# perform k_means based on known number of clusters
kmeans = KMeans(n_clusters=data_params[SELECT]['num_clusters'], random_state=42, n_init=10)
clusters = kmeans.fit_predict(data)


data = np.array(result)
print(data.shape)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

cluster_id_list = list(set(clusters.tolist()))

plots = []
for cluster_id in cluster_id_list:
    print('-'*10)
    print("cluster_id: ", cluster_id)
    print("number points: ", sum(clusters==cluster_id))
    selected_data = reduced_data[clusters==cluster_id]
    selected_labels = np.array(labels)[clusters==cluster_id]
    plots.append(go.Scatter(x=selected_data[:, 0], y=selected_data[:, 1], mode='markers', text=selected_labels))


fig = go.Figure(data=plots)
# fig.update_traces(visible='legendonly')
fig.show()



# %%
