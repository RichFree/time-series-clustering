# for CustomImageDataset
import os
import glob
import random
import pandas as pd
import torch
from torchvision.io import read_image
from pathlib import Path
from PIL import Image
from pyts.image import GramianAngularField, MarkovTransitionField
# for transforms
from torchvision import transforms
import attr
# for CustomDataloader
from torch.utils.data import DataLoader
import numpy as np


# this recreates the ImageFolder function 
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, img_size=400):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.path_mapper(img_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.gaf_function = GramianAngularField(image_size=img_size, 
                                                method="difference",
                                                sample_range=(0,1))
        self.mtf_function = MarkovTransitionField(image_size=img_size, 
                                                  n_bins=5)


    def path_mapper(self, img_dir):
        df = pd.DataFrame()
        img_path = Path(img_dir)
        # grab all the label folders
        dirs = [f for f in img_path.iterdir() if f.is_dir()]
        for dir in dirs:
            for f in dir.iterdir():
                new_row = {'file': f.name, 'label': dir.name}
                # we have the file name, and then the label of the folder it belongs to
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        return df

    def __len__(self):
        return len(self.img_labels)

    def normalize(self, data):
        data_normalize = ((data - data.min()) / (data.max() - data.min()))
        return data_normalize


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 
                                self.img_labels.iloc[idx,1],
                                self.img_labels.iloc[idx,0])
        data = np.load(img_path)['data']
        
        data = data.reshape((1,-1))
        gaf_image = self.normalize(self.gaf_function.transform(data)[0])
        # mtf_image = gaf_image # to turn off mtf
        mtf_image = self.normalize(self.mtf_function.transform(data)[0])
        image = torch.from_numpy((np.stack([gaf_image, mtf_image], axis=0)).astype(np.float32))
        # assert image.dtype == torch.float32, "Tensor is not float32!"

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

@attr.s(auto_attribs=True)
class ImageTransforms:
    img_size: int = 400
    crop_size: tuple[int,int] = (224,224)
    normalize_means: list = [0.0, 0.0]
    normalize_stds:  list = [1.0, 1.0]

    def split_transform(self, img) -> torch.Tensor:
        transform = self.single_transform()
        return torch.stack((transform(img), transform(img)))

    def single_transform(self):
        transform_list = [
            # transforms.ToTensor(),
            transforms.RandomResizedCrop(self.crop_size, 
                                         scale=(0.3,0.7),
                                         antialias=False),
            # transforms.RandomCrop(size=(int(self.img_size * 0.4), int(self.img_size * 0.4)))
            transforms.Normalize(mean=self.normalize_means, std=self.normalize_stds)
        ]
        return transforms.Compose(transform_list)

class CustomDataloader():

    def __init__(self, 
                 img_dir: str,
                 img_size: int = 400,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 persistent_workers: bool = True,
                 shuffle: bool = True
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.shuffle = shuffle
        normalize_means, normalize_stds = normalize_params(img_dir).calculate_mean_std()
        self.image_transforms = ImageTransforms(img_size=img_size,
                                                crop_size=(224,224),
                                                normalize_means=normalize_means,
                                                normalize_stds=normalize_stds)
        self.dataset = CustomImageDataset(img_dir=img_dir,
                                          img_size=img_size,
                                          transform=self.image_transforms.split_transform)

    def get_dataloader(self):
        return DataLoader(self.dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers,
                          shuffle=self.shuffle)


class normalize_params():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_size = len(np.load(self.get_random_file())['data'])
        self.gaf_function = GramianAngularField(image_size=self.img_size, 
                                                method="difference",
                                                sample_range=(0,1))
        self.mtf_function = MarkovTransitionField(image_size=self.img_size,
                                                n_bins = 5)


    def get_random_file(self):
        # Search for all files in the directory and subdirectories
        file_list = glob.glob(os.path.join(self.root_dir, '**', '*'), recursive=True)
        # Filter out directories from the list
        file_list = [f for f in file_list if os.path.isfile(f)]
        # If there are no files found, return None or raise an exception
        if not file_list:
            raise FileNotFoundError("No files found in the specified directory")
        # Select and return a random file path
        return random.choice(file_list)

    def normalize(self, data):
        data_normalize = ((data - data.min()) / (data.max() - data.min()))
        return data_normalize


    def load_image(self, filepath):
        data = np.load(filepath)['data'].astype(np.float32)
        data = data.reshape((1,-1))
        gaf_image = self.gaf_function.transform(data)[0]
        # mtf_image = gaf_image
        mtf_image = self.mtf_function.transform(data)[0]
        image = (np.stack([gaf_image, mtf_image], axis=0)).astype(np.float32)
        return image

    def calculate_mean_std(self):
        # Initialize lists to store the sum and squared sum of pixel values
        mean_1, mean_2 = 0.0, 0.0
        std_1, std_2 = 0.0, 0.0
        num_pixels = 0
        image_dir = self.root_dir

        # Iterate through all images in the directory
        for dirpath, dirnames, filenames in os.walk(image_dir):
            for filename in filenames:
                # Full path of the file
                file_path = os.path.join(dirpath, filename)

                if os.path.isfile(file_path) and file_path.endswith(('npz')):
                    img_np = self.load_image(file_path)
                    # img_np = np.array(img) / 255.0  # Normalize to range [0, 1]
                    
                    num_pixels += img_np.shape[1] * img_np.shape[2]
                    
                    mean_1 += np.sum(img_np[0, :, :])
                    mean_2 += np.sum(img_np[1, :, :])
                    
                    std_1 += np.sum(img_np[0, :, :] ** 2)
                    std_2 += np.sum(img_np[1, :, :] ** 2)

        # Calculate mean
        mean_1 /= num_pixels
        mean_2 /= num_pixels

        # Calculate standard deviation
        std_1 = (std_1 / num_pixels - mean_1 ** 2) ** 0.5
        std_2 = (std_2 / num_pixels - mean_2 ** 2) ** 0.5

        return [mean_1, mean_2], [std_1, std_2]



