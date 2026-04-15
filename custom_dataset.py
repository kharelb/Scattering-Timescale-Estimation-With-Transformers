import pandas as pd
import numpy as np
import torch
from pathlib import Path
import re
from torch.utils.data import Dataset
from torchvision.transforms import v2

import random

class CustomDataset(Dataset):
    def __init__(self, image_dir_path, target_path, transform=None, normalize=True, add_noise_data=True):
        self.img_dir = Path(image_dir_path)
        self.transform = transform

        img_files = list(self.img_dir.glob("*.npy"))
        target_df = pd.read_csv(target_path)
        self.valid_tar_df = target_df.drop_duplicates(subset="event_id").dropna(subset="scat_time")

        self.valid_img_files = []
        tar_df_set = set(self.valid_tar_df["event_id"])
        
        for img_file in img_files:
            event_id = int(re.split('[_.]', img_file.name)[4])
            if event_id in tar_df_set:
                self.valid_img_files.append(img_file)

            else:
                continue

                                                                                
        self.normalize = normalize
        self.noise_data = add_noise_data

    def __len__(self):
        return len(self.valid_img_files)

    
    def __getitem__(self, idx):  
        img_path = self.valid_img_files[idx]
        event_id = int(re.split('[_.]', img_path.name)[4])
        target_value = self.valid_tar_df[self.valid_tar_df.event_id == event_id].scat_time.item()
        # error_value = self.valid_tar_df[self.valid_tar_df.event_id == event_id].scat_time_err.item()
        
        # Load dynamic spectrum
        spectrum = np.load(img_path)

        if self.noise_data and np.random.choice([0, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == 0:
            spectrum = np.random.normal(loc=0, scale=1, size=spectrum.shape)
            if np.random.choice([0,1]) == 0:
                max_channels_range = np.random.choice([1,2,3,4,5])
                start_channel = np.random.choice(np.arange(spectrum.shape[0] - 5))
                if np.random.choice([0,1]) == 0:
                    spectrum[start_channel:max_channels_range, :] = 0
                else:
                    spectrum[start_channel:max_channels_range, :] = np.random.choice([0.1, 0.4, 0.5, 0.6, 0.7])
            target_value = 0.0

        # Convert to torch tensors
        spectrum = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0)
        target_value = torch.tensor(target_value, dtype=torch.float32)
        
        if self.transform:
            spectrum = self.transform(spectrum)
            timeseries = spectrum.mean(dim=1, keepdim=True).squeeze(0)
        else:
            timeseries = spectrum.mean(dim=1, keepdim=True).squeeze(0)

        if self.normalize:
            spectrum = v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))(spectrum)
            timeseries = v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))(timeseries)

        return spectrum, timeseries, target_value, event_id

class CustomDatasetPCAtimeseries(Dataset):
    def __init__(self, image_dir_path, target_path, transform=None, normalize=True):
        self.img_dir = Path(image_dir_path)
        self.transform = transform

        self.img_files = list(self.img_dir.glob("*.npy"))
        self.target_df = pd.read_csv(target_path)
        self.normalize = normalize

    def __len__(self):
        return len(self.img_files)


    def pca_timeseries(self, dynamic_spectrum):
        mean = torch.mean(dynamic_spectrum, dim=1)
        mean_diff = dynamic_spectrum - mean.unsqueeze(1)
        cov_matrix = torch.cov(dynamic_spectrum)
        cov_matrix = cov_matrix + torch.eye(cov_matrix.shape[1]) * 1e-6
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        projected = mean_diff.T @ eigenvectors[:, -1:]
        return projected.transpose(0,1)
    
    def __getitem__(self, idx):
        error = True
        idx = idx
        while error:
            try:
                img_path = self.img_files[idx]
                event_id = int(re.split('[_.]', img_path.name)[4])
                target_value = self.target_df[self.target_df.event_id == event_id].scat_time.item()
    
                # Load dynamic spectrum
                spectrum = np.load(img_path)
                # timeseries = spectrum.mean(axis=0)
    
                # Convert to torch tensors
                spectrum = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0)
                # timeseries = torch.tensor(timeseries, dtype=torch.float32).unsqueeze(0)
                target_value = torch.tensor(target_value, dtype=torch.float32)
                error = False
    
            except Exception as e:
                # spectrum = np.random.normal(0, random.choice([1, 6]), (256, 162))
                # # timeseries = spectrum.mean(axis=0)
                # # Convert to torch tensors
                # spectrum = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0)
                # # timeseries = torch.tensor(timeseries, dtype=torch.float32).unsqueeze(0)
                # target_value = torch.tensor(0, dtype=torch.float32)
                error = True
                idx = idx + np.random.choice([-1, -2, -3, -4, -5, 1, 2, 3, 4, 5])

                

        
        if self.transform:
            spectrum = self.transform(spectrum)
            # timeseries = self.transform(timeseries)
            timeseries = self.pca_timeseries(spectrum.squeeze(0))
        else:
            timeseries = self.pca_timeseries(spectrum.squeeze(0))
        

        if self.normalize:
            spectrum = v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))(spectrum)
            timeseries = v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))(timeseries)

        return spectrum, timeseries, target_value


class CustomDatasetQuantileTimeseries(Dataset):
    def __init__(self, image_dir_path, target_path, transform=None, normalize=True, quantile=0.25):
        self.img_dir = Path(image_dir_path)
        self.transform = transform

        self.img_files = list(self.img_dir.glob("*.npy"))
        self.target_df = pd.read_csv(target_path)
        self.normalize = normalize
        self.quantile = quantile

    def __len__(self):
        return len(self.img_files)


    def quantile_timeseries(self, dynamic_spectrum):
        timeseries = torch.quantile(dynamic_spectrum, q=self.quantile, dim=0)
        return timeseries
    
    def __getitem__(self, idx):
        error = True
        idx = idx
        while error:
            try:
                img_path = self.img_files[idx]
                event_id = int(re.split('[_.]', img_path.name)[4])
                target_value = self.target_df[self.target_df.event_id == event_id].scat_time.item()
    
                # Load dynamic spectrum
                spectrum = np.load(img_path)
                # timeseries = spectrum.mean(axis=0)
    
                # Convert to torch tensors
                spectrum = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0)
                # timeseries = torch.tensor(timeseries, dtype=torch.float32).unsqueeze(0)
                target_value = torch.tensor(target_value, dtype=torch.float32)
                error = False
    
            except Exception as e:
                # spectrum = np.random.normal(0, random.choice([1, 6]), (256, 162))
                # # timeseries = spectrum.mean(axis=0)
                # # Convert to torch tensors
                # spectrum = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0)
                # # timeseries = torch.tensor(timeseries, dtype=torch.float32).unsqueeze(0)
                # target_value = torch.tensor(0, dtype=torch.float32)
                error = True
                idx = idx + np.random.choice([-1, -2, -3, -4, -5, 1, 2, 3, 4, 5])

                

        
        if self.transform:
            spectrum = self.transform(spectrum)
            # timeseries = self.transform(timeseries)
            timeseries = self.quantile_timeseries(spectrum.squeeze(0))
        else:
            timeseries = self.quantile_timeseries(spectrum.squeeze(0))
        

        if self.normalize:
            spectrum = v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))(spectrum)
            timeseries = v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))(timeseries)

        return spectrum, timeseries, target_value

