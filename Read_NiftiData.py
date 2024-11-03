import nibabel as nib
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

def read_nifti_file(nifti_file):
    nii_image = nib.load(nifti_file)
    nii_data = nii_image.get_fdata()
    return nii_data

# creates a custom Dataset class that loads all .nii.gz files from both 'health' and 'patient' folders
# assigning labels accordingly
# reates DataLoaders for both training and testing sets
class BrainMRIDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        
        for label, class_name in enumerate(['health', 'patient']):
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith('.nii.gz'):
                    self.samples.append((os.path.join(class_dir, file), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = read_nifti_file(file_path)
        data = torch.from_numpy(data).float().unsqueeze(0)  # Add channel dimension
        return data, label

# Create datasets
train_dataset = BrainMRIDataset('./Training')
test_dataset = BrainMRIDataset('./Testing')

# Create data loaders
batch_size = 2 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")

# Iterating through the data
for batch_data, batch_labels in train_loader:
    print(f"Batch shape: {batch_data.shape}, Labels: {batch_labels}")
    break

for batch_data, batch_labels in test_loader:
    print(f"Batch shape: {batch_data.shape}, Labels: {batch_labels}")
    break