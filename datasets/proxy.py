import os
from torch.utils.data import Dataset
from torchvision.io import read_image


# Define dataset class
class ProxyDataset(Dataset):
    def __init__(self, images, labels, transform=None, return_path=False, soft_label=None):
        self.images = images
        self.labels = labels
        self.soft_label = soft_label
        
        self.transform = transform
        
        self.return_path = return_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = read_image(self.images[idx])
        image = image / 255.0
        
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
            
        if self.soft_label:
            soft_label = self.soft_label[idx]
        else:
            soft_label = 0
            
        if self.return_path:
            return image, label, self.images[idx], soft_label
        
        return image, label, soft_label