import os
from torch import tensor
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
            soft_label = tensor(soft_label)
        else:
            soft_label = 0
    
        return {
            "image": image,
            "image_path": self.images[idx],
            "hard_label": label,
            "soft_label": soft_label
        }
