import os
from torch.utils.data import Dataset
from torchvision.io import read_image


# Define dataset class
class ProxyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        
        self.transform = transform

        gen_dataset_path = 'generated_dataset'
        self.proxy_imgs_path = os.path.join(gen_dataset_path,'images')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = read_image(f'{self.proxy_imgs_path}/{self.images[idx]}')
        image = image / 255.0
        
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, label
    