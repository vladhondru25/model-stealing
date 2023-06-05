from torchvision import transforms
import torch
import torchvision


class FOOD101(object):
    def __init__(self, input_size = 224):
        self.n_classes = 10
        transform = transforms.Compose([
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (.5,), (.5,)
            ),
        ])

        self.train_dataset = torchvision.datasets.Food101(
            root='../data', 
            split='train', 
            download=True,
            transform=transform
        )
        self.test_dataset = torchvision.datasets.Food101(
            root='../data', 
            split='test', 
            download=True,
            transform=transform
        )

    def train_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(
            self.train_dataset,  
            batch_size = kwargs.get('batch_size', 32), 
            shuffle = True,
            num_workers=2, 
            pin_memory=True,
            drop_last = True
        )

    def test_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(
            self.test_dataset,  
            batch_size = kwargs.get('batch_size', 32), 
            num_workers=2, 
            pin_memory=True,
            drop_last = False
        )

