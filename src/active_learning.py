import random
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets import CIFAR10, ProxyDataset

from src.utils import *


def active_learning_step(student_model, total_examples, label_mapper, device,
                         images_train_dataset, classes_train_dataset, labels_train_dataset, soft_labels_train_dataset,
                         valid_images, valid_labels, valid_soft_labels):
    collected_images, collected_labels, collected_soft_labels = [], [], []

    if student_model.num_classes == 101 and total_examples > 800:
        total_examples = 800
    num_examples = total_examples // 2 if total_examples > 1 else 1

    # Define optimizer
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
    # Early stopping 
    early_stopping = EarlyStopping(student_model, tolerance=100, min_delta=0.01)
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Define the proxy dataset -> compute 
    proxy_transforms = T.Compose([
        # T.Resize((32,32)),
        T.Normalize((0.5,), (0.5,))
    ])

    for ith in range(min(total_examples,2)):
        proxy_dataset = ProxyDataset(images_train_dataset, classes_train_dataset, proxy_transforms, True)
        proxy_dataloader  = DataLoader(proxy_dataset, num_workers=2, batch_size=32)

        with torch.no_grad():           
            mean_latent = torch.zeros(size=(student_model.num_classes,student_model.latent_space_size), device=device)

            student_model.return_feature_domain = True
            fc2_images = [[] for _ in range(len(label_mapper))]

            for batch in tqdm(proxy_dataloader, leave=False):
                batch['image'] = batch['image'].to(device=device)
                batch['hard_label'] = batch['hard_label'].to(device=device)

                logits, fc2 = student_model(batch['image']) # shape -> (batch,128), pt resnet (512, 4, 4)
                fc2 = fc2.view(fc2.size(0), -1)
                
                # Compute mean
                batch_size = logits.size(0)
                num_classes = student_model.num_classes
                indices = torch.stack((
                    batch['hard_label'],
                    torch.arange(batch_size, device=batch['hard_label'].device)
                ))
                values = torch.ones_like(batch['hard_label'], dtype=torch.float)
                one_hot = torch.sparse_coo_tensor(indices, values, size=(num_classes, batch_size))

                sums_per_subset = torch.mm(one_hot, fc2)

                mean_latent = mean_latent + sums_per_subset

                # Add images latent space (i.e. fc2)
                for c in batch['hard_label'].unique():
                    fc2_images[c].extend(torch.split(fc2[batch['hard_label'] == c], split_size_or_sections=1))

            mean_latent = mean_latent / (len(proxy_dataset) / student_model.num_classes) # torch.Size([10, 128])

            # Get probabilities - cu cat cresti sigma, cu atat prob sunt mai apropiate i.e. sigma mai mic, valori mai sparse
            sigma = 600.0
            probabilities = torch.zeros(size=(len(fc2_images), len(fc2_images[0])))

            for c in range(len(fc2_images)):
                for i,fc2_image in enumerate(fc2_images[c]):
                    probabilities[c][i] = (fc2_image - mean_latent[c]).pow(2).sum().sqrt()
                    probabilities[c][i] = torch.exp(-probabilities[c][i] / sigma)

            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True) # torch.Size([10, 5429])
            cum_sum = probabilities.cumsum(dim=1) # torch.Size([10, 5120])

            intervals = {classes_train_dataset[0]: 0}
            for i in range(1,len(images_train_dataset)):
                if classes_train_dataset[i] != classes_train_dataset[i-1]:
                    intervals[classes_train_dataset[i]] = i
            # print(intervals)
            # Sample examples
            idxs_used = set()
            for c in range(len(cum_sum)):
                idxs = torch.searchsorted(cum_sum[c], torch.rand(num_examples))
                deduplicated_idxs = idxs.unique()

                while len(deduplicated_idxs) < num_examples:
                    temp_idxs = torch.searchsorted(cum_sum[c], torch.rand(num_examples))
                    idxs = torch.cat((idxs, temp_idxs))
                    deduplicated_idxs = idxs.unique()

                deduplicated_idxs = deduplicated_idxs[:num_examples] 

                collected_images.extend([images_train_dataset[intervals[c] + idx] for idx in deduplicated_idxs])
                collected_labels.extend([labels_train_dataset[intervals[c] + idx] for idx in deduplicated_idxs])
                collected_soft_labels.extend([soft_labels_train_dataset[intervals[c] + idx] for idx in deduplicated_idxs])

                idxs_used.update([(intervals[c] + idx).item() for idx in deduplicated_idxs])

        # Delete from images and labels
        idxs_unused = set(range(len(images_train_dataset))) - idxs_used
        images_train_dataset = [images_train_dataset[i] for i in idxs_unused]
        classes_train_dataset = [classes_train_dataset[i] for i in idxs_unused]
        labels_train_dataset = [labels_train_dataset[i] for i in idxs_unused]
        soft_labels_train_dataset = [soft_labels_train_dataset[i] for i in idxs_unused]

        if ith == 0:
            # Define the transformations
            train_transforms = T.Compose([
                # T.Resize((32,32)),
                # T.RandomCrop(224, padding=4),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize((0.5,), (0.5,))
            ])
            valid_transforms = T.Compose([
                # T.Resize((32,32)),
                T.Normalize((0.5,), (0.5,))
            ])

            # Define the proxy datasets
            proxy_train_dataset = ProxyDataset(collected_images, collected_labels, train_transforms, False, collected_soft_labels)
            proxy_valid_dataset = ProxyDataset(valid_images, valid_labels, valid_transforms, False, valid_soft_labels)

            # Define the proxy dataloaders
            train_dataloader = DataLoader(proxy_train_dataset, batch_size=32, shuffle=True)
            valid_dataloader = DataLoader(proxy_valid_dataset, batch_size=32)

            # Training loop
            early_stopping.reset()

            start_training(student_model, optimizer, steplr, criterion, early_stopping, device, train_dataloader, valid_dataloader, 5)
            
    return collected_images, collected_labels, collected_soft_labels, \
        images_train_dataset, labels_train_dataset, soft_labels_train_dataset, classes_train_dataset
