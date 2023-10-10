import numpy as np
import torch
from tqdm import trange


def get_neighs(student_model, device, proxy_unused_dataset, db_dataset, db_dataloader, k,
               train_images, train_labels, train_soft_labels, estimations, ground_truth,
               use_soft_labels, distance, use_og_labels, unused_classes=None, student_name=None, activation=None):
    """
    Method used to get the labels according the neighbours
    """
    num_classes = student_model.num_classes

    if use_soft_labels and distance == 'euclidean':
        student_model.eval()
        with torch.no_grad():
            for i in trange(len(proxy_unused_dataset), desc='Assigning labels on extra data', leave=False):
                batch = proxy_unused_dataset[i]
                image = batch['image'].to(device=device)
                image_path = batch['image_path']
                label = batch['hard_label']
                soft_label = batch['soft_label']

                
                if student_name == "resnet_food":
                    _ = student_model(image.unsqueeze(dim=0))
                    latent_fm = activation['latent_space']
                else:
                    _,latent_fm = student_model(image.unsqueeze(dim=0))

                # Calculate distances
                distances = torch.zeros(size=(len(db_dataset),), device=device)
                for j, (img_db, softl_db) in enumerate(db_dataloader):
                    img_db = img_db.to(device=device)
                    
                    dist = (img_db - latent_fm).pow(2).sum(dim=(-1,-2,-3)).sqrt()
                    
                    distances[j*128:(j+1)*128] = dist

                smallest_d, smallest_d_indices = torch.topk(distances, k=k, largest=False)
                
                closest_slabels = torch.vstack([db_dataset[l][1] for l in smallest_d_indices]).to(device=device)

                weighted_slabels = smallest_d.unsqueeze(1) * closest_slabels
                estimated_soft_label = weighted_slabels.sum(0) / smallest_d.sum()
                estimated_hard_label = estimated_soft_label.argmax().item()
            
                if not use_og_labels or (use_og_labels and estimated_hard_label == unused_classes[i]):
                    train_images.append(image_path)
                    train_labels.append(estimated_hard_label)
                    train_soft_labels.append(estimated_soft_label.tolist())

                    estimations.append(estimated_hard_label)
                    ground_truth.append(label)
    elif use_soft_labels and distance == 'cosine':
        student_model.eval()
        with torch.no_grad():
            for i in trange(len(proxy_unused_dataset), desc='Assigning labels on extra data', leave=False):
                # image, label, image_path, soft_label = proxy_unused_dataset[i]
                # image = image.to(device=device)
                batch = proxy_unused_dataset[i]
                image = batch['image'].to(device=device)
                image_path = batch['image_path']
                label = batch['hard_label']
                soft_label = batch['soft_label']

                if student_name == "resnet_food":
                    _ = student_model(image.unsqueeze(dim=0))
                    latent_fm = activation['latent_space']
                else:
                    _,latent_fm = student_model(image.unsqueeze(dim=0))

                norm_unkwn = torch.sqrt(torch.sum(torch.square(latent_fm), dim=(-1,-2,-3), keepdim=True))

                # Calculate distances
                distances = torch.zeros(size=(len(db_dataset),), device=device)
                for j, (img_db, softl_db) in enumerate(db_dataloader):
                    img_db = img_db.to(device=device)
                    
                    norm_db = torch.sqrt(torch.sum(torch.square(img_db), dim=(-1,-2,-3), keepdim=True))

                    dot_prod = torch.tensordot(img_db, latent_fm, dims=[[-1,-2,-3],[-1,-2,-3]])
                    denominator = (norm_db * norm_unkwn).squeeze(dim=-1).squeeze(dim=-1)
                    cosine_similarities = (dot_prod / denominator).squeeze(dim=-1)
                    
                    distances[j*128:(j+1)*128] = cosine_similarities

                smallest_d, smallest_d_indices = torch.topk(distances, k=k, largest=True)
                
                closest_slabels = torch.vstack([db_dataset[l][1] for l in smallest_d_indices]).to(device=device)
        
                weighted_slabels = smallest_d.unsqueeze(1) * closest_slabels
                estimated_soft_label = weighted_slabels.sum(0) / smallest_d.sum()
                estimated_hard_label = estimated_soft_label.argmax().item()
                
                if not use_og_labels or (use_og_labels and estimated_hard_label == unused_classes[i]):
                    train_images.append(image_path)
                    train_labels.append(estimated_hard_label)
                    train_soft_labels.append(estimated_soft_label.tolist())

                    estimations.append(estimated_hard_label)
                    ground_truth.append(label)
    elif not use_soft_labels and distance == 'euclidean':
        student_model.eval()
        with torch.no_grad():
            for i in trange(len(proxy_unused_dataset), desc='Assigning labels on extra data', leave=False):
                # image, label, image_path, soft_label = proxy_unused_dataset[i]
                # image = image.to(device=device)
                batch = proxy_unused_dataset[i]
                image = batch['image'].to(device=device)
                image_path = batch['image_path']
                label = batch['hard_label']
                soft_label = batch['soft_label']

                if student_name == "resnet_food":
                    _ = student_model(image.unsqueeze(dim=0))
                    latent_fm = activation['latent_space']
                else:
                    _,latent_fm = student_model(image.unsqueeze(dim=0))

                # Calculate distances
                distances = torch.zeros(size=(len(db_dataset),), device=device)
                for j, (img_db, softl_db) in enumerate(db_dataloader):
                    img_db = img_db.to(device=device)
                    
                    dist = (img_db - latent_fm).pow(2).sum(dim=(-1,-2,-3)).sqrt()
                    
                    distances[j*128:(j+1)*128] = dist

                smallest_d, smallest_d_indices = torch.topk(distances, k=k, largest=False)
                closest_labels = torch.Tensor([db_dataset[l][1] for l in smallest_d_indices]).to(dtype=torch.int32)

                counts = torch.bincount(closest_labels, minlength=num_classes)
                max_occ = (counts == counts.max()).nonzero()

                if len(max_occ) == 1:
                    # print(f'Easy. Label = {max_occ.item()}')
                    estimated_hard_label = max_occ.item()
                else:
                    min_dist = float('inf')
                    estimated_label = None
                    for idx,x in enumerate(max_occ):
                        dist = smallest_d[x==closest_labels].sum()
                        if dist < min_dist:
                            min_dist = dist
                            estimated_label = x

                    estimated_hard_label = estimated_label.item()
                
                if not use_og_labels or (use_og_labels and estimated_hard_label == unused_classes[i]):
                    train_images.append(image_path)
                    train_labels.append(estimated_hard_label)

                    estimated_soft_label = torch.zeros(size=(num_classes,), device=device)
                    estimated_soft_label[estimated_hard_label] = 1
                    train_soft_labels.append(estimated_soft_label.tolist())

                    estimations.append(estimated_hard_label)
                    ground_truth.append(label)
    elif not use_soft_labels and distance == 'cosine':
        student_model.eval()
        with torch.no_grad():
            for i in trange(len(proxy_unused_dataset), desc='Assigning labels on extra data', leave=False):
                batch = proxy_unused_dataset[i]
                image = batch['image'].to(device=device)
                image_path = batch['image_path']
                label = batch['hard_label']
                soft_label = batch['soft_label']

                if student_name == "resnet_food":
                    _ = student_model(image.unsqueeze(dim=0))
                    latent_fm = activation['latent_space']
                else:
                    _,latent_fm = student_model(image.unsqueeze(dim=0))

                norm_unkwn = torch.sqrt(torch.sum(torch.square(latent_fm), dim=(-1,-2,-3), keepdim=True))

                # Calculate distances
                distances = torch.zeros(size=(len(db_dataset),), device=device)
                for j, (img_db, softl_db) in enumerate(db_dataloader):
                    img_db = img_db.to(device=device)
                    
                    norm_db = torch.sqrt(torch.sum(torch.square(img_db), dim=(-1,-2,-3), keepdim=True))

                    dot_prod = torch.tensordot(img_db, latent_fm, dims=[[-1,-2,-3],[-1,-2,-3]])
                    denominator = (norm_db * norm_unkwn).squeeze(dim=-1).squeeze(dim=-1)
                    cosine_similarities = (dot_prod / denominator).squeeze(dim=-1)
                    
                    distances[j*128:(j+1)*128] = cosine_similarities

                smallest_d, smallest_d_indices = torch.topk(distances, k=k, largest=True)
                closest_labels = torch.Tensor([db_dataset[l][1] for l in smallest_d_indices]).to(dtype=torch.int32)

                counts = torch.bincount(closest_labels, minlength=num_classes)
                max_occ = (counts == counts.max()).nonzero()

                if len(max_occ) == 1:
                    # print(f'Easy. Label = {max_occ.item()}')
                    estimated_hard_label = max_occ.item()
                else:
                    min_dist = float('inf')
                    estimated_label = None
                    for idx,x in enumerate(max_occ):
                        dist = smallest_d[x==closest_labels].sum()
                        if dist < min_dist:
                            min_dist = dist
                            estimated_label = x

                    estimated_hard_label = estimated_label.item()
                
                if not use_og_labels or (use_og_labels and estimated_hard_label == unused_classes[i]):
                    train_images.append(image_path)
                    train_labels.append(estimated_hard_label)

                    estimated_soft_label = torch.zeros(size=(num_classes,), device=device)
                    estimated_soft_label[estimated_hard_label] = 1
                    train_soft_labels.append(estimated_soft_label.tolist())

                    estimations.append(estimated_hard_label)
                    ground_truth.append(label)
    else:
        raise Exception('Not a valid combination')