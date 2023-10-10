import os
import json
import numpy as np
import torch
import sys

from shutil import rmtree
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from torch.nn import Module
from torch.nn.functional import softmax
from torch.utils.data import Dataset
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

from predictors.alexnet import Alexnet, AlexnetFood
from predictors.half_alexnet import HalfAlexnet, HalfAlexnet2, HalfAlexnetFood
from predictors.resnet18 import ResNet18Custom


# Class for early stopping
class EarlyStopping():
    def __init__(self, model, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta

        self.counter = 0
        self.min_valid_loss = np.inf

        self.early_stop = False

        self.best_weights = None
        self.model = model

    def __call__(self, validation_loss):
        self.counter += 1

        if validation_loss < self.min_valid_loss - self.min_delta:
          self.min_valid_loss = validation_loss
          self.counter = 0

          # Save weights
          self.best_weights = self.model.state_dict()
          
        if self.counter >= self.tolerance:  
            self.early_stop = True
            if self.best_weights:
                self.model.load_state_dict(self.best_weights)

    def reset(self):
        self.counter = 0
        self.early_stop = False
    
    def restart(self, model):
        self.counter = 0
        self.min_valid_loss = np.inf

        self.early_stop = False

        self.best_weights = None
        self.model = model

class DBDataset(Dataset):
  def __init__(self, images_db, labels_db):
    self.images_db = images_db
    self.labels_db = labels_db

  def __len__(self):
    return len(self.images_db)

  def __getitem__(self, idx):
    image = self.images_db[idx]
    label = self.labels_db[idx]

    return image, label


def get_teacher(device) -> Alexnet:
    # Teacher
    teacher_model = Alexnet(name=None, n_outputs=10)

    ckpt_path = 'checkpoints/teacher_alexnet_for_cifar10_state_dict.pt'
    teacher_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    sys.exit(0)

    teacher_model.eval()
    teacher_model = teacher_model.to(device)

    return teacher_model

def get_teacher_food(device, teacher) -> AlexnetFood:
    # Teacher
    if teacher == "alexnet":
        teacher_model = AlexnetFood(name=None, n_outputs=101)

        ckpt_path = 'checkpoints/teacher_food101_alexnet.pt'
    elif teacher == "resnet50":
        teacher_model = resnet50(weights=None)
        teacher_model.fc = torch.nn.Linear(in_features=2048, out_features=101)

        ckpt_path = 'checkpoints/teacher_food101_resnet50.pt'
    else:
        raise ValueError("Wrong teacher name for Food101")
    teacher_model.load_state_dict(torch.load(ckpt_path, map_location=device))

    teacher_model.eval()
    teacher_model = teacher_model.to(device)

    return teacher_model

def get_student(student_name, device):
    if student_name == 'half_alexnet':
        student_model = HalfAlexnet(name=None, n_outputs=10)

        path_to_save = 'pretrained_student.pt'
    elif student_name == 'half_alexnet2':
        # For active learning
        student_model = HalfAlexnet2(name=None, n_outputs=10)

        path_to_save = 'pretrained_student.pt'
    elif student_name == 'resnet':
        student_model = ResNet18Custom(name=None, n_outputs=10)

        path_to_save = 'checkpoints/resnet18_pretrained_cifar10.pth'
    elif student_name == 'half_alexnet_food':
        student_model = HalfAlexnetFood(name=None, n_outputs=101)

        path_to_save = 'half_alex_net_food101.pt'
    elif student_name == "resnet_food":
        student_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        student_model.num_classes = 101        
        student_model.latent_space_size = 512 * 4 * 4

        num_ftrs = student_model.fc.in_features
        student_model.fc = torch.nn.Linear(num_ftrs, 101)
    else:
        raise Exception('No student model found')

    if student_name != "resnet_food":
        if torch.cuda.is_available():
            student_model.load_state_dict(torch.load(path_to_save))
        else:
            student_model.load_state_dict(torch.load(path_to_save, map_location ='cpu'))

    student_model.to(device)

    return student_model

def read_dataset(dataset_path, label_mapper, 
                 images_train, classes_trian, labels_train, soft_labels_train,
                 images_valid, labels_valid, soft_labels_valid) -> None:
    # Read train images
    with open(f'{dataset_path}/train_labels.json') as in_json:
        train_labels_dict = json.load(in_json)
    
    train_path = os.path.join(dataset_path,'train')
    for class_folder in sorted(os.listdir(train_path)):
        for image_name in os.listdir(os.path.join(train_path,class_folder)):
            image_path = os.path.join(train_path,class_folder,image_name)

            images_train.append(image_path)
            classes_trian.append(label_mapper[class_folder])
            labels_train.append(train_labels_dict[class_folder][image_path]['label'])
            soft_labels_train.append(train_labels_dict[class_folder][image_path]['soft_label'])

    # Read valid images
    with open(f'{dataset_path}/valid_labels.json') as in_json:
        valid_labels_dict = json.load(in_json)
    
    valid_path = os.path.join(dataset_path,'valid')
    for class_folder in sorted(os.listdir(valid_path)):
        for image_name in os.listdir(os.path.join(valid_path,class_folder)):
            image_path = os.path.join(valid_path,class_folder,image_name)

            images_valid.append(image_path)
            labels_valid.append(valid_labels_dict[class_folder][image_path]['label'])
            soft_labels_valid.append(valid_labels_dict[class_folder][image_path]['soft_label'])

def split_data(num_examples, random_seed, use_all_data, classes_train_dataset, images_train_dataset, labels_train_dataset, soft_labels_train_dataset, 
                num_classes=10, return_og_labels=False):
    train_size = num_examples*num_classes
    if num_classes == 101:
        if num_examples > 800:
            train_size = 800*101

    # Do a stratified split of the data
    train_images, unused_images, train_labels, unused_labels, train_soft_labels, unused_soft_labels, _, unused_classes = \
        train_test_split(images_train_dataset, labels_train_dataset, soft_labels_train_dataset, classes_train_dataset,
                         train_size=train_size, stratify=classes_train_dataset, random_state=random_seed)  

    if not use_all_data:
        if len(unused_images) > len(train_images):
            unused_images, _, unused_labels, _, unused_soft_labels, _, unused_classes, _ = \
                train_test_split(unused_images, unused_labels, unused_soft_labels, unused_classes, train_size=len(train_images), 
                                 stratify=unused_classes, random_state=random_seed)

    if return_og_labels:
        return train_images, train_labels, train_soft_labels, unused_images, unused_labels, unused_soft_labels, unused_classes

    return train_images, train_labels, train_soft_labels, unused_images, unused_labels, unused_soft_labels, unused_classes

def start_training(student_model, optimizer, steplr, criterion, early_stopping, device, train_dataloader, valid_dataloader, epochs, 
                   writer=None, use_mixup=False) -> None:
    student_model.return_feature_domain = False

    # Training the student
    for epoch in range(epochs):
        # Define progress bar
        loop = tqdm(train_dataloader, desc='Training', leave=False)
        
        # Training loop
        student_model.train()
        training_loss_epoch = []
        for batch in loop:
            optimizer.zero_grad()
            
            x = batch['image'].to(device=device)
            y = batch['hard_label'].to(device=device)
            soft_y = batch['soft_label'].to(device=device)

            mixup_prob = np.random.rand()
            if use_mixup and mixup_prob >= 0.5:
                """
                mixed_x, targets_a, targets_b, lam = mixup_data(x, soft_y)

                logits = student_model(mixed_x)

                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                """
                # generate mixed sample
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(x.size()[0]).cuda()
                target_a = soft_y
                target_b = soft_y[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

                # compute output
                output = student_model(x)
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            else:
            
                # Forward pass
                logits = student_model(x)

                loss = criterion(input=logits, target=soft_y)

            # Backward pass
            training_loss_epoch.append(loss.item())
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            # Update progress bar
            loop.set_description(f'Epoch {epoch+1}/{epochs}')
            loop.set_postfix(training_loss=loss.item())
        
        # Validation loop on proxy validation dataset
        student_model.eval()
        validation_loss_epoch = []  
        acc = 0
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc='Validating', leave=False):
                x = batch['image'].to(device=device)
                y = batch['hard_label'].to(device=device)
                
            
                logits = student_model(x)
                pred = softmax(logits, dim=1)
                
                confidence,y_hat = torch.max(pred, dim=1)
                
                loss = criterion(input=logits, target=y)
                validation_loss_epoch.append(loss.item())
                
                acc += torch.sum(y_hat==y).item()
            
        # loop.write(f'validation_loss on proxy = {sum(validation_loss_epoch)/len(validation_loss_epoch):.4f}')
        # loop.write(f'validation_accuracy on proxy = {100*acc/len(proxy_valid_dataset):.2f}%')

        # Save best model
        # valid_proxy_acc = acc/len(proxy_valid_dataset)
        # if valid_proxy_acc > best_accuracy:
        #     best_accuracy = valid_proxy_acc

        steplr.step()
        early_stopping(sum(validation_loss_epoch)/len(validation_loss_epoch))
        if early_stopping.early_stop:
            # print(f"We are at epoch {epoch}")
            # print(f'Model saved from epoch {best_epoch}.')
            break

def start_evaluation_true_gt(student_model, criterion, device, true_dataset, label_mapper_inv):
    true_dataloader = true_dataset.test_dataloader()
    acc_per_class = {k:[0,0,0] for k,v in label_mapper_inv.items()}

    student_model.eval()
    with torch.no_grad():
        test_loss = []
        acc = 0
        for x,y in true_dataloader:
            x = x.to(device=device)
            y = y.to(device=device)
        
            logits = student_model(x)
            pred = softmax(logits, dim=1)
            
            confidence,y_hat = torch.max(pred, dim=1)
            
            loss = criterion(input=logits, target=y)
            test_loss.append(loss.item())
            
            acc += torch.sum(y_hat==y)
            
            for i in range(len(y)):
                a = y[i].item()
                p = y_hat[i].item()
                
                
                if a == p:
                    acc_per_class[a][0] += 1 # correct predictions
                acc_per_class[a][1] += 1     # total number 
                acc_per_class[p][2] += 1     # predictions of class

    # for k,v in acc_per_class.items():
    #     print(f'Class {label_mapper_inv[k]}: correct_pred={v[0]}, actual={v[1]} => acc={v[0]*100/v[1]:.2f}%, total_pred={v[2]}')

    return 100*acc/len(true_dataset.test_dataset)

def start_evaluation_teacher_gt(teacher_model, student_model, criterion, device, true_dataset, label_mapper_inv):
    true_dataloader = true_dataset.test_dataloader()
    # acc_per_class = {k:[0,0,0] for k,v in label_mapper_inv.items()}

    student_model.eval()
    teacher_model.eval()
    with torch.no_grad():
        test_loss = []
        acc = 0
        for x,y in tqdm(true_dataloader, desc='Testing', leave=False):
            x = x.to(device=device)
            # y = y.to(device=device)

            teacher_pred = softmax(teacher_model(x), dim=1)
            _, y = torch.max(teacher_pred, dim=1)
        
            logits = student_model(x)
            pred = softmax(logits, dim=1)
            
            confidence,y_hat = torch.max(pred, dim=1)
            
            loss = criterion(input=logits, target=y)
            test_loss.append(loss.item())
            
            acc += torch.sum(y_hat==y).item()
            
            # for i in range(len(y)):
            #     a = y[i].item()
            #     p = y_hat[i].item()
                
                
            #     if a == p:
            #         acc_per_class[a][0] += 1
            #     acc_per_class[a][1] += 1
            #     acc_per_class[p][2] += 1
        
    # for k,v in acc_per_class.items():
    #     print(f'Class {label_mapper_inv[k]}: correct_pred={v[0]}, actual={v[1]} => acc={v[0]*100/v[1]:.2f}%, total_pred={v[2]}')

    return 100*acc/len(true_dataset.test_dataset)

def build_db(student_model, device, train_dataloader, return_soft, images_db, labels_db, student_name=None, activation=None):
    # Create the database
    student_model.eval()
    with torch.no_grad():
        for batch in train_dataloader:
            x = batch['image'].to(device=device)
            y = batch['hard_label']
            soft_y = batch['soft_label']

            if student_name == "resnet_food":
                _ = student_model(x)
                latent_fm = activation['latent_space']
            else:
                _,latent_fm = student_model(x)

            for i in range(latent_fm.shape[0]):
                images_db.append(latent_fm[i])
                if return_soft:
                    labels_db.append(soft_y[i])
                else:
                    labels_db.append(y[i])

def save_results(filename, num_examples, acc_before, acc_after):
    with open(f'experiments2/{filename}', 'a') as log_file:
        log_file.write(f'{num_examples},{acc_before},{acc_after}\n')

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
