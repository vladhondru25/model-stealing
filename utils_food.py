import os
import numpy as np
import torch

from shutil import rmtree
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm
from torch.nn import Module
from torch.nn.functional import softmax
from torch.utils.data import Dataset

from predictors.alexnet import AlexnetFood
from predictors.half_alexnet import HalfAlexnetFood


# Class for early stopping
class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.min_valid_loss = np.inf
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss < self.min_valid_loss:
          self.min_valid_loss = validation_loss
          self.counter = 0
        elif validation_loss > self.min_valid_loss + self.min_delta:
          self.counter +=1
          
          if self.counter >= self.tolerance:  
              self.early_stop = True

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


def get_teacher(device) -> AlexnetFood:
    # Teacher
    teacher_model = AlexnetFood(name=None, n_outputs=101)

    ckpt_path = 'checkpoints/teacher_food101.pt'
    teacher_model.load_state_dict(torch.load(ckpt_path, map_location=device))

    teacher_model.eval()
    teacher_model = teacher_model.to(device)

    return teacher_model

def get_student(device) -> Module:
    student_model = HalfAlexnetFood(name=None, n_outputs=101)

    path_to_save = 'half_alex_net_food101.pt'
    if torch.cuda.is_available():
        student_model.load_state_dict(torch.load(path_to_save))
    else:
        student_model.load_state_dict(torch.load(path_to_save, map_location ='cpu'))

    student_model.to(device)

    return student_model

def read_dataset(dataset_path, label_mapper, images, labels) -> None:
    for folder in os.listdir(dataset_path):
        if 'imagenet' in folder:
            continue
        class_path = os.path.join(dataset_path,folder)
        images_names = os.listdir(class_path)
        
        for image_name in images_names:
            images.append(os.path.join(class_path,image_name))
            labels.append(label_mapper[folder])

def get_teacher_preds(teacher_model, device, proxy_dataloader, filtered_images, filtered_labels, filtered_soft_labels) -> None:
    with torch.no_grad():
        # for img,label,path,_ in tqdm(proxy_dataloader):
        for img,label,path,_ in tqdm(proxy_dataloader, leave=False):
            img = img.to(device=device)
            label = label.to(device=device)

            logits = teacher_model(img)
            pred = softmax(logits, dim=1)
            
            confidence,y_hat = torch.max(pred, dim=1)

            filtered_images.extend(list(path))
            filtered_labels.extend(y_hat.tolist())
            filtered_soft_labels.extend(pred)
            
        # Display results of filtering
        # print(f'A total of {len(filtered_images)} remained out of {len(proxy_dataset)}')
        # print()
        # counter_per_class = {v:0 for k,v in label_mapper.items()}
        # for label in filtered_labels:
        #     counter_per_class[label] += 1
        # for clasa in counter_per_class:
        #     print(f'Class {clasa}({label_mapper_inv[clasa]}) has {counter_per_class[clasa]} entries')

def split_data(num_examples, RANDOM_SEED, filtered_images, filtered_labels, filtered_soft_labels):
    # Do a stratified split of the data
    NO_IMGS_TO_USE = num_examples
    
    filtered_images_subset, filtered_images_unused, filtered_labels_subset, filtered_labels_unused, filtered_soft_labels_subset, filtered_soft_labels_unused = \
        train_test_split(filtered_images, filtered_labels, filtered_soft_labels, train_size=NO_IMGS_TO_USE, stratify=filtered_labels)#, random_state=RANDOM_SEED)  

    # print(filtered_labels_subset)
    train_images, validation_images, train_labels, validation_labels, train_soft_labels, validation_soft_labels = \
        train_test_split(filtered_images_subset, filtered_labels_subset, filtered_soft_labels_subset, train_size=0.8, stratify=filtered_labels_subset)#, random_state=RANDOM_SEED)
    valid_images, test_images, valid_labels, test_labels, valid_soft_labels, test_soft_labels = \
        train_test_split(validation_images, validation_labels, validation_soft_labels, test_size=0.5, stratify=validation_labels)#, random_state=RANDOM_SEED)

    if len(filtered_images_unused) > len(train_images):
        filtered_images_unused, _, filtered_labels_unused, _, filtered_soft_labels_unused, _ = \
            train_test_split(filtered_images_unused, filtered_labels_unused, filtered_soft_labels_unused, train_size=len(train_images), stratify=filtered_labels_unused)

    return train_images, train_labels, train_soft_labels, \
            valid_images, valid_labels, valid_soft_labels, \
            test_images,  test_labels,  test_soft_labels, \
            filtered_images_unused, filtered_labels_unused, filtered_soft_labels_unused

def start_training(student_model, optimizer, steplr, criterion, best_accuracy, device, train_dataloader, proxy_valid_dataset, valid_dataloader, epochs, temp_best_model, writer=None) -> None:
    student_model.return_feature_domain = False
    
    early_stopping = EarlyStopping(tolerance=5, min_delta=0.05)
    # early_stopping = EarlyStopping(tolerance=5, min_delta=0.001)

    best_epoch = -1
    # Training the student
    for epoch in range(epochs):
        # Define progress bar
        # loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        loop = tqdm(train_dataloader, leave=False)
        
        # Training loop
        student_model.train()
        training_loss_epoch = []
        # for batch_idx, (x,y,soft_y) in loop:
        for x,y,soft_y in loop:#tqdm(train_dataloader, leave=False):
            optimizer.zero_grad()
            
            x = x.to(device=device)
            y = y.to(device=device)
            soft_y = soft_y.to(device=device)
            
            # Forward pass
            logits = student_model(x)
            # Backward pass
            loss = criterion(input=logits, target=soft_y)
            training_loss_epoch.append(loss.item())
            # Prob trb inlocuit criterionul, sa adaugi soft-labels
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
            for x,y,_ in tqdm(valid_dataloader, leave=False):
                x = x.to(device=device)
                y = y.to(device=device)
            
                logits = student_model(x)
                pred = softmax(logits, dim=1)
                
                confidence,y_hat = torch.max(pred, dim=1)
                
                loss = criterion(input=logits, target=y)
                validation_loss_epoch.append(loss.item())
                
                acc += torch.sum(y_hat==y).item()
            
        # loop.write(f'validation_loss on proxy = {sum(validation_loss_epoch)/len(validation_loss_epoch):.4f}')
        # loop.write(f'validation_accuracy on proxy = {100*acc/len(proxy_valid_dataset):.2f}%')
        # loop.set_postfix(validation_loss=f'{sum(validation_loss_epoch)/len(validation_loss_epoch):.4f}')
        # loop.set_postfix(validation_accuracy=f'{100*acc/len(proxy_valid_dataset):.2f}%')
        if writer:
            writer.add_scalar(f'train/training_loss_{temp_best_model}', sum(training_loss_epoch) / len(training_loss_epoch), epoch)
            writer.add_scalar(f'valid/validation_loss_{temp_best_model}', sum(validation_loss_epoch)/len(validation_loss_epoch), epoch)
            writer.add_scalar(f'valid/validation_accuracy_{temp_best_model}', 100*acc/len(proxy_valid_dataset), epoch)

        # Save best model
        valid_proxy_acc = acc/len(proxy_valid_dataset)
        if valid_proxy_acc > best_accuracy:
            best_accuracy = valid_proxy_acc
            best_epoch = epoch
            torch.save(student_model.state_dict(), f'temp_models/{temp_best_model}.pt')
            # print(f'Saved at epoch {epoch}')

        steplr.step()
        early_stopping(sum(validation_loss_epoch)/len(validation_loss_epoch))
        if early_stopping.early_stop:
            # print(f"We are at epoch {epoch}")
            print(f'Model saved from epoch {best_epoch}.')
            break

    return best_accuracy

def start_evaluation_true_gt(student_model, criterion, device, true_dataloader, len_ds, label_mapper_inv):
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
            
        # print('Student with true dataset:')
        # print(f'test_loss = {sum(test_loss)/len(test_loss):.4f}')                  # 1.7439;  1.6199
        # print(f'test_accuracy = {100*acc/len(true_dataset.test_dataset):.2f}%')    # 46.09%;  48.57%
        # print()

    # for k,v in acc_per_class.items():
    #     print(f'Class {label_mapper_inv[k]}: correct_pred={v[0]}, actual={v[1]} => acc={v[0]*100/v[1]:.2f}%, total_pred={v[2]}')

    return 100*acc/len_ds

def start_evaluation_teacher_gt(teacher_model, student_model, criterion, device, true_dataloader, len_ds, label_mapper_inv):
    acc_per_class = {k:[0,0,0] for k,v in label_mapper_inv.items()}

    student_model.eval()
    teacher_model.eval()
    with torch.no_grad():
        test_loss = []
        acc = 0
        for x,y in true_dataloader:
            x = x.to(device=device)
            # y = y.to(device=device)

            teacher_pred = softmax(teacher_model(x), dim=1)
            _, y = torch.max(teacher_pred, dim=1)
        
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
                    acc_per_class[a][0] += 1
                acc_per_class[a][1] += 1
                acc_per_class[p][2] += 1
            
        # print('Student with true dataset:')
        # print(f'test_loss = {sum(test_loss)/len(test_loss):.4f}')                  # 1.7439;  1.6199
        # print(f'test_accuracy = {100*acc/len(true_dataset.test_dataset):.2f}%')    # 46.09%;  48.57%
        # print()
        
    # for k,v in acc_per_class.items():
    #     print(f'Class {label_mapper_inv[k]}: correct_pred={v[0]}, actual={v[1]} => acc={v[0]*100/v[1]:.2f}%, total_pred={v[2]}')

    return 100*acc/len_ds

def build_db(student_model, device, train_dataloader, return_soft, images_db, labels_db):
    if os.path.exists('images_db'):
        rmtree('images_db')
    db_path = 'images_db'
    images_path = os.path.join(db_path,'images')
    labels_path = os.path.join(db_path,'labels')

    # if not os.path.exists('images_db'):
    #     os.makedirs(db_path)
    #     os.makedirs(images_path)
    #     os.makedirs(labels_path)

    # os.makedirs(os.path.join(images_path,f'class{i}'))
    # os.makedirs(os.path.join(labels_path,f'class{i}'))

    # Create the database
    student_model.eval()
    with torch.no_grad():
        # for dataloader in [train_dataloader, valid_dataloader, test_dataloader]:
        for dataloader in [train_dataloader]:
            for x,y,soft_y in dataloader:
                x = x.to(device=device)

                _,latent_fm = student_model(x)

                for i in range(latent_fm.shape[0]):
                    # label = y[i].item()
                    # path_image = f'{images_path}/class{label}'
                    # path_label = f'{labels_path}/class{label}'
                    # idx = len(os.listdir(path_image))
                    # torch.save(latent_fm[i].cpu(), f'{path_image}/t{idx}.pt')
                    # torch.save(soft_y[i].cpu(), f'{path_label}/t{idx}.pt')

                    images_db.append(latent_fm[i])
                    if return_soft:
                        labels_db.append(soft_y[i])
                    else:
                        labels_db.append(y[i])

def get_results(filename, num_examples, acc_before, acc_after):
    with open(f'experiments/{filename}', 'a') as log_file:
        log_file.write(f'{num_examples},{acc_before},{acc_after}\n')