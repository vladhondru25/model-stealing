import numpy as np
import os
import setup
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

# Class for early stopping
class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, validation_loss):
        if self.best_score is None:
          self.best_score = validation_loss
        elif validation_loss - self.best_score < self.min_delta:
          self.best_score = validation_loss
        else:
          self.counter +=1
          if self.counter >= self.tolerance:  
              self.early_stop = True
              
              
def train_proxy_dataset(
    model, dataset,
    n_epochs = 200
):
    model.to(setup.device)

    lr = 0.001
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = lr,
    )
    loss_function = torch.nn.CrossEntropyLoss()
    
    proxy_dataset_train, proxy_dataset_valid = dataset
    proxy_datalaoder_train = DataLoader(dataset=proxy_dataset_train, batch_size=64, shuffle=True)
    proxy_datalaoder_valid = DataLoader(dataset=proxy_dataset_valid, batch_size=64, shuffle=True)
    
    early_stopping = EarlyStopping(tolerance=20, min_delta=0.001)
    
    # Training the student
    for epoch in range(n_epochs):
        # Define progress bar
        loop = tqdm(enumerate(proxy_datalaoder_train), total=len(proxy_datalaoder_train))
        
        # Training loop
        model.train()
        training_loss_epoch = []
        # for batch_idx, (x,soft_y) in loop:
        for batch_idx, (x,soft_y) in loop:
            optimizer.zero_grad()
            
            x = x.to(device=setup.device)
            soft_y = soft_y.to(device=setup.device)
            
            # Forward pass
            logits = torch.softmax(model(x), dim = -1)
            # Backward pass
            loss = loss_function(input=logits, target=soft_y)
            training_loss_epoch.append(loss.item())

            loss.backward()
            
            # Optimize
            optimizer.step()
            
            # Update progress bar
            loop.set_description(f'Epoch {epoch+1}/{n_epochs}')
            loop.set_postfix(training_loss=loss.item())
        
        # Validation loop on proxy validation dataset
        model.eval()
        validation_loss_epoch = []  
        acc = 0
        with torch.no_grad():
            for x,y in proxy_datalaoder_valid:
                x = x.to(device=setup.device)
                y = y.to(device=setup.device)
            
                logits = model(x)
                _,y_hat = torch.max(logits, dim=1)
                
                _,y_true = torch.max(y, dim=1)
                
                loss = loss_function(input=torch.softmax(logits, dim=-1), target=y_true)
                validation_loss_epoch.append(loss.item())
                
                acc += torch.sum(y_hat==y_true).item()
            
        loop.write(f'validation_loss on proxy = {sum(validation_loss_epoch)/len(validation_loss_epoch):.4f}')
        loop.write(f'validation_accuracy on proxy = {100*acc/len(proxy_dataset_valid):.2f}%')

        early_stopping(sum(validation_loss_epoch)/len(validation_loss_epoch))
        if early_stopping.early_stop:
            print(f"We are at epoch {epoch}")
            break
