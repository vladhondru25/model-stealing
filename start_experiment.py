import argparse
import random
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets import CIFAR10, ProxyDataset

import src.knn as knn
from src.utils import *


parser = argparse.ArgumentParser(description='Get the inputs.')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--dataset', type=str, default='glide', help='dataset name: glide, stable')
parser.add_argument('--student', type=str, default='resnet', help='the student architecture')
parser.add_argument('--use_soft_labels', default=True, type=lambda x: x == 'True')
parser.add_argument('--distance', type=str, default='euclidean', help='the distance for knn')
parser.add_argument('--use_mixup', default=False, type=lambda x: x == 'True')
args = parser.parse_args()

LR = args.lr
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
CONFIDENCE_TH = 0.8

DATASET_PATH = f'images_generated_{args.dataset}'
EXPERIMENT_NAME = f'{args.use_soft_labels}_{args.distance}_{args.dataset}_{args.student}'
USE_ALL_DATA = False
# print(f'Use all data?: {USE_ALL_DATA}')

# Set random seed for replicating testing results
RANDOM_SEED = 0
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device = {device}')

# Define true dataset
true_testing_dataset = CIFAR10(input_size = 32)

# Define the teacher model
teacher_model = get_teacher(device)

for _ in range(5):
    # Define dataset
    label_mapper = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9
    }
    label_mapper_inv = {v:k for k,v in label_mapper.items()}

    # Get images paths and labels
    images_train_dataset = []
    classes_train_dataset = []
    labels_train_dataset = []
    soft_labels_train_dataset = []

    valid_images = []
    valid_labels = []
    valid_soft_labels = []
    read_dataset(DATASET_PATH, label_mapper, images_train_dataset, classes_train_dataset, labels_train_dataset, soft_labels_train_dataset,
                 valid_images, valid_labels, valid_soft_labels)
    
    print(len(soft_labels_train_dataset))
    print(np.std(soft_labels_train_dataset, axis=1).shape)
    break

    # for num_examples in [1, 2, 4, 8, 16, 32, 64]:
    for power in range(13):
        num_examples = 2**power
        # Define the student model
        student_model = get_student(args.student, device)
        # Define optimizer
        optimizer = torch.optim.Adam(student_model.parameters(), lr=LR)
        steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.95)
        # Define loss function
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        # Early stopping 
        early_stopping = EarlyStopping(student_model, tolerance=5, min_delta=0.01)

        # Split the dataset
        train_images, train_labels, train_soft_labels, \
        unused_images, unused_labels, unused_soft_labels = split_data(num_examples, RANDOM_SEED, USE_ALL_DATA, classes_train_dataset,
                                                                      images_train_dataset, labels_train_dataset, soft_labels_train_dataset)
        
        # Define the transformations
        train_transforms = T.Compose([
            # T.Resize((32,32)),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize((0.5,), (0.5,))
        ])
        valid_transforms = T.Compose([
            # T.Resize((32,32)),
            T.Normalize((0.5,), (0.5,))
        ])

        # Define the proxy datasets
        proxy_train_dataset = ProxyDataset(train_images, train_labels, train_transforms, False, train_soft_labels)
        proxy_valid_dataset = ProxyDataset(valid_images, valid_labels, valid_transforms, False, valid_soft_labels)

        # Define the proxy dataloaders
        train_dataloader = DataLoader(proxy_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_dataloader = DataLoader(proxy_valid_dataset, batch_size=BATCH_SIZE)

        # Training loop
        start_training(student_model, optimizer, steplr, criterion, early_stopping, device, train_dataloader, valid_dataloader, EPOCHS)

        # Testing on CIFAR10 ground truth
        student_model.return_feature_domain = False
        # acc_true_gt_before = start_evaluation_true_gt(student_model, criterion, device, true_testing_dataset, label_mapper_inv)
        # Testing using labels predicted with teacher
        acc_teacher_gt_before = start_evaluation_teacher_gt(teacher_model, student_model, criterion, device, true_testing_dataset, label_mapper_inv)


        # Redefine training dataset and dataloader with no augmentation
        proxy_train_dataset = ProxyDataset(train_images, train_labels, valid_transforms, False, train_soft_labels)
        train_dataloader = DataLoader(proxy_train_dataset, batch_size=BATCH_SIZE, shuffle=False)

        student_model.return_feature_domain = True
            
        # Build the database
        images_db = []
        labels_db = []
        build_db(student_model, device, train_dataloader, args.use_soft_labels, images_db, labels_db)

        # Define the database i.e. feature maps with label from teacher
        db_dataset = DBDataset(images_db=images_db, labels_db=labels_db)
        db_dataloader = DataLoader(db_dataset, batch_size=128, shuffle=False)

        # Define the extra examples (which we will try to assign label to)
        proxy_unused_dataset  = ProxyDataset(unused_images, unused_labels,  valid_transforms, True, unused_soft_labels)

        # Assign labels to extra data
        for k in [5]:
            estimations = []
            ground_truth = []

            knn.get_neighs(student_model, device, proxy_unused_dataset, db_dataset, db_dataloader, k,
               train_images, train_labels, train_soft_labels, estimations, ground_truth,
               args.use_soft_labels, args.distance
            )

            # correct = (np.array(estimations) == np.array(ground_truth)).sum()
            # print(f'Final score for k={k}: {correct} / {len(estimations)}, acc = {correct/len(estimations)}')

        # Re-define the training dataset and dataloader
        proxy_train_dataset = ProxyDataset(train_images, train_labels, train_transforms, False, train_soft_labels)
        train_dataloader = DataLoader(proxy_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Re-start training with extra data
        early_stopping.reset()
        # for g in optimizer.param_groups:
        #         g['lr'] = LR / 2
        start_training(student_model, optimizer, steplr, criterion, early_stopping, device, train_dataloader, valid_dataloader, 
                       EPOCHS, use_mixup=args.use_mixup)

        # Testing on CIFAR10 ground truth
        student_model.return_feature_domain = False
        # acc_true_gt_after = start_evaluation_true_gt(student_model, criterion, device, true_testing_dataset, label_mapper_inv)
        # Testing using labels predicted with teacher
        acc_teacher_gt_after = start_evaluation_teacher_gt(teacher_model, student_model, criterion, device, true_testing_dataset, label_mapper_inv)

        print(f'For {num_examples} examples, accuracy went from {acc_teacher_gt_before} to {acc_teacher_gt_after}')
        # save_results(f'{EXPERIMENT_NAME}.txt', num_examples, acc_teacher_gt_before, acc_teacher_gt_after)
