import argparse
import random
import sys
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets import CIFAR10, FOOD101, ProxyDataset

import src.knn as knn
import src.active_learning as al
from src.utils import *


parser = argparse.ArgumentParser(description='Get the inputs.')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--step_size', type=int, default=20, help='step size')
parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('--no_runs', type=int, default=5, help='number of runs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--dataset', type=str, default='glide', help='dataset name: glide, stable, food101')
parser.add_argument('--student', type=str, default='resnet', help='the student architecture')
parser.add_argument('--teacher', type=str, default='alexnet', help='the teacher architecture for food101')
parser.add_argument('--use_soft_labels', default=True, type=lambda x: x == 'True')
parser.add_argument('--use_active_learning', default=True, type=lambda x: x == 'True')
parser.add_argument('--use_og_labels', default=False, type=lambda x: x == 'True')
parser.add_argument('--distance', type=str, default='euclidean', help='the distance for knn')
parser.add_argument('--use_all_data', default=False, type=lambda x: x == 'True')
parser.add_argument('--use_mixup', default=False, type=lambda x: x == 'True')
parser.add_argument('--save_results', default=True, type=lambda x: x == 'True')
args = parser.parse_args()

LR = args.lr
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

DATASET_PATH = f'images_generated_{args.dataset}'
EXPERIMENT_NAME = f'{args.use_soft_labels}_{args.distance}_{args.dataset}_{args.student}_{args.teacher}'
USE_ALL_DATA = args.use_all_data # False
# print(f'Use all data?: {USE_ALL_DATA}')
IMAGE_SIZE = 224 if 'food' in args.dataset else 32

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

# seed_everything(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device = {device}')

# Define true dataset
true_testing_dataset = FOOD101(input_size=IMAGE_SIZE) if 'food' in args.dataset else CIFAR10(input_size=IMAGE_SIZE)

# Define the teacher model
if 'food' in args.dataset:
    teacher_model = get_teacher_food(device, args.teacher)
else:
    teacher_model = get_teacher(device)

for _ in range(args.no_runs):
    # Define dataset
    if 'food' in args.dataset:
        with open('food101_classes.txt') as input_file:
            food101_classes = input_file.read().split()
        label_mapper = {food101_classes[i]: i for i in range(len(food101_classes))}
        label_mapper_inv = {v:k for k,v in label_mapper.items()}
    else:
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
    
    max_power = 13 if 'food' not in args.dataset else 11
    # for power in [10]:    
    for power in range(max_power):      
        num_examples = 2**power
        # Define the student model
        student_model = get_student(args.student, device)
        activation = None
        if args.student == "resnet_food":
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = input[0].detach()
                return hook


            student_model.avgpool.register_forward_hook(get_activation('latent_space'))
        # Define optimizer
        optimizer = torch.optim.Adam(student_model.parameters(), lr=LR)
        steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.95)        
        # Define loss function
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        # Early stopping 
        early_stopping = EarlyStopping(student_model, tolerance=5, min_delta=0.01)

        # Split the dataset
        if args.use_active_learning:
            train_images, train_labels, train_soft_labels, \
            unused_images, unused_labels, unused_soft_labels, unused_classes = \
                al.active_learning_step(student_model, num_examples, label_mapper, device,
                         images_train_dataset, classes_train_dataset, labels_train_dataset, soft_labels_train_dataset,
                         valid_images, valid_labels, valid_soft_labels)
        else:
            train_images, train_labels, train_soft_labels, \
            unused_images, unused_labels, unused_soft_labels, unused_classes = \
                split_data(num_examples, RANDOM_SEED, USE_ALL_DATA, classes_train_dataset,
                            images_train_dataset, labels_train_dataset, soft_labels_train_dataset, len(label_mapper))

        # Define the transformations
        train_transforms = T.Compose([
            # T.Resize((IMAGE_SIZE,IMAGE_SIZE)),
            T.RandomCrop(IMAGE_SIZE, padding=4),
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize((0.5,), (0.5,))
        ])
        valid_transforms = T.Compose([
            # T.Resize((IMAGE_SIZE,IMAGE_SIZE)),
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
        build_db(student_model, device, train_dataloader, args.use_soft_labels, images_db, labels_db, args.student, activation)

        # Define the database i.e. feature maps with label from teacher
        db_dataset = DBDataset(images_db=images_db, labels_db=labels_db)
        db_dataloader = DataLoader(db_dataset, batch_size=128, shuffle=False)

        # Define the extra examples (which we will try to assign label to)
        proxy_unused_dataset  = ProxyDataset(unused_images, unused_labels, valid_transforms, True, unused_soft_labels)

        # Assign labels to extra data
        for k in [5]:
            estimations = []
            ground_truth = []

            knn.get_neighs(student_model, device, proxy_unused_dataset, db_dataset, db_dataloader, k,
               train_images, train_labels, train_soft_labels, estimations, ground_truth,
               args.use_soft_labels, args.distance, args.use_og_labels, unused_classes, args.student, activation
            )

            correct = (np.array(estimations) == np.array(ground_truth)).sum()
            # print(f'Final score for k={k}: {correct} / {len(estimations)}, acc = {correct/len(estimations)}')
            # print(f'Final score for {num_examples=}: {correct} / {len(estimations)}, acc = {correct/len(estimations)}')

        # Re-define the training dataset and dataloader
        proxy_train_dataset = ProxyDataset(train_images, train_labels, train_transforms, False, train_soft_labels)
        train_dataloader = DataLoader(proxy_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

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
        if args.save_results:
            save_results(f'{EXPERIMENT_NAME}.txt', num_examples, acc_teacher_gt_before, acc_teacher_gt_after)
