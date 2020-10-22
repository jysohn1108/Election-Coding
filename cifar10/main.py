'''
This is the code for running in a single machine (without using MPI)
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import os
import sys
import pdb
import time
import copy
import random
import argparse
import numpy as np
from functools import reduce

# From DRACO
from model import resnet
from resnet_draco import ResNet18, ResNet50
from sgd_modified_draco import SGDModified

parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--network', default='ResNet18', type=str, choices = ['ResNet18', 'ResNet50'], help='Network I will use')
parser.add_argument('--gpu_num', default=6, type=int, help='GPU number I will use')
parser.add_argument('--lr', default=1e-4, type=float, help='')
parser.add_argument('--batch_size', default=120, type=int, help='') 
parser.add_argument('--batch_size_test', default=100, help='')
parser.add_argument('--load_worker', default=4, help='')
parser.add_argument('--num_epochs', default=100, type=int, help='number of training Epochs')
parser.add_argument('--signed_sum', default=False, type=bool, help='Whether we run signed sum scheme (compared scheme) instead of taking the majority vote at each node')
parser.add_argument('--coord_median', default=False, type=bool, help='Whether we use coordinate-wise median (with full gradient) instead of Coded SignSGD-MV')
parser.add_argument('--noisy_gradient', default=False, type=bool, help='Whether we add Gaussian noise to the gradient before applying the sign function')
parser.add_argument('--variance', default=0, type=float, help='the variance of Gaussian noise (added at the computed gradient)')  

parser.add_argument('--trial_idx', default=1, type=int, help='trial index')
parser.add_argument('--deterministic', default=False, type=bool, help='Whether the code is probabilistic or deterministic')
parser.add_argument('--redundancy', default=1.0, type=float, help='expected redundancy (E[r] = np) in probabilistic codes')
parser.add_argument('--num_nodes', default=5, type=int, help='number of nodes at the system')
parser.add_argument('--num_Byz_nodes', default=0, type=int, help='number of Byzantine nodes')
parser.add_argument('--attack_type', default='reverse', type=str, choices = ['reverse', 'directional', 'random'], help='attack type')

args = parser.parse_args()
print(args)


device = 'cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu' 
content_train = ['epoch, train_batch_idx, training time'] # store data 
content_test = ['epoch, train_batch_idx, test accuracy'] # store data 
start_training_time = 0
train_time = 0



def load_data(batch_size, batch_size_test, num_workers):

    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset_train = CIFAR10(root='../data', train=True, 
                            download=True, transform=transforms_train)
    dataset_test = CIFAR10(root='../data', train=False, 
                           download=True, transform=transforms_test)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset_test, batch_size=batch_size_test, 
                             shuffle=False, num_workers=num_workers)

    return dataset_train, dataset_test, train_loader, test_loader


def train(epoch, global_steps, batch_idx, inputs, targets, EncMatrix, ByzIdx):

    global start_training_time
    global train_time 

    net.train() 
    net.to(device) 
    global_steps += 1

    _grad_buffer_global = [] # list of gradient (each element corresponds to each layer)

    if args.coord_median == False: # Coded/Uncoded SignSGD
        for param_idx, param in enumerate(net.parameters()):
            _grad_buffer_global.append(torch.zeros(param.size()).to(device)) # initialize
        for rank in np.arange(0, num_nodes)[::-1]:    
            if rank == 0:   # master
                for param_idx, param in enumerate(net.parameters()): 
                    _grad_buffer_global[param_idx] = torch.sign(_grad_buffer_global[param_idx]) # signSGD-MV: take majority vote at master              

                ## Update the model (@ master)
                optimizer_master.step(grads = _grad_buffer_global, mode = "normal")
                step_lr_scheduler.step()

            else: # Distributed nodes
                grad_collector = {}
                indices = np.where(EncMatrix[rank-1] != 0)[0]
                numBatchInPart = len(inputs)/EncMatrix.shape[1] # number of data points in each batch 
            
                for idx, dataIdx  in enumerate(indices):
                    inputs_part = inputs[(int)(dataIdx*numBatchInPart):(int)((dataIdx+1)*numBatchInPart)]
                    targets_part = targets[(int)(dataIdx*numBatchInPart):(int)((dataIdx+1)*numBatchInPart)]

                    inputs_part = inputs_part.to(device)
                    targets_part = targets_part.to(device)
                    outputs_part = net(inputs_part)        
                    loss = criterion(outputs_part, targets_part)
                    optimizer_master.zero_grad()
                    loss.backward()
                    
                    # extract gradient...
                    _grad_buffer = [] # list of gradient (each element corresponds to each layer)
                    for param_idx, param in enumerate(net.parameters()):
                        _grad_buffer.append(np.zeros(param.size())) # initialize
                        grad_extract = param.grad
                        
                        if rank in ByzIdx: #Byzantine workers
                            if args.attack_type == 'random':
                                randVec = np.random.randint(2,size = param.size()) # random attack
                                randVec = 2 * randVec - 1 
                                _grad_buffer[param_idx] = torch.from_numpy(randVec).to(device)
                            elif args.attack_type == 'reverse':
                                _grad_buffer[param_idx] = -torch.sign(grad_extract).to(device) # reverse attack
                            elif args.attack_type == 'directional':
                                _grad_buffer[param_idx] = -torch.ones(param.size()).to(device) # directional attack
                            else:
                                print('args.attack_type is wrong')
                        else:
                            if args.signed_sum == True:
                                _grad_buffer[param_idx] = grad_extract.to(device) # original
                            else:
                                if args.noisy_gradient == True:
                                    grad_extract += args.variance * torch.randn_like(grad_extract)
                                _grad_buffer[param_idx] = torch.sign(grad_extract).to(device) # signSGD

                    grad_collector[dataIdx] = _grad_buffer # fill in the grad_collector "dictionary" (key,value pairs)

                ## send gradient to master
                for param_idx, param in enumerate(net.parameters()):
                    aggregated_grad = torch.zeros(param.shape).to(device)
                    #1) linear combination
                    for key, value in grad_collector.items():
                        aggregated_grad += value[param_idx]
                    aggregated_grad = torch.sign(aggregated_grad)
                    #2) send (store)
                    _grad_buffer_global[param_idx] += aggregated_grad     


    else: # coordinate-wise median
        for param_idx, param in enumerate(net.parameters()):
            _grad_buffer_global.append([]) # initialize
    
        for rank in np.arange(0, num_nodes)[::-1]:
            if rank == 0: # master
                for g_idx, grads in enumerate(_grad_buffer_global):
                    val, ind = torch.median(torch.cat(grads, dim=1), 1)
                    coord_median = val.to(device)
                    _grad_buffer_global[g_idx] = coord_median

                ## Update the model (@ master)
                optimizer_master.step(grads = _grad_buffer_global, mode = "median")
                step_lr_scheduler.step()

            else:
                dataIdx = rank - 1
                numBatchInPart = len(inputs)/EncMatrix.shape[1] # number of data points in each batch 
                inputs_part = inputs[(int)(dataIdx*numBatchInPart):(int)((dataIdx+1)*numBatchInPart)]
                targets_part = targets[(int)(dataIdx*numBatchInPart):(int)((dataIdx+1)*numBatchInPart)]

                inputs_part = inputs_part.to(device)
                targets_part = targets_part.to(device)
                outputs_part = net(inputs_part)        
                loss = criterion(outputs_part, targets_part)
                optimizer_master.zero_grad()
                loss.backward()

                for param_idx, param in enumerate(net.parameters()):
                    grad_extract = param.grad
                    _shape = grad_extract.shape
                    if rank in ByzIdx:
                        if args.attack_type == 'reverse':
                            _grad_buffer_global[param_idx].append(-1 * grad_extract.reshape((reduce(lambda x, y: x * y, _shape),)).reshape(-1,1))
                        else:
                            print('TODO: insert here')
                            exit()
                    else:
                        _grad_buffer_global[param_idx].append(grad_extract.reshape((reduce(lambda x, y: x * y, _shape),)).reshape(-1,1))


    # store the training time
    elapsed_time = time.time()-start_training_time
    train_time = train_time + elapsed_time
    content_train.append('%i,%i,%.3f' %(epoch, batch_idx, train_time))   
    if batch_idx%50 == 0:    print('Epoch: %i, bach_idx: %i, train_time: %.3f' %(epoch, batch_idx, train_time))

    return global_steps


def test(epoch, best_acc, global_steps, train_batch_idx):
    net.eval()
    net.to(device) 

    global content_test

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
              
        acc = 100 * correct / total
        print('test epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(epoch, train_batch_idx, len(train_loader), test_loss/(train_batch_idx+1), acc))
        content_test.append('%i,%i,%.3f' %(epoch, train_batch_idx, acc))        

    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('save_model'):
            os.mkdir('save_model')
        torch.save(state, './save_model/ckpt.pth')
        best_acc = acc
    net.to('cpu')
    return best_acc


def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__=='__main__':
    best_acc = 0
    epoch = 0
    global_steps = 0
    train_time = 0
    print('device:', device)
    print('Trial ', args.trial_idx)
    print("="*88)

    print('==> Preparing data..')
    seed(args.trial_idx)
    dataset_train, dataset_test, train_loader, test_loader = load_data(args.batch_size, args.batch_size_test, args.load_worker)
    num_train = len(dataset_train) # number of training data in cifar-10


    print('==> Set the Byzantine attack setup..')
    num_nodes = args.num_nodes
    num_batch = (int)(num_train/args.batch_size)
    Byz = args.num_Byz_nodes # number of Byzantine nodes
    if args.deterministic == False: connect_prob = args.redundancy/num_nodes #probability p that an element of encoding matrix is 1
    ByzIdx = np.random.choice(num_nodes,Byz,replace = False) + 1 # randomly select 'Byz' Byzantines  #ByzIdx = [4,5,6]
    print('ByzIdx: ', ByzIdx)


    print('==> Making model..')
    if args.network == 'ResNet18':  net = ResNet18()
    elif args.network == 'ResNet50':    net = ResNet50()
    net = net.to(device)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    print('==> Set training option..')
    criterion = nn.CrossEntropyLoss()
    optimizer_master = SGDModified(net.parameters(), lr=args.lr, 
                          momentum=0.9, weight_decay=1e-4)
    if num_nodes in [5,9]:    decay_epoch = [40*num_batch, 80*num_batch]
    else:   decay_epoch = [20*num_batch, 40*num_batch, 80*num_batch]
    print('decay_epoch :', (np.array(decay_epoch)/num_batch).astype('int'))
    step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_master, 
                                     milestones=decay_epoch, gamma=0.1)

    print('==> Set allocation matrix..')
    if args.deterministic == False and args.coord_median == False: # probabilistic codes
        checkValidG = 0
        while checkValidG < num_nodes:
            checkValidG = 0
            EncMatrix = np.random.choice(2, size=(num_nodes,num_nodes), p=[1-connect_prob, connect_prob]) # encoding matrix (G) w/ average redundancy r=n*connect_prob
            for j in range(num_nodes):
                if np.sum(EncMatrix[j]) != 0:
                    checkValidG = checkValidG + 1
    else: # deterministic codes (or coordinate-wise median)
        if args.redundancy == 1.0 or args.coord_median == True:
            EncMatrix = np.identity(num_nodes)  # r=1 (uncoded)
        elif args.redundancy == 3.8:    
            EncMatrix = np.array([[1,0,0,0,0], [0,1,1,1,0], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1] ]) # encoding matrix (G) w/ r=3.8
    print('Enc Matrix: \n ', EncMatrix)
    print('sum of ones in Enc Matrix: ', sum(sum(EncMatrix)))


    print('==> Training start!')
    for epoch in np.arange(1, args.num_epochs+1):
        print('Epoch: ', epoch)   
        for param_group in optimizer_master.param_groups:
                print('lr :', param_group['lr'])
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):  
            inputs, targets = inputs.to(device), targets.to(device)
            start_training_time = time.time() # start of training
            global_steps = train(epoch, global_steps, batch_idx, inputs, targets, EncMatrix, ByzIdx)
            if batch_idx == num_batch: # test every epoch
                best_acc = test(epoch, best_acc, global_steps, batch_idx)            
                print('best test accuracy is ', best_acc)

        # store results at files
        if args.coord_median == True:
            f = open('{}_coord_median_trial_{}_batch_{}_num_nodes_{}_num_Byz_{}_redundancy_{}_attack_{}_test.csv'.format(args.network, args.trial_idx, args.batch_size, num_nodes, Byz, args.redundancy, args.attack_type),'w')
        else:
            f = open('{}_noise_variance_{}_signSGD_trial_{}_batch_{}_num_nodes_{}_num_Byz_{}_redundancy_{}_attack_{}_test.csv'.format(args.network, args.variance, args.trial_idx, args.batch_size, num_nodes, Byz, args.redundancy, args.attack_type),'w')
        f.write('\n'.join(content_test))
        f.close()

