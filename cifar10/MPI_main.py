# Code for MPI  
#
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import sys
import time
import random
import argparse
import numpy as np

from mpi4py import MPI  ## MPI
from model import resnet
from resnet_draco import ResNet18
from sgd_modified_draco import SGDModified

parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=1e-4, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', default=120, help='')
parser.add_argument('--batch_size_test', default=100, help='')
parser.add_argument('--num_worker', default=4, help='')
parser.add_argument('--logdir', type=str, default='logs', help='')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

content_train = ['epoch, train_batch_idx, training time'] # store data 
content_test = ['epoch, train_batch_idx, test accuracy'] # store data 

start_training_time = 0
train_time = 0

train_loss = 0
train_correct = 0
train_total = 0

print('==> Preparing data..')
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
train_loader = DataLoader(dataset_train, batch_size=args.batch_size, 
                          shuffle=True, num_workers=args.num_worker)
test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, 
                         shuffle=False, num_workers=args.num_worker)

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
numWorkers = size - 1



## Configuration
num_train = 50000 # number of training data in cifar-10
size_batch = args.batch_size # batch size
num_epoch = 300

Byz = 1 # number of Byzantine nodes
ByzIdx = []
attackType = -1 # 0: random, 1: directional (all-one), -1: reverse

ProbDesign = 0 # 1 if we use probabilistic code, and 0 otherwise (if we use algebraic code)
trialNum = 1 # the number of trial in probabilistic code (for taking the average)
probEnc = 1/3 #probability that an element of encoding matrix is 1
EncMatrix = np.identity(numWorkers)  # r=1 (uncoded)




num_batch = (int)(num_train/args.batch_size)
print('num_batch : ', num_batch)
print('==> Making model..')

net = ResNet18()
net = net.to(device)
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('The number of parameters of model is', num_params)

if args.resume is not None:
    checkpoint = torch.load('./save_model/' + args.resume)
    net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()
optimizer_master = SGDModified(net.parameters(), lr=args.lr, 
                      momentum=0.9, weight_decay=1e-4)

decay_epoch = [80*num_batch, 120*num_batch]
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_master, 
                                 milestones=decay_epoch, gamma=0.1)

def train(epoch, global_steps, batch_idx, inputs, targets):
    net.train() 

    global train_loss
    global train_correct
    global train_total
    global content_train


    global start_training_time
    global train_time

    global rank
    global EncMatrix
    global ByzIdx
    
    net.to(device)
    global_steps += 1
    step_lr_scheduler.step()
    grad_collector = {}
    
    if rank == 0:  
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)
    
        loss = criterion(outputs, targets)

        _grad_buffer = [] # list of gradient (each element corresponds to each layer)
        for param_idx, param in enumerate(net.parameters()):
            receive_buffer = np.zeros((size-1,) + param.shape)    
            requests = [ MPI.REQUEST_NULL ] * (size-1)
            for sender in range(1,size):
                requests[sender-1] = comm.Irecv(receive_buffer[sender-1],source=sender, tag=88+param_idx)                
            status = [ MPI . Status () for i in range (1 , size )]    
            # Wait for all the messages
            MPI.Request.Waitall (requests , status ) # connection btw requests and status made at here...
    
            updated_grad = np.zeros(param.shape) # aggregated velocity (sign)
            for sender2 in range(1,size):
                updated_grad = updated_grad + receive_buffer[sender2-1]
            _grad_buffer.append(torch.zeros(param.size())) # initialize
            _grad_buffer[param_idx] = torch.sign(torch.from_numpy(updated_grad)).to(device) # signSGD: take majority vote at master              

        ## Update the model (@ master)
        optimizer_master.step(grads = _grad_buffer, mode = "normal")
        

    else:
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
                grad_extract = param.grad.cpu().data.numpy().astype(np.float64)
                #_grad_buffer[param_idx] = grad_extract # original
                #_grad_buffer[param_idx] = -grad_extract # when reverse attack is applied to all workers
                _grad_buffer[param_idx] = np.sign(grad_extract) # signSGD
                
                if rank in ByzIdx: 
                    if attackType == 0:
                        randVec = np.random.randint(2,size = param.size()) # random attack
                        randVec = 2 * randVec - 1 
                        _grad_buffer[param_idx] = randVec
                    elif attackType == -1:
                        _grad_buffer[param_idx] = -np.sign(grad_extract) # reverse attack
                    elif attackType == 1:
                        _grad_buffer[param_idx] = np.ones(param.size()) # directional attack
                    else:
                        print('attackType is wrong')
                    
            grad_collector[dataIdx] = _grad_buffer # fill in the grad_collector "dictionary" (key,value pairs)

        ##send gradient to master
        for param_idx, param in enumerate(net.parameters()):
            aggregated_grad = np.zeros(param.shape)
            #1) linear combination
            for key, value in grad_collector.items():
                aggregated_grad = np.add(aggregated_grad, value[param_idx])
            aggregated_grad = np.sign(aggregated_grad)
            #2) comm.isend
            request = comm.Isend (aggregated_grad, dest=0, tag=88+param_idx)
            request.Wait ()

    net.to('cpu')
    for param_idx, param in enumerate(net.parameters()):
        comm.Bcast(param.cpu().data.numpy(), root=0) # broadcast the array from rank 0 (master) to all workers

    if rank == 0:
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()
        
        elapsed_time = time.time()-start_training_time
        train_time = train_time + elapsed_time
        acc = 100 * train_correct / train_total
        content_train.append('%i,%i,%.3f' %(epoch, batch_idx, train_time))   

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
        print('==> Saving model..')
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


if __name__=='__main__':
    best_acc = 0
    epoch = 0
    global_steps = 0
    if args.resume is not None:
        test(epoch=0, best_acc=0)
    else:
        print('device:', device)
        for randomIdx in range(trialNum):        
            train_time = 0
            ## Define encoding matrix  ex.EncMatrix = np.identity(numWorkers)  # r=1 (uncoded)
            if rank == 0:
                ## Probabilistic Coding (Encoding matrix)
                if ProbDesign == 1:
                    checkValidG = 0
                    while checkValidG < numWorkers:
                        checkValidG = 0
                        EncMatrix = np.random.choice(2, size=(numWorkers,numWorkers), p=[1-probEnc, probEnc]) # encoding matrix (G) w/ average redundancy r=n*probEnc
                        for j in range(numWorkers):
                            if np.sum(EncMatrix[j]) != 0:
                                checkValidG = checkValidG + 1
                else:
                    EncMatrix = np.identity(numWorkers)  # r=1 (uncoded)
                    #EncMatrix = np.array([[1,0,0,0,0], [0,1,1,1,0], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1] ]) # encoding matrix (G) w/ r=3.8 (deterministic for n=5, b=1)
                    #EncMatrix = np.ones((numWorkers, numWorkers)) # r=numWorkers 
                    sumEnc = np.sum(EncMatrix)

                # Store EncMatrix in matapp variable..
                if randomIdx == 0:
                    matapp = EncMatrix
                else:
                    matapp = np.vstack((matapp, EncMatrix))
            else:
                EncMatrix = np.zeros((numWorkers, numWorkers))
            
            EncMatrix = comm.bcast(EncMatrix, root=0)
            
            ## Set Byzantine nodes
            if rank == 0:
                ByzIdx = np.random.choice(numWorkers,Byz,replace = False) + 1 # randomly select 'Byz' Byzantines
            else:
                ByzIdx = []
            ByzIdx = comm.bcast(ByzIdx, root = 0)

            while True:
                epoch += 1
                
                train_loss = 0
                train_correct = 0
                train_total = 0

                for batch_idx, (inputs, targets) in enumerate(train_loader):  
                    start_training_time = time.time() # start of training
                    global_steps = train(epoch, global_steps, batch_idx, inputs, targets)
                    if rank == 0: # test every batch #and batch_idx == 0: # test every epoch
                        best_acc = test(epoch, best_acc, global_steps, batch_idx)            
                
                if epoch >= num_epoch: 
                    break
            
            content_train.append('epoch, train_batch_idx, training time')
            content_test.append('epoch, train_batch_idx, test accuracy')
 
                 
        f = open('signSGD_lr_1e-4_test.csv','w')
        f.write('\n'.join(content_test))
        f.close()
        f2 = open('signSGD_lr_1e-4_train.csv','w')
        f2.write('\n'.join(content_train))
        f2.close()
        print("Saved")
            
