"""
This program is based on the code of Benjamin Solecki for Amazon Employee Access Challenge. 
(Original source code: https://github.com/pyduan/amazonaccess)

We apply the election coding schemes on multiple workers with MPI.
Please find the configuration at line 254
"""

from numpy import array, hstack
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn import preprocessing
from scipy import sparse
from itertools import combinations

from sklearn.preprocessing import OneHotEncoder
from mpi4py import MPI

import numpy as np
import pandas as pd
import random
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
numWorkers = size - 1

content = ['epoch, time, auc'] # stored data format
SEED = 55

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    # input: vector (1 x m)
    # output: vector (1 x m)
    # return 1 / (1 + np.exp(-x))
    return 1 / (2 + np.expm1(-x))  # expm1(x) is equivalent to exp(x) - 1  


def net_input(theta, x):
    # Computes the weighted sum of inputs
    # input: vector (1 x d), matrix (m x d)
    # output: vector (1 x m)
    return np.dot(x, theta)   

def probability(theta, x):
    # Returns the probability after passing through sigmoid
    # input: vector (1 x d), matrix (m x d)
    # output: vector (1 x m)
    return sigmoid(net_input(theta, x))

def cost_function(theta, x, y):
    # Computes the cost function for all the training samples
    # input: vector (1 x d), matrix (m x d), vector (1 x m)
    # output: scalar
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost

def gradient(theta, x, y):
    # Computes the gradient of the cost function at the point theta
    # input: vector (1 x d), matrix (m x d), vector (1 x m)
    # output: scalar
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, probability(theta,   x) - y)



def group_data(data, degree=3, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return array(new_data).T

def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print("Saved")

def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = train_test_split(
                                       X, y, test_size=1.0/float(N), 
                                       random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.roc_auc_score(y_cv, preds)
        mean_auc += auc
    return mean_auc/N


learner = "log"
if rank==0:
    print("Reading dataset...")
train_data = pd.read_csv('data/train.csv')
submit=learner + '.csv'
all_data = np.array(train_data.iloc[:,1:])
if numWorkers == 5:
    num_train= 26325
elif numWorkers == 7:
    num_train= 26208 
elif numWorkers == 15:
    num_train= 26325
elif numWorkers == 49:
    num_train= 28665

print('num_train:', num_train)

# Transform data
if rank==0:
    print("Transforming data...")
# Relabel the variable values to smallest possible so that I can use bincount
# on them later.
relabler = preprocessing.LabelEncoder()
for col in range(len(all_data[0,:])):
    relabler.fit(all_data[:, col])
    all_data[:, col] = relabler.transform(all_data[:, col])
########################## 2nd order features ################################
dp = group_data(all_data, degree=2) 
for col in range(len(dp[0,:])):
    relabler.fit(dp[:, col])
    dp[:, col] = relabler.transform(dp[:, col])
    uniques = len(set(dp[:,col]))
    maximum = max(dp[:,col])
    if maximum < 65534:
        count_map = np.bincount((dp[:, col]).astype('uint16'))
        for n,i in enumerate(dp[:, col]):
            if count_map[i] <= 1:
                dp[n, col] = uniques
            elif count_map[i] == 2:
                dp[n, col] = uniques+1
    else:
        for n,i in enumerate(dp[:, col]):
            if (dp[:, col] == i).sum() <= 1:
                dp[n, col] = uniques
            elif (dp[:, col] == i).sum() == 2:
                dp[n, col] = uniques+1
    uniques = len(set(dp[:,col]))
    relabler.fit(dp[:, col])
    dp[:, col] = relabler.transform(dp[:, col])
########################## 3rd order features ################################
dt = group_data(all_data, degree=3)
for col in range(len(dt[0,:])):
    relabler.fit(dt[:, col])
    dt[:, col] = relabler.transform(dt[:, col])
    uniques = len(set(dt[:,col]))
    maximum = max(dt[:,col])
    if maximum < 65534:
        count_map = np.bincount((dt[:, col]).astype('uint16'))
        for n,i in enumerate(dt[:, col]):
            if count_map[i] <= 1:
                dt[n, col] = uniques
            elif count_map[i] == 2:
                dt[n, col] = uniques+1
    else:
        for n,i in enumerate(dt[:, col]):
            if (dt[:, col] == i).sum() <= 1:
                dt[n, col] = uniques
            elif (dt[:, col] == i).sum() == 2:
                dt[n, col] = uniques+1
    uniques = len(set(dt[:,col]))
    relabler.fit(dt[:, col])
    dt[:, col] = relabler.transform(dt[:, col])

########################## 1st order features ################################
for col in range(len(all_data[0,:])):
    relabler.fit(all_data[:, col])
    all_data[:, col] = relabler.transform(all_data[:, col])
    uniques = len(set(all_data[:,col]))
    maximum = max(all_data[:,col])
    if maximum < 65534:
        count_map = np.bincount((all_data[:, col]).astype('uint16'))
        for n,i in enumerate(all_data[:, col]):
            if count_map[i] <= 1:
                all_data[n, col] = uniques
            elif count_map[i] == 2:
                all_data[n, col] = uniques+1
    else:
        for n,i in enumerate(all_data[:, col]):
            if (all_data[:, col] == i).sum() <= 1:
                all_data[n, col] = uniques
            elif (all_data[:, col] == i).sum() == 2:
                all_data[n, col] = uniques+1
    uniques = len(set(all_data[:,col]))
    relabler.fit(all_data[:, col])
    all_data[:, col] = relabler.transform(all_data[:, col])

# Collect the training features together
y = np.array(train_data.ACTION)
y = np.reshape(y,(len(y),1))
y_train = y[:num_train]
y_test = y[num_train:]
X = all_data[:num_train]
X_2 = dp[:num_train]
X_3 = dt[:num_train]
# Collect the testing features together
X_test = all_data[num_train:]
X_test_2 = dp[num_train:]
X_test_3 = dt[num_train:]

X_train_all = np.hstack((X, X_2, X_3))
X_test_all = np.hstack((X_test, X_test_2, X_test_3))
num_features = X_train_all.shape[1]
    
if learner == 'NB':
    model = naive_bayes.BernoulliNB(alpha=0.03)
else:
    model = linear_model.LogisticRegression(class_weight='balanced', penalty='l2')
    
# Xts holds one hot encodings for each individual feature in memory
# speeding up feature selection 

if rank==0:
    print("Performing One Hot Encoding on entire dataset...")
Xt = np.vstack((X_train_all, X_test_all))
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(Xt)

X_train = Xt[:num_train]
X_test = Xt[num_train:]
encX_train = enc.transform(X_train[0:1]).toarray()

if rank==0:
    print("Training full model...")
    print("Making prediction and saving results...")

learn_decay = 0 # 1 if learning rate decay is applied (gamma = 1/(c1+tc2)), 0 otherwise
if learn_decay == 1:
    c1 = 10
    c2 = 0.01

else:
    gamma = 0.001
eta= 0.9 # momentum term (when eta=0, this scheme reduces to SGD)
model_dimension  = encX_train.shape[1] # 263500
if rank == 0:
    encX_test = enc.transform(X_test).toarray()

## Configuration
Byz = 5 #5 # number of Byzantine nodes (b)
attackType = 1 # 0: random, 1: directional (all-one), -1: reverse
ProbDesign = 1 # 1 if we use probabilistic code, and 0 otherwise (if test SignSGD-MV)
trialNum = 1 #20 # the number of trials in probabilistic code (for taking the average)
probEnc = 2/15 #0.9  #probability that an element of encoding matrix is 1 (p=r/n)
dummy = 4 * np.ones((1,numWorkers)) # 4 is chosen to distinguish from 0 and 1
auc = 0
if numWorkers == 15:
    batch_size = 15
elif numWorkers == 49:
    batch_size = 5 

for randomIdx in range(trialNum):
    if rank == 0:
        ## Probabilistic Coding (Encoding matrix)
        if ProbDesign == 1:
            EncMatrix = np.random.choice(2, size=(numWorkers,numWorkers), p=[1-probEnc, probEnc]) # encoding matrix (G) w/ average redundancy r=n*probEnc
        else:
            EncMatrix = np.identity(numWorkers)  # r=1 (uncoded)
            sumEnc = np.sum(EncMatrix)
        # Store EncMatrix in matapp variable..
        if randomIdx == 0:
            matapp = EncMatrix
        else:
            matapp = np.vstack((matapp, dummy))
            matapp = np.vstack((matapp, EncMatrix))
        print('EncMatrix: ', EncMatrix)
    else:
        EncMatrix = np.zeros((numWorkers, numWorkers))
    
    EncMatrix = comm.bcast(EncMatrix, root=0)

    # Data Allcation for Workers
    if rank != 0:  
        # extract the allocated data partition indices for a given rank
        Idxvec = np.nonzero(EncMatrix[rank - 1]) 
        Idxvec = np.asarray(Idxvec)
        if Idxvec.shape[1] != 0: # if at least one data is allocated to the worker 
            X_train_tensor = np.zeros((Idxvec.shape[1],(int)(num_train/numWorkers),Xt.shape[1])) 
            y_train_mat = np.zeros((Idxvec.shape[1],(int)(num_train/numWorkers))) 
        
            for Idx in range(Idxvec.shape[1]):
                dataIdx = Idxvec[0][Idx] # data index for worker with rank "rank"
                X_train_tensor[Idx] = Xt[dataIdx*int(num_train/numWorkers):(dataIdx+1)*int(num_train/numWorkers)]
                y_train = y_train.T
                y_train_mat[Idx] = y_train[0][dataIdx*int(num_train/numWorkers):(dataIdx+1)*int(num_train/numWorkers)]
                y_train = y_train.T    
            vel_mat = np.zeros((encX_train.shape[1], Idxvec.shape[1])) # velocity vector
    
    ## Syncronize the workers (and master)
    if rank == 0:
        dummyData = rank  
    else:
        dummyData = None
    
    if rank == 0:
        receive_buffer2 = np.zeros(((size-1),1))
    
        requests2 = [ MPI.REQUEST_NULL ] * (size-1)
        for sender in range(1,size):
            requests2[sender-1] = comm.Irecv(receive_buffer2[(sender-1):],source=sender)
    
        status2 = [ MPI . Status () for i in range (1 , size )]
        # Wait for all the messages
        MPI.Request.Waitall (requests2 , status2 ) # connection btw requests and status made at here...
    else:
        data_rank = np . array ([ rank ]) 
        request2 = comm.Isend ([ data_rank [:] , 1 , MPI . INT ], 0, rank )
        request2.Wait () 
    
    # broadcast dummyData to sync workers
    dummyData = comm.bcast(dummyData, root=0)
    
     
    ## Model update for each iteration 
    temp_time = 0
    NumEpoch = 8
    theta = np.zeros((encX_train.shape[1], 1)) # model parameter
    vel_agg = np.zeros((encX_train.shape[1], 1)) # aggregated velocity
    
    if rank == 0:
        ByzIdx = np.random.choice(numWorkers,Byz, replace = False) + 1 # randomly select 'Byz' Byzantines
    else:
        ByzIdx = []

    ByzIdx = comm.bcast(ByzIdx, root = 0)
    print('Byzantine nodes: ', ByzIdx)

    for epoch in range(NumEpoch): 
        for batchIdx in range(int(num_train/(size-1)/batch_size)):
            if rank == 0:
                start_time=time.time()
                receive_buffer = np.zeros((model_dimension * (size-1),1))    
                requests = [ MPI.REQUEST_NULL ] * (size-1)
                for sender in range(1,size):
                    requests[sender-1] = comm.Irecv(receive_buffer[model_dimension*(sender-1):(sender)*model_dimension],source=sender)                
                status = [ MPI . Status () for i in range (1 , size )]    
                # Wait for all the messages
                MPI.Request.Waitall (requests , status ) # connection btw requests and status made at here...
        
                vel_agg = np.zeros((model_dimension,1)) # aggregated velocity (sign)
                for sender2 in range(1,size):
                    vel_agg = vel_agg + receive_buffer[model_dimension*(sender2-1):(sender2)*model_dimension]
                vel_agg = np.sign(vel_agg) # take majority vote at master
            else :
                if Idxvec.shape[1] != 0: # if at least one data is allocated to the worker 
                    g_iter = np.zeros((model_dimension,1))
                    for partition in range(X_train_tensor.shape[0]):
                        X_train_temp = X_train_tensor[partition]
                        y_train_temp = y_train_mat[partition]
                        y_train_temp = np.reshape(y_train_temp,(len(y_train_temp),1))
                        encX1 = enc.transform(X_train_temp[batchIdx*batch_size:(batchIdx+1)*batch_size]).toarray()
                        g_iter = gradient(theta,encX1,np.array(y_train_temp[batchIdx*batch_size:(batchIdx+1)*batch_size])) # calculate gradient
                        
                        vel_mat[:,[partition]] = eta * vel_mat[:,[partition]] + (1-eta) * g_iter # update velocity 
                    vel_sign_mat = np.sign(vel_mat)
                    vel_sign = vel_sign_mat.sum(axis=1)
                    vel_sign = np.sign(vel_sign) # calculate the sign of velocity
                else: # no data partition received
                    vel_sign = np.zeros((model_dimension,1))
                if rank in ByzIdx: #rank == 1 or rank == 2:
                    if attackType == 0:
                        randVec = np.random.randint(2,size = (encX_train.shape[1],1))
                        randVec = 2 * randVec - 1
                        request = comm.Isend (randVec[:], 0, rank ) # random attack
                    elif attackType == -1:
                        request = comm.Isend (-vel_sign[:], 0, rank ) # reverse attack
                    elif attackType == 1:
                        allOneVec = np.ones((encX_train.shape[1],1))
                        request = comm.Isend (allOneVec[:], 0, rank ) # directional attack
                    else:
                        print('attackType is wrong')
                    request.Wait ()
                else:
                    request = comm.Isend (vel_sign[:], 0, rank )
                    request.Wait ()
                
            comm.Bcast(vel_agg, root=0) # broadcast the array from rank 0 to all others
            
            if learn_decay == 1:
                currStep = epoch * (int(num_train/(size-1)/batch_size)-1) + batchIdx
                gamma = 1/(c1 + currStep * c2)
            else:
                if epoch > 0:
                    gamma = 0.0001
            theta = theta - gamma * vel_agg # update parameter at all workers and master
            
            if rank == 0:
                elapsed_time =  time.time()-start_time
                temp_time = temp_time + elapsed_time
                preds = probability(theta,encX_test)
                auc = metrics.roc_auc_score(y_test, preds)
                content.append('%f, %f,%f' %(epoch, temp_time,auc))
                print('trial ', randomIdx, ',epoch:', epoch,', temp_time:', temp_time, ', auc:', auc) 

    content.append('epoch, time, auc')
    
if rank == 0:
    if learn_decay == 1:
        f = open('Trial_%d_NumEpoch_%d_ElectionCode_Batch_%d_c1_%d_c2_%f_Eta_%f_numWorkers_%d_Byz_%d_probEnc_%f_Reverse.csv' %(trialNum, NumEpoch, batch_size,c1,c2, eta, numWorkers, Byz, probEnc), 'w')
    else:
        if ProbDesign == 1:
            f = open('1st_Trial_%d_NumEpoch_%d_ElectionCode_Batch_%d_Gamma_%f_Eta_%f_numWorkers_%d_Byz_%d_probEnc_%f_attackType_%d.csv' %(trialNum, NumEpoch, batch_size,gamma, eta, numWorkers, Byz, probEnc, attackType), 'w')
        else:
            f = open('1st_Trial_%d_NumEpoch_%d_ElectionCode_Batch_%d_Gamma_%f_Eta_%f_numWorkers_%d_Byz_%d_sumEnc_%d_attackType_%d.csv' %(trialNum, NumEpoch, batch_size,gamma, eta, numWorkers, Byz, sumEnc, attackType), 'w')


    f.write('\n'.join(content))
    f.close()
    if ProbDesign == 1:
        np.savetxt("G_matrix_probEnc_%f.csv" %(probEnc), matapp, fmt='%d', delimiter=',')
    print("Saved")



