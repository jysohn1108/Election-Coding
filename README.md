# Election Coding

This repository is the official implementation for the following NeurIPS 2020 paper:

[Election Coding for Distributed Learning: Protecting SignSGD Against Byzantine Attacks](https://arxiv.org/abs/1910.06093).

## Overview
Current distributed learning systems suffer from serious performance degradation under Byzantine attacks. This paper proposes Election Coding, a codingtheoretic
framework to guarantee Byzantine-robustness for distributed learning algorithms based on signed stochastic gradient descent (SignSGD) that minimizes the worker-master communication load. The suggested framework explores new information-theoretic limits of finding the majority opinion when some workers could be attacked by adversary, and paves the road to implement robust and communication-efficient distributed learning algorithms. Under this framework, we construct two types of codes, random Bernoulli codes and deterministic algebraic codes, that tolerate Byzantine attacks with a controlled amount of computational redundancy and guarantee convergence in general non-convex scenarios. For the Bernoulli codes, we provide an upper bound on the error probability in estimating the signs of the true gradients, which gives useful insights into code design for Byzantine tolerance. The proposed deterministic codes are proven to perfectly tolerate arbitrary Byzantine attacks. Experiments on real datasets confirm that the suggested codes provide substantial improvement in Byzantine tolerance of distributed learning systems employing SignSGD.


## Depdendencies 
Tested stable depdencises:
* python 3.7.7 (Anaconda)
* PyTorch 1.6.0
* torchvision 0.7.0
* CUDA 10.2.89
* cuDNN 7.6.5
* MPI4Py 3.0.3


## Training (+ evaluation)
Please refer to README.md at each sub-folder for each dataset (CIFAR-10, [Amazon Employee Access](https://www.kaggle.com/c/amazon-employee-access-challenge)).

