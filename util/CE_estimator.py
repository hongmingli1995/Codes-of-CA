# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:43:21 2019

@author: 61995
"""

import torch
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import matplotlib.pyplot as plt
#Recommendation sigma: sigma = (input.size()[0])**(-1/(4+(input.size()[1])))
#variable1: past samples X^(t-1) size:(batch_size,lag)
#variable2: current samples X(t) size:(batch_size,1)

import scipy.io as scio

data = scio.loadmat('mp3_data.mat')
vd = data['y']
#vd[:,0] = (vd[:,0] - (vd[:,0]).min())/((vd[:,0]).max()-(vd[:,0]).min())
#vd[:,1] = (vd[:,1] - (vd[:,1]).min())/((vd[:,1]).max()-(vd[:,1]).min())
vd = vd.flatten()


def GaussianMatrix(X,sigma):
    G = torch.mm(X, X.T)
    K = 2*G-(torch.diag(G).reshape([1,G.size()[0]]))
    K = 1/(2*sigma**2)*(K-(torch.diag(G).reshape([G.size()[0],1])))
    K = torch.exp(K)
    
#    X = X.cpu().detach().numpy()
#    K = rbf_kernel(X,X,1/(2*sigma**2))
#    K = torch.tensor(K)
    
    return K









def CE(variable1,variable2,sigma1,sigma2,alpha):
    input1 = variable1
    K_x = GaussianMatrix(input1,sigma1)/(input1.size(dim=0))
    L_x,_ = torch.symeig(K_x,eigenvectors=True)
    lambda_x = torch.abs(L_x)
    
    #lambda_x = L_x
    H_x = (1/(1-alpha))*torch.log((torch.sum(lambda_x ** alpha)))
    
    
    
    input2 = variable2
    K_y = GaussianMatrix(input2,sigma2)/(input2.size(dim=0))
    L_y,_ = torch.symeig(K_y,eigenvectors=True)
    lambda_y = torch.abs(L_y)
    #lambda_y = L_y
    H_y = (1/(1-alpha))*torch.log((torch.sum(lambda_y ** alpha)))
    
    K_xy = K_x*K_y*(input1.size(dim=0))
    K_xy = K_xy / torch.sum(torch.diag(K_xy))
    
    L_xy,_ = torch.symeig(K_xy,eigenvectors=True)
    lambda_xy = torch.abs(L_xy)
    #lambda_xy = L_xy
    H_xy =  (1/(1-alpha))*torch.log((torch.sum(lambda_xy ** alpha)))
    
    
    return H_xy - H_x

ce_l = []
s_l =[]
for i in range(8000):#s = 0.2
    s = i
    lag =100

    #plt.plot(x)
    #plt.show()
    y = vd[int(i*1024):int(i*1024)+1024]
#    plt.plot(y)
#    plt.show()
    if i%10 == 0:
        print(i)
    
    label_all =[]
    bag_all = []
    for idx2 in range(len(y)-lag-1):
        bag_all.append(y[idx2:idx2+lag])
        label_all.append(y[idx2+lag])
    bag_all = torch.Tensor(np.array(bag_all).reshape([len(bag_all),lag]))
    label_all = torch.Tensor(np.array(label_all).reshape([len(label_all),1]))
    sigma1 = (bag_all.size()[0])**(-1/(4+(bag_all.size()[1])))
    sigma2 = (label_all.size()[0])**(-1/(4+(label_all.size()[1])))
    ce = CE(bag_all,label_all,sigma1,sigma2,1.01)
    ce =  ce.cpu().detach().numpy()
    ce_l.append(ce)
    s_l.append(s)
#%%    
plt.plot(s_l,ce_l)
plt.xlabel('time')
plt.ylabel('CE')

plt.show()

