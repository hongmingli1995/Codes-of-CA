# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:43:21 2019

@author: 61995
"""

import torch

def GaussianMatrix(X,sigma):
    G = torch.mm(X, X.T)
    K = 2*G-(torch.diag(G).reshape([1,G.size()[0]]))
    K = 1/(2*sigma**2)*(K-(torch.diag(G).reshape([G.size()[0],1])))
    K = torch.exp(K)
    
    return K



#A = torch.empty([128,512]).normal_(0,1)
#b = GaussianMatrix(A,1)
#b2 = torch.symeig(b)
##variable1,variable2,sigma1,sigma2,alpha
#variable1 = torch.ones([660,512]).normal_(0,0.1).cuda()
#sigma1 = 1
#variable2 = (torch.ones([660,512]).normal_(0,0.1)).cuda()
#sigma2 = 1
#alpha = 2

def MI(variable1,variable2,sigma1,sigma2,alpha):
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
    
    mutual_information = H_x + H_y - H_xy
    return mutual_information

def J_entripy(variable1,variable2,sigma1,sigma2,alpha):
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
    
    #mutual_information = H_x + H_y - H_xy
    return H_xy - H_x - H_y
    
    
    
def J_entripy_3(variable1,variable2,variable3,sigma1,sigma2,sigma3,alpha):
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
    
    
    
    input3 = variable3
    K_z = GaussianMatrix(input3,sigma3)/(input3.size(dim=0))
    L_z,_ = torch.symeig(K_z,eigenvectors=True)
    lambda_z = torch.abs(L_z)
    #lambda_y = L_y
    H_z = (1/(1-alpha))*torch.log((torch.sum(lambda_z ** alpha)))
    
    
    K_xz = K_x*K_z*(input1.size(dim=0))
    K_xz = K_xz / torch.sum(torch.diag(K_xz))
    
    L_xz,_ = torch.symeig(K_xz,eigenvectors=True)
    lambda_xz = torch.abs(L_xz)
    #lambda_xy = L_xy
    H_xz =  (1/(1-alpha))*torch.log((torch.sum(lambda_xz ** alpha)))
    
    
    K_xyz = K_x*K_y*K_z*(input1.size(dim=0))
    K_xyz = K_xyz / torch.sum(torch.diag(K_xyz))
    
    L_xyz,_ = torch.symeig(K_xyz,eigenvectors=True)
    lambda_xyz = torch.abs(L_xyz)
    #lambda_xy = L_xy
    H_xyz =  (1/(1-alpha))*torch.log((torch.sum(lambda_xyz ** alpha)))
    
    #mutual_information = H_x + H_y - H_xy
    return H_y - (H_xyz - H_xz)


#m = MI(variable1,variable2,sigma1,sigma2,alpha)