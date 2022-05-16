# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:43:21 2019

@author: 61995
"""

import torch
from sklearn.metrics.pairwise import rbf_kernel


def GaussianMatrix(X,sigma):
    G = torch.mm(X, X.T)
    K = 2*G-(torch.diag(G).reshape([1,G.size()[0]]))
    K = 1/(2*sigma**2)*(K-(torch.diag(G).reshape([G.size()[0],1])))
    K = torch.exp(K)
    
#    X = X.cpu().detach().numpy()
#    K = rbf_kernel(X,X,1/(2*sigma**2))
#    K = torch.tensor(K)
    
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

#def J_entripy(variable1,variable2,sigma1,sigma2,alpha):
#    input1 = variable1
#    K_x = GaussianMatrix(input1,sigma1)/(input1.size(dim=0))
#    L_x,_ = torch.symeig(K_x,eigenvectors=True)
#    lambda_x = torch.abs(L_x)
#    
#    #lambda_x = L_x
#    H_x = (1/(1-alpha))*torch.log((torch.sum(lambda_x ** alpha)))
#    
#    
#    
#    input2 = variable2
#    K_y = GaussianMatrix(input2,sigma2)/(input2.size(dim=0))
#    L_y,_ = torch.symeig(K_y,eigenvectors=True)
#    lambda_y = torch.abs(L_y)
#    #lambda_y = L_y
#    H_y = (1/(1-alpha))*torch.log((torch.sum(lambda_y ** alpha)))
#    
#    K_xy = K_x*K_y*(input1.size(dim=0))
#    K_xy = K_xy / torch.sum(torch.diag(K_xy))
#    
#    L_xy,_ = torch.symeig(K_xy,eigenvectors=True)
#    lambda_xy = torch.abs(L_xy)
#    #lambda_xy = L_xy
#    H_xy =  (1/(1-alpha))*torch.log((torch.sum(lambda_xy ** alpha)))
#    
#    #mutual_information = H_x + H_y - H_xy
#    return H_xy - H_x



#def J_entripy(variable1,variable2,sigma1,sigma2,alpha):
#    input1 = variable1
#    K_x = GaussianMatrix(input1,sigma1)/(input1.size(dim=0))
#    L_x,_ = torch.symeig(K_x,eigenvectors=True)
#    lambda_x = torch.abs(L_x)
#    
#    #lambda_x = L_x
#    H_x = (1/(1-alpha))*torch.log((torch.sum(lambda_x ** alpha)))
#    
#    
#    
#    input2 = variable2
#    K_y = GaussianMatrix(input2,sigma2)/(input2.size(dim=0))
#    L_y,_ = torch.symeig(K_y,eigenvectors=True)
#    lambda_y = torch.abs(L_y)
#    #lambda_y = L_y
#    H_y = (1/(1-alpha))*torch.log((torch.sum(lambda_y ** alpha)))
#    
#    K_xy = K_x*K_y*(input1.size(dim=0))
#    K_xy = K_xy / torch.sum(torch.diag(K_xy))
#    
#    L_xy,_ = torch.symeig(K_xy,eigenvectors=True)
#    lambda_xy = torch.abs(L_xy)
#    #lambda_xy = L_xy
#    H_xy =  (1/(1-alpha))*torch.log((torch.sum(lambda_xy ** alpha)))
#    
#    #mutual_information = H_x + H_y - H_xy
#    return H_x + H_y - H_xy  




def J_entripy_3(variable1,variable2,variable3,sigma1,sigma2,sigma3,alpha):
    input1 = variable1
    K_x = GaussianMatrix(input1,sigma1)/(input1.size(dim=0))
    L_x,_ = torch.symeig(K_x,eigenvectors=True)
    lambda_x = torch.abs(L_x)
    
    #lambda_x = L_x
    H_x = (1/(1-alpha))*torch.log2((torch.sum(lambda_x ** alpha)))
    
    
    
    input2 = variable2
    K_y = GaussianMatrix(input2,sigma2)/(input2.size(dim=0))
    L_y,_ = torch.symeig(K_y,eigenvectors=True)
    lambda_y = torch.abs(L_y)
    #lambda_y = L_y
    H_y = (1/(1-alpha))*torch.log2((torch.sum(lambda_y ** alpha)))
    
    
    
    input3 = variable3
    K_z = GaussianMatrix(input3,sigma3)/(input3.size(dim=0))
    L_z,_ = torch.symeig(K_z,eigenvectors=True)
    lambda_z = torch.abs(L_z)
    #lambda_y = L_y
    H_z = (1/(1-alpha))*torch.log2((torch.sum(lambda_z ** alpha)))
    
    
    K_xz = K_x*K_z*(input1.size(dim=0))
    K_xz = K_xz / torch.sum(torch.diag(K_xz))
    
    L_xz,_ = torch.symeig(K_xz,eigenvectors=True)
    lambda_xz = torch.abs(L_xz)
    #lambda_xy = L_xy
    H_xz =  (1/(1-alpha))*torch.log2((torch.sum(lambda_xz ** alpha)))
    
    
    K_xyz = K_x*K_y*K_z*(input1.size(dim=0))
    K_xyz = K_xyz / torch.sum(torch.diag(K_xyz))
    
    L_xyz,_ = torch.symeig(K_xyz,eigenvectors=True)
    lambda_xyz = torch.abs(L_xyz)
    #lambda_xy = L_xy
    H_xyz =  (1/(1-alpha))*torch.log2((torch.sum(lambda_xyz ** alpha)))
    
    #mutual_information = H_x + H_y - H_xy
    return H_y, (H_xyz - H_xz)

def J_entripy_3_21(variable1,variable2,variable3,sigma1,sigma2,sigma3,alpha):
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
    
    
    input4 = torch.cat((variable1,variable2),1)
    sigma4 = (input4.size()[0])**(-1/(4+(input4.size()[1])))
    
    
    input6 = torch.cat((variable1,variable2),1)
    sigma6 = (input6.size()[0])**(-1/(4+(input6.size()[1])))
    
    K_xy = GaussianMatrix(input6,sigma6)/(input6.size(dim=0))
    L_xy,_ = torch.symeig(K_xy,eigenvectors=True)
    lambda_xy = torch.abs(L_xy)
    #lambda_y = L_y
    H_xy = (1/(1-alpha))*torch.log((torch.sum(lambda_xy ** alpha)))
    
    
    K_xz = K_x*K_z*(input1.size(dim=0))
    K_xz = K_xz / torch.sum(torch.diag(K_xz))
    
    L_xz,_ = torch.symeig(K_xz,eigenvectors=True)
    lambda_xz = torch.abs(L_xz)
    #lambda_xy = L_xy
    H_xz =  (1/(1-alpha))*torch.log((torch.sum(lambda_xz ** alpha)))
    
    
    K_yz = K_y*K_z*(input2.size(dim=0))
    K_yz = K_yz / torch.sum(torch.diag(K_yz))
    
    L_yz,_ = torch.symeig(K_yz,eigenvectors=True)
    lambda_yz = torch.abs(L_yz)
    #lambda_xy = L_xy
    H_yz =  (1/(1-alpha))*torch.log((torch.sum(lambda_yz ** alpha)))
    
    
    K_xyz = K_xy*K_z*(input1.size(dim=0))
    K_xyz = K_xyz / torch.sum(torch.diag(K_xyz))
    
    L_xyz,_ = torch.symeig(K_xyz,eigenvectors=True)
    lambda_xyz = torch.abs(L_xyz)
    #lambda_xy = L_xy
    H_xyz =  (1/(1-alpha))*torch.log((torch.sum(lambda_xyz ** alpha)))
    
    
    #mutual_information = H_x + H_y - H_xy
    return H_xz+H_yz-H_xyz-H_z#
    
def J_entripy_5(variable1,variable2,variable3,variable4,variable5,sigma1,sigma2,sigma3,sigma4,sigma5,alpha):
    input1 = variable1
    K_x = GaussianMatrix(input1,sigma1)/(input1.size(dim=0))
    L_x,_ = torch.symeig(K_x,eigenvectors=True)
    lambda_x = torch.abs(L_x)
    
    #lambda_x = L_x
    H_x = (1/(1-alpha))*torch.log2((torch.sum(lambda_x ** alpha)))
    
    
    
    input2 = variable2
    K_y = GaussianMatrix(input2,sigma2)/(input2.size(dim=0))
    L_y,_ = torch.symeig(K_y,eigenvectors=True)
    lambda_y = torch.abs(L_y)
    #lambda_y = L_y
    H_y = (1/(1-alpha))*torch.log2((torch.sum(lambda_y ** alpha)))
    
    
    
    input3 = variable3
    K_z = GaussianMatrix(input3,sigma3)/(input3.size(dim=0))
    L_z,_ = torch.symeig(K_z,eigenvectors=True)
    lambda_z = torch.abs(L_z)
    #lambda_y = L_y
    H_z = (1/(1-alpha))*torch.log2((torch.sum(lambda_z ** alpha)))
    
    
    input4 = variable4
    K_w = GaussianMatrix(input4,sigma4)/(input4.size(dim=0))
    L_w,_ = torch.symeig(K_w,eigenvectors=True)
    lambda_w = torch.abs(L_w)
    #lambda_y = L_y
    H_w = (1/(1-alpha))*torch.log2((torch.sum(lambda_w ** alpha)))
    
    input5 = variable5
    K_a = GaussianMatrix(input5,sigma5)/(input5.size(dim=0))
    L_a,_ = torch.symeig(K_a,eigenvectors=True)
    lambda_a = torch.abs(L_a)
    #lambda_y = L_y
    H_a = (1/(1-alpha))*torch.log2((torch.sum(lambda_a ** alpha)))
    
    
    input6 = torch.cat((variable1,variable2),1)
    sigma6 = (input6.size()[0])**(-1/(4+(input6.size()[1])))
    
    K_xy = GaussianMatrix(input6,sigma6)/(input6.size(dim=0))
    L_xy,_ = torch.symeig(K_xy,eigenvectors=True)
    lambda_xy = torch.abs(L_xy)
    #lambda_y = L_y
    H_xy = (1/(1-alpha))*torch.log2((torch.sum(lambda_xy ** alpha)))
    
    

    
    
    K_xzw = K_x*K_z*K_w*(input1.size(dim=0))
    K_xzw = K_xzw / torch.sum(torch.diag(K_xzw))
    
    L_xzw,_ = torch.symeig(K_xzw,eigenvectors=True)
    lambda_xzw = torch.abs(L_xzw)
    #lambda_xy = L_xy
    H_xzw =  (1/(1-alpha))*torch.log2((torch.sum(lambda_xzw ** alpha)))
    
    
    
    K_yzw = K_y*K_z*K_w*(input2.size(dim=0))
    K_yzw = K_yzw / torch.sum(torch.diag(K_yzw))
    
    L_yzw,_ = torch.symeig(K_yzw,eigenvectors=True)
    lambda_yzw = torch.abs(L_yzw)
    #lambda_xy = L_xy
    H_yzw =  (1/(1-alpha))*torch.log2((torch.sum(lambda_yzw ** alpha)))
    
    K_zw = K_z*K_w*(input3.size(dim=0))
    K_zw = K_zw / torch.sum(torch.diag(K_zw))
    
    L_zw,_ = torch.symeig(K_zw,eigenvectors=True)
    lambda_zw = torch.abs(L_zw)
    #lambda_xy = L_xy
    H_zw =  (1/(1-alpha))*torch.log2((torch.sum(lambda_zw ** alpha)))
    
    

    
    
    
    K_ya = K_y*K_a*(input2.size(dim=0))
    K_ya = K_ya / torch.sum(torch.diag(K_ya))
    
    L_ya,_ = torch.symeig(K_ya,eigenvectors=True)
    lambda_ya = torch.abs(L_ya)
    #lambda_xy = L_xy
    H_ya =  (1/(1-alpha))*torch.log2((torch.sum(lambda_ya ** alpha)))

    
    
#    K_xyzw = K_x*K_y*K_z*K_w*(input1.size(dim=0))
#    K_xyzw = K_xyzw / torch.sum(torch.diag(K_xyzw))
#    
#    L_xyzw,_ = torch.symeig(K_xyzw,eigenvectors=True)
#    lambda_xyzw = torch.abs(L_xyzw)
#    #lambda_xy = L_xy
#    H_xyzw =  (1/(1-alpha))*torch.log((torch.sum(lambda_xyzw ** alpha)))
    
    
    
    K_xyzw = K_xy*K_z*K_w*(input1.size(dim=0))
    K_xyzw = K_xyzw / torch.sum(torch.diag(K_xyzw))
    
    L_xyzw,_ = torch.symeig(K_xyzw,eigenvectors=True)
    lambda_xyzw = torch.abs(L_xyzw)
    #lambda_xy = L_xy
    H_xyzw =  (1/(1-alpha))*torch.log2((torch.sum(lambda_xyzw ** alpha)))
    
    
    #mutual_information = H_x + H_y - H_xy
    return (H_xzw+H_yzw-H_xyzw-H_zw),H_y#


def J_entripy_4(variable1,variable2,variable3,variable4,sigma1,sigma2,sigma3,sigma4,alpha):
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
    
    
    input4 = variable4
    K_w = GaussianMatrix(input4,sigma4)/(input4.size(dim=0))
    L_w,_ = torch.symeig(K_w,eigenvectors=True)
    lambda_w = torch.abs(L_w)
    #lambda_y = L_y
    H_w = (1/(1-alpha))*torch.log((torch.sum(lambda_w ** alpha)))
    

    
    
    input6 = torch.cat((variable1,variable2),1)
    sigma6 = (input6.size()[0])**(-1/(4+(input6.size()[1])))
    
    K_xy = GaussianMatrix(input6,sigma6)/(input6.size(dim=0))
    L_xy,_ = torch.symeig(K_xy,eigenvectors=True)
    lambda_xy = torch.abs(L_xy)
    #lambda_y = L_y
    H_xy = (1/(1-alpha))*torch.log((torch.sum(lambda_xy ** alpha)))
    
    
    K_xzw = K_x*K_z*K_w*(input1.size(dim=0))
    K_xzw = K_xzw / torch.sum(torch.diag(K_xzw))
    
    L_xzw,_ = torch.symeig(K_xzw,eigenvectors=True)
    lambda_xzw = torch.abs(L_xzw)
    #lambda_xy = L_xy
    H_xzw =  (1/(1-alpha))*torch.log((torch.sum(lambda_xzw ** alpha)))
    
    
    
    K_yzw = K_y*K_z*K_w*(input2.size(dim=0))
    K_yzw = K_yzw / torch.sum(torch.diag(K_yzw))
    
    L_yzw,_ = torch.symeig(K_yzw,eigenvectors=True)
    lambda_yzw = torch.abs(L_yzw)
    #lambda_xy = L_xy
    H_yzw =  (1/(1-alpha))*torch.log((torch.sum(lambda_yzw ** alpha)))
    
    K_zw = K_z*K_w*(input3.size(dim=0))
    K_zw = K_zw / torch.sum(torch.diag(K_zw))
    
    L_zw,_ = torch.symeig(K_zw,eigenvectors=True)
    lambda_zw = torch.abs(L_zw)
    #lambda_xy = L_xy
    H_zw =  (1/(1-alpha))*torch.log((torch.sum(lambda_zw ** alpha)))
    
    


    
    
    K_xyzw = K_x*K_y*K_z*K_w*(input1.size(dim=0))
    K_xyzw = K_xyzw / torch.sum(torch.diag(K_xyzw))
    
    L_xyzw,_ = torch.symeig(K_xyzw,eigenvectors=True)
    lambda_xyzw = torch.abs(L_xyzw)
    #lambda_xy = L_xy
    H_xyzw =  (1/(1-alpha))*torch.log((torch.sum(lambda_xyzw ** alpha)))
    
    
    
#    K_xyzw = K_xy*K_z*K_w*(input1.size(dim=0))
#    K_xyzw = K_xyzw / torch.sum(torch.diag(K_xyzw))
#    
#    L_xyzw,_ = torch.symeig(K_xyzw,eigenvectors=True)
#    lambda_xyzw = torch.abs(L_xyzw)
#    #lambda_xy = L_xy
#    H_xyzw =  (1/(1-alpha))*torch.log((torch.sum(lambda_xyzw ** alpha)))
    
    
    #mutual_information = H_x + H_y - H_xy
    return (H_xzw+H_yzw-H_xyzw-H_zw)#


#m = MI(variable1,variable2,sigma1,sigma2,alpha)