# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:27:40 2021

@author: 61995
"""

import cv2

import scipy.io as scio
#from VGG11 import *
from skimage import io
#from data_neighbor_sobel import *
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torch

import torch.nn.functional as F
from util.MI_estimator_s import *
N= 512


dmi_all = []
dmi_all_r = []
s = 0.01

for er in range(1):
    

    x1 =[]
    x2 =[]
    x3 =[]

    
    for i in range(2):
        x1t = int(np.random.normal())
        x2t = int(np.random.normal())
        x3t = int(np.random.normal())

    
        x1.append(x1t)
        x2.append(x2t)
        x3.append(x3t)

    a = 1.8
    for i in range(N+100):
        x1t = 0.1*x1[-1]+s*(np.random.normal())#(np.random.normal())#3.4*x1[-1]*(1-(x1[-1])**2)*np.exp(-(x1[-1])**2)+s*(np.random.normal())
        x2t = 0.1*x2[-1]+3*x1[-1]+s*(np.random.normal())
        x3t = 0.1*x3[-1]+2*x2[-1]+5*x1[-1]+s*(np.random.normal())#+0.4*x1[-2]#+int(bool(x1[-1]) != bool(x1[-2]))

        x1.append(x1t)
        x2.append(x2t)
        x3.append(x3t)

    x1 = x1[100:]
    x2 = x2[100:]
    x3 = x3[100:]

#
    x1 = (x1-np.min(x1))/(np.max(x1)-np.min(x1))
    x2 = (x2-np.min(x2))/(np.max(x2)-np.min(x2))
    x3 = (x3-np.min(x3))/(np.max(x3)-np.min(x3))

    xt_ = []
    x1_ = np.array(x1[:])
    x2_ = np.array(x2[:])
    x3_ = np.array(x3[:])

    xt_.append(x1_)
    xt_.append(x2_)
    xt_.append(x3_)

    
    x_t = []
    x_1 = np.array(x1[1:])
    x_2 = np.array(x2[1:])
    x_3 = np.array(x3[1:])

    
    x_t.append(x_1)
    x_t.append(x_2)
    x_t.append(x_3)

seq_list = np.arange(3)    
dmi_all = [] 

dmi_y = []   
lag = 1
for all_i in range(3):  
    for all_j in range(3):
        print(all_j)
        dmi = []
        dma = []
        for delay in range(lag):
            delay+=1
            
            #delay = 5
        
            dmi_r =[]
            dmi_a =  []
            for idx_label in range(1):#
         
                label_all =[]
                bag_all = []
                label_d = []
                side_d = []
                d_all = []

                for idx2 in range(N-0-lag+1):
                    
                    side_info = []
                    all_info = []
                    
                    bag_all.append(xt_[all_i][idx2:idx2+delay])
                    #label_all.append(x_t[all_j][idx2:idx2+1])
                    label_all.append(xt_[all_j][idx2+delay])
                    label_d.append(xt_[all_j][idx2:idx2+delay])
                    for idx3 in seq_list:
                        all_info.append(xt_[idx3][idx2:idx2+delay])
                        
                        if (idx3 != all_j) and (idx3 != all_i):
                            side_info.append(xt_[idx3][idx2:idx2+delay])
                    side_info = np.array(side_info).flatten()
                    side_d.append(side_info)
                    
                    all_info = np.array(all_info).flatten()
                    d_all.append(all_info)
        
                
                bag_all = torch.Tensor(np.array(bag_all).reshape([len(bag_all),delay]))
                label_all = torch.Tensor(np.array(label_all).reshape([len(label_all),1]))
                label_d = torch.Tensor(np.array(label_d).reshape([len(label_d),delay]))
                side_d = torch.Tensor(np.array(side_d).reshape([len(side_d),len(side_d[0])]))
                d_all = torch.Tensor(np.array(d_all).reshape([len(d_all),len(d_all[0])]))
                
                
                
                sigma1 = (bag_all.size()[0])**(-1/(4+(bag_all.size()[1])))
                sigma2 = (label_all.size()[0])**(-1/(4+(label_all.size()[1])))
                sigma3 = (label_d.size()[0])**(-1/(4+(label_d.size()[1])))
                sigma4 = (side_d.size()[0])**(-1/(4+(side_d.size()[1])))
                sigma5 = (d_all.size()[0])**(-1/(4+(d_all.size()[1])))
                
                
                
                bag_all = torch.Tensor(np.array(bag_all).reshape([len(bag_all),delay]))
                label_all = torch.Tensor(np.array(label_all).reshape([len(label_all),1]))
                label_d = torch.Tensor(np.array(label_d).reshape([len(label_d),delay]))
                side_d = torch.Tensor(np.array(side_d).reshape([len(side_d),len(side_d[0])]))
                d_all = torch.Tensor(np.array(d_all).reshape([len(d_all),len(d_all[0])]))
                
                start = time.time()
                #je = J_entripy(bag_all,label_all,sigma1,sigma2,1.01).cpu().detach().numpy()
#                Hy,je = J_entripy_3(bag_all,label_all,label_d,sigma1,sigma2,sigma3,1.01)
#                Hy =  Hy.cpu().detach().numpy()
#                je =  je.cpu().detach().numpy()
#                
#                ce = J_entripy(label_d,label_all,sigma1,sigma2,1.01)
#                ce =  ce.cpu().detach().numpy()
                je,ja = J_entripy_5(bag_all,label_all,label_d,side_d,d_all,sigma1,sigma2,sigma3,sigma4,sigma5,1.01)
#                je = J_entripy_3(bag_all,label_all,label_d,sigma1,sigma2,sigma3,1.01)
#                ja = J_entripy_3(bag_all,label_all,label_d,sigma1,sigma2,sigma3,1.01)
                je =  je.cpu().detach().numpy()
                ja =  ja.cpu().detach().numpy()
                
                
                
                
                stop = time.time()
                #print("Salience Map Generation: ", stop - start, " seconds")
                dmi_r.append(je)
                dmi_a.append(ja)
        
            if delay == 1:
                dmi = np.array(dmi_r)#/delay
                dma = np.array(dmi_a)
            else:
                dmi = dmi + np.array(dmi_r)#/delay
                #dma = dma + np.array(dmi_a)
        #dmi_all.append(1+dmi/ce)
        #dmi_all.append(Hy + dmi)
        dmi_all.append((dmi+ja))
        dmi_y.append(ja)
    
dmi_all = np.array(dmi_all).reshape([3,3])
for i in range(3):
    dmi_all[:,i] = dmi_all[:,i]-dmi_all[i,i]
dmi_all = (dmi_all-np.min(dmi_all))/(np.max(dmi_all)-np.min(dmi_all))
plt.imshow(dmi_all)
#%%

scio.savemat('x1_l.mat',{"x1": x1})
scio.savemat('x2_l.mat',{"x2": x2})
scio.savemat('x3_l.mat',{"x3": x3})