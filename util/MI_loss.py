# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:09:51 2019

@author: 61995
"""

import sys

import torch
import torchvision.transforms.functional as TF

#def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=1e-6):
#  # has had softmax applied
#  _, k = x_out.size()
#  p_i_j = compute_joint(x_out, x_tf_out)
#
#
#  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
#  p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
#                                           k)  # but should be same, symmetric
#
#  # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
#  p_i_j[(p_i_j < EPS).data] = EPS
#  p_j[(p_j < EPS).data] = EPS
#  p_i[(p_i < EPS).data] = EPS
#
#  loss = - p_i_j * (torch.log(p_i_j) \
#                    - lamb * torch.log(p_j) \
#                    - lamb * torch.log(p_i))
#
#  loss = loss.sum()
#
#  loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
#                            - torch.log(p_j) \
#                            - torch.log(p_i))
#
#  loss_no_lamb = loss_no_lamb.sum()
#
#  return loss#, loss_no_lamb
#
#
#def compute_joint(x_out, x_tf_out):
#  # produces variable that requires grad (since args require grad)
#
#  bn, k = x_out.size()
#
#
#  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
#  p_i_j = p_i_j.sum(dim=0)  # k, k
#  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
#  p_i_j = p_i_j / p_i_j.sum()  # normalise
#
#  return p_i_j

def IID_loss(pi_x, pi_gx):
    _, k = pi_x.size()
    p = pi_x.t() @ pi_gx
    p = (p + p.t()) / 2
    #p[(p < 1e-6).data] = 1e-6
    p = torch.clamp(p,1e-6,3.4e+30)
    p = p/(p.sum())
    
    pi = p.sum(dim=0).view(k, 1).expand(k, k)
    pj = p.sum(dim=1).view(1, k).expand(k, k)
    loss = -((p * (torch.log(p) - torch.log(pi) - torch.log(pj))).sum())
    return loss
    

#       self.conv1 = nn.Conv2d(in_channels = 3, out_channels = num_channel, kernel_size = 3,)        
#        torch.nn.init.uniform(self.conv1.weight,-np.sqrt(6/(self.conv1.weight.shape[0]*self.conv1.weight.shape[1]*self.conv1.weight.shape[2])), np.sqrt(6/(self.conv1.weight.shape[0]*self.conv1.weight.shape[1]*self.conv1.weight.shape[2])))
#        self.conv2 = nn.Conv2d(in_channels = num_channel, out_channels = num_channel*2, kernel_size = 3,padding=1)
#        torch.nn.init.uniform(self.conv2.weight,-np.sqrt(6/(self.conv2.weight.shape[0]*self.conv2.weight.shape[1]*self.conv2.weight.shape[2])), np.sqrt(6/(self.conv2.weight.shape[0]*self.conv2.weight.shape[1]*self.conv2.weight.shape[2])))
#        self.conv3 = nn.Conv2d(in_channels = num_channel*2, out_channels = num_channel*4, kernel_size = 3,padding=1)
#        torch.nn.init.uniform(self.conv3.weight,-np.sqrt(6/(self.conv3.weight.shape[0]*self.conv3.weight.shape[1]*self.conv3.weight.shape[2])), np.sqrt(6/(self.conv3.weight.shape[0]*self.conv3.weight.shape[1]*self.conv3.weight.shape[2])))
#        self.conv4 = nn.Conv2d(in_channels = num_channel*4, out_channels = num_channel*4, kernel_size = 3,padding=1)
#        torch.nn.init.uniform(self.conv4.weight,-np.sqrt(6/(self.conv4.weight.shape[0]*self.conv4.weight.shape[1]*self.conv4.weight.shape[2])), np.sqrt(6/(self.conv4.weight.shape[0]*self.conv4.weight.shape[1]*self.conv4.weight.shape[2])))
#        self.conv5 = nn.Conv2d(in_channels = num_channel*4, out_channels = num_channel*8, kernel_size = 3,padding=1)
#        torch.nn.init.uniform(self.conv5.weight,-np.sqrt(6/(self.conv5.weight.shape[0]*self.conv5.weight.shape[1]*self.conv5.weight.shape[2])), np.sqrt(6/(self.conv5.weight.shape[0]*self.conv5.weight.shape[1]*self.conv5.weight.shape[2])))
#        self.conv6 = nn.Conv2d(in_channels = num_channel*8, out_channels = num_channel*8, kernel_size = 3,padding=1)
#        torch.nn.init.uniform(self.conv6.weight,-np.sqrt(6/(self.conv6.weight.shape[0]*self.conv6.weight.shape[1]*self.conv6.weight.shape[2])), np.sqrt(6/(self.conv6.weight.shape[0]*self.conv6.weight.shape[1]*self.conv6.weight.shape[2])))
#        self.conv7 = nn.Conv2d(in_channels = num_channel*8, out_channels = num_channel*8, kernel_size = 3,padding=1)
#        torch.nn.init.uniform(self.conv7.weight,-np.sqrt(6/(self.conv7.weight.shape[0]*self.conv7.weight.shape[1]*self.conv7.weight.shape[2])), np.sqrt(6/(self.conv7.weight.shape[0]*self.conv7.weight.shape[1]*self.conv7.weight.shape[2])))
#        self.conv8 = nn.Conv2d(in_channels = num_channel*8, out_channels = num_channel*8, kernel_size = 3,padding=1)
#        torch.nn.init.uniform(self.conv8.weight,-np.sqrt(6/(self.conv8.weight.shape[0]*self.conv8.weight.shape[1]*self.conv8.weight.shape[2])), np.sqrt(6/(self.conv8.weight.shape[0]*self.conv8.weight.shape[1]*self.conv8.weight.shape[2])))
#

#        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = num_channel, kernel_size = 3,)        
#        torch.nn.init.uniform(self.conv1.weight,-np.sqrt(6/(np.prod(np.array(self.conv1.weight.shape)))), np.sqrt(6/(np.prod(np.array(self.conv1.weight.shape)))))
#        self.conv2 = nn.Conv2d(in_channels = num_channel, out_channels = num_channel*2, kernel_size = 3,padding=1)
#        torch.nn.init.uniform(self.conv2.weight,-np.sqrt(6/(np.prod(np.array(self.conv2.weight.shape)))), np.sqrt(6/(np.prod(np.array(self.conv2.weight.shape)))))
#        self.conv3 = nn.Conv2d(in_channels = num_channel*2, out_channels = num_channel*4, kernel_size = 3,padding=1)
#        torch.nn.init.uniform(self.conv3.weight,-np.sqrt(6/(np.prod(np.array(self.conv3.weight.shape)))), np.sqrt(6/(np.prod(np.array(self.conv3.weight.shape)))))
#        self.conv4 = nn.Conv2d(in_channels = num_channel*4, out_channels = num_channel*4, kernel_size = 3,padding=1)
#        torch.nn.init.uniform(self.conv4.weight,-np.sqrt(6/(np.prod(np.array(self.conv4.weight.shape)))), np.sqrt(6/(np.prod(np.array(self.conv4.weight.shape)))))
#        self.conv5 = nn.Conv2d(in_channels = num_channel*4, out_channels = num_channel*8, kernel_size = 3,padding=1)
#        torch.nn.init.uniform(self.conv5.weight,-np.sqrt(6/(np.prod(np.array(self.conv5.weight.shape)))), np.sqrt(6/(np.prod(np.array(self.conv5.weight.shape)))))
#        self.conv6 = nn.Conv2d(in_channels = num_channel*8, out_channels = num_channel*8, kernel_size = 3,padding=1)
#        torch.nn.init.uniform(self.conv6.weight,-np.sqrt(6/(np.prod(np.array(self.conv6.weight.shape)))), np.sqrt(6/(np.prod(np.array(self.conv6.weight.shape)))))
#        self.conv7 = nn.Conv2d(in_channels = num_channel*8, out_channels = num_channel*8, kernel_size = 3,padding=1)
#        torch.nn.init.uniform(self.conv7.weight,-np.sqrt(6/(np.prod(np.array(self.conv7.weight.shape)))), np.sqrt(6/(np.prod(np.array(self.conv7.weight.shape)))))
#        self.conv8 = nn.Conv2d(in_channels = num_channel*8, out_channels = num_channel*8, kernel_size = 3,padding=1)
#        torch.nn.init.uniform(self.conv8.weight,-np.sqrt(6/(np.prod(np.array(self.conv8.weight.shape)))), np.sqrt(6/(np.prod(np.array(self.conv8.weight.shape)))))

