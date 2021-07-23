#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:58:37 2018

@author: nephilim
"""
from numba import jit
import numpy as np
from skimage import filters
import scipy.io as scio
#
@jit(nopython=True)
def Abnormal_Model(xl,zl,CPML):
    beta_=377
    epsilon=np.ones((xl+2*CPML,zl+2*CPML))*4
    sigma=np.ones((xl+2*CPML,zl+2*CPML))*0.003*beta_
    p=34+CPML
    l=14
    w=3
    for i in range(p-l,p+l):
        for j in range(p-w,p+w):
            epsilon[i][j]=1
            sigma[i][j]=0*beta_
    for i in range(p-w,p+w):
        for j in range(p-l,p+l):
            epsilon[i][j]=1
            sigma[i][j]=0*beta_
    p = zl+2*CPML-p
    for i in range(p-l,p+l):
        for j in range(p-w,p+w):
            epsilon[i][j]=8
            sigma[i][j]=0.01*beta_
    for i in range(p-w,p+w):
        for j in range(p-l,p+l):
            epsilon[i][j]=8
            sigma[i][j]=0.01*beta_
    return epsilon,sigma

@jit(nopython=True)
def Create_iModel(xl,zl,CPML):
    beta_=377
    epsilon=np.ones((xl+2*CPML,zl+2*CPML))*4
    sigma=np.ones((xl+2*CPML,zl+2*CPML))*0.003*beta_
    return epsilon,sigma

def Overthrust_Model(xl,zl,CPML):
    beta_=377
    epsilon=np.ones((xl+2*CPML,zl+2*CPML))
    sigma=np.ones((xl+2*CPML,zl+2*CPML))
    data_=scio.loadmat('./overthrust.mat')
    data_sigma=data_['sigma']*beta_
    data_epsilon=data_['ep']
    sigma[10:-10,10:-10]=data_sigma
    epsilon[10:-10,10:-10]=data_epsilon
    
    epsilon[:CPML,:]=epsilon[CPML,:]
    epsilon[-CPML:,:]=epsilon[-CPML-1,:]
    epsilon[:,:CPML]=epsilon[:,CPML].reshape((len(epsilon[:,CPML]),-1))
    epsilon[:,-CPML:]=epsilon[:,-CPML-1].reshape((len(epsilon[:,-CPML-1]),-1))
    
    sigma[:CPML,:]=sigma[CPML,:]
    sigma[-CPML:,:]=sigma[-CPML-1,:]
    sigma[:,:CPML]=sigma[:,CPML].reshape((len(sigma[:,CPML]),-1))
    sigma[:,-CPML:]=sigma[:,-CPML-1].reshape((len(sigma[:,-CPML-1]),-1))
    
    return epsilon,sigma

@jit(nopython=True)
def Overthrust_iModel(xl,zl,CPML):
    beta_=377
    epsilon=np.ones((xl+2*CPML,zl+2*CPML))
    sigma=np.ones((xl+2*CPML,zl+2*CPML))
    data_=scio.loadmat('./overthrust.mat')
    data_sigma=data_['sigma1']*beta_
    data_epsilon=data_['ep1']
    sigma[10:-10,10:-10]=data_sigma
    epsilon[10:-10,10:-10]=data_epsilon
    
    epsilon[:CPML,:]=epsilon[CPML,:]
    epsilon[-CPML:,:]=epsilon[-CPML-1,:]
    epsilon[:,:CPML]=epsilon[:,CPML].reshape((len(epsilon[:,CPML]),-1))
    epsilon[:,-CPML:]=epsilon[:,-CPML-1].reshape((len(epsilon[:,-CPML-1]),-1))
    
    sigma[:CPML,:]=sigma[CPML,:]
    sigma[-CPML:,:]=sigma[-CPML-1,:]
    sigma[:,:CPML]=sigma[:,CPML].reshape((len(sigma[:,CPML]),-1))
    sigma[:,-CPML:]=sigma[:,-CPML-1].reshape((len(sigma[:,-CPML-1]),-1))
    return epsilon,sigma

# @jit(nopython=True)
def Tunnel_Model(xl,zl,CPML):
    epsilon=np.ones((xl+2*CPML,zl+2*CPML))*4
    sigma=np.ones((xl+2*CPML,zl+2*CPML))*0.0*377
    ep=np.load('epsilon_complex.npy')
    epsilon[CPML:-CPML,CPML:-CPML]=ep
    epsilon[:CPML,:]=epsilon[CPML,:]
    epsilon[-CPML:,:]=epsilon[-CPML-1,:]
    epsilon[:,:CPML]=epsilon[:,CPML].reshape((len(epsilon[:,CPML]),-1))
    epsilon[:,-CPML:]=epsilon[:,-CPML-1].reshape((len(epsilon[:,-CPML-1]),-1))
    return epsilon,sigma

#Create Initial_Overthrust Model
def Initial_Smooth_Model(epsilon_,sigma_,sig):
    iepsilon=filters.gaussian(epsilon_,sigma=sig)
    isigma=filters.gaussian(sigma_,sigma=sig)
    iepsilon[:10,:]=1
    isigma[:10,:]=0
    return iepsilon,isigma