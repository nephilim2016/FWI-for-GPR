#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 00:44:55 2021

@author: nephilim
"""

import Forward2D
import Calculate_Gradient_NoSave_Pool
import numpy as np
import time
from Optimization import para,options,Optimization
from pathlib import Path
from matplotlib import pyplot,cm
from skimage import filters
import Create_Model

if __name__=='__main__':  
    start_time=time.time() 
    #Multiscale
    frequence=[0.6e8,1e8,4e8]
    for idx,freq_ in enumerate(frequence):
        # if idx==1:
        #     continue
        #Model Params
        print(freq_)
        para.xl=111
        para.zl=201
        para.dx=0.05
        para.dz=0.05
        para.k_max=3000
        para.dt=5e-11
        para.beta=377
        #Ricker wavelet main frequence
        para.Freq=freq_
        #True Model
        # epsilon=4*np.ones((para.xl,para.zl))
        # epsilon[40:50,40:60]=1
        # sigma=1e-5*np.ones((para.xl,para.zl))
        epsilon,sigma=Create_Model.Overthrust_Model(para.xl,para.zl,10)
        epsilon=epsilon[10:-10,10:-10]
        sigma=sigma[10:-10,10:-10]
        mu=np.ones((para.xl,para.zl))
        #Source Position
        para.source_site=[]
        para.receiver_site=[]
        
        para.source_site=[]
        para.receiver_site=[]
        for index in range(10,211,10):
            para.source_site.append((10,index))
        
        for index in range(10,211,1):
            para.receiver_site.append((10,index))
        para.receiver_site=np.array(para.receiver_site)
        #Get True Model Data
        Forward2D.Forward_2D(sigma,epsilon,mu,para)
        
        print('Forward Done !')
        print('Elapsed time is %s seconds !'%str(time.time()-start_time))
        #Save Profile Data
        para.data=[]
        for i in range(len(para.source_site)):
            data=np.load('./%sHz_forward_data_file/%sx_%sz_record.npy'%(para.Freq,para.source_site[i][0],para.source_site[i][1]))
            para.data.append(data)
        #If the first frequence,Create Initial Model
        if idx==0:
            iepsilon,isigma=Create_Model.Initial_Smooth_Model(epsilon,sigma,sig=10)
            # iepsilon=iepsilon[10:-10,10:-10]
            # isigma=isigma[10:-10,10:-10]
        #If the first frequence,Using the last final model
        else:
            dir_path='./%sHz_imodel_file'%frequence[idx-1]
            file_num=int(len(list(Path(dir_path).iterdir())))-1
            data=np.load('./%sHz_imodel_file/%s_imodel.npy'%(frequence[idx-1],file_num))
            iepsilon=data.reshape((para.xl,-1))
        # Anonymous function for Gradient Calculate
        fh=lambda x,y:Calculate_Gradient_NoSave_Pool.misfit(x,y,para)    
        
        # # Test Gradient
        # f,g=Calculate_Gradient_NoSave_Pool.misfit(isigma,iepsilon,para)
        # pyplot.figure()
        # pyplot.imshow(g[:int(len(g)/2)].reshape((111,-1)))
        # pyplot.colorbar()
        # pyplot.figure()
        # pyplot.imshow(g[int(len(g)/2):].reshape((111,-1)))
        # pyplot.colorbar()
        
        #Options Params
        options.method='lbfgs'
        options.tol=1e-4
        options.maxiter=50
        Optimization_=Optimization(fh,isigma,iepsilon)
        imodel,info=Optimization_.optimization()
        

        pyplot.figure()
        pyplot.imshow(imodel[:int(len(imodel)/2)].reshape((para.xl,-1)),cmap=cm.seismic)
        pyplot.colorbar()
        
        pyplot.figure()
        pyplot.imshow(imodel[int(len(imodel)/2):].reshape((para.xl,-1)),cmap=cm.seismic)
        pyplot.colorbar()
        #Plot Error Data
        pyplot.figure()
        data_=[]
        for info_ in info:
            data_.append(info_[3])
        pyplot.plot(data_/data_[0])
        pyplot.yscale('log')
    print('Elapsed time is %s seconds !'%str(time.time()-start_time))