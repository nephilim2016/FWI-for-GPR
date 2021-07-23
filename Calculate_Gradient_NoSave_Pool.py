#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 01:20:43 2021

@author: nephilim
"""

from multiprocessing import Pool
import numpy as np
import time
import Add_CPML
import Wavelet
import Time_loop
import Reverse_time_loop

def calculate_gradient(sigma,epsilon,mu,index,CPML_Params,para):
    #Get Forward Params
    k_max=para.k_max
    ep0=8.841941282883074e-12
    t=np.arange(k_max)*para.dt
    f=Wavelet.ricker(t,para.Freq)
    #True Model Profile Data
    data=para.data[index]
    #Get Forward Data ----> <Generator>
    Forward_data=Time_loop.time_loop(para.xl,para.zl,para.dx,para.dz,para.dt,\
                                     sigma,epsilon,mu,CPML_Params,f,k_max,\
                                     para.source_site[index],para.receiver_site)
    #Get Generator Data
    For_data=[]
    idata=np.zeros((para.k_max,len(para.receiver_site)))
    for idx in range(para.k_max):
        tmp=Forward_data.__next__()
        For_data.append(np.array(tmp[0]))
        idata[idx,:]=tmp[1]
    #Get Residual Data
    rhs_data=idata-data
    #Get Reversion Data ----> <Generator>
    Reverse_data=Reverse_time_loop.reverse_time_loop(para.xl,para.zl,para.dx,para.dz,\
                                                     para.dt,sigma,epsilon,mu,CPML_Params,f,k_max,\
                                                     para.receiver_site,rhs_data)
    #Get Generator Data
    RT_data=[]
    for i in range(para.k_max):
        tmp=Reverse_data.__next__()
        RT_data.append(np.array(tmp[0]))
    RT_data.reverse()
    
    time_sum_eps=np.zeros((para.xl+2*CPML_Params.npml,para.zl+2*CPML_Params.npml))
    time_sum_sig=np.zeros((para.xl+2*CPML_Params.npml,para.zl+2*CPML_Params.npml))

    for k in range(1,k_max-1):
        u1=For_data[k+1]
        u0=For_data[k-1]
        u=For_data[k]
        p1=RT_data[k]
        time_sum_eps+=p1*(u1-u0)/para.dt/2
        time_sum_sig+=p1*u

    g_eps=ep0*time_sum_eps[10:-10,10:-10]
    g_sig=time_sum_sig[10:-10,10:-10]/para.beta
    
    g_eps[:10,:]=0
    g_sig[:10,:]=0

    return rhs_data.flatten(),g_sig.flatten(),g_eps.flatten()    

def misfit(sigma,epsilon,para): 
    sigma=sigma/para.beta
    mu=np.ones((para.xl,para.zl))
    start_time=time.time()  
    CPML_Params=Add_CPML.Add_CPML(para.xl,para.zl,sigma,epsilon,mu,para.dx,para.dz,para.dt)
    g_eps=0.0
    g_sig=0.0
    rhs=[]
    pool=Pool(processes=8)
    res_l=[]
    
    for index,value in enumerate(para.source_site):
        res=pool.apply_async(calculate_gradient,args=(sigma,epsilon,mu,index,CPML_Params,para))
        res_l.append(res)
    pool.close()
    pool.join()

    
    for res in res_l:
        result=res.get()
        rhs.append(result[0])
        g_sig+=result[1]
        g_eps+=result[2]
        del result
    rhs=np.array(rhs)        
    f=0.5*np.linalg.norm(rhs.flatten(),2)**2
    g=np.hstack((g_sig,g_eps))
    pool.terminate() 
#    print('**********',lambda_,'**********')
    print('Misfit elapsed time is %s seconds !'%str(time.time()-start_time))
    return f,g