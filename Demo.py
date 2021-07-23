#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 19:32:53 2021

@author: nephilim
"""

from multiprocessing import Pool
import numpy as np
import Add_CPML
import Time_loop
import Wavelet
import time
    
def Forward_2D(xl,zl,Freq,sigma,epsilon,mu,dx,dz,dt,source_site,receiver_site):
    pool=Pool(processes=8)
    Profile=np.empty((k_max,len(source_site)))
    res_l=[]
    for index,data_position in enumerate(zip(source_site,receiver_site)):
        res=pool.apply_async(Forward2D,args=(xl,zl,Freq,sigma,epsilon,mu,dx,dz,dt,data_position[0],data_position[1],index))
        res_l.append(res)
    pool.close()
    pool.join()
    for res in res_l:
        result=res.get()
        Profile[:,result[0]]=result[1]
        del result
    del res_l
    pool.terminate() 
    return Profile

def Forward2D(xl,zl,Freq,sigma,epsilon,mu,dx,dz,dt,value_source,value_receiver,index):
    t=np.arange(k_max)*dt
    f=Wavelet.ricker(t,Freq)
    CPML_Params=Add_CPML.Add_CPML(xl,zl,sigma,epsilon,mu,dx,dz,dt)
    Forward_data=Time_loop.time_loop(xl,zl,dx,dz,dt,sigma,epsilon,mu,CPML_Params,f,k_max,value_source,value_receiver)
    Profile=np.empty((k_max))
    for idx in range(k_max):
        tmp=Forward_data.__next__()
        Profile[idx]=tmp[0]
    return index,Profile

if __name__=='__main__':
    start_time=time.time()
    xl=101
    zl=101
    dx=dz=0.01
    dt=1e-11
    k_max=4000
    Freq=4e8
    epsilon=4*np.ones((xl,zl))
    epsilon[40:60,40:60]=15
    sigma=1e-5*np.ones((xl,zl))
    mu=np.ones((xl,zl))
    source_site=[]
    receiver_site=[]
    for idx in range(10,111,1):
        source_site.append((10,idx))
        receiver_site.append((10,idx))
        
    Profile=Forward_2D(xl,zl,Freq,sigma,epsilon,mu,dx,dz,dt,source_site,receiver_site)
    print(time.time()-start_time)
    
    from matplotlib import pyplot,cm
    pyplot.figure(1)
    pyplot.imshow(Profile,extent=(0,1,0,1),vmin=-100,vmax=100,cmap=cm.seismic)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,1,6))
    ax.set_xticklabels([0,20,40,60,80,100])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_yticklabels([40,30,20,10,0])
    pyplot.xlabel('Tracks')
    pyplot.ylabel('Time (ns)')
    pyplot.figure(2)
    pyplot.imshow(epsilon,cmap=cm.seismic,vmin=0,vmax=20)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,100,6))
    ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1])
    ax.set_yticks(np.linspace(0,100,6))
    ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1])
    pyplot.xlabel('x (m)')
    pyplot.ylabel('z (m)')

    
    