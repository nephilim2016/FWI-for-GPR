#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:46:58 2018

@author: nephilim
"""
import numpy as np
import shutil
import os
import cmath
import matplotlib.pyplot as plt

class para():
    def __init__(self):
        pass
class options():
    def __init__(self):
        pass

class Optimization(para,options):
    def __init__(self,fh,isigma,iepsilon):
        super().__init__()
        self.fh=fh
        self.data=np.hstack((isigma.flatten(),iepsilon.flatten()))
    def optimization(self):
        if not hasattr(options,'maxiter'):
            options.maxiter=10
        if not hasattr(options,'maxit_ls'):
            options.maxit_ls=5
        if not hasattr(options,'M'):
            options.M=5
        if not hasattr(options,'tol'):
            options.tol=1e-2
        if not hasattr(options,'fid'):
            options.fid=1
        if not hasattr(options,'method'):
            options.method='lbfgs'
        if not hasattr(options,'c1'):
            options.c1=1e-4
        if not hasattr(options,'c2'):
            options.c2=0.9
        if not hasattr(options,'cpudate'):
            options.cpudate=1
        if not hasattr(options,'write'):
            options.write=1
        if not hasattr(options,'debug'):
            options.debug=1
        if not hasattr(options,'doplot'):
            options.doplot=1
        if not hasattr(options,'ls_int'):
            options.ls_int=2
        if not hasattr(options,'progTol'):
            options.progTol=1e-9
            
        if not os.path.exists('./%sHz_imodel_file'%para.Freq):
            os.makedirs('./%sHz_imodel_file'%para.Freq)
        else:
            shutil.rmtree('./%sHz_imodel_file'%para.Freq)
            os.makedirs('./%sHz_imodel_file'%para.Freq)
        
        n=len(self.data)
        d=np.zeros(n)
        x=self.data
        iter_=0
        alpha0=1.0
#        alpha=0
        print('%5s,%6s,%15s,%15s,%15s\n'%('iter','eval','step length','function value','||g(x)||_2'))
        f,g=self.fh(x[:int(len(x)/2)].reshape((para.xl,-1)),
                    x[int(len(x)/2):].reshape((para.xl,-1)))
        
#        fig=plt.figure()
#        axes1=fig.add_subplot(111) 
#        line1,=axes1.plot(g)
        
        f0=f
        fevals=1
        info=[[iter_,fevals,alpha0,f,np.linalg.norm(g,2)]]
        print('%5d,%6d,%15.5e,%15.5e,%15.5e\n'%(iter_,fevals,alpha0,f,np.linalg.norm(g,2)))
        for iter_ in range(0,options.maxiter):
            if options.method=='sd':
                d=-g
            
            if options.method=='cg':
                if iter_==0:
                    d=-g
                else:
                    gotgo=np.dot(g_old,g_old)
                
                if options.cgupdate==0:
                    beta=np.dot(g,g)/(gotgo)
                elif options.cgupdate==1:
                    beta=np.dot(g,(g-g_old))/(gotgo)
                elif options.cgupdate==2:
                    beta=np.dot(g,(g-g_old))/np.dot((g-g_old),d)
                else:
                    beta_FR=np.dot(g,(g-g_old))/(gotgo)
                    beta_PR=np.dot(g,g)-np.dot(g,g_old)/(gotgo)
                    beta=max(-beta_FR,min(beta_PR,beta_FR))
                d=-g+beta*d
                if np.dot(g*d)>-options.progTol:
                    beta=0
                    d=-g
                g_old=g
            
            if options.method=='lbfgs':
                if iter_==0:
                    S=np.zeros((n,options.M))
                    Y=np.zeros((n,options.M))
                else:
                    S=np.hstack((S[:,1:],S[:,0][:,np.newaxis]))
                    Y=np.hstack((Y[:,1:],Y[:,0][:,np.newaxis]))
                    S=np.hstack((S[:,0:-1],alpha*d[:,np.newaxis]))
                    Y=np.hstack((Y[:,0:-1],(g-g_old)[:,np.newaxis]))
            
                d=self.__B(-g,S,Y,np.array([]))
                p=-np.dot(d,g)/np.dot(g,g)
                if p<=0:
                    S=np.zeros((n,options.M))
                    Y=np.zeros((n,options.M))
                g_old=g
            
            gtd=np.dot(g,d)
            if iter_==0:
                alpha0=-f/gtd/10
            else:
                alpha0=1
#                alpha0=min(1,2*(f-f_old)/(gtd))
                if alpha0<=0:
                    alpha0=1
            f_old=f
            gtd_old=gtd
            alpha,f,g,lsiter=self.__WolfeLineSearch(x,alpha0,d,f,g,gtd,options.c1,options.c2,options.ls_int,options.maxit_ls,options.debug,options.doplot,self.fh)
    
            fevals=fevals+lsiter
            x=x+alpha*d
            x=self.__counts(x)
            np.save('./%sHz_imodel_file/%d_imodel.npy'%(para.Freq,iter_),x)
            info.append([iter_,fevals,alpha,f,np.linalg.norm(g,2)])
            print('%5d,%6d,%15.5e,%15.5e,%15.5e\n'%(iter_+1,fevals,alpha,f,np.linalg.norm(g,2)))
            if f/f0<options.tol:
                print('Function Value less than funTol\n')
                break
            if alpha==0:
                print('alpha Value equal 0\n')
                break
       
            if fevals>=1000:
                print('Reached Maximum Number of Function Evaluations\n')
                break
    
            if iter_==options.maxiter:
                print('Reached Maximum Number of Iterations\n')
                break   
        
        return x,info
    def __B(self,x,S,Y,H0):
        J=np.sum(abs(S),0).nonzero()[0]
        S=S[:,J]
        Y=Y[:,J]
        M=S.shape[1]
        n=len(x)
        

        if (H0.size==0) and (M>0):
            H0=np.linalg.norm(Y[:,-1],2)**2/np.dot(S[:,-1],Y[:,-1])*np.ones(n)
        else:
            H0=np.ones(n)
            
        alpha=np.zeros(M)
        rho=np.zeros(M)
        for k in range(M):
            rho[k]=1/np.dot(Y[:,k],S[:,k])
        q=x
        for k in range(M-1,-1,-1):
            alpha[k]=rho[k]*np.dot(S[:,k],q)
            q-=alpha[k]*Y[:,k]
        z=q/H0
        for k in range(M):
            beta=rho[k]*np.dot(Y[:,k],z)
            z+=(alpha[k]-beta)*S[:,k]
        return z
    
    def __WolfeLineSearch(self,x,t,d,f,g,gtd,c1,c2,LS_interp,maxLS,debug,doPlot,fh):
        x_new=x+t*d
        x_new=self.__counts(x_new)
        f_new,g_new=fh(x_new[:int(len(x_new)/2)].reshape((para.xl,-1)),
                       x_new[int(len(x_new)/2):].reshape((para.xl,-1)))
        lsiter=1
        gtd_new=np.dot(g_new,d)
        LSiter=0
        t_prev=0.0
        f_prev=f
        g_prev=g
        gtd_prev=gtd
        done=0
        while LSiter<maxLS:
            
            if (f_new>(f+c1*t*gtd)) or ((LSiter>1) and (f_new>=f_prev)):
                bracket=np.hstack((t_prev,t))
                bracketFval=np.hstack((f_prev,f_new))
                bracketGval=np.hstack((g_prev[:,np.newaxis],g_new[:,np.newaxis]))
                break
            elif abs(gtd_new)<=-c2*gtd:
                bracket=t
                bracketFval=f_new
                bracketGval=g_new
                done=1
                break
            elif gtd_new>=0:
                bracket=np.hstack((t_prev,t))
                bracketFval=np.hstack((f_prev,f_new))
                bracketGval=np.hstack((g_prev[:,np.newaxis],g_new[:,np.newaxis]))
                break
            temp=t_prev
            t_prev=t
            minStep=t+0.01*(t-temp)
            maxStep=t*10
            if LS_interp<=1:
                if debug:
                    print('Extending Braket\n')
                t=maxStep
            elif LS_interp==2:
                args=np.array([[temp,f_prev,gtd_prev],[t,f_new,gtd_new]])
                t=self.__polyinterp(args,doPlot,minStep,maxStep)
                if debug:
                    print('Lines Search Cubic Extrapolation Iteration %i , alpha=%d\n'%(LSiter,t))
            f_prev=f_new
            g_prev=g_new
            gtd_prev=gtd_new
    
            x_new=x+t*d
            x_new=self.__counts(x_new)
            f_new,g_new=fh(x_new[:int(len(x_new)/2)].reshape((para.xl,-1)),
                           x_new[int(len(x_new)/2):].reshape((para.xl,-1)))
    
            lsiter+=1
            gtd_new=np.dot(g_new,d)
            LSiter+=1
            
        if LSiter==maxLS:
            bracket=np.hstack((0,t))
            bracketFval=np.hstack((f,f_new))
            bracketGval=np.hstack((g_prev[:,np.newaxis],g_new[:,np.newaxis]))
        insufProgress=0
        while (not done) and (LSiter<maxLS):
            LOpos=np.argmin(bracketFval)
            f_LO=bracketFval[LOpos]
            HIpos=-LOpos+1
            if LS_interp<=1:
                if debug:
                    print('Bisecting\n')
                t=np.mean(bracket)
            else:
                args=np.array([[bracket[0],bracketFval[0],np.dot(bracketGval[:,0],d)],[bracket[1],bracketFval[1],np.dot(bracketGval[:,1],d)]])
                t=self.__polyinterp(args,doPlot)
                if debug:
                    print('Lines Search Grad-Cubic Interpolation Iteration',LSiter,'alpha=',t)  
            if np.min((np.max(bracket)-t,t-np.min(bracket)))/(np.max(bracket)-np.min(bracket))<0.1:
                if debug:
                    print('Interpolation close to boundary')
                if insufProgress or (t>=np.max(bracket)) or (t<=np.min(bracket)):
                    if abs(t-np.max(bracket))<abs(t-np.min(bracket)):
                        t=np.max(bracket)-0.1*(np.max(bracket)-np.min(bracket))
                    else:
                        t=np.min(bracket)+0.1*(np.max(bracket)-np.min(bracket))
                    if debug:
                        print(', Evaluating at 0.1 away from boundary, alpha=',t)
                    insufProgress=0
                else:
                    if debug:
                        print('\n')
                    insufProgress=1
            else:
                insufProgress=0

            x_new=x+t*d
            x_new=self.__counts(x_new)
            f_new,g_new=fh(x_new[:int(len(x_new)/2)].reshape((para.xl,-1)),
                           x_new[int(len(x_new)/2):].reshape((para.xl,-1)))
            lsiter+=1
            gtd_new=np.dot(g_new,d)
            LSiter+=1
    
            armijo=(f_new<f+c1*t*gtd)
            if (not armijo) or (f_new>=f_LO):
                bracket[HIpos]=t
                bracketFval[HIpos]=f_new
                bracketGval[:,HIpos]=g_new
        
            else:
                if abs(gtd_new)<=-c2*gtd:
                    done=1
                elif gtd_new*(bracket[HIpos]-bracket[LOpos])>=0:
                    bracket[HIpos]=bracket[LOpos]
                    bracketFval[HIpos]=bracketFval[LOpos]
                    bracketGval[:,HIpos]=bracketGval[:,LOpos]
                bracket[LOpos]=t
                bracketFval[LOpos]=f_new
                bracketGval[:,LOpos]=g_new
        if LSiter==maxLS:
            if debug:
                print('Line Search Exceeded Maximum Line Search Iterations\n')
        if type(bracket)==np.ndarray:
            LOpos=np.argmin(bracketFval)
            f_LO=bracketFval[LOpos]
            t=bracket[LOpos]
            f_new=bracketFval[LOpos]
            g_new=bracketGval[:,LOpos]
        else:
            f_LO=bracketFval
            t=bracket
            f_new=bracketFval
            g_new=bracketGval
            
        return t,f_new,g_new,lsiter
    
    def __polyinterp(self,points,*vargs):
        xmin=np.min(points[:,0])
        xmax=np.max(points[:,0])
        print(xmin,xmax)
        if len(vargs)<1:
            doPlot=0
        else:
            doPlot=vargs[0]
        if len(vargs)<2:
            xminBound=xmin
        else:
            xminBound=vargs[1]
        if len(vargs)<3:
            xmaxBound=xmax
        else:
            xmaxBound=vargs[2]

        minPos=np.argmin(points[:,0])
        notMinPos=-minPos+1
        if (points[minPos,0]-points[notMinPos,0])==0:
            return (xmaxBound+xminBound)/2
        d1=points[minPos,2]+points[notMinPos,2]-3*(points[minPos,1]-points[notMinPos,1])/(points[minPos,0]-points[notMinPos,0])
        d2=cmath.sqrt(d1**2-points[minPos,2]*points[notMinPos,2])
        if d2.imag==0.0:
            t=points[notMinPos,0]-(points[notMinPos,0]-points[minPos,0])*((points[notMinPos,2]+d2.real-d1)/(points[notMinPos,2]-points[minPos,2]+2*d2.real))
            minPos=np.min((np.max((t,xminBound)),xmaxBound))
        else:
            minPos=(xmaxBound+xminBound)/2
        return minPos
        
    def __counts(self,x):
        sigma=x[:int(len(x)/2)]
        epsilon=x[int(len(x)/2):]
        sigma[sigma<0]=0
        epsilon[epsilon>81]=81
        epsilon[epsilon<1]=1
        x=np.hstack((sigma,epsilon))
        return x