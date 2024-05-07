import numpy as np
import sys, os
import pandas as pd
from scipy import optimize

class Model_Potts():
    
    def __init__(self, L, Q, X, Y, u = None, margin = 1):
        
        self.data_len = L
        self.data_voc = Q
        self.data = X
        self.labels = Y
        self.reg = u if u is not None else L*Q
        self.weights = np.random.normal(0,1,size = L*Q + 1)
        self.kappa = margin
        self.trained = False
            
        self.size = int(len(self.data))
        _deltas = np.zeros(shape=(self.size,self.data_len,self.data_voc),dtype=np.int8)
    
        for p in range(self.size):
            for i in range(self.data_len):
                _deltas[p,i,self.data[p,i]]=1

        self.delta_tmp = _deltas 
        
    def loss_func(self):
        """
        Returns
        Loss function of the linear model across the dataset
        """
        w, data, labels = self.weights, self.data, self.labels
        
        b=w[0]; J = (w[1:]).reshape(self.data_len,self.data_voc)
        
        seqs = np.reshape(data,(-1,self.data_len,1))
        res_J = J[np.arange(J.shape[0])[:, None], seqs]
    
        cons = np.linalg.norm(J)*np.linalg.norm(J) - self.data_len*self.data_voc
        pred = np.sum(res_J/np.sqrt(self.data_len),axis=1) + b
        sum_vec = np.sum(J,axis=1)
    
        _single_loss = np.abs(self.kappa-labels*pred[:,0]) * np.heaviside(self.kappa-labels*pred[:,0],0)
        
        return np.sum(_single_loss,axis=0)+(self.reg/4)*cons*cons+(self.reg * self.reg/2)*np.sum(sum_vec*sum_vec)


    def der_loss(self):
        """
        Returns
        Derivative of the loss function of the linear model across the dataset
        """
        w, data, labels = self.weights, self.data, self.labels
        b=w[0]; J = (w[1:]).reshape(self.data_len,self.data_voc)
        
        seqs = np.reshape(data,(-1,self.data_len,1))
        res_J = J[np.arange(J.shape[0])[:, None], seqs]
        
        pred = np.sum(res_J/np.sqrt(self.data_len),axis=1) + b
        cons = np.linalg.norm(J)*np.linalg.norm(J) - self.reg
        sum_vec = (np.sum(J,axis=1))
    
        der_b = np.sum((-labels)*np.heaviside(self.kappa-labels*pred[:,0],0))
        
        der_J =self.reg*cons*J + self.reg*self.reg*sum_vec[:,np.newaxis]+ np.sum( ((-self.delta_tmp*labels[:,np.newaxis,np.newaxis])/np.sqrt(L))*((np.heaviside(self.kappa-labels*pred[:,0],0))[:,np.newaxis,np.newaxis]),axis=0 )   
    
        return np.hstack([der_b,der_J.flatten()])
    
    
    def training(self):
    
        w, data, labels = self.weights, self.data, self.labels
        optimized_weigths = optimize.minimize(self.loss_func,x0=w,args=(data,labels),jac=self.der_loss,tol=1e-12,method='L-BFGS-B')
        #w = optimized_weigths.x[1:]; b = optimized_weigths.x[0]; 
        self.weights = optimized_weigths.x
        self.trained = True
        
        return optimized_weigths.x
       
        
    def acc_scores(self):
        
        w, data, labels = self.weights, self.data, self.labels
        b=w[0]; J = w[1:]
        pred = np.sum(J*data/np.sqrt(self.data_len),axis=1) + b    
        y_pred = np.sign(pred).astype(np.int8)
        
        return np.sum(y_pred==labels)/len(y_pred)
    
    
    
    
class Model_Spins():
    
    def __init__(self, L, X, Y, u = None, margin = 1):
        
        self.data_len = L
        self.data = X
        self.labels = Y
        self.reg = u if u is not None else L
        self.weights = np.random.normal(0,1,size = L + 1)
        self.kappa = margin
        self.trained = False 
            
        self.size = int(len(self.data))
        
        
    def loss_func(self, w):
        """
        Returns
        Loss function of the linear model across the dataset
        """
        data, labels = self.data, self.labels
        b=w[0]; J = w[1:]
        
        cons = np.linalg.norm(J)*np.linalg.norm(J) - self.data_len
        pred = np.sum((J*data)/np.sqrt(self.data_len),axis=1) + b
        _single_loss = np.abs(self.kappa-labels*pred) * np.heaviside(self.kappa-labels*pred,0)
        
        return np.sum(_single_loss)+(self.reg/4)*cons*cons
    
    
    def der_loss(self, w):
        """
        Returns
        Derivative of the loss function of the linear model across the dataset
        """
        data, labels = self.data, self.labels
        b=w[0]; J = w[1:]
        pred = np.sum((J*data)/np.sqrt(self.data_len),axis=1) + b
        cons = np.linalg.norm(J)*np.linalg.norm(J) - self.data_len
    
        der_b = np.sum((-labels)*np.heaviside(self.kappa-labels*pred,0))
        
        der_J =self.reg*cons*J + np.sum(((-data*labels[:,np.newaxis])/np.sqrt(self.data_len))*(np.heaviside(self.kappa-labels*pred,0)[:,np.newaxis]),axis=0)
        
        return np.hstack([der_b,der_J])
    
    
    def training(self):
        
        w, data, labels = self.weights, self.data, self.labels
        #optimized_weigths = optimize.minimize(self.loss_func,x0=w,args=(data,labels),jac=self.der_loss,tol=1e-12,method='L-BFGS-B')
        optimized_weigths = optimize.minimize(self.loss_func,x0=w,jac=self.der_loss,tol=1e-12,method='L-BFGS-B')

        #w = optimized_weigths.x[1:]; b = optimized_weigths.x[0]; 
        self.weights = optimized_weigths.x
        self.trained = True
        
        return optimized_weigths.x
    
    def e_training(self):
        
        
        w= self.weights
        b=w[0]; J = w[1:]
        
        pred = np.sum((J*self.data)/np.sqrt(self.data_len),axis=1) + b
        E_train = np.sum(np.abs(self.kappa-self.labels*pred)*np.heaviside(self.kappa-self.labels*pred,0))
        
        return E_train

       
        
    def acc_scores(self, test_data, test_labels):
        
        w =  self.weights
        b=w[0]; J = w[1:]
        pred = np.sum(J*test_data/np.sqrt(self.data_len),axis=1) + b    
        y_pred =np.array([-1 for _ in range(len(test_data))])
        y_pred[np.where(pred>0)]=1
        #y_pred = np.sign(pred).astype(np.int8)
        
        return np.sum(y_pred==test_labels)/len(y_pred)