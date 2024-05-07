import sys
sys.setrecursionlimit(10000)
import numpy as np
from scipy.special import erfc
import scipy.integrate as integrate


class SP_equations:
    def __init__(self, L,  mode = 'Spins', C = None, d = None, G = None, D = None, Q = 1, margin = 1.0, c = 1.0):
        
        self.size = L
        self.kappa = margin
        self.dupl = c
        
        assert mode in ['Spins', 'Potts']
        self.mode = mode
        
        if self.mode == 'Potts':
            self.voc_size = Q
            self.corr_matrix = G
            self.displacement = D
            
        elif self.mode == 'Spins':
            self.corr_matrix = C
            self.displacement = d
            self.voc_size = 1
            
        

    def GP(self, x,q,r,b):
        """
        Positive patterns contribution to the free energy
        """
        κ = self.kappa
        K1 = κ - r/2 + b
        return 1/(2*x)*(-(K1**2 + q)*erfc(-K1/np.sqrt(2*q)) +
                        ((K1-x*self.dupl)**2 + q)*erfc(-(K1-x*self.dupl)/np.sqrt(2*q)) - 
                        np.sqrt(2*q/np.pi)*K1*np.exp(-K1**2/(2*q)) + 
                        np.sqrt(2*q/np.pi)*(K1-x*self.dupl)*np.exp(-(K1-x*self.dupl)**2/(2*q)))
    
    
    def GN(self, x,q,r,b):
        """
        Negative patterns contribution to the free energy
        """
        κ = self.kappa
        K2 = - κ + r/2 + b
        return 1/(2*x)*(-(K2**2 + q)*erfc(K2/np.sqrt(2*q)) +
                        ((K2+x)**2 + q)*erfc((K2+x)/np.sqrt(2*q)) + 
                        np.sqrt(2*q/np.pi)*K2*np.exp(-K2**2/(2*q)) - 
                        np.sqrt(2*q/np.pi)*(K2+x)*np.exp(-(K2+x)**2/(2*q)))
    
    
    def dGP(self, x,q,r,b):
        """
        Gradient of GP
        """
        κ = self.kappa
        K1 = κ - r/2 + b
        dGP_dx = (-self.GP(x,q,r,b)/x - 
                  self.dupl/x*(K1-x*self.dupl)*erfc(-(K1-x*self.dupl)/np.sqrt(2*q))-
                  self.dupl/x*np.sqrt(2*q/np.pi)*np.exp(-(K1-x*self.dupl)**2/(2*q)) )
        dGP_dq = - 1/(2*x)*( erfc(-K1/np.sqrt(2*q)) - erfc(-(K1-x*self.dupl)/np.sqrt(2*q)) )
        dGP_dr = (1/(2*x)*( K1*erfc(-K1/np.sqrt(2*q)) - (K1- x*self.dupl)*erfc(-(K1-x*self.dupl)/np.sqrt(2*q)) ) +
                  1/x*np.sqrt(q/(2*np.pi))*(np.exp(-K1**2/(2*q)) - np.exp(-(K1-x*self.dupl)**2/(2*q))))
        return dGP_dx,dGP_dq,dGP_dr
    
    
    def dGN(self, x,q,r,b):
        """
        Gradient of GN
        """
        κ = self.kappa
        K2 = -κ + r/2 + b
        dGN_dx = (-self.GN(x,q,r,b)/x + 
                  1/x*(K2+x)*erfc((K2+x)/np.sqrt(2*q))-
                  1/x*np.sqrt(2*q/np.pi)*np.exp(-(K2+x)**2/(2*q)) )
        dGN_dq = - 1/(2*x)*( erfc(K2/np.sqrt(2*q)) - erfc((K2+x)/np.sqrt(2*q)) )
        dGN_dr = (1/(2*x)*(- K2*erfc(K2/np.sqrt(2*q)) + (K2 + x)*erfc((K2+x)/np.sqrt(2*q)) ) +
                  1/x*np.sqrt(q/(2*np.pi))*(np.exp(-K2**2/(2*q)) - np.exp(-(K2+x)**2/(2*q))))
        return dGN_dx,dGN_dq,dGN_dr
    
    
    def GS(self, hk,hx,hq,hr):
        """
        Entropy contribution to the free energy
        """
        L, C, δ = self.size, self.corr_matrix, self.displacement
        A = hk*np.identity(L) + hx*C
        Ainv = np.linalg.inv(A)
        return -1/L*(hq*np.trace(np.matmul(Ainv,C)) -
                     hr**2*np.dot(δ,np.dot(Ainv,δ)))
    

    def ET(self,hk,hx,hq,hr,x,q,r,b, aP, aN):
        """
        Free energy
        """
        return self.GS(hk,hx,hq,hr)/2 +(hk + hx*q + hq*x + 2*hr*r)/2 + aP*self.GP(x,q,r,b)/2 +  aN* self.GN(x,q,r,b)/2
    
    def GS_wPotts(self,hk,hx,hq,hr):
        """
        Entropy contribution to the free energy
        
        Parameters
        ----------
        L : int
            size of Γ and Δ
        Q : int
            number of Potts states
        Γ : L-dimensional array of QxQ matrices
            Γⱼ = diag(Mⱼ) - Mⱼ(Mⱼ)ᵀ
        Δ : L-dimensional array of QxQ matrices
            Δⱼ = δⱼ(δⱼ)ᵀ
        """
        L, Q, Γ,Δ = self.size, self.voc_size, self.corr_matrix, self.displacement

        ϵ = 1e-8   # to invert A when hk = 0
        A = (hk + ϵ)*np.identity(Q) + hx*Γ
        Ainv = np.linalg.inv(A)
        # dets = np.linalg.det(A)
        trsΓ = np.trace(np.matmul(Ainv,Γ),axis1=1, axis2=2)
        trsΔ = np.trace(np.matmul(Ainv,Δ),axis1=1, axis2=2)
        return - hq*np.mean(trsΓ) + (hr**2)*np.mean(trsΔ)
    
    def ET_wPotts(self,hk,hx,hq,hr,x,q,r,b, aP, aN):
        """
        Free energy
        """
        return self.GS_wPotts(hk,hx,hq,hr)/2 +(self.voc_size*hk + hx*q + hq*x + 2*hr*r)/2 + aP*self.GP(x,q,r,b)/2 +  aN*self.GN(x,q,r,b)/2
    
    
    def eq_b(self, hx,x,q,r,b,aP, aN):
        """
        Equation for b
        """
        κ = self.kappa
        K1 = κ - r/2 + b
        K2 = -κ + r/2 + b
        return 1/(4*x*hx)*(aP*(2*np.sqrt(2*q/np.pi)*(-np.exp(-K1**2/(2*q)) + np.exp(-(K1-x*self.dupl)**2/(2*q)) ) 
                               + (r - 2*κ)*erfc(-K1/np.sqrt(2*q)) - (r+2*x*self.dupl - 2*κ)*erfc(-(K1-x*self.dupl)/np.sqrt(2*q)) )+
                           aN*(2*np.sqrt(2*q/np.pi)*(np.exp(-K2**2/(2*q)) - np.exp(-(K2+x)**2/(2*q)) ) 
                               - (r - 2*κ)*erfc(K2/np.sqrt(2*q)) + (r+2*x - 2*κ)*erfc((K2+x)/np.sqrt(2*q)) ))

    
    def saddleUNSATcov1(self, aP, aN, η,s):
        """
        Parameters
        ----------
        η : learning rate
        s : step of the recursion
        Returns
        ----------
        Order parameters
        """
        L, C,δ  = self.size, self.corr_matrix , self.displacement
        if s==0: ### set inizialization
            return 10,2,-2,-1,0.59,0.84,1,1.12
        else:
            
            old_hk, old_hx, old_hq, old_hr, old_x, old_q, old_r, old_b = self.saddleUNSATcov1(aP, aN, η,s-1)
            dGP_dx,dGP_dq,dGP_dr = self.dGP(old_x,old_q,old_r,old_b)
            dGN_dx,dGN_dq,dGN_dr = self.dGN(old_x,old_q,old_r,old_b)
            
            A = old_hk*np.identity(L) + old_hx*C
            Ainv = np.linalg.inv(A)
            Ainv2 = np.linalg.matrix_power(Ainv,2)
            CA = np.matmul(C,Ainv)
            ACA = np.matmul(Ainv,CA)
            ACAC = np.matmul(ACA,C)
            
            v00 = np.trace(Ainv)/L
            v01 = np.trace(CA)/L
            v11 = np.trace(Ainv2)/L
            v12 = np.trace(ACA)/L
            v13 = np.dot(δ,np.dot(Ainv2,δ))/L
            v22 = np.trace(ACAC)/L
            v23 = np.dot(δ,np.dot(ACA,δ))/L
            
            hx = η*( - aP*dGP_dq - aN*dGN_dq) + (1-η)*old_hx
            hq = η*( - aP*dGP_dx - aN*dGN_dx) + (1-η)*old_hq
            hr = η*( - aP*dGP_dr - aN*dGN_dr)/2 + (1-η)*old_hr
            
            x = η*(old_hk*v12 + old_hx*v22)+(1-η)*old_x
            q = η*((old_hr**2)*v23 - old_hq*v22) +(1-η)*old_q
            r = η*(- old_hr*old_hk*v13 - old_hr*old_hx*v23) +(1-η)*old_r
        
            hk = η*(- old_hx*v01 - old_hq*v12 + old_hr**2*v13)/v00 + (1-η)*old_hk
            b = η*self.eq_b(old_hx,old_x,old_q,old_r,old_b,aP,aN) + (1-η)*old_b
            
            return hk, hx, hq, hr, x, q, r, b
        
    def saddleUNSAT_wPotts(self, aP, aN, η,s):
        """
        Parameters
        ----------
        η : double in (0,1)
            learning rate
        s : int
            step of the recursion
        
        Returns
        ----------
        hk, hx, hq, hr, x, q, r, b
        """
        L, Q, Γ, Δ = self.size, self.voc_size, self.corr_matrix, self.displacement
        if s==0:
            return -0.07,2,-1.5,-0.9,0.6,0.8,0.6,1.
    
        else:
            
            old_hk, old_hx, old_hq, old_hr, old_x, old_q, old_r, old_b = self.saddleUNSAT_wPotts(aP,aN,η,s-1)
            dGP_dx,dGP_dq,dGP_dr = self.dGP(old_x,old_q,old_r,old_b)
            dGN_dx,dGN_dq,dGN_dr = self.dGN(old_x,old_q,old_r,old_b)
            
            ϵ = 1e-8   # to invert A when hk = 0
            A = (old_hk + ϵ)*np.identity(Q) + old_hx*Γ
            Ainv = np.linalg.inv(A)
            Ainv2 = np.linalg.matrix_power(Ainv, 2)
            Γ2 = np.linalg.matrix_power(Γ, 2)
            
            v11 = np.mean(np.trace(Ainv2,axis1=1, axis2=2)) - 1/(old_hk+ϵ)**2
            v12 = np.mean(np.trace(np.matmul(Ainv2,Γ),axis1=1, axis2=2))
            v13 = np.mean(np.trace(np.matmul(Ainv2,Δ),axis1=1, axis2=2))
            v22 = np.mean(np.trace(np.matmul(Ainv2,Γ2),axis1=1, axis2=2))
            v23 = np.mean(np.trace(np.matmul(np.matmul(Ainv2,Γ),Δ),axis1=1, axis2=2))
            
            hx = η*( - aP*dGP_dq - aN*dGN_dq) + (1-η)*old_hx
            hq = η*( - aP*dGP_dx - aN*dGN_dx) + (1-η)*old_hq
            hr = η*( - aP*dGP_dr - aN*dGN_dr)/2 + (1-η)*old_hr
            
            x = η*(old_hk*v12 + old_hx*v22)+(1-η)*old_x
            q = η*((old_hr**2)*v23 - old_hq*v22) +(1-η)*old_q
            r = η*(- old_hr*old_hk*v13 - old_hr*old_hx*v23) +(1-η)*old_r
            
            hk = η*( (-(v12/v11)*old_hx + 
                      np.sqrt(-old_hq*(v12/v11) + old_hr**2*(v13/v11) + 
                              ((v12**2/v11**2) - v22/v11)*old_hx**2 - 1/v11)) )  + (1-η)*old_hk
    
            b = η*self.eq_b(old_hx,old_x,old_q,old_r,old_b,aP,aN) + (1-η)*old_b
            
            return hk, hx, hq, hr, x, q, r, b
    

        
    def execute_saddpeqs_aN_cov(self, aP, aN, η = .1, iters = 150):
        """
        Run the saddle point iterations to find the order parameters to compute the theoretical predictions of metrics
        """
        if self.mode == 'Potts':
            hk, hx, hq, hr, x, q, r, b = self.saddleUNSAT_wPotts(aP,aN,η,iters)
            eT = -self.ET_wPotts(hk,hx,hq,hr,x,q,r,b, aP, aN)/(aP+aN)
            
        elif self.mode == 'Spins':
            hk, hx, hq, hr, x, q, r, b = self.saddleUNSATcov1(aP,aN,η,iters)
            eT = -self.ET(hk,hx,hq,hr,x,q,r,b, aP, aN)/(self.dupl*aP+aN)
        #eGsR = eG(L,CC,δ,αP,αN,κ,hk, hx, hq, hr, x, q, r, b)
        
        return [aP,aN,q,r,b, eT]
    

    #def eT(L,C,δ,αP,αN,κ,hk, hx, hq, hr, x, q, r, b):
    #    """    
    #    Parameters
    #    ----------
    #    L : data dimensionality
    #    C : covariance matrix
    #    δ : δ = m+ - m-
    #    αP: density of positive class P/L
    #    αN: density of negative class N/L
    #    κ : margin of the classifier
    #
    #    Returns
    #    ----------
    #    Training energy
    #    """
    #        
    #    return -ET(L,C,δ,αP,αN,κ,hk,hx,hq,hr,x,q,r,b)/(αP+αN)
     
    
class Test_metrics:
    def __init__(self, Q, R ,B):
        self.overlap = Q
        self.magn = R
        self.bias = B
        
    def eG(self):
        """
        Generalization energy (in-sample) after optimization of SP equations
        """
        q = self.overlap, r = self.magn , b = self.bias
        return 1/2*( np.exp(-(b-r/2)**2/(2*q))/np.sqrt(2*np.pi)*np.sqrt(q)  + 
                    (b-r/2)/2 *erfc(-(b-r/2)/np.sqrt(2*q)) +
                   np.exp(-(b+r/2)**2/(2*q))/np.sqrt(2*np.pi)*np.sqrt(q)  - 
                    (b+r/2)/2 *erfc((b+r/2)/np.sqrt(2*q)))


    def TPR(self, γ):
        return 0.5*erfc((γ+self.bias-self.magn/2)/np.sqrt(2*self.overlap))
    
    def FPR(self, γ):
        return 0.5*erfc((γ+self.bias+self.magn/2)/np.sqrt(2*self.overlap))
    
    def TNR(self, γ):
        return 1 - self.FPR(γ)
    
    def FNR(self, γ):
        return 1 - self.TPR(γ)
    
    def BA(self):
        return 0.5*self.TPR(0) + 0.5*self.TNR(0)
    
    def ROC(self):
        xROC, yROC = [], []
        maxGamma = 20*(np.abs(self.bias)+np.abs(self.magn)/2 + np.sqrt(2*self.overlap) )
        gammas = np.linspace(-maxGamma,maxGamma,num=5000)
        for γ in gammas:
            xROC.append(self.FPR(γ))
            yROC.append(self.TPR(γ))
        
        ind = np.argsort(np.array(xROC))
        xROC, yROC = np.array(xROC)[ind], np.array(yROC)[ind]
        
        return xROC,yROC

    def AUC(self):
        xROC,yROC = self.ROC()
        return np.abs(integrate.trapezoid(yROC,x=xROC))
    
    def PPV(self, φP, φN, γ):
        return φP*self.TPR(γ)/(φP*self.TPR(γ) + φN*self.FPR(γ))
    
    def PRC(self, φP, φN):
        xPRC, yPRC = [], []
        maxGamma = 20*(np.abs(self.bias)+np.abs(self.magn)/2 + np.sqrt(2*self.overlap) )
        gammas = np.linspace(-maxGamma,maxGamma,num=5000)
        for γ in gammas:
            xPRC.append(self.TPR(γ))
            yPRC.append(self.PPV(φP,φN, γ))
        
        ind = np.argsort(np.array(xPRC))
        xPRC, yPRC = np.array(xPRC)[ind], np.array(yPRC)[ind]
        
        return xPRC,yPRC
    
    def AUPRC(self, φP, φN):
        xPRC,yPRC = self.PRC(φP,φN)
        return np.abs(integrate.trapezoid(yPRC[xPRC>0.001],x=xPRC[xPRC>0.001])) 
    
    def NPV(self, φP, φN):
        return φN*self.TNR(0)/(φN*self.TNR(0) + φP*self.FNR(0))
    
    def F1(self, φP, φN):
        TP = φP*self.TPR(0)
        TN = φN*self.TNR(0)
        FP = φN*self.FPR(0)
        FN = φP*self.FNR(0)
        return 2*TP/(2*TP+FP+FN)
    
    def FM(self, φP, φN):
        TP = φP*self.TPR(0)
        TN = φN*self.TNR(0)
        FP = φN*self.FPR(0)
        FN = φP*self.FNR(0)
        return np.sqrt(TP/(TP+FP)*TP/(TP+FN))
    
    def MCC(self, φP, φN):
        TP = φP*self.TPR(0)
        TN = φN*self.TNR(0)
        FP = φN*self.FPR(0)
        FN = φP*self.FNR(0)
        return (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))