import numpy as np

#def eG(L,Q,Γ,Δ,αP,αN,κ,hk, hx, hq, hr, x, q, r, b):
#    """
#    Parameters
#    ----------
#    L : int
#        size of Γ and Δ
#    Q : int
#        number of Potts states
#    Γ : L-dimensional array of QxQ matrices
#        Γⱼ = diag(Mⱼ) - Mⱼ(Mⱼ)ᵀ
#    Δ : L-dimensional array of QxQ matrices
#        Δⱼ = δⱼ(δⱼ)ᵀ
#    αP: double
#        density of positive samples
#    αN: double
#        density of negative samples
#    κ : double
#        margin
#
#    Returns
#    ----------
#    Generalization energy (in-sample)
#    """
#    return 1/2*( np.exp(-(b-r/2)**2/(2*q))/np.sqrt(2*np.pi)*np.sqrt(q)  + 
#                (b-r/2)/2 *erfc(-(b-r/2)/np.sqrt(2*q)) +
#               np.exp(-(b+r/2)**2/(2*q))/np.sqrt(2*np.pi)*np.sqrt(q)  - 
#                (b+r/2)/2 *erfc((b+r/2)/np.sqrt(2*q)))
#


def dataRnd_wPotts(L,Q):
    """
    Parameters
    ----------
    L : int
        size of Γ and Δ
    Q : int
        number of Potts states
    
    Returns
    ----------
    M : L x Q matrix
        M\_jt freq of state t on site j
        sum\_t M\_jt = 1
    \delta : L x Q matrix
        \delta\_jt pos-to-neg distances
        sum\_t \delta\_jt = 0
    """
    M = np.random.uniform(0,0.9,(L,Q))
    M = np.dot(np.diag(1/np.sum(M,axis=1)),M)
    δ = np.random.uniform(-0.9,0.9,(L,Q))
    δ = δ - np.dot(np.diag(np.mean(δ,axis=1)),np.ones((L,Q)))
    return M,δ

def dataFromIsing(mIsing,δIsing):
    """
    Parameters
    ----------
    mIsing :L-dim array
            vector of magnetizations for Ising
    \deltaIsing :L-dim array
            vector of pos-to-neg distance for Ising
    
    Returns
    ----------
    M : L x Q matrix
        M\_up = (1+mIsing)/2
        M\_down = (1-mIsing)/2
    \delta : L x Q matrix
        \delta\_up = \deltaIsing/2
        \delta\_down = -\deltaIsing/2
    """
    M = np.transpose([(1+mIsing)/2,(1-mIsing)/2])
    δ = np.transpose([δIsing/2,-δIsing/2])
    
    return M,δ

def buildMatrices_Potts(M,δ):
    """
    Parameters
    ----------
    M : L x Q matrix
        M\_jt freq of state t on site j
        sum\_t M\_jt = 1
    \delta : L x Q matrix
        \delta\_jt pos-to-neg distances
        sum\_t \delta\_jt = 0
        
    Returns
    ----------
    Γ : L-dimensional array of QxQ matrices
        Γⱼ = diag(Mⱼ) - Mⱼ(Mⱼ)ᵀ
    Δ : L-dimensional array of QxQ matrices
        Δⱼ = δⱼ(δⱼ)ᵀ
    """
    L,Q = M.shape
    Γ = []
    Δ = []
    for i in range(L):
        Γ.append(np.diag(M[i]) - np.outer(M[i],M[i]) )
        Δ.append(np.outer(δ[i],δ[i]))
    return np.array(Γ),np.array(Δ)

def average_Potts(data, Q):
    
    """
    Q: number of total states
    Returns the Frequency Matrix over the data
    """
    
    dataset_onehot = np.eye(Q)[data]
    freqs = dataset_onehot.mean(0)
    
    return freqs
    