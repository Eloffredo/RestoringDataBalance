import numpy as np
import pandas as pd
from utils import buildMatrices_Potts, average_Potts


class Dataset:
    
    def __init__(self, TRAIN_PATH, TEST_PATH = None, split = None):
        
        assert TEST_PATH is not None and split is None, "Either provide a path for test dataset or a fraction of splitting the dataset"
        

        if TEST_PATH is not None:
            self.train_dataset = pd.read_csv(TRAIN_PATH)
            self.test_dataset = pd.read_csv(TEST_PATH)
            
        else:
            df = pd.read_csv(TRAIN_PATH)
            self.train_dataset = pd.sample(f = 1 - split)
            self.test_dataset = df.drop(self.train_dataset.index)
            
            
    def normalize_and_rescale(self, datapoints):
        
        datapoints = datapoints/datapoints.max()
        datapoints = np.where(datapoints>=0.5,1,-1)
        
        return datapoints
    
        
    def prepare_dataset(self, P_labels, N_labels, P_size, N_size):
        
        positive_dataset = self.train_dataset[self.train_dataset['label'].isin(P_labels)].sample(P_size)
        negative_dataset = self.train_dataset[self.train_dataset['label'].isin(N_labels)].sample(N_size)
        positive_datapoints = np.array(positive_dataset.drop( columns = ['label']) ).astype(np.int8)
        negative_datapoints = np.array(negative_dataset.drop( columns = ['label']) ).astype(np.int8)
        Y = np.array([1 for _ in range(len(self.train_dataset))])
        Y[len(positive_datapoints):] = -1
        
        self.train_dataset = np.concatenate(( self.normalize_and_rescale(positive_datapoints), self.normalize_and_rescale(negative_datapoints) ))
        self.train_labels = Y
        
        positive_dataset = self.test_dataset[self.test_dataset['label'].isin(P_labels)].sample(500)
        negative_dataset = self.test_dataset[self.test_dataset['label'].isin(N_labels)].sample(500)
        positive_datapoints = np.array(positive_dataset.drop( columns = ['label']) ).astype(np.int8)
        negative_datapoints = np.array(negative_dataset.drop( columns = ['label']) ).astype(np.int8)
        Y = np.array([1 for _ in range(len(self.test_dataset))])
        Y[len(positive_datapoints):] = -1
        
        self.test_dataset = np.concatenate(( self.normalize_and_rescale(positive_datapoints), self.normalize_and_rescale(negative_datapoints) ))
        self.test_labels = Y
        
        
class Sample_dataset:
    def __init__(self, mode, L, m_positive, m_negative):
        
        assert mode in ['Spins', 'Potts']
        
        self.size = L
        self.m_plus = m_positive
        self.m_minus = m_negative
        self.nature = mode
        
        if self.nature == 'Spins':
            assert len(m_positive) == L and sum(m_positive) <= L 
            
            self.voc_size = 2
            self.p_plus = np.array([0.5*(1-m_positive),0.5*(1+m_positive)]).T
            self.p_minus = np.array([0.5*(1-m_negative),0.5*(1+m_negative)]).T
            
        elif self.nature == 'Potts':
            
            self.voc_size = m_positive.shape[1]
            self.p_plus = m_positive
            self.p_minus = m_negative
            
        
    def sampling(self, S_num,p_):
        
        Samples = np.zeros(shape=(S_num,self.size))
        for t in range(self.size):
            if self.nature == 'Spins':
                Samples[:,t] = 2*np.random.choice(self.voc_size,p=p_[t], size = S_num) -1
            elif self.nature == 'Potts':
                Samples[:,t] = np.random.choice(self.voc_size,p=p_[t],size=S_num)

        return Samples.astype(np.int8)
                    
    def prepare_dataset(self, P_size, N_size):
        
        positive_datapoints = self.sampling(P_size,self.p_plus)
        negative_datapoints = self.sampling(N_size,self.p_minus)
        Y = np.array([1 for _ in range(P_size + N_size)])
        Y[P_size:] = -1
        self.train_dataset = np.concatenate(( positive_datapoints, negative_datapoints ))
        self.train_labels = Y
        
        positive_datapoints = self.sampling(500,self.p_plus)
        negative_datapoints = self.sampling(500,self.p_minus)
        Y = np.array([1 for _ in range(1000)])
        Y[500:] = -1
        self.test_dataset = np.concatenate(( positive_datapoints, negative_datapoints ))
        self.test_labels = Y
        
        
class Dataset_wPotts:
    def __init__(self, PATH,  split = 0.1):
    
        df = pd.read_csv(PATH)
        self.path = PATH
            
        self.train_dataset = df.sample(frac = 1 - split)
        self.test_dataset = df.drop(self.train_dataset.index)
        
    def prepare_dataset(P_size, N_size):
            
        positive_dataset = self.train_dataset[self.train_dataset['label']== 1].sample(P_size)
        negative_dataset = self.train_dataset[self.train_dataset['label']== -1].sample(N_size)
        positive_datapoints = np.array(positive_dataset.drop( columns = ['label']) ).astype(np.int8)
        negative_datapoints = np.array(negative_dataset.drop( columns = ['label']) ).astype(np.int8)
        Y = np.array([1 for _ in range(len(self.train_dataset))])
        Y[len(positive_datapoints):] = -1
        
        self.train_dataset = np.concatenate(( self.normalize_and_rescale(positive_datapoints), self.normalize_and_rescale(negative_datapoints) ))
        self.train_labels = Y
        
        positive_dataset = self.test_dataset[self.test_dataset['label'].isin(P_labels)].sample(500)
        negative_dataset = self.test_dataset[self.test_dataset['label'].isin(N_labels)].sample(500)
        positive_datapoints = np.array(positive_dataset.drop( columns = ['label']) ).astype(np.int8)
        negative_datapoints = np.array(negative_dataset.drop( columns = ['label']) ).astype(np.int8)
        Y = np.array([1 for _ in range(len(self.test_dataset))])
        Y[len(positive_datapoints):] = -1
        
        self.test_dataset = np.concatenate(( self.normalize_and_rescale(positive_datapoints), self.normalize_and_rescale(negative_datapoints) ))
        self.test_labels = Y
        
    def buildGD(self):
        
        df = pd.read_csv(self.path)
        P_data = df[df['label'] == 1].drop(columns = ['label'])
        N_data = df[df['label'] == -1].drop(columns = ['label'])
        
        Q = max(P_data.max().max(), N_data.max().max() ) + 1
        L = P_data.shape[1]

        Mpos = average_Potts(P_data,Q = Q) ###compute the frequency matrix LxQ
        Mneg = average_Potts(N_data,Q = Q)

        M_LP = 0.5*(Mpos+Mneg)
        δ_LP = np.sqrt(Mpos.shape[0]) *(Mpos-Mneg) 

        Γ,Δ = buildMatrices_Potts(M_LP,δ_LP) ### build Gamma, Delta matrices
        
        self.matrices = [Γ,Δ]
        
        return Γ,Δ

        