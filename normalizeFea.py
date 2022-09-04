import numpy as np
from scipy.sparse import csr_matrix

def normalizeFea(X=None,*args,**kwargs):
    varargin = normalizeFea.varargin
    nargin = normalizeFea.nargin

    # X: n*m
    Length=np.sqrt(np.sum(X ** 2))

    Length[Length <= 0]=1e-08

    
    Lambda=1.0 / Length

    X=np.dot(np.diag(csr_matrix(Lambda)),X)
