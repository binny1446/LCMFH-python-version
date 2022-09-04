import numpy as np
from scipy.sparse import csr_matrix


def map_rank(traingnd=None,testgnd=None,IX=None,*args,**kwargs):
    varargin = map_rank.varargin
    nargin = map_rank.nargin

    ## Label Matrix
    if type(traingnd) is list:
        traingnd=csr_matrix(np.arange(1,len(traingnd)), np.double(traingnd),1)
      

    
    if type(testgnd) is list:
        testgnd=csr_matrix(np.arange(1,len(testgnd)),np.double(testgnd),1)
        

    
    numtrain,numtest=np.shape(IX)
    apall=np.zeros(numtrain,numtest)
    aa=np.arange(1,numtrain)

    for i in np.arange(1,numtest).reshape(-1):
        y=IX(np.arange(),i)
        new_label=np.zeros(1,numtrain)
        new_label[np.dot(traingnd,np.transpose(testgnd(i,np.arange()))) > 0]=1
        xx=np.cumsum(new_label(y))
        x=np.dot(xx,new_label(y))

        p=x / aa

        p=np.cumsum(p)
        id=np.where(p != 0)
        p[id]=p(id) / xx(id)
        apall[np.arange(),i]=np.transpose(p)

    
    ap=np.mean(apall,2)
