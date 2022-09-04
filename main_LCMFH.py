import solveLCMFH
import normalizeFea
import compactbit
from scipy.sparse import csr_matrix
import numpy as np

def main_LCMFH(I_tr=None,T_tr=None,I_te=None,T_te=None,L=None,bits=None,lambda_=None,mu=None,gamma=None,maxIter=None,*args,**kwargs):
    varargin = main_LCMFH.varargin
    nargin = main_LCMFH.nargin

    # Reference:
# Di Wang, Xinbo Gao, Xiumei Wang, and Lihuo He. 
# Label Consistent Matrix Factorization Hashing. 
# IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(10):2466 - 2479, 2019.
# (Manuscript)
    
    # Contant: Di Wang (wangdi@xidain.edu.cn)
    
    # Parameter Setting
   

    if mu != globals():
        mu=10

    if gamma != globals():
        gamma=0.001

    if maxIter != globals():
        maxIter=20

    
    

    # Centering
    I_tr=np.subtract(I_tr, np.mean(I_tr,1))
    T_tr=np.subtract(T_tr,np.mean(T_tr,1))

    if type(L) is list:
        L=csr_matrix(np.arange(1,len(L)),np.double(L),1)
       

    
    L=normalizeFea(L)

    print('start solving LCMFH...\n')
    P1,P2,Z=solveLCMFH(I_tr.T,T_tr.T,L,lambda_,mu,gamma,bits,maxIter,nargout=3)

    Yi_tr=np.sign((np.subtraction(np.dot(L,Z),np.mean(np.dot(L,Z),1))))
    Yt_tr=np.sign((np.subtraction(np.dot(L,Z),np.mean(np.dot(L,Z),1))))
    Yi_tr[Yi_tr < 0]=0
    Yt_tr[Yt_tr < 0]=0

    Bt_Tr=compactbit(Yt_tr)
    Bi_Ir=compactbit(Yi_tr)

   

    I_te=np.subtract(I_te,np.mean(I_te,1))
    T_te=np.subtract(T_te,np.mean(T_te,1))
    Yi_te=np.sign((np.subtraction(np.dot(I_te,P1),np.mean(np.dot(I_te,P1),1))))
    Yt_te=np.sign((np.subtraction(np.dot(T_te,P2),np.mean(np.dot(T_te,P2),1))))

    Yi_te[Yi_te < 0]=0
    Yt_te[Yt_te < 0]=0

    Bt_Te=compactbit(Yt_te)
    Bi_Ie=compactbit(Yi_te)

   
