from pickle import TRUE
import numpy as np

def solveLCMFH(X1=None,X2=None,L=None,lambda_=None,mu=None,gamma=None,bits=None,maxIter=None,*args,**kwargs):
    varargin = solveLCMFH.varargin
    nargin = solveLCMFH.nargin

    #solveLCMFH Summary of this function goes here
# Label Consistent Matrix Factorization Hashing
#   minimize_{U1, U2, Z1, Z2}    lambda*||X1 - U1 * Z' * L'||^2 + 
#      (1 - lambda)||X2 - U2 * Z' * L'||^2 + 
#      mu * ||L * Z - X1' * P1||^2 + mu * ||L * Z - X2' * P2||^2
#      gamma * (||U1||^2 + ||U2||^2 + ||L * Z||^2 + ||P1||^2  + ||P2||^2)
# Notation:
# X1: data matrix of View1, each column is a sample vector
# X2: data matrix of View2, each column is a sample vector
# L: label matrix of X1 and X2, each row is a label vector
# lambda: trade off between different views
# mu: trade off between matrix factorization and cross correlations
# gamma: parameter to control the model complexity
    
    # Version1.0 -- May/2015
# Written by Di Wang
    
    
    ## Initialization
    row=np.shape(X1)[0]
    rowt=np.shape(X2)[0]
    colL=np.shape(L)[1]
    U1=np.random.rand(row,bits)
    U2=np.random.rand(rowt,bits)
    Z=np.random.rand(colL,bits)
    P1=np.random.rand(row,bits)
    P2=np.random.rand(rowt,bits)

    threshold=0.01
    lastF=99999999
    iter=1
    obj=np.zeros(maxIter,1)
    x = TRUE

    while x:

        # update U1 and U2
        U1 = np.linalg.solve(np.dot(np.dot(X1 , L) , Z) , np.sum(np.dot(np.dot(np.transpose(Z), np.transpose(L)) , np.dot(L , Z)) , np.dot(np.linalg.solve(gamma, lambda_) , np.eye(bits))))
        U2 = np.linalg.solve(np.dot(np.dot(X2 , L) , Z) , np.sum(np.dot(np.dot(np.transpose(Z), np.transpose(L)) , np.dot(L , Z)) + np.dot(np.linalg.solve(gamma,(1-lambda_) ), np.eye(bits)))

        Z_left=np.linalg.solve(np.dot(np.transpose(L),L) , np.dot(np.dot(lambda_ , np.transpose(L)),np.dot(np.transpose(X1), U1)) + np.dot(np.dot((1 - lambda_),np.transpose(L)),np.dot(np.transpose(X2),U2)) + np.dot(np.dot(mu,np.transpose(L)),np.dot(np.transpose(X1),P1)) + np.dot(np.dot(mu,np.transpose(L)),np.dot(np.transpose(X2),P2)))
        Z=np.linalg.solve(Z_left , np.dot(np.dot(lambda_,np.transpose(U1)),U1) + np.dot(np.dot((1 - lambda_),np.transpose(U2)),U2) + np.dot((np.dot(2,mu) + gamma),np.eye(bits)))

        P1=np.linalg.solve(np.dot(X1,np.transpose(X1)) + np.linalg.solve(gamma , np.dot(mu,np.eye(row))),(np.dot(np.dot(X1,L),Z))
        P2=np.linalg.solve(np.dot(X2,np.transpose(X2) + np.linalg.solve(gamma , np.dot(mu,np.eye(rowt)), np.dot(np.dot(X2,L),Z)

        norm1=np.dot(lambda_,np.linalg.cond(X1 - np.dot(np.dot(U1,np.transpose(Z)),np.transpose(L)),'fro'))
        norm2=np.dot((1 - lambda_),np.linalg.cond(X2 - np.dot(np.dot(U2,np.transpose(Z)),np.transpose(L)),'fro'))
        norm3=np.dot(mu,np.linalg.cond(np.dot(L,Z) - np.dot(np.transpose(X1),P1),'fro')) + np.dot(mu,np.linalg.cond(np.dot(L,Z) - np.dot(np.transpose(X2),P2),'fro'))
        norm4=np.dot(gamma,(np.linalg.cond(U1,'fro') + np.linalg.cond(U2,'fro') + np.linalg.cond(dot(L,Z),'fro') + np.linalg.cond(P1,'fro') + np.linalg.cond(P2,'fro')))
        currentF=norm1 + norm2 + norm3 + norm4
        obj[iter]=currentF

        print('\nobj at iteration %d: %.4f\n reconstruction error for image: %.4f,\n reconstruction error for text: %.4f,\n reconstruction error for linear projection: %.4f,\n regularization term: %.4f\n\n',iter,currentF,norm1,norm2,norm3,norm4)
        if (lastF - currentF) < threshold:
            print('algorithm converges...\n')
            print('final obj: %.4f\n reconstruction error for image: %.4f,\n reconstruction error for text: %.4f,\n reconstruction error for linear projection: %.4f,\n regularization term: %.4f\n\n',currentF,norm1,norm2,norm3,norm4)
            return P1,P2,Z,U1,U2,obj
        if iter >= maxIter:
            return P1,P2,Z,U1,U2,obj
        iter=iter + 1

        lastF=copy(currentF)


    
    return P1,P2,Z,U1,U2,obj
    return P1,P2,Z,U1,U2,obj
    
if __name__ == '__main__':
    pass
    