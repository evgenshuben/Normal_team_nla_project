import numpy as np
from scipy import linalg, sparse 
from math import ceil

class RandPca:
    def __init__(self) :
        return
  
    def randomized_subspace_iteration(self, A, l, q, u) : 
        (m,n) = A.shape
        l = np.min([m,l])
        Y = A.dot(np.random.normal(size=[n,l]))
        
        [Q,R] = np.linalg.qr(Y, mode='reduced')
    
        Q = Q.astype('float32') 
        R = R.astype('float32') 
        u = u.astype('float32') 
        v = np.ones([R.shape[1],1], dtype='float32')
        o = np.ones([n,1], dtype='float32')
    
        if np.sum(u) != 0: 
            [Q,R] = linalg.qr_update(Q, R, u, v)
        for i in range(q):
            [G,R] = np.linalg.qr(A.T.dot(Q) + o.dot(u.T.dot(Q)), mode='reduced')
            [Q,R] = np.linalg.qr(A.dot(G) + u.dot(o.T.dot(G)), mode='reduced')
        return Q,R
  
    def rsvd(self, A, k, q, l, centre = True) :
        (m,n) = A.shape
        l = np.max([l, 2*q]) 
        E = A.mean(1).reshape([m,1])
        u = -E if centre else np.zeros(E.shape)
        
        [Q,R] = self.randomized_subspace_iteration(A, l, q, u)
        #B = Q.T.dot(A) - (Q.T.dot(E) if centre else 0)
        B = (A.T.dot(Q) - (E.T.dot(Q) if centre else 0)).T # to speed up the operation in case A is a sparse matrix
    
        [U,S,V] = linalg.svd(B, full_matrices=False)
        U = Q.dot(U)
        k = np.min([k, U.shape[1]])
        U = U[:,0:k]
        S = S[0:k]
        V = V[0:k,:]
        return U, S, V
  
    def rpca(self, A, k, centre = True) :
        (m,n) = A.shape
        assert(k <= m)
        [U,S,V] = self.rsvd(A, k, 1, 2*k)
        return sparse.diags(S).dot(V)
  
    def reconstruction_error(self, A, k, centre = True, q=0) :
        (m,n) = A.shape
        assert(k <= m)
        [U,S,V] = self.rsvd(A, k, q, 2*k, centre)
    
        if centre == True :
            E = A.mean(1).reshape([m,1])
        else :
            E = np.zeros([m,1])
    
        #if centre: 
        #  B = A - A.mean(1)
        #else:
        #  B = A
        #[U,S,V] = np.linalg.svd(B)
        #U = U[:,0:k]
        #S = S[0:k]
        #V = V[0:k,:]
    
        bs = 5*1024
        nb = int(ceil(n*1.0 / bs))
        col_loss = np.zeros([n,])
        US = U.dot(np.diag(S))
        for b in range(0, nb) :
            i = b*bs 
            j = np.min([i + bs, n])
            v = V[:,i:j]
            B = US.dot(v) + E
            col_loss[i:j] = np.sum(np.power(A[:,i:j] - B, 2), 0)
          
        return col_loss
  
    def low_rank_approximation(self, A, k, q=0, centre = True) :
        (m,n) = A.shape
        assert(k <= m)
        [U,S,V] = self.rsvd(A, k, q, 2*k, centre)
    
        if centre == True :
            E = A.mean(1).reshape([m,1])
        else :
            E = np.zeros([m,1])
    
        B = U.dot(np.diag(S).dot(V)) + E
    
        Err = np.sum(np.power(A - B, 2),0)
        return B, Err
  