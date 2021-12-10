import enum
import numpy as np
import gzip,pickle
import random
import math
import sys


def FairRec(U,P,k,V,alpha,Value=None):
    # Allocation set for each customer, initially it is set to empty set
    # feasible set for each customer, initially it is set to P
    A,F ={},{}
    for _,u in enumerate(U):
        A[u]=[]

    for _,u in enumerate(U):
        F[u]=P[:]
    
    # number of copies of each producer
    l = alpha*m*k
    l /= (n+0.0)
    l=int(l)

    # R= number of rounds of allocation to be done in first GRR
    R = (l*n)
    R /= (m+0.0)
    R = math.ceil(R)
    R = int(R)  

    # total number of copies to be allocated
    T = l*n

    u_less=[] # customers allocated with <k products till now
       
    # first greedy round-robin allocation
    [B,F1]=greedy_round_robin(m,n,R,l,T,V,U[:],F.copy())
    F={}
    F=F1.copy()
    print("GRR done")
    # adding the allocation
    for _,u in enumerate(U):        
        A[u]=A[u][:]+B[u][:]
    
    # second phase
    for _,u in enumerate(A):
        if len(A[u])>=k:
            pass
        else:
            u_less.append(u)

    # allocating every customer till k products
    for _, u in enumerate(u_less):
        new=(V[u,:]).argsort()[-(k+k):][::-1]
        for _, p in enumerate(new):
            if p in A[u]:
                pass
            else:
                A[u].append(p)

            if len(A[u])==k:
                break

    return A

def greedy_round_robin(m,n,R,l,T,V,U,F,Value=None): 
    # greedy round robin allocation based on a specific ordering of customers (assuming the ordering is done in the relevance scoring matrix before passing it here)
    
    # creating empty allocations, total availability
    B,Z={},{}
    for _, u in enumerate(U):
        B[u]=[]
    
    # available number of copies of each producer
    P=range(0,n,1) # set of producers
    for _, p in enumerate(P):
        Z[p]=l
    
    # allocating the producers to customers
    for t in range(1,R+1,1):
        print("GRR round number==============================",t)
        for i in range(0,m,1):
            if T!=0:
                u=U[i]
                # choosing the p_ which is available and also in feasible set for the user
                possible=[(Z[p]>0)*(p in F[u])*V[u,p] for p in range(0,n,1)] 
                p_=np.argmax(possible) 
                
                if  len(F[u])>0 and (p_ in F[u]):
                    if (Z[p_]>0):
                        B[u].append(p_)
                        F[u].remove(p_)
                        Z[p_]-=1
                        T+=-1
                else:
                    return B,F
            else:
                return B,F

    # returning the allocation
    return B,F

if __name__== "__main__":
    # dataset
    dataset=sys.argv[1]
    # relevance scoring data
    V=np.loadtxt(dataset,delimiter=',')
    
    print("relevance scoring data loaded")

    m,n=V.shape # number of customers, number of producers
    
    
    U,P=range(0,m,1),range(0,n,1) # list of customers, list of producers
 
    # size of recommendation
    arg2 = sys.argv[2]
    reco_size=int(arg2)

    # fraction of MMS to be guaranteed to every producer
    arg3 = sys.argv[3]
    alpha=float(arg3)

    # calling FairRec
    A=FairRec(U,P,reco_size,V,alpha)

    # saving the results in pickle format (dictionary format { <customer> : <recommended_products_list> })
    file_name = dataset[:-4]+"_"+str(alpha)+"_k_"+str(reco_size)+".pkl.gz"
    f_out=gzip.open(file_name,"wb")
    pickle.dump(A,f_out,-1)
    f_out.close()

