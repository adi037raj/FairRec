import enum
import numpy as np
import gzip,pickle
import math
import random
import networkx as nx
from itertools import permutations
import sys
import datetime

def remove_envy_cycle(B,U,V,Value=None):
    r=0
    while True:
        r-=-1
        print("In envy cycle removal:",r)
        # edges
        E=[]
        # create empty graph
        G=nx.DiGraph()
        # add nodes
        G.add_nodes_from(U)
        
        # find edges
        print("In envy cycle removal: finding edges")
        for _, u in enumerate(U):
            for _, v in enumerate(U):
                if u==v:
                    pass
                else:
                    V_u,V_v=0,0
                    for _,p in enumerate(B[u]):
                        V_u=V_u+V[u,p]
                    
                    for _,p in enumerate(B[v]):
                        V_v=V_v+V[u,p]
                    
                    if V_v<=V_u:
                        pass
                    else:
                        E.append((u,v))
        
        # add edges to the graph
        G.add_edges_from(E) 
        # find cycle and remove
        print("In envy cycle removal: graph done, finding and removing cycles")        
        
        try:
            cycle=nx.find_cycle(G,orientation="original")
            cyc_val = cycle[0][0]
            temp=(B[cyc_val])[:]
            for _,pair in enumerate(cycle):
                B[pair[0]]=B[pair[1]][:]
            cyc_val = cycle[-1][0]
            B[cyc_val]=temp[:]
        except:
            break
    # topological sort
    t_list = nx.topological_sort(G)
    U=list(t_list)
    return B.copy(),U[:]


def FairRecPlus(U,P,k,V,alpha):    
    # Allocation set for each customer, initially it is set to empty set
    A, F = {},{}
    for _, u in enumerate(U):
        A[u]=[]
    
    # feasible set for each customer, initially it is set to P
    for _, u in enumerate(U):
        F[u]=P[:]
   
    # l= number of copies of each producer, equal to the exposure guarantee for producers
    l = alpha*m*k
    l/=(n+0.0)
    l=int(l)

    # R= number of rounds of allocation to be done in first GRR
    R = (l*n)
    R /= (m+0.0)
    R = math.ceil(R)
    R=int(R)    

    # T= total number of products to be allocated
    T= l*n
       
    # first greedy round-robin allocation
    B={}
    [B,F1]=greedy_round_robin(m,n,R,l,T,V,U[:],F.copy())
    F={}
    F=F1.copy()
    print("GRR done")
    # adding the allocation
    for _, u in enumerate(U):
        A[u]=A[u][:]+B[u][:]
        
    # filling the recommendation set upto size k
    u_less=[]
    for _,u in enumerate(A):
        if len(A[u])<k:
            u_less.append(u)
    for _,u in enumerate(u_less):
        scores=V[u,:]
        new=scores.argsort()[-(k+k):][::-1]
        for _, p in enumerate(new):
            if p in A[u]:
                pass
            else:
                A[u].append(p)

            if len(A[u])==k:
                break
    
    end_time=datetime.datetime.now()    
    
    return A

# greedy round robin allocation based on a specific ordering of customers
# This is the modified greedy round robin where we remove envy cycles
def greedy_round_robin(m,n,R,l,T,V,U,F,Value=None):
    print(m,n,R,l,T,V.shape,len(U))

    # creating empty allocations, total availability
    B,Z={},{}
    for _, u in enumerate(U):
        B[u]=[]
    
    # available number of copies of each producer
    P=range(0,n,1) # set of producers
    for _,p in enumerate(P):
        Z[p]=l
    
    # number of rounds
    r=0
    while True:
        # number of rounds
        r-=-1
        # allocating the producers to customers
        print("GRR round number==============================",r)
        
        for i in range(0,m,1):
            #user
            u=U[i]
            
            # choosing the p_ which is available and also in feasible set for the user
            possible=[(Z[p]>0)*(p in F[u])*V[u,p] for p in range(0,n,1)] 
            p_=np.argmax(possible)                             
                
            if (p_ in F[u]) and len(F[u])>0:
                if (Z[p_]>0):
                    B[u].append(p_)
                    F[u].remove(p_)
                    Z[p_]=Z[p_]-1
                    T=T-1    
            else: #stopping criteria                
                print("now doing envy cycle removal")
                B,U=remove_envy_cycle(B.copy(),U[:],V)
                return B.copy(),F.copy()
            
            if T!=0: #stopping criteria
                pass
            else:                
                print("now doing envy cycle removal")
                B,U=remove_envy_cycle(B.copy(),U[:],V)              
                return B.copy(),F.copy()
        # envy-based manipulations, m, U, V, B.copy()
        print("GRR done")        
      
        # remove envy cycle
        print("now doing envy cycle removal")
        B,U=remove_envy_cycle(B.copy(),U[:],V)
        t_sum = sum([len(B[u]) for u in B])
        print(t_sum,T,n*l)
    # returning the allocation
    
    return B.copy(),F.copy()

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
    A=FairRecPlus(U,P,reco_size,V,alpha)

    # saving the results in pickle format (dictionary format { <customer> : <recommended_products_list> })
    file_name = dataset[:-4]+"_"+str(alpha)+"_k_"+str(reco_size)+".pkl.gz"
    f_out=gzip.open(file_name,"wb")
    pickle.dump(A,f_out,-1)
    f_out.close()
