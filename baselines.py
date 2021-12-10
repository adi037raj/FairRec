import enum
import numpy as np
import gzip,pickle
import random,math
import operator

# Baseline: mixedTP-k (mix of top-k/2 and poorest-k/2)
# INPUTS: fname= csv file with relevance scores, k= recommendation size
def generate_mixedTP_k(fname,k,delim=','):
    # relevance scores
    V=np.loadtxt(fname,delimiter=delim)
    
    m,n =V.shape #number of customers, number of producers
    
    U,P = range(m), range(n) #Customers, Producers
    
    # Recommendations , Exposures
    B,E={},{}

    for p in range(n):
        E[p]=0.0

    for _, u in enumerate(U):
        B[u]=[]
        scores=V[u,:]
        #top-k/2
        top_half=scores.argsort()[-int(math.ceil((k+0.0)/2)):][::-1]
        for _, p in enumerate(top_half):
            B[u].append(p)
        # producers sorted based on increasing exposures and allocating the first feasible producer
        prod_sorted=sorted(E.items(),key=operator.itemgetter(1))
        prod_index=0
        while True:
            if len(B[u])<k:
                break
            p=prod_sorted[prod_index][0]
            if p in B[u]:
                prod_index= prod_index + 1
            else:
                B[u].append(p)
                prod_index=prod_index + 1
        for _, p in enumerate(B[u]):
            E[p]=E[p] + 1.0
    # Saving the results in pickle format
    file_name = fname[:-4]+"_mixedTP_k_"+str(k)+".pkl.gz"
    f_out=gzip.open(file_name,"wb")
    pickle.dump(B,f_out,-1)
    f_out.close()

# Baseline: poorest-k
# INPUTS: fname= csv file with relevance scores, k= recommendation size
def generate_poorest_k(fname,k,delim=','):
    # relevance scores
    V=np.loadtxt(fname,delimiter=delim)
    
    (m,n) = V.shape #number of customers, number of producers

    U, P=range(m),range(n) #Customers, Producers

    # Exposures
    E={}
    for _, p in enumerate(P):
        E[p]=0.0
        
    # poorest-k
    B={}
    for _, u in enumerate(U):
        B[u]=[]
    
    # greedy round robin of producer-centric allocation: poorest-k
    for i in range(k):
        for _, u in enumerate(U):
            # producers sorted based on increasing exposures and allocating the first feasible producer
            items = E.items()
            op_key = operator.itemgetter(1)
            prod_sorted=sorted(items,key=op_key)
            for _, p_tuple in enumerate(prod_sorted):
                p=p_tuple[0]
                if p in B[u]:
                    pass
                else:
                    E[p]+=1
                    B[u].append(p)
                    break
    # Saving the results in pickle format
    file_name = fname[:-4]+"_poorest_k_"+str(k)+".pkl.gz"
    f_out=gzip.open(file_name,"wb")
    pickle.dump(B,f_out,-1)
    f_out.close()

# Baseline: mixedTR-k (mix of top-k/2 and random-k/2)
# INPUTS: fname= csv file with relevance scores, k= recommendation size
def generate_mixedTR_k(fname,k,delim=','):
    # relevance scores
    V=np.loadtxt(fname,delimiter=delim)
    
    (m,n)=V.shape #number of customers
    
    U,P=range(m),range(n) #Customers, Producers

    # mixedTR-k
    B={}

    for _, u in enumerate(U):
        scores=V[u,:]
        l = (k+0.0)/2
        l = math.ceil(l)
        l=int(l)
        half=(scores.argsort()[-l:])[::-1]
        remaining_P=[]
        for _,p in enumerate(P):
            if p in half:
                pass
            else:
                remaining_P.append(p)

        other_half=random.sample(remaining_P,int(k-l))

        B[u]=[]
        # first half from top-k/2
        for _, i in enumerate(half):
            B[u].append(i)
        # second half from random-k/2
        for _, i in enumerate(other_half):
            B[u].append(i)        
    # Saving the results in pickle format
    file_name = fname[:-4]+"_mixedTR_k_"+str(k)+".pkl.gz"
    f_out=gzip.open(file_name,"wb")
    pickle.dump(B,f_out,-1)
    f_out.close()
    

# Baselines: random-k and top-k
# INPUTS: fname= csv file with relevance scores, k= recommendation size
def generate_random_n_top_k(fname,k,delim=','):
    # relevance scores
    V=np.loadtxt(fname,delimiter=delim)
    
    m,n=V.shape #number of customers, number of producers
    

    U,P=range(m),range(n) #Customers, Producers 
 
    # baseline1 = top-k recommendations, baseline2 = random-k recommendations
    B1,B2={},{}

    for _, u in enumerate(U):
        B1[u]=(V[u,:]).argsort()[-k:][::-1]
        B2[u]=random.sample(P,k)
    # Saving the results in pickle format
    file_name1 = fname[:-4]+"_top_k_"+str(k)+".pkl.gz"
    f_out=gzip.open(file_name1,"wb")
    pickle.dump(B1,f_out,-1)
    f_out.close()
    
    file_name1 = fname[:-4]+"_random_k_"+str(k)+".pkl.gz"
    f_out=gzip.open(fname[:-4]+"_random_k_"+str(k)+".pkl.gz","wb")
    pickle.dump(B2,f_out,-1)
    f_out.close()
    