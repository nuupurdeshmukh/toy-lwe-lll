# primal lattice attack on toy lwe
# b=A*s+e mod q with e small so b is close to the lattice point A*s
# we add b as an extra basis vector so (e,1) becomes a very short vector
# then lll finds it and we read off e and solve for s

import math

from lll import lll


def inv_mod_matrix(M,q):
    # gauss jordan mod a prime q
    n=len(M)
    A=[[x%q for x in row]+[int(i==j) for j in range(n)] for i,row in enumerate(M)]
    for c in range(n):
        p=next((r for r in range(c,n) if A[r][c]!=0),None)
        if p is None:
            raise ValueError("matrix not invertible mod q")
        A[c],A[p]=A[p],A[c]
        inv=pow(A[c][c],-1,q)
        A[c]=[x*inv%q for x in A[c]]
        for r in range(n):
            if r!=c and A[r][c]!=0:
                f=A[r][c]
                A[r]=[(a-f*b)%q for a,b in zip(A[r],A[c])]
    return [row[n:] for row in A]


def qary_basis(A,q):
    # basis of L={x in Z^m : x=A*z mod q}
    # split A into A1 (first n rows) and A2 (the rest)
    # any x in L has x[n:]=A2*A1^-1*x[:n] mod q
    # so the basis is [I | C^T] on top and [0 | q*I] below with C=A2*A1^-1
    # det of this basis is q^(m-n)
    m,n=len(A),len(A[0])
    A1inv=inv_mod_matrix(A[:n],q)
    A2=A[n:]
    C=[[sum(A2[i][k]*A1inv[k][j] for k in range(n))%q for j in range(n)] for i in range(m-n)]
    B=[]
    for i in range(n):
        B.append([int(i==j) for j in range(n)]+[C[k][i] for k in range(m-n)])
    for j in range(m-n):
        B.append([0]*n+[q*int(j==k) for k in range(m-n)])
    return B,A1inv


def embed(B,b,t):
    # kannan embedding
    # the new lattice contains (b-A*s-q*k,t)=(e,t)
    return [row+[0] for row in B]+[list(b)+[t]]


def attack(A,b,q,eta,reduce=lll):
    # returns s or None
    # reduce can be swapped for fpylll in bigger experiments
    m,n=len(A),len(A[0])
    B,A1inv=qary_basis(A,q)
    R=reduce(embed(B,b,1))
    for x in R:
        if abs(x[-1])==1 and all(abs(y)<=eta for y in x[:m]):
            e=[y*x[-1] for y in x[:m]]
            rhs=[(b[i]-e[i])%q for i in range(n)]
            s=[sum(A1inv[i][k]*rhs[k] for k in range(n))%q for i in range(n)]
            ok=all((sum(A[i][j]*s[j] for j in range(n))+e[i]-b[i])%q==0 for i in range(m))
            if ok:
                return s
    return None


def predict_success(n,m,q,eta,delta0,tau):
    # lll finds (e,1) when it is shorter than a typical shortest vector
    # by more than what lll loses which is about delta0^d
    # lam1 is the expected length of (e,1) since each e_i has variance eta/2
    # gh is the gaussian heuristic for the shortest vector of a random lattice
    # tau is a constant fitted on data
    d=m+1
    lam1=math.sqrt(m*eta/2+1)
    gh=math.sqrt(d/(2*math.pi*math.e))*math.exp((m-n)*math.log(q)/d)
    return gh/lam1>=tau*delta0**d


if __name__=="__main__":
    from lwe import keygen
    pk,s,e=keygen(4,10,97,1,b"demo")
    print("secret   ",s)
    print("recovered",attack(pk[0],pk[1],97,1))
