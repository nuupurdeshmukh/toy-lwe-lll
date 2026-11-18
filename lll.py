# lll reduction with exact fractions
# a basis is a list of rows so B[i] is the ith vector
# indices start at 0

from fractions import Fraction
import math


def dot(u,v):
    return sum(a*b for a,b in zip(u,v))


def gram_schmidt(B):
    # returns Bstar mu Bn
    # Bstar[i] is b*_i (no normalising)
    # mu[i][j] is <b_i,b*_j>/<b*_j,b*_j> for j<i
    # Bn[i] is <b*_i,b*_i>
    # raise ValueError if the rows are dependent
    n=len(B)
    Bstar=[]
    Bn=[]
    mu=[[Fraction(0)]*n for _ in range(n)]
    for i in range(n):
        v=[Fraction(x) for x in B[i]]
        for j in range(i):
            mu[i][j]=Fraction(dot(B[i],Bstar[j]))/Bn[j]
            v=[a-mu[i][j]*c for a,c in zip(v,Bstar[j])]
        nv=dot(v,v)
        if nv==0:
            raise ValueError("rows are linearly dependent")
        Bstar.append(v)
        Bn.append(nv)
    return Bstar,mu,Bn


def nearest_int(x):
    # nearest integer, halves round up (python's round() rounds halves to even)
    return math.floor(x+Fraction(1,2))


def lll(B,delta=Fraction(99,100)):
    # returns an lll reduced basis of the same lattice
    # dont change the input B
    B=[list(row) for row in B]
    n=len(B)
    Bstar,mu,Bn=gram_schmidt(B)
    k=1
    while k<n:
        # size reduce b_k against b_{k-1}...b_0
        for j in range(k-1,-1,-1):
            q=nearest_int(mu[k][j])
            if q!=0:
                B[k]=[a-q*c for a,c in zip(B[k],B[j])]
                # b*_k stays the same, only row k of mu changes
                for i in range(j):
                    mu[k][i]-=q*mu[j][i]
                mu[k][j]-=q
        # lovasz condition
        if Bn[k]+mu[k][k-1]**2*Bn[k-1]>=delta*Bn[k-1]:
            k+=1
        else:
            B[k],B[k-1]=B[k-1],B[k]
            # recompute from scratch, simple and fine for toy sizes
            Bstar,mu,Bn=gram_schmidt(B)
            k=max(k-1,1)
    return B


def is_lll_reduced(B,delta=Fraction(99,100)):
    # true if |mu[i][j]|<=1/2 for all j<i
    # and Bn[k]+mu[k][k-1]^2*Bn[k-1]>=delta*Bn[k-1] for all k>=1
    Bstar,mu,Bn=gram_schmidt(B)
    n=len(B)
    for i in range(n):
        for j in range(i):
            if abs(mu[i][j])>Fraction(1,2):
                return False
    for k in range(1,n):
        if Bn[k]+mu[k][k-1]**2*Bn[k-1]<delta*Bn[k-1]:
            return False
    return True


if __name__=="__main__":
    # ans is [[-1, 1], [2, 2]]
    print(lll([[3,1],[2,2]]))
