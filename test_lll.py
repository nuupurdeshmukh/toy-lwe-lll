# tests for lll.py
# run with python3 test_lll.py or pytest test_lll.py
# everything checked here is a proven property so a failure means a bug

import random
from fractions import Fraction

from lll import dot,gram_schmidt,lll,is_lll_reduced

DELTA=Fraction(99,100)
ALPHA=1/(DELTA-Fraction(1,4))


# exact determinant using gaussian elimination
def det(M):
    A=[[Fraction(x) for x in row] for row in M]
    n=len(A)
    d=Fraction(1)
    for c in range(n):
        p=next((r for r in range(c,n) if A[r][c]!=0),None)
        if p is None:
            return Fraction(0)
        if p!=c:
            A[c],A[p]=A[p],A[c]
            d=-d
        d*=A[c][c]
        for r in range(c+1,n):
            f=A[r][c]/A[c][c]
            A[r]=[a-f*b for a,b in zip(A[r],A[c])]
    return d


# exact inverse using gauss jordan
def inverse(M):
    n=len(M)
    A=[[Fraction(x) for x in row]+[Fraction(int(i==j)) for j in range(n)]
       for i,row in enumerate(M)]
    for c in range(n):
        p=next(r for r in range(c,n) if A[r][c]!=0)
        A[c],A[p]=A[p],A[c]
        piv=A[c][c]
        A[c]=[x/piv for x in A[c]]
        for r in range(n):
            if r!=c and A[r][c]!=0:
                f=A[r][c]
                A[r]=[a-f*b for a,b in zip(A[r],A[c])]
    return [row[n:] for row in A]


def matmul(X,Y):
    return [[sum(X[i][k]*Y[k][j] for k in range(len(Y))) for j in range(len(Y[0]))]
            for i in range(len(X))]


# same lattice means B_out=U*B_in with U integer and det U=+-1
def same_lattice(B_in,B_out):
    U=matmul(B_out,inverse(B_in))
    if any(Fraction(x).denominator!=1 for row in U for x in row):
        return False
    return abs(det(U))==1


def random_basis(n,bound,rng):
    while True:
        B=[[rng.randint(-bound,bound) for _ in range(n)] for _ in range(n)]
        if det(B)!=0:
            return B


# start from identity and do random integer row ops so det stays +-1
def random_unimodular(n,steps,rng):
    U=[[int(i==j) for j in range(n)] for i in range(n)]
    for _ in range(steps):
        i,j=rng.sample(range(n),2)
        c=rng.choice([-2,-1,1,2])
        U[i]=[a+c*b for a,b in zip(U[i],U[j])]
        if rng.random()<0.3:
            U[i],U[j]=U[j],U[i]
    return U


def test_gram_schmidt_example():
    Bstar,mu,Bn=gram_schmidt([[3,1],[2,2]])
    assert mu[1][0]==Fraction(4,5)
    assert Bstar[1]==[Fraction(-2,5),Fraction(6,5)]
    assert Bn==[10,Fraction(8,5)]


def test_gram_schmidt_orthogonal_and_det():
    rng=random.Random(1)
    for n in range(2,7):
        B=random_basis(n,20,rng)
        Bstar,mu,Bn=gram_schmidt(B)
        for i in range(n):
            for j in range(i):
                assert dot(Bstar[i],Bstar[j])==0
        # product of |b*_i|^2 should be det^2
        prod=Fraction(1)
        for x in Bn:
            prod*=x
        assert prod==det(B)**2


def test_gram_schmidt_rejects_dependent():
    try:
        gram_schmidt([[1,2],[2,4]])
    except ValueError:
        return
    raise AssertionError("dependent rows should raise ValueError")


def test_hand_example():
    assert lll([[3,1],[2,2]])==[[-1,1],[2,2]]


def test_input_not_modified():
    B=[[3,1],[2,2]]
    lll(B)
    assert B==[[3,1],[2,2]]


def test_is_lll_reduced_detects_bad_bases():
    assert is_lll_reduced([[-1,1],[2,2]])
    assert not is_lll_reduced([[3,1],[2,2]])    # mu is 0.8
    assert not is_lll_reduced([[10,0],[0,1]])   # lovasz fails


def test_random_bases():
    rng=random.Random(2)
    for n in range(2,9):
        for _ in range(4):
            B=random_basis(n,50,rng)
            R=lll(B)
            assert is_lll_reduced(R),(n,B)
            assert same_lattice(B,R),(n,B)
            # hermite bound |b1|^(2n)<=alpha^(n(n-1)/2)*det^2
            assert Fraction(dot(R[0],R[0]))**n<=ALPHA**(n*(n-1)//2)*det(B)**2


# the lattice here is Z^n so lambda_1=1
# lll guarantees |b1|^2<=alpha^(n-1)
def test_hidden_Zn():
    rng=random.Random(3)
    exact_units=0
    trials=0
    for n in range(2,9):
        for _ in range(4):
            U=random_unimodular(n,6*n,rng)
            R=lll(U)
            assert is_lll_reduced(R)
            assert dot(R[0],R[0])<=ALPHA**(n-1)
            trials+=1
            exact_units+=all(dot(r,r)==1 for r in R)
    print(f"  hidden Z^n: got the exact unit basis in {exact_units}/{trials} cases")


if __name__=="__main__":
    tests=[(name,f) for name,f in sorted(globals().items())
           if name.startswith("test_") and callable(f)]
    failed=0
    for name,f in tests:
        try:
            f()
            print(f"PASS  {name}")
        except NotImplementedError:
            failed+=1
            print(f"TODO  {name}")
        except Exception as e:
            failed+=1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
