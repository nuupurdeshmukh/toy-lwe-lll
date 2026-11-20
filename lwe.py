# toy regev style lwe encrypting one bit
# n is the secret length m the number of samples q a prime below 65536
# noise is centered binomial cbd(eta) like in ml-kem
# randomness comes from shake-256

import hashlib
import math
from fractions import Fraction
from math import comb

from lll import dot


class Xof:
    # shake-256 as a stream of bytes
    # a longer digest of the same seed starts with the shorter one so we just regrow the buffer
    def __init__(self,seed):
        self.seed=seed
        self.buf=b""
        self.pos=0

    def read(self,k):
        if self.pos+k>len(self.buf):
            self.buf=hashlib.shake_256(self.seed).digest(max(2*len(self.buf),self.pos+k,1024))
        out=self.buf[self.pos:self.pos+k]
        self.pos+=k
        return out


def uniform(xof,q):
    # rejection sampling so there is no modulo bias
    limit=(65536//q)*q
    while True:
        x=int.from_bytes(xof.read(2),"little")
        if x<limit:
            return x%q


def cbd(xof,eta):
    # sum of eta coins minus sum of eta coins
    bits=xof.read(2*eta)
    a=sum(x&1 for x in bits[:eta])
    b=sum(x&1 for x in bits[eta:])
    return a-b


def keygen(n,m,q,eta,seed):
    # e is returned only for experiments and is never used to decrypt
    xof=Xof(seed)
    A=[[uniform(xof,q) for _ in range(n)] for _ in range(m)]
    s=[uniform(xof,q) for _ in range(n)]
    e=[cbd(xof,eta) for _ in range(m)]
    b=[(dot(A[i],s)+e[i])%q for i in range(m)]
    return (A,b),s,e


def encrypt(pk,bit,q,seed):
    # r picks a random subset of the m samples
    A,b=pk
    m,n=len(A),len(A[0])
    xof=Xof(seed)
    r=[x&1 for x in xof.read(m)]
    u=[sum(r[i]*A[i][j] for i in range(m))%q for j in range(n)]
    v=(dot(r,b)+bit*(q//2))%q
    return u,v


def decrypt(s,ct,q):
    # v-u.s is r.e+bit*(q//2) mod q so round to 0 or q/2
    u,v=ct
    d=(v-dot(u,s))%q
    c=d if d<=q//2 else d-q
    return 1 if abs(c)>q//4 else 0


def failure_exact(m,q,eta):
    # exact probability that bit 0 decrypts wrong
    # that is P(|r.e|>q//4) with r uniform in {0,1}^m and each e_i from cbd(eta)
    # return a Fraction
    # if h of the r_i are 1 then r.e is a sum of h cbd(eta) values
    # that is h*eta coins minus h*eta coins
    # a subtracted coin c equals (1-c)-1 and 1-c is again a fair coin
    # so r.e = X-h*eta with X ~ Binomial(2*h*eta,1/2)
    t=q//4
    total=Fraction(0)
    for h in range(m+1):
        k=h*eta
        tail=sum(comb(2*k,j) for j in range(2*k+1) if abs(j-k)>t)
        total+=Fraction(comb(m,h)*tail,2**m*4**k)
    return total


def failure_bound(m,q,eta):
    # an upper bound on failure_exact from a tail inequality
    # return a float
    # given h, r.e = X-h*eta is a centered sum of 2*h*eta fair coins
    # r.e is an integer so |r.e|>t means |r.e|>=t+1
    # hoeffding gives P(|X-EX|>=a) <= 2*exp(-2a^2/(2*h*eta)) = 2*exp(-a^2/(h*eta))
    # this grows with h so h=m bounds every term of the average over h
    a=q//4+1
    return min(1.0,2*math.exp(-a*a/(m*eta)))


if __name__=="__main__":
    pk,s,e=keygen(8,16,3329,2,b"demo")
    for bit in [0,1]:
        ct=encrypt(pk,bit,3329,b"r%d"%bit)
        print(bit,decrypt(s,ct,3329))
