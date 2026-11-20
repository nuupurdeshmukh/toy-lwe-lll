# tests for lwe.py
# run with python3 test_lwe.py

import itertools
from fractions import Fraction

from lwe import Xof,uniform,cbd,keygen,encrypt,decrypt,failure_exact,failure_bound


def test_xof_same_seed_same_bytes():
    assert Xof(b"a").read(50)==Xof(b"a").read(50)
    assert Xof(b"a").read(50)!=Xof(b"b").read(50)


def test_xof_stream_matches_one_read():
    x=Xof(b"a")
    out=x.read(10)+x.read(3000)+x.read(7)
    assert out==Xof(b"a").read(3017)


def test_uniform_range():
    x=Xof(b"u")
    for q in [2,7,97,3329]:
        vals=[uniform(x,q) for _ in range(2000)]
        assert min(vals)>=0 and max(vals)<q


def test_cbd_range_mean_variance():
    x=Xof(b"c")
    for eta in [1,2,3]:
        vals=[cbd(x,eta) for _ in range(100000)]
        assert min(vals)>=-eta and max(vals)<=eta
        mean=sum(vals)/len(vals)
        var=sum(v*v for v in vals)/len(vals)-mean**2
        assert abs(mean)<0.02
        assert abs(var-eta/2)<0.03


def test_decrypt_correct_small_noise():
    for t in range(100):
        pk,s,e=keygen(8,16,3329,1,b"k%d"%t)
        bit=t%2
        assert decrypt(s,encrypt(pk,bit,3329,b"r%d"%t),3329)==bit


# brute force over every r and every e for tiny m
def brute_failure(m,q,eta):
    probs={}
    for bits in itertools.product([0,1],repeat=2*eta):
        v=sum(bits[:eta])-sum(bits[eta:])
        probs[v]=probs.get(v,Fraction(0))+Fraction(1,2**(2*eta))
    total=Fraction(0)
    for r in itertools.product([0,1],repeat=m):
        for e in itertools.product(probs,repeat=m):
            p=Fraction(1,2**m)
            for ei in e:
                p*=probs[ei]
            if abs(sum(a*b for a,b in zip(r,e)))>q//4:
                total+=p
    return total


def test_failure_exact_small_case():
    assert failure_exact(3,7,1)==Fraction(19,256)


def test_failure_exact_matches_brute_force():
    for m,q,eta in [(2,5,1),(3,7,1),(3,5,2),(4,9,1),(2,7,3)]:
        assert failure_exact(m,q,eta)==brute_failure(m,q,eta),(m,q,eta)


def test_failure_bound_is_above_exact():
    for m in [10,40,100]:
        for q in [23,97,257,3329]:
            for eta in [1,2,3]:
                assert failure_bound(m,q,eta)>=failure_exact(m,q,eta),(m,q,eta)


def test_failure_exact_matches_measured():
    m,q,eta,T=40,23,2,3000
    fails=0
    for t in range(T):
        pk,s,e=keygen(4,m,q,eta,b"K%d"%t)
        fails+=decrypt(s,encrypt(pk,0,q,b"R%d"%t),q)!=0
    p=float(failure_exact(m,q,eta))
    sd=(p*(1-p)/T)**0.5
    assert abs(fails/T-p)<4*sd,(fails/T,p)


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
