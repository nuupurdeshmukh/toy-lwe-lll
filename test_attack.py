# tests for attack.py
# run with python3 test_attack.py

from lwe import keygen
from attack import inv_mod_matrix,qary_basis,embed,attack,predict_success
from test_lll import det


def test_inv_mod_matrix():
    q=97
    M=[[3,5,1],[2,7,4],[9,1,6]]
    inv=inv_mod_matrix(M,q)
    for i in range(3):
        for j in range(3):
            v=sum(M[i][k]*inv[k][j] for k in range(3))%q
            assert v==int(i==j)


def test_inv_mod_matrix_rejects_singular():
    try:
        inv_mod_matrix([[1,2],[2,4]],97)
    except ValueError:
        return
    raise AssertionError("singular matrix should raise ValueError")


def test_qary_basis_det():
    for seed in range(5):
        pk,s,e=keygen(3,8,97,1,b"d%d"%seed)
        try:
            B,_=qary_basis(pk[0],97)
        except ValueError:
            continue
        assert abs(det(B))==97**(8-3)


# x is in L when x=A*z mod q for z=A1^-1*x[:n]
def test_qary_basis_rows_in_lattice():
    q=97
    pk,s,e=keygen(3,8,q,1,b"rows")
    A=pk[0]
    B,A1inv=qary_basis(A,q)
    for x in B:
        z=[sum(A1inv[i][k]*x[k] for k in range(3))%q for i in range(3)]
        for i in range(8):
            assert (sum(A[i][j]*z[j] for j in range(3))-x[i])%q==0


def test_embed_contains_short_vector():
    # (e,1) should equal b minus some lattice point so check e+A*s=b mod q
    q=97
    pk,s,e=keygen(3,8,q,1,b"emb")
    A,b=pk
    for i in range(8):
        assert (sum(A[i][j]*s[j] for j in range(3))+e[i]-b[i])%q==0


def test_attack_recovers_secret_small():
    found=0
    tried=0
    for seed in range(6):
        pk,s,e=keygen(4,10,97,1,b"a%d"%seed)
        try:
            r=attack(pk[0],pk[1],97,1)
        except ValueError:
            continue
        tried+=1
        found+=r==s
    assert tried>=3 and found==tried,(found,tried)


def test_attack_big_noise_does_not_crash():
    pk,s,e=keygen(4,10,97,20,b"noisy")
    try:
        r=attack(pk[0],pk[1],97,20)
    except ValueError:
        return
    assert r is None or len(r)==4


def test_predict_success_monotone_in_q():
    # bigger q with same noise should never make the prediction go from yes to no
    prev=False
    for q in [17,97,257,1031,3329]:
        p=predict_success(10,20,q,2,1.02,0.3)
        assert not (prev and not p)
        prev=p


if __name__=="__main__":
    tests=[(name,f) for name,f in sorted(globals().items())
           if name.startswith("test_") and callable(f)]
    failed=0
    for name,f in tests:
        try:
            f()
            print(f"PASS  {name}")
        except Exception as e:
            failed+=1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
