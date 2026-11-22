# decryption failure: measured vs exact vs hoeffding bound
# m eta fixed and q varies
# measured rates come from running keygen encrypt decrypt for real with bit 0

import csv
import math
import os
import sys

import matplotlib.pyplot as plt

from lwe import keygen,encrypt,decrypt,failure_exact,failure_bound
from experiments.common import setup_axes,save,RESULTS,BLUE,ORANGE,AQUA,INK,INK2

M=64
ETA=2
N=4
T=4000
Q_MEASURED=[23,29,31,37,41,43,47,53,59,61,67,71]
Q_ALL=Q_MEASURED+[79,89,97,109,127,151,179,211,251]


def log10_fraction(x):
    # log10 of a Fraction even when it is too small for a float
    if x==0:
        return float("-inf")
    return (math.log(x.numerator)-math.log(x.denominator))/math.log(10)


def run():
    os.makedirs(RESULTS,exist_ok=True)
    path=os.path.join(RESULTS,"failure.csv")
    rows=[]
    for q in Q_ALL:
        fails=None
        if q in Q_MEASURED:
            fails=0
            for t in range(T):
                pk,s,e=keygen(N,M,q,ETA,b"f%d_%d"%(q,t))
                fails+=decrypt(s,encrypt(pk,0,q,b"g%d_%d"%(q,t)),q)!=0
        ex=failure_exact(M,q,ETA)
        bd=failure_bound(M,q,ETA)
        rows.append([q,"" if fails is None else fails,T,log10_fraction(ex),math.log10(bd) if bd>0 else ""])
        print(q,fails,float(ex),bd)
    with open(path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["q","fails","trials","log10_exact","log10_bound"])
        w.writerows(rows)
    print("saved",path)


def plot():
    with open(os.path.join(RESULTS,"failure.csv")) as f:
        rows=list(csv.DictReader(f))
    qs=[int(r["q"]) for r in rows]
    ex=[float(r["log10_exact"]) for r in rows]
    bd=[float(r["log10_bound"]) for r in rows]
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    setup_axes(ax)
    ax.plot(qs,[10**x for x in bd],color=ORANGE,lw=2,label="hoeffding bound")
    ax.plot(qs,[10**x for x in ex],color=BLUE,lw=2,label="exact")
    mq,mr,lo,hi=[],[],[],[]
    for r in rows:
        if r["fails"]!="" and int(r["fails"])>0:
            k=int(r["fails"])
            n=int(r["trials"])
            p=k/n
            # wilson 95% interval since normal bars break when there are only a few failures
            z=1.96
            c=(p+z*z/(2*n))/(1+z*z/n)
            h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            mq.append(int(r["q"]))
            mr.append(p)
            lo.append(p-(c-h))
            hi.append(c+h-p)
    ax.errorbar(mq,mr,yerr=[lo,hi],fmt="o",ms=6,color=AQUA,ecolor=AQUA,elinewidth=1.2,capsize=0,
                markeredgecolor="white",markeredgewidth=1.2,label=f"measured ({T} runs each, 95% interval)")
    ax.set_yscale("log")
    ax.set_xlabel("modulus q",color=INK,fontsize=10)
    ax.set_ylabel("probability that bit 0 decrypts wrong",color=INK,fontsize=10)
    ax.set_title(f"m={M}  eta={ETA}",color=INK2,fontsize=10,loc="left")
    ax.legend(frameon=False,fontsize=9,loc="lower left")
    save(fig,"failure.png")


if __name__=="__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
