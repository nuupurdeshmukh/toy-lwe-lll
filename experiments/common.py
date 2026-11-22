# shared helpers for the experiments
# run every experiment from the repo root like python3 -m experiments.hermite

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE="#2a78d6"
ORANGE="#eb6834"
AQUA="#1baf7a"
INK="#0b0b0b"
INK2="#52514e"
GRID="#e4e3df"

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS=os.path.join(ROOT,"results")
FIGURES=os.path.join(ROOT,"figures")


def fpylll_lll(B):
    # same interface as lll.lll but uses fplll underneath
    from fpylll import IntegerMatrix,LLL
    M=IntegerMatrix.from_matrix(B)
    LLL.reduction(M,delta=0.99)
    return [[M[i,j] for j in range(M.ncols)] for i in range(M.nrows)]


def is_prime(n):
    # miller rabin with fixed bases is exact for n<3.3*10^24 and very safe above
    if n<2:
        return False
    small=[2,3,5,7,11,13,17,19,23,29,31,37]
    for p in small:
        if n%p==0:
            return n==p
    d=n-1
    s=0
    while d%2==0:
        d//=2
        s+=1
    for a in small:
        x=pow(a,d,n)
        if x==1 or x==n-1:
            continue
        for _ in range(s-1):
            x=x*x%n
            if x==n-1:
                break
        else:
            return False
    return True


def next_prime(x):
    while not is_prime(x):
        x+=1
    return x


def log_norm(v):
    # log of the euclidean length that works for huge ints
    return 0.5*math.log(sum(x*x for x in v))


def setup_axes(ax):
    for side in ["top","right"]:
        ax.spines[side].set_visible(False)
    for side in ["left","bottom"]:
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2,labelsize=9)
    ax.grid(True,color=GRID,linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig,name):
    os.makedirs(FIGURES,exist_ok=True)
    path=os.path.join(FIGURES,name)
    fig.savefig(path,dpi=160,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print("saved",path)
