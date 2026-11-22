# where does the primal attack stop working
# runs the attack on a grid of (n,q) with m=2n and eta fixed
# then fits tau in predict_success on half the grid and tests it on the other half
# delta0 for each dimension comes from the fpylll runs in results/hermite.csv

import csv
import math
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from lwe import keygen
from attack import attack,predict_success
from experiments.common import fpylll_lll,setup_axes,save,RESULTS,BLUE,INK,INK2

ETA=2
NS=[5,10,15,20,25,30,35,40,45,50,55,60]
QS=[17,31,61,127,257,521,1031,2053]
TRIALS=10


def delta0_table():
    # mean fpylll delta0 per dimension then linear interpolation
    by={}
    with open(os.path.join(RESULTS,"hermite.csv")) as f:
        for r in csv.DictReader(f):
            if r["impl"]=="fpylll":
                by.setdefault(int(r["d"]),[]).append(float(r["delta0"]))
    ds=sorted(by)
    ms=[sum(by[d])/len(by[d]) for d in ds]

    def at(d):
        if d<=ds[0]:
            return ms[0]
        if d>=ds[-1]:
            return ms[-1]
        for i in range(len(ds)-1):
            if ds[i]<=d<=ds[i+1]:
                w=(d-ds[i])/(ds[i+1]-ds[i])
                return ms[i]*(1-w)+ms[i+1]*w
    return at


def run():
    os.makedirs(RESULTS,exist_ok=True)
    path=os.path.join(RESULTS,"boundary.csv")
    rows=[]
    for n in NS:
        m=2*n
        for q in QS:
            found=0
            tried=0
            seed=0
            while tried<TRIALS:
                pk,s,e=keygen(n,m,q,ETA,b"b%d_%d_%d"%(n,q,seed))
                seed+=1
                try:
                    r=attack(pk[0],pk[1],q,ETA,reduce=fpylll_lll)
                except ValueError:
                    # first n rows of A not invertible mod q so skip this instance
                    continue
                tried+=1
                found+=r==s
            rows.append([n,m,q,found,tried])
            print(n,q,found,"/",tried)
    with open(path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["n","m","q","found","tried"])
        w.writerows(rows)
    print("saved",path)


def load():
    with open(os.path.join(RESULTS,"boundary.csv")) as f:
        return [(int(r["n"]),int(r["m"]),int(r["q"]),int(r["found"]),int(r["tried"])) for r in csv.DictReader(f)]


def accuracy(cells,tau,d0):
    ok=0
    for n,m,q,found,tried in cells:
        pred=predict_success(n,m,q,ETA,d0(m+1),tau)
        ok+=pred==(found/tried>=0.5)
    return ok/len(cells)


def fit():
    rows=load()
    d0=delta0_table()
    # checkerboard split so both halves cover the whole grid
    train=[r for r in rows if (NS.index(r[0])+QS.index(r[2]))%2==0]
    test=[r for r in rows if (NS.index(r[0])+QS.index(r[2]))%2==1]
    taus=[i/100 for i in range(5,201)]
    scores=[accuracy(train,t,d0) for t in taus]
    best=max(scores)
    good=[t for t,s in zip(taus,scores) if s==best]
    tau=good[len(good)//2]
    print("tau range with best train accuracy",good[0],"to",good[-1],"picked",tau)
    print("train accuracy",round(best,3),"on",len(train),"cells")
    print("test accuracy",round(accuracy(test,tau,d0),3),"on",len(test),"cells")
    return tau,d0


def plot(tau,d0):
    rows=load()
    grid=[[0.0]*len(NS) for _ in QS]
    for n,m,q,found,tried in rows:
        grid[QS.index(q)][NS.index(n)]=found/tried
    cmap=LinearSegmentedColormap.from_list("seq",["#f3f7fd","#9cc3ef",BLUE,"#0d3f7f"])
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    im=ax.imshow(grid,origin="lower",cmap=cmap,vmin=0,vmax=1,aspect="auto")
    for i,q in enumerate(QS):
        for j,n in enumerate(NS):
            v=grid[i][j]
            ax.text(j,i,f"{v:.1f}",ha="center",va="center",fontsize=8,
                    color="white" if v>0.6 else INK)
    # predicted boundary: for each q the cells to the left are predicted to succeed
    xs,ys=[],[]
    for i,q in enumerate(QS):
        last=-1
        for j,n in enumerate(NS):
            if predict_success(n,2*n,q,ETA,d0(2*n+1),tau):
                last=j
        xs+=[last+0.5,last+0.5]
        ys+=[i-0.5,i+0.5]
    ax.plot(xs,ys,color="#eb6834",lw=2.2,label=f"predicted boundary (tau={tau})")
    ax.set_xticks(range(len(NS)))
    ax.set_xticklabels(NS)
    ax.set_yticks(range(len(QS)))
    ax.set_yticklabels(QS)
    ax.set_xlabel("secret length n  (m=2n samples)",color=INK,fontsize=10)
    ax.set_ylabel("modulus q",color=INK,fontsize=10)
    ax.tick_params(colors=INK2,labelsize=9)
    for side in ["top","right","left","bottom"]:
        ax.spines[side].set_visible(False)
    cb=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.02)
    cb.set_label("attack success rate",color=INK2,fontsize=9)
    cb.ax.tick_params(colors=INK2,labelsize=8)
    cb.outline.set_visible(False)
    ax.set_title(f"primal attack with LLL  eta={ETA}  {TRIALS} instances per cell",color=INK2,fontsize=10,loc="left")
    ax.legend(frameon=False,fontsize=9,loc="upper left",bbox_to_anchor=(0,-0.14))
    save(fig,"boundary.png")


if __name__=="__main__":
    if "--plot-only" not in sys.argv:
        run()
    tau,d0=fit()
    plot(tau,d0)
