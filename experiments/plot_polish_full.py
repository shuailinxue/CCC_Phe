"""Render final frozen full-slice labels and the most variable program."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument("inference",type=Path)
    a=p.parse_args()
    path=a.inference
    if not (path/"complete").exists():
        raise ValueError("Inference is incomplete")
    with np.load(path/"cells.npz") as z:
        xy,y,activity=z["xyz"],z["niche"],z["activity"]
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(10,4.75),layout="constrained")
    color=plt.get_cmap("tab20")
    for k in np.unique(y):
        pos=xy[y==k]
        axes[0].scatter(*pos.T,s=.32,color=color(k%20),label=f"{k}",rasterized=True)
    axes[0].legend(frameon=False,ncol=4,markerscale=5,title="Niche")
    r=int(activity.std(0).argmax())
    lo,hi=np.quantile(activity[:,r],[.01,.99])
    im=axes[1].scatter(*xy.T,c=activity[:,r],vmin=lo,vmax=hi,cmap="viridis",s=.35,rasterized=True)
    fig.colorbar(im,ax=axes[1],label=f"Program {r+1} activity")
    for ax,title in zip(axes,["Frozen full-slice niche assignments","Independent communication activity"]):
        ax.set(title=title,aspect="equal",xlabel="X (input units)",ylabel="Y (input units)")
    fig.savefig(path/"full_spatial.pdf",dpi=230)
    fig.savefig(path/"full_spatial.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    main()
