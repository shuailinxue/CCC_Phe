"""Plot stagewise comparisons and reliability from saved final metrics."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument("selected",type=Path)
    a=p.parse_args()
    sel=a.selected
    records=json.loads((sel/"selection.json").read_text())["decisions"]
    metric=json.loads((sel/"test_metrics.json").read_text())
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"axes.spines.right":False,"axes.spines.top":False})
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout="constrained")
    entries=[r for r in records if r["stage"] in ("A","B","C","D")]
    names=[f'{r["stage"]} · R{r["rank"]}' for r in entries]
    y=[r["utility"] for r in entries]
    axes[0].barh(names,y,color=["#4477aa" if q["selected"] else ".72" for q in entries])
    axes[0].set(xlabel="Internal prediction utility ↓",xlim=(min(y)-.015,max(y)+.01))
    for r,n in zip(entries,names):
        axes[1].scatter(r["matched"],r["utility"])
        axes[1].annotate(n,(r["matched"],r["utility"]),xytext=(3,3),textcoords="offset points",fontsize=7)
    axes[1].set(xlabel="Matched program cosine",ylabel="Internal prediction utility ↓")
    e=[r for r in records if r["stage"]=="E"]
    axes[2].plot([q["k"] for q in e],[q["utility"] for q in e],"o-",color="#4477aa")
    axes[2].set(xlabel="Niche count",ylabel="Internal prediction utility ↓",xticks=[q["k"] for q in e])
    t=axes[2].twinx()
    t.plot([q["k"] for q in e],[q["ari"] for q in e],"s--",color="#aa6644")
    t.set_ylabel("Seed ARI",color="#aa6644")
    def save(fig,name):
        for ext in ("pdf","png"):
            fig.savefig(sel/f"{name}.{ext}",dpi=180)
        plt.close(fig)
    save(fig,"stagewise_selection")
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout="constrained")
    saved=[]
    for ax,subset in zip(axes,("random","grouped")):
        for model,color in zip(("A73","final73","composition","shuffle"),(".35","#4477aa","#ba8844","#228833")):
            scores=metric[model]["niche"][subset]
            n,c,estimate=np.asarray(scores["calibration"]).T
            keep=n>0
            ax.plot(estimate[keep]/n[keep],c[keep]/n[keep],"o-",label=model,color=color,ms=3)
        ax.plot([0,1],[0,1],":",color=".65")
        ax.set(xlim=(0,.4),ylim=(0,.4),xlabel="Mean predicted detection",ylabel="Observed detection fraction",
               title=f"{subset.title()} edge holdout")
        ax.legend(frameon=False,fontsize=8)
        for model in metric:
            for target in ("dictionary","niche"):
                m=metric[model][target][subset]
                saved.append(dict(model=model,target=target,holdout=subset,
                                  **{k:v for k,v in m.items() if isinstance(v,(int,float))}))
    pd.DataFrame(saved).to_csv(sel/"test_summary.tsv",sep="\t",index=False)
    save(fig,"heldout_calibration")


if __name__=="__main__":
    main()
