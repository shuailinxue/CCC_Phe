from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phenoniche.v1002.simulation_figures import NICHE_COLORS, _style


def run():
    root=Path('outputs/v1002_repair')
    summary=json.loads((root/'summary.json').read_text())
    data=np.load(root/'primary_maps.npz')
    xy=data['coordinates'];labels=data['labels']
    reports=summary['reports'];final=reports[summary['final_method']][0]
    _style()
    def scatter(ax,values,title,categorical=False):
        color=[NICHE_COLORS[int(k)] for k in values] if categorical else values
        result=ax.scatter(xy[:,0],xy[:,1],c=color,s=5,linewidths=0,**({} if categorical else {'cmap':'viridis','vmin':0,'vmax':1}))
        ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,fontsize=8)
        return result
    fig,axes=plt.subplots(2,2,figsize=(7.2,7.5),layout='constrained')
    scatter(axes[0,0],labels,'A  True labels',True)
    scatter(axes[0,1],final['predicted'],'B  Final CCC-anchored staged labels',True)
    for ax,k,title in [(axes[1,0],1,'C  Risk Niche1 activity'),(axes[1,1],3,'D  Protective Niche3 activity')]:
        im=scatter(ax,np.array(final['activity'])[:,k],title)
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('K=6: CCC-anchored staged factorization',fontsize=10)
    for ext in ('pdf','png'):fig.savefig(root/f'Figure_example_maps_fixed.{ext}',dpi=300)
    plt.close(fig)
    fig=plt.figure(figsize=(12,8.5),layout='constrained');grid=fig.add_gridspec(3,4)
    methods=['CCC-only-K6','Full-Random-K6',summary['final_method']]
    for row,k in enumerate((1,2)):
        scatter(fig.add_subplot(grid[row,0]),(labels==k).astype(float),f'Niche{k} true region')
        for col,m in enumerate(methods,1):
            scatter(fig.add_subplot(grid[row,col]),np.array(reports[m][0]['activity'])[:,k],f'N{k}: '+{'CCC-only-K6':'CCC-only','Full-Random-K6':'Full random'}.get(m,'Final staged'))
    ax=fig.add_subplot(grid[2,:2]);hc=data['hc']
    ax.plot(hc[1],'o-',label='Niche1',color=NICHE_COLORS[1]);ax.plot(hc[2],'s--',label='Niche2',color=NICHE_COLORS[2])
    ax.set_title('Identical generating composition');ax.set_xlabel('Cell type');ax.set_ylabel('Probability');ax.legend()
    ax=fig.add_subplot(grid[2,2:]);hi=data['hi'];idx=np.union1d(np.argsort(-hi[1])[:8],np.argsort(-hi[2])[:8]);x=np.arange(len(idx))
    ax.bar(x-.18,hi[1,idx],.36,color=NICHE_COLORS[1],label='Niche1');ax.bar(x+.18,hi[2,idx],.36,color=NICHE_COLORS[2],label='Niche2')
    ax.set_title('Distinct generating directed CCC (display only)');ax.set_xticks(x,idx,rotation=60,fontsize=5);ax.set_xlabel('Filtered directed feature index');ax.legend()
    fig.suptitle('Twin diagnostic: identical biology and noise across methods',fontsize=10)
    for ext in ('pdf','png'):fig.savefig(root/f'Figure_twin_niche_diagnostic.{ext}',dpi=300)
    plt.close(fig)

def main_figure():
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    root=Path('outputs/v1002_repair')
    records=json.loads((root/'final_robustness_grid.json').read_text())
    data=np.load(root/'primary_maps.npz');_style()
    fig=plt.figure(figsize=(12,8),layout='constrained');outer=fig.add_gridspec(2,3)
    ax=fig.add_subplot(outer[0,0]);xy=data['coordinates']
    ax.scatter(xy[:,0],xy[:,1],c=[NICHE_COLORS[k] for k in data['labels']],s=4)
    ax.set_aspect('equal');ax.set_title('A  Cell-level simulation (unchanged)');ax.set_xticks([]);ax.set_yticks([])
    for slot,letter,metric in [(outer[0,1],'B','sensitivity'),(outer[0,2],'C','auc'),(outer[1,0],'D','CCC_AUPRC')]:
        inner=GridSpecFromSubplotSpec(3,3,subplot_spec=slot,wspace=.3,hspace=.35)
        for i,size in enumerate((50,100,150)):
            for j,purity in enumerate((.3,.5,.7)):
                ax=fig.add_subplot(inner[i,j]);means=[];sds=[]
                for noise in (0,.05,.1):
                    rows=[r for r in records if r['size']==size and r['purity']==purity and r['noise']==noise]
                    values=[r['CCC_AUPRC'] if metric=='CCC_AUPRC' else np.mean([v[metric] for v in r['spatial']['per_factor'][1:]]) for r in rows]
                    means.append(np.mean(values));sds.append(np.std(values,ddof=1))
                ax.errorbar([0,.05,.1],means,yerr=sds,marker='o',ms=2,color='#2F6B9A',lw=.8,capsize=2)
                ax.set_ylim(0,1.02);ax.set_xticks([0,.1]);ax.tick_params(labelsize=5)
                if i==0:ax.set_title(f'{letter} {metric}, p={purity}' if j==0 else f'p={purity}',fontsize=6)
                if j==0:ax.set_ylabel(f'size={size}',fontsize=6)
                if i==2:ax.set_xlabel('noise',fontsize=6)
    ax=fig.add_subplot(outer[1,1])
    for name,color in [('Risk_WB',NICHE_COLORS[1]),('Neutral_WB',NICHE_COLORS[2]),('Protective_WB',NICHE_COLORS[3])]:
        means=[];sds=[]
        for noise in (0,.05,.1):
            vals=[r['bulk'][name] for r in records if r['noise']==noise and r['purity']==.5 and r['size']==100]
            means.append(np.mean(vals));sds.append(np.std(vals,ddof=1))
        ax.errorbar([0,.05,.1],means,yerr=sds,label=name,color=color,marker='o',capsize=2)
    ax.set_title('E  Final staged bulk transfer');ax.set_xlabel('noise');ax.set_ylabel('WB correlation');ax.legend(fontsize=6)
    ax=fig.add_subplot(outer[1,2]);prim=[r for r in records if r['noise']==.05 and r['size']==100 and r['purity']==.5]
    means=[np.mean([r['bulk'][k] for r in prim]) for k in ('gamma_1','gamma_2','gamma_3')]
    sd=[np.std([r['bulk'][k] for r in prim],ddof=1) for k in ('gamma_1','gamma_2','gamma_3')]
    ax.bar(range(3),means,yerr=sd,color=NICHE_COLORS[1:4],capsize=3);ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(3),['Risk','Neutral','Protective']);ax.set_ylabel('Cox gamma')
    ax.set_title('F  Final staged clinical phenotype')
    ax.text(.02,.98,'Test C = '+format(np.mean([r['bulk']['Test_C'] for r in prim]),'.3f'),transform=ax.transAxes,va='top')
    for ext in ('pdf','png'):fig.savefig(root/f'Figure_simulation_main.{ext}',dpi=300)
    plt.close(fig)

if __name__=='__main__':
    run()
    if Path('outputs/v1002_repair/final_robustness_grid.json').exists():main_figure()
