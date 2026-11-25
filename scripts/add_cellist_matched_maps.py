from pathlib import Path
import base64
import numpy as np,pandas as pd,anndata as ad,nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1_ST5/composition_consensus_analysis'); q=np.load(p/'labels.npz');pairs=pd.read_csv(p/'matched_pairs.csv');a=ad.read_h5ad('/data1/xueshuailin/CCC_Phe/workspace/data/nsclc_zhencaiwei/ST1_ST5_joint_v3_input.h5ad',backed='r');assert np.array_equal(a.obs_names.astype(str),q['cell_ids'].astype(str));xy=np.asarray(a.obsm['spatial']);a.file.close();sid=q['sample_id'];labels=q['labels'];K=int(labels.max())+1;pal=plt.get_cmap('tab20')(np.arange(K));regions=['tumor_core','tumor_edge','immune']
fig,axs=plt.subplots(2,3,figsize=(14,8.5),layout='constrained')
for row in (0,1):
 bg=np.flatnonzero(sid==row)
 for col,region in enumerate(regions):
  ax=axs[row,col];ax.scatter(xy[bg,0],xy[bg,1],c='#e8e8e8',s=.16,lw=0,rasterized=True)
  part=pairs[pairs.region==region];ix=part[['ST1_index','ST5_index'][row]].to_numpy(int);ax.scatter(xy[ix,0],xy[ix,1],c=pal[labels[ix]],s=1.0,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');ax.set_title(f'{["ST1","ST5"][row]} · {region.replace("_"," ")}\nmatched windows: {len(ix):,}')
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([],[],marker='o',ls='',color=pal[j],label=f'N{j+1}',markersize=5) for j in range(K)],loc='outside lower center',ncol=K,fontsize=7,frameon=False)
fig.savefig(p/'matched_regions.png',dpi=145);plt.close(fig)
nbpath=Path('/home/xueshuailin/CCC_Phe/Cellist_ST1_ST5_Joint_V3.ipynb');nb=nbformat.read(nbpath,4);tag='st1_st5_composition_consensus_matched_20261116';found=[c for c in nb.cells if tag in c.get('metadata',{}).get('tags',[])];assert len(found)==1
cell=found[0];needle="display(Image(filename=str(p/'overview.png'),width=1100))";assert needle in cell.source;cell.source=cell.source.replace(needle,needle+"\nprint('下图彩色点为参与严格组成配对的位置，颜色代表联合组成类；灰色为其他细胞，不表示通讯差异显著。')\ndisplay(Image(filename=str(p/'matched_regions.png'),width=1100))")
cell.outputs.insert(1,nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((p/'matched_regions.png').read_bytes()).decode()},metadata={}))
nbformat.write(nb,nbpath);print('added matched location map')
