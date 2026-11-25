"""Show author myeloid annotations in ST1 boundary N8 and three comparators."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import base64,json
import numpy as np,pandas as pd,anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nbformat
W=Path('/data1/xueshuailin/CCC_Phe/workspace');R=W/'outputs/nsclc_zhencaiwei/ST1_ST5/v3_joint_K12';O=W/'outputs/nsclc_zhencaiwei/ST1_ST5/st1_boundary_k12_n8';O.mkdir(parents=True,exist_ok=True)
q=np.load(R/'predictions.npz');ids=q['cell_ids'].astype(str);sid=q['sample_id'];labels=q['labels'];frames=[];coords=[]
for s in ('ST1','ST5'):
 a=ad.read_h5ad(W/f'data/nsclc_zhencaiwei/processed/{s}/{s}_counts.h5ad',backed='r');assert np.array_equal(np.char.add(s+'::',a.obs_names.to_numpy(dtype=str)),ids[sid==({'ST1':0,'ST5':1}[s])]);frames.append(a.obs[['assign.copyKAT','assign.curated.Myeloid.inte','Tumor_boundary']].copy());coords.append(np.asarray(a.obsm['spatial_native']));a.file.close()
obs=pd.concat(frames,ignore_index=True);xy=np.concatenate(coords);bd=obs.Tumor_boundary.astype(str).isin(['Inner 1','Inner 2','Outer 1','Outer 2']).to_numpy();celltype=obs['assign.copyKAT'].astype(str).to_numpy();sub=obs['assign.curated.Myeloid.inte'].astype(object).fillna('Unannotated').astype(str).to_numpy();N8=labels==7
sets=[('ST1 boundary N8',(sid==0)&bd&N8),('ST1 boundary other',(sid==0)&bd&~N8),('ST5 boundary all',(sid==1)&bd),('ST1 outside boundary N8',(sid==0)&~bd&N8)]
subtypes=['Macro_CXCL9','Macro_SPP1','Macro_SLPI','AM','Macro_TREM2'];out=[]
for group,sel in sets:
 typ=sel&(celltype=='Myeloid');labeled=typ&np.isin(sub,subtypes)
 for t in subtypes:
  count=int((typ&(sub==t)).sum());out.append({'group':group,'subtype':t,'count':count,'all_centers':int(sel.sum()),'myeloid_centers':int(typ.sum()),'annotated_myeloid':int(labeled.sum()),'fraction_all_centers':count/max(int(sel.sum()),1),'fraction_myeloid_centers':count/max(int(typ.sum()),1),'fraction_annotated_myeloid':count/max(int(labeled.sum()),1),'unannotated_myeloid':int((typ&~labeled).sum())})
df=pd.DataFrame(out);df.to_csv(O/'author_myeloid_four_groups.csv',index=False)
# Spatial panel: author types at N8 boundary centers only; point locations are cells.
sel=sets[0][1];grey=np.flatnonzero((sid==0)&~sel);center=np.flatnonzero(sel);cx=np.flatnonzero(sel&(celltype=='Myeloid')&(sub=='Macro_CXCL9'));sp=np.flatnonzero(sel&(celltype=='Myeloid')&(sub=='Macro_SPP1'));fig,axs=plt.subplots(1,3,figsize=(16,5.2),layout='constrained');ax=axs[0];ax.scatter(xy[grey,0],xy[grey,1],c='#ededed',s=.16,lw=0,rasterized=True);ax.scatter(xy[center,0],xy[center,1],c='#bfc7ca',s=.5,lw=0,rasterized=True);ax.scatter(xy[cx,0],xy[cx,1],c='#d54343',s=7,lw=0,rasterized=True);ax.scatter(xy[sp,0],xy[sp,1],c='#345da5',s=7,lw=0,rasterized=True);ax.axis('off');ax.set_aspect('equal');ax.set_title('ST1 boundary N8 · center cell annotations');ax.legend(handles=[Line2D([],[],marker='o',ls='',color='#d54343',label=f'CXCL9+ macrophage (n={len(cx)})',markersize=5),Line2D([],[],marker='o',ls='',color='#345da5',label=f'SPP1+ macrophage (n={len(sp)})',markersize=5),Line2D([],[],marker='o',ls='',color='#bfc7ca',label='Other N8 boundary centers',markersize=5)],fontsize=7,loc='lower left',frameon=False)
colors={'Macro_CXCL9':'#d54343','Macro_SPP1':'#345da5','Macro_SLPI':'#e49b41','AM':'#71a08d','Macro_TREM2':'#9a7db2'};x=np.arange(len(sets));width=.16
for j,t in enumerate(subtypes):
 v=df[df.subtype==t];axs[1].bar(x+(j-2)*width,v.fraction_all_centers*100,width,color=colors[t],label=t);axs[2].bar(x+(j-2)*width,v.fraction_annotated_myeloid*100,width,color=colors[t],label=t)
xt=['ST1 border\nN8','ST1 border\nother niches','ST5 border\nall niches','ST1 N8\noutside border']
for ax in axs[1:]:ax.set_xticks(x,xt,fontsize=8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axs[1].set_ylabel('% of all center cells');axs[1].set_title('Fraction of all center cells');axs[2].set_ylabel('% of annotated myeloid center cells');axs[2].set_title('Among annotated myeloid centers');axs[2].legend(fontsize=7,frameon=False,loc='upper right');fig.savefig(O/'n8_myeloid_four_groups.png',dpi=150);plt.close(fig)
summary=df.groupby('group',sort=False).first()[['all_centers','myeloid_centers','annotated_myeloid','unannotated_myeloid']];summary.to_csv(O/'author_myeloid_four_group_denominators.csv');print(summary.to_string());print('\n',df[df.subtype.isin(['Macro_CXCL9','Macro_SPP1'])][['group','subtype','count','fraction_all_centers','fraction_annotated_myeloid']].to_string(index=False))
nbpath=Path('/home/xueshuailin/CCC_Phe/Cellist_ST1_ST5_Joint_V3.ipynb');n=nbformat.read(nbpath,4);tag='st1_n8_myeloid_author_four_groups';source='''# ST1 边界 N8：作者髓系亚群，四组比较\nfrom pathlib import Path\nimport pandas as pd\nfrom IPython.display import display,Image\np=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1_ST5/st1_boundary_k12_n8')\ndisplay(Image(filename=str(p/'n8_myeloid_four_groups.png'),width=1150))\nprint('左图：ST1 边界 N8 的中心细胞，红色为作者标注的 CXCL9⁺ 巨噬细胞，蓝色为 SPP1⁺。')\nprint('中图以各组全部中心细胞为分母；右图以作者已标注亚型的髓系中心细胞为分母。')\ndisplay(pd.read_csv(p/'author_myeloid_four_group_denominators.csv'))\ndisplay(pd.read_csv(p/'author_myeloid_four_groups.csv').round(3))\nprint('边界为作者 Inner 1–2 + Outer 1–2；其他位置 N8 包括 ST1 肿瘤核心、基质及更外层。各组不独立，切片间比较不能证明因果。')\n''';cell=nbformat.v4.new_code_cell(source,metadata={'tags':[tag]});txt='分母：\n'+summary.to_string()+'\n\nCXCL9⁺ / SPP1⁺ 巨噬细胞：\n'+df[df.subtype.isin(['Macro_CXCL9','Macro_SPP1'])][['group','subtype','count','fraction_all_centers','fraction_annotated_myeloid']].round(3).to_string(index=False)+'\n';cell.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((O/'n8_myeloid_four_groups.png').read_bytes()).decode()},metadata={}),nbformat.v4.new_output('stream',name='stdout',text=txt)];found=[i for i,c in enumerate(n.cells) if tag in c.get('metadata',{}).get('tags',[])];
if found:n.cells[found[0]]=cell
else:n.cells.append(cell)
nbformat.write(n,nbpath);print('COMPLETE')
