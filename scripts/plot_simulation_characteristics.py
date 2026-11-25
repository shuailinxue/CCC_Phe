"""Describe the simulated niches using composition, LR and functional scores."""
from pathlib import Path
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,LinearSegmentedColormap
from matplotlib.patches import Patch,Rectangle
import pandas as pd
import sys
sys.path.insert(0,'/home/xueshuailin/CCC_Phe/src')
from phenoniche.ccc.lr_atlas import canonicalize_complex

W=Path('/data1/xueshuailin/CCC_Phe/workspace');DATA=W/'data/tumor_immune_interface_v2_epithelial_majority';OUT=W/'outputs/tumor_immune_interface/v2_single_tissue_epithelial_majority'
TYPES=['Cancer epithelial','T cell','B cell','CAF','Endothelial']
COLORS=['#dc7181','#2b9e9b','#7cad58','#c99f59','#a7bfd0']
NICHE=['Background','Tumor core','Interface A','Interface B','Infiltrated tumor']

def make_plot():
 a=ad.read_h5ad(DATA/'tissue.h5ad',backed='r');ct=a.obs.cell_type.astype(str).to_numpy();region=a.obs.design_region_id.to_numpy();a.file.close();truth=np.array([0,1,2,3,4,4])[region]
 percentages=np.asarray([[np.mean(ct[truth==k]==t)*100 for t in TYPES] for k in range(5)]);assert np.allclose(percentages.sum(1),100)
 views=np.load(OUT/'simulation_characteristics_views.npz');fig=plt.figure(figsize=(15,7.3),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[1.3,1]);ax=fig.add_subplot(grid[0,:]);axes=[fig.add_subplot(grid[1,i]) for i in range(3)];ax.set_box_aspect(.34);bottom=np.zeros(5)
 for j,(name,color) in enumerate(zip(TYPES,COLORS)):
  vals=percentages[:,j];ax.bar(np.arange(5),vals,bottom=bottom,color=color,width=.68,label=name)
  for k,v in enumerate(vals):
   if v>=8:ax.text(k,bottom[k]+v/2,f'{v:.0f}%',ha='center',va='center',fontsize=8,color='#222')
  bottom+=vals
 ax.set_xticks(range(5),['Background','Tumor core','Interface A','Interface B','Infiltrated tumor'],fontsize=9);ax.set_ylim(0,110);ax.set_yticks(np.arange(0,101,20));ax.set_ylabel('Cell fraction (%)');ax.set_title('Cell composition across five niches',fontsize=12);ax.spines[['top','right']].set_visible(False)
 ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=5,fontsize=8,frameon=False)
 for k in range(5):ax.text(k,102,f'n={(truth==k).sum():,}',ha='center',fontsize=8,color='#555')

 # This is a compact illustration, with examples selected only for display.
 # The simulation perturbs gene expression, not individual CCC features.
 perturb=pd.read_csv(DATA/'biological_perturbations.csv');perturb=perturb[perturb.state.isin(['IFNg_boundary','TGFb_boundary'])]
 planted=perturb.pivot_table(index=['cell_type','gene'],columns='state',values='fold_change',aggfunc='prod').dropna();planted=planted[~np.isclose(planted.IFNg_boundary,planted.TGFb_boundary)]
 changed={cell:set(block.index.get_level_values('gene')) for cell,block in planted.groupby(level='cell_type')}
 import json
 folder=Path(json.loads((DATA/'target_cache.json').read_text())['folder']);features=pd.read_csv(folder/'features.csv')
 affected=np.asarray([bool(set(canonicalize_complex(r.ligand))&changed.get(r.sender,set()) or set(canonicalize_complex(r.receptor))&changed.get(r.receiver,set())) for r in features.itertuples()])
 mint=LinearSegmentedColormap.from_list('niche_mint',['#f5faf8','#a9ddd3','#40aa9b']);pink='#ea62a9';orange='#e99c45'
 result=[];ims={}
 for ax,key,title in zip(axes,['ccc','pathway','response'],['CCC','Communication pathways','Receiver response']):
  values=np.asarray(views[key],dtype=float);names=np.asarray(views[key+'_names']);a,b=values[2],values[3];valid=np.isfinite(a)&np.isfinite(b)
  if key!='response':
   strength=np.maximum(a,b);contrast=(a-b)/(np.abs(a)+np.abs(b)+1e-8);scaled_strength=strength/np.maximum(views['ccc_feature_scale'],1e-8) if key=='ccc' else strength;chosen=valid&(scaled_strength>=.1)&(np.abs(contrast)>1e-8)
   if key=='ccc':chosen&=affected&(np.abs(contrast)>=.2)
   delta=contrast[chosen];v=values[[2,3]][:,chosen]/np.maximum(strength[chosen],1e-8);cmap=mint;norm=plt.Normalize(vmin=0,vmax=1)
   rule='planted-gene linked; A/B ≥1.5-fold; feature-scaled strength ≥0.1' if key=='ccc' else 'top 6 by A/B difference in each direction; strength ≥0.1; not a significance filter'
  else:
   finite=np.isfinite(values);count=finite.sum(0);mean=np.where(finite,values,0).sum(0)/np.maximum(count,1);sd=np.sqrt(np.where(finite,(values-mean)**2,0).sum(0)/np.maximum(count,1));valid&=(count>=3)&(sd>1e-8);z=(values-mean)/np.maximum(sd,1e-8);chosen=valid&(np.abs(z[2]-z[3])>=1.);delta=(z[2]-z[3])[chosen];v=z[[2,3]][:,chosen];cmap=mint;norm=TwoSlopeNorm(vmin=-1.8,vcenter=0,vmax=1.8);rule='A/B ≥1 SD across five niche means'
  selected=np.flatnonzero(chosen);a_high=np.flatnonzero(delta>0);b_high=np.flatnonzero(delta<0);a_high=a_high[np.argsort(-delta[a_high])[:6]];b_high=b_high[np.argsort(delta[b_high])[:6]];order=np.r_[a_high,b_high];v=v[:,order]
  if not len(order):raise ValueError(f'No features pass the display filter for {key}')
  im=ax.imshow(v,aspect='auto',interpolation='nearest',cmap=cmap,norm=norm)
  ims[key]=im
  for q in order:
   j=selected[q];result.append((key,str(names[j]),float(a[j]),float(b[j]),float(delta[q]),rule))
  ax.set_box_aspect(.23)
  for xpos in np.arange(len(order)+1)-.5:ax.axvline(xpos,color='white',linewidth=.5,alpha=.8)
  ax.axhline(.5,color='white',linewidth=.7)
  if len(a_high):ax.add_patch(Rectangle((-.5,-.5),len(a_high),1,fill=False,edgecolor=pink,linewidth=2,linestyle=(0,(1,1))))
  if len(b_high):ax.add_patch(Rectangle((len(a_high)-.5,.5),len(b_high),1,fill=False,edgecolor=orange,linewidth=2,linestyle=(0,(1,1))))
  ax.set_yticks([0,1],['Interface A','Interface B'],fontsize=9);ax.set_xticks([]);ax.tick_params(length=0)
  ax.set_title(title,fontsize=11)
  ax.text(.5,-.08,'A higher  |  B higher',transform=ax.transAxes,ha='center',va='top',fontsize=8,color='#555')
  for spine in ax.spines.values():spine.set_linewidth(.4);spine.set_color('#c6cbd0')
 cb=fig.colorbar(ims['ccc'],ax=axes[0],orientation='horizontal',fraction=.07,pad=.20,shrink=.75);cb.set_label('Fraction of stronger A/B mean',fontsize=9);cb.ax.tick_params(labelsize=8)
 cbp=fig.colorbar(ims['pathway'],ax=axes[1],orientation='horizontal',fraction=.07,pad=.20,shrink=.75);cbp.set_label('Fraction of stronger A/B mean',fontsize=9);cbp.ax.tick_params(labelsize=8)
 cb2=fig.colorbar(ims['response'],ax=axes[2],orientation='horizontal',fraction=.07,pad=.20,shrink=.75);cb2.set_label('Response z-score across five niche means',fontsize=9);cb2.ax.tick_params(labelsize=8)
 pd.DataFrame(result,columns=['view','feature','A_mean','B_mean','selection_contrast','display_filter']).to_csv(OUT/'simulation_AB_display_features.csv',index=False)
 fig.suptitle('Simulation overview: composition and communication examples\nColumns selected for display; not a complete list or significance test',fontsize=13)
 return fig

if __name__=='__main__':
 dest=OUT/'simulation_characteristics.png';fig=make_plot();fig.savefig(dest,dpi=170,bbox_inches='tight');print(dest)
