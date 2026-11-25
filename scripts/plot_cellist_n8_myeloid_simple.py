"""Clear four-way comparison of two author-annotated macrophage populations."""
from pathlib import Path
import base64
import pandas as pd,numpy as np,nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
p=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1_ST5/st1_boundary_k12_n8');d=pd.read_csv(p/'author_myeloid_four_groups.csv');keys=['ST1 boundary N8','ST1 boundary other','ST5 boundary all'];labels=['ST1 边界 N8','ST1 边界其他类','ST5 全部边界'];fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
fig,axs=plt.subplots(1,2,figsize=(16,5.4),layout='constrained');yy=np.arange(3)
for ax,(sub,title,color) in zip(axs,[('Macro_CXCL9','CXCL9+ 巨噬细胞','#cf5150'),('Macro_SPP1','SPP1+ 巨噬细胞','#386ba6')]):
 df=d[d.subtype==sub].set_index('group').loc[keys];v=df.fraction_all_centers.to_numpy()*100;bars=ax.barh(yy,v,color=color,alpha=.8,height=.53);bars[0].set_alpha(1);ax.set_yticks(yy,labels,fontproperties=font);[tick.set_fontsize(19) for tick in ax.get_yticklabels()];ax.invert_yaxis();ax.set_title(title,fontproperties=font,fontsize=19,pad=12);ax.set_xlim(0,9.5);ax.set_xlabel('占该组全部中心细胞的比例',fontproperties=font,fontsize=15);ax.tick_params(axis='x',labelsize=13);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
 for i,(x,n) in enumerate(zip(v,df['count'])):ax.text(x+.13,i,f'{x:.2f}%  ({n} 个)',ha='left',va='center',fontsize=15,fontproperties=font)
 for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
fig.suptitle('ST1 边界 N8：两类巨噬细胞的对比',fontproperties=font,fontsize=21);fig.savefig(p/'n8_myeloid_simple.png',dpi=160);plt.close(fig)
nb=Path('/home/xueshuailin/CCC_Phe/Cellist_ST1_ST5_Joint_V3.ipynb');n=nbformat.read(nb,4);tag='st1_n8_myeloid_author_four_groups';source='''# ST1 边界 N8：两类巨噬细胞对比\nfrom pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\np=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1_ST5/st1_boundary_k12_n8')\ndisplay(Image(filename=str(p/'n8_myeloid_simple.png'),width=1100))\nprint('两个图均以各组全部中心细胞为分母；括号是作者标注的细胞数量。')\nprint('边界使用作者 Inner 1–2 + Outer 1–2。')\ndenom=pd.read_csv(p/'author_myeloid_four_group_denominators.csv');display(denom[denom['group'].isin(['ST1 boundary N8','ST1 boundary other','ST5 boundary all'])])\nprint('髓系亚群标注有缺失；数据来自两张切片，比例是描述性比较。')\n''';cell=nbformat.v4.new_code_cell(source,metadata={'tags':[tag]});cell.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((p/'n8_myeloid_simple.png').read_bytes()).decode()},metadata={}),nbformat.v4.new_output('stream',name='stdout',text='两图的分母都是各组全部中心细胞；括号为作者标注的细胞数量。髓系亚群标注存在缺失。\n')];found=[i for i,c in enumerate(n.cells) if tag in c.get('metadata',{}).get('tags',[])];assert len(found)==1;n.cells[found[0]]=cell;nbformat.write(n,nb);print('updated cell',found[0]+1)
