from pathlib import Path
import base64
import numpy as np,pandas as pd,nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
p=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1_ST5/st1_boundary_k12_n8');d=pd.read_csv(p/'author_myeloid_four_group_denominators.csv').set_index('group');keys=['ST1 boundary N8','ST1 boundary other','ST5 boundary all'];labels=['ST1 边界 N8','ST1 边界其他类','ST5 全部边界'];d=d.loc[keys];percentage=d.annotated_myeloid/d.all_centers*100
fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(8,5.4),layout='constrained');y=np.arange(3);bars=ax.barh(y,percentage,color='#6b88a1',alpha=.8,height=.53);bars[0].set_alpha(1);ax.set_yticks(y,labels,fontproperties=font);[t.set_fontsize(19) for t in ax.get_yticklabels()];ax.invert_yaxis();ax.set_xlim(0,max(percentage)*1.6);ax.set_title('已标注巨噬细胞总比例',fontproperties=font,fontsize=19,pad=12);ax.set_xlabel('占该组全部中心细胞的比例',fontproperties=font,fontsize=15);ax.tick_params(axis='x',labelsize=13);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
for i,(v,n) in enumerate(zip(percentage,d.annotated_myeloid)):ax.text(v+max(percentage)*.018,i,f'{v:.2f}%  ({n} 个)',va='center',fontsize=15,fontproperties=font)
for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
fig.savefig(p/'n8_total_macrophages.png',dpi=160);plt.close(fig)
nb=Path('/home/xueshuailin/CCC_Phe/Cellist_ST1_ST5_Joint_V3.ipynb');n=nbformat.read(nb,4);tag='st1_n8_total_author_macrophages';source='''# ST1 边界 N8：整体巨噬细胞比例\nfrom pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\np=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1_ST5/st1_boundary_k12_n8')\ndisplay(Image(filename=str(p/'n8_total_macrophages.png'),width=700))\nprint('作者标注的 CXCL9+、SPP1+、SLPI+、TREM2+ 和肺泡巨噬细胞合计；分母是各组全部中心细胞。')\nprint('未标注亚型的髓系细胞没有算入巨噬细胞，实际数量可能被低估。')\n''';cell=nbformat.v4.new_code_cell(source,metadata={'tags':[tag]});cell.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((p/'n8_total_macrophages.png').read_bytes()).decode()},metadata={}),nbformat.v4.new_output('stream',name='stdout',text='作者标注的五类巨噬细胞合计；未标注亚型的髓系细胞未计入。\n')];found=[i for i,c in enumerate(n.cells) if tag in c.get('metadata',{}).get('tags',[])];
if found:n.cells[found[0]]=cell
else:n.cells.append(cell)
nbformat.write(n,nb);print('proportions',percentage.to_dict(),'cell',found[0]+1 if found else len(n.cells))
