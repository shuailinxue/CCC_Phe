"""Raw and evaluation-aligned spatial maps plus common-center metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from phenoniche.simulation.evaluation import label_mapping,mapped
from phenoniche.simulation.joint_reporting import JointExpressionReport


class BaselineReport:
    def __init__(self,experiment_output,comparison_output):
        self.experiment=Path(experiment_output);self.output=Path(comparison_output)
        self.reference=JointExpressionReport(self.experiment)
        self.frame=pd.read_csv(self.output/'labels.csv');self.metrics=pd.read_csv(self.output/'metrics.csv')
        self.protocol=json.loads((self.output/'protocol.json').read_text());self.train=np.load(self.output/'split_rows.npz')['train']
        self.methods=['Cell Composition','Expression PCA','CCC-AE','GraphST',self.protocol.get('cellcharter_name','CellCharter (PCA)'),*self.protocol.get('additional_methods',[]),'CCC-only GNN','CCC + gene GNN']
        self.figures=self.output/'figures';self.figures.mkdir(exist_ok=True)

    def save(self,fig,name):
        for ext in ('png','pdf','svg'):fig.savefig(self.figures/(name+'.'+ext),dpi=300,bbox_inches='tight')
        return fig

    def spatial(self,align=False):
        rows=int(np.ceil((len(self.methods)+1)/4))
        fig,axes=plt.subplots(rows,4,figsize=(12,3.35*rows),layout='constrained');axes=axes.ravel()
        self.reference.panel(axes[0],self.reference.truth,'Ground truth',True,names=['Tumor core','Normal exterior','Edge A','Edge B'])
        metric=self.metrics[self.metrics.scope=='all_centers_descriptive'].set_index('method')
        for ax,method in zip(axes[1:],self.methods):
            raw=self.frame[method].to_numpy()
            values=mapped(raw,label_mapping(raw,self.reference.truth,self.train)) if align else raw
            self.reference.panel(ax,values,f'{method}\nWhole-tissue ARI={metric.loc[method,"overall_ARI"]:.3f}',True,names=None if align else [f'Cluster {j}' for j in range(4)])
        for ax in axes[len(self.methods)+1:]:ax.set_visible(False)
        fig.suptitle('Evaluation color matching on training centers' if align else 'Original clustering results · raw cluster IDs',fontsize=12)
        note='\nGNN niche assignment: full KMeans + 20% local pooling.' if self.protocol.get('GNN_niche_postprocessing') else ''
        if self.protocol.get('raw_GNN_controls_in_metric_table'):note+=' Original GNN controls are retained in the metric table.'
        fig.supxlabel('Same cells; K=4 for every method. Labels are used only after clustering for evaluation/color matching. Spatial maps include training centers.'+note,fontsize=7)
        return self.save(fig,'baseline_spatial_aligned' if align else 'baseline_spatial_raw')

    def metric_table(self):
        columns=['method','K','n','n_true_edge','overall_ARI','overall_NMI','edge_F1','edge_IoU','edge_AB_ARI','edge_AB_accuracy']
        return self.metrics[self.metrics.scope=='common_spatial_test'][columns].reset_index(drop=True)

    def metric_plot(self):
        frame=self.metric_table().set_index('method').loc[self.methods]
        fig,axes=plt.subplots(1,4,figsize=(12,max(4,.43*len(frame)+1)),layout='constrained')
        from phenoniche.gnn.reporting import COLORS
        for ax,key,title in zip(axes,['overall_ARI','overall_NMI','edge_F1','edge_AB_accuracy'],['Overall ARI','Overall NMI','Edge detection F1','Edge A/B accuracy']):
            ax.barh(np.arange(len(frame)),frame[key],color=[COLORS[2] if method in ('CCC-only GNN','CCC + gene GNN') else COLORS[0] for method in frame.index])
            ax.set(xlim=(-.05,1.03),yticks=np.arange(len(frame)),yticklabels=list(frame.index) if ax is axes[0] else [],title=title)
            ax.invert_yaxis()
        counts=self.protocol['common_test_true_region_counts']
        fig.supxlabel(f'Common spatial test: n={self.protocol["common_test_centers"]}; truth-region counts {counts}. One simulation/training seed; no uncertainty claim.',fontsize=7)
        return self.save(fig,'baseline_common_spatial_test_metrics')
