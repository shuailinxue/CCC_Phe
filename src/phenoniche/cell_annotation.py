"""Independent SingleR reference transfer for inferred expression pilots.

No imports from PhenoMap and no outcome/niche inputs. Correlation scores are
not probabilities. Reference-calibrated rejection is our wrapper, not SingleR's
R-specific pruneScores implementation; inferred-RNA accuracy needs separate QC.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

REFERENCE_LABELS = {
    'Cancer Epithelial': 'Cancer_epithelial',
    'Normal Epithelial': 'Normal_epithelial',
    'T-cells': 'T_cells', 'B-cells': 'B_cells',
    'Plasmablasts': 'Plasma_cells', 'Myeloid': 'Myeloid',
    'Endothelial': 'Endothelial', 'CAFs': 'Fibroblast', 'PVL': 'Perivascular',
}
CORE_MARKERS = {
    'Cancer_epithelial': ['EPCAM','KRT8','KRT18','KRT19','KRT7'],
    'Normal_epithelial': ['EPCAM','KRT8','KRT18','KRT19','KRT7'],
    'T_cells': ['CD3D','CD3E','CD3G','TRAC','CD247','CD2'],
    'B_cells': ['MS4A1','CD79A','CD79B','CD19','CD22','BANK1'],
    'Plasma_cells': ['MZB1','JCHAIN','SDC1'],
    'Myeloid': ['LYZ','TYROBP','FCER1G','AIF1','C1QA','C1QB','LST1'],
    'Endothelial': ['PECAM1','VWF','KDR','EMCN','PLVAP','RAMP2','CDH5','CLEC14A'],
    'Fibroblast': ['COL1A1','COL1A2','COL3A1','DCN','LUM','PDGFRA'],
    'Perivascular': ['RGS5','PDGFRB','ACTA2','TAGLN','CSPG4','MCAM'],
}


def validate_matrix(expression, genes):
    x=np.asarray(expression, dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=len(genes) or len(set(genes))!=len(genes):
        raise ValueError('Unique genes must match cells-by-genes expression')
    if not np.isfinite(x).all() or (x<0).any():
        raise ValueError('Expression must be finite and nonnegative')
    return x


def marker_ranks(expression, genes, labels):
    """Mean within-cell ranks; never treat positive inferred rates as detection."""
    x=validate_matrix(expression, genes)
    ranks=rankdata(x, axis=1, method='average')/x.shape[1]
    lookup={g:i for i,g in enumerate(genes)}
    result={}
    for label in labels:
        ids=[lookup[g] for g in CORE_MARKERS[label] if g in lookup]
        result[label]=ranks[:,ids].mean(1) if len(ids)>=2 else np.full(len(x),np.nan)
    return pd.DataFrame(result)


class SingleRAnnotator:
    """Train from a labelled reference; query must have identical shared genes."""
    def __init__(self, expression, labels, genes, threads=4):
        import singler
        x=validate_matrix(expression, genes)
        labels=np.asarray(labels, dtype=str)
        if len(labels)!=len(x) or not set(labels)<=set(CORE_MARKERS):
            raise ValueError('Unsupported reference labels')
        self.genes=list(genes);self.threads=threads;self.thresholds=None
        self.reference=singler.train_single(x.T, labels.tolist(), self.genes,
                                            num_threads=threads)

    def predict(self, expression):
        import singler
        x=validate_matrix(expression,self.genes)
        result=singler.classify_single(x.T,self.reference,num_threads=self.threads)
        score_frame=result['scores']
        labels=list(score_frame.get_column_names())
        scores=np.column_stack([np.asarray(score_frame[label]) for label in labels])
        best=np.asarray(result['best'],dtype=str)
        frame=pd.DataFrame({'singleR_label':best,'singleR_delta':np.asarray(result['delta'])})
        for j,label in enumerate(labels):frame['score_'+label]=scores[:,j]
        frame['singleR_score']=[scores[i,labels.index(label)] for i,label in enumerate(best)]
        ranks=marker_ranks(x,self.genes,labels)
        frame['core_marker_rank']=[ranks[label].iloc[i] for i,label in enumerate(best)]
        frame['cell_type']=best
        if self.thresholds is not None:
            for label in labels:
                ids=best==label; t=self.thresholds[label]
                supported=(frame.singleR_score>=t['score']) & (frame.singleR_delta>=t['delta']) & (frame.core_marker_rank>=t['marker_rank'])
                frame.loc[ids & ~supported,'cell_type']='Unknown'
        return frame

    def calibrate(self, expression, true_labels):
        """Fix class thresholds on separate reference donors, never target counts."""
        result=self.predict(expression);true=np.asarray(true_labels)
        thresholds={}
        ranks=marker_ranks(expression,self.genes,sorted(set(true)))
        for label in sorted(set(true)):
            ids=(true==label)&result.singleR_label.eq(label).to_numpy()
            if ids.sum()<10:raise ValueError(f'Insufficient correct calibration cells: {label}')
            thresholds[label]={
                'score':float(np.quantile(result.loc[ids,'singleR_score'],.01)),
                'delta':max(0.,float(np.quantile(result.loc[ids,'singleR_delta'],.01))),
                'marker_rank':float(np.quantile(ranks.loc[true==label,label],.05)),
            }
        self.thresholds=thresholds
        return thresholds
