"""Synthetic, non-LR receiver responses; never fabricate biological priors."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import anndata as ad
import numpy as np
import pandas as pd
import torch
from ..storage import STORAGE_ROOT
from ..gnn.model import physical_weights
from .generation import TYPES
from .programs import program_specs

ROOT=Path(__file__).resolve().parents[3]

@dataclass(frozen=True)
class ExpressionExperimentConfig:
    base_simulation_config:str=str(ROOT/'config/tumor_edge_simulation.json')
    output_dir:str=str(STORAGE_ROOT/'outputs/tumor_edge_ccc_expression')
    response_genes_per_program:int=6
    negative_control_genes:int=12
    response_log_fold:float=1.0
    nuisance_sd:float=.35
    expression_epochs:int=40
    patience:int=6
    batch_size:int=128
    lr:float=.005
    lambda_dictionary:float=.0001
    device:str='cuda:2'
    seed:int=40700
    figure_dpi:int=300
    training_mode:str='frozen'
    frozen_control_dir:str=str(STORAGE_ROOT/'outputs/tumor_edge_ccc_expression')
    joint_epochs:int=40
    joint_lr:float=.0001
    lambda_gene:float=.01
    auxiliary_gradient_cap:float=.1
    ccc_validation_tolerance:float=.005
    def __post_init__(self):
        if self.response_genes_per_program<2 or self.negative_control_genes<2 or self.response_log_fold<=0:
            raise ValueError('Require positive response strength, up/down genes and negative controls')
        if self.nuisance_sd<0 or min(self.expression_epochs,self.patience,self.batch_size)<1 or self.lr<=0 or self.lambda_dictionary<0:
            raise ValueError('Invalid expression training settings')
        if self.training_mode not in ('frozen','joint') or self.joint_epochs<1 or self.joint_lr<=0 or self.lambda_gene<0 or not 0<=self.auxiliary_gradient_cap<=1 or not 0<=self.ccc_validation_tolerance<=.05:
            raise ValueError('Invalid protected joint training settings')
    def to_dict(self):return asdict(self)
    @classmethod
    def from_json(cls,path):return cls(**json.loads(Path(path).read_text()))


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def receiver_exposure(adata,lr,data,base_config):
    """Observed incoming LR signal on existing physical pairs into each center.

    Neither anatomical labels nor designed MI activity is read. Program specs
    define synthetic LR groups/roles; the inputs are measured L/R + geometry.
    Same sigma/anchor weights as CCC; self physical pairs are excluded.
    """
    raw=np.asarray(adata.X,np.float32);lookup={g:i for i,g in enumerate(adata.var_names)}
    result=np.zeros((len(adata),base_config.mi_count),np.float32)
    for start in range(0,len(adata),256):
        centers=np.arange(start,min(start+256,len(adata)));members=np.asarray(data.members[centers])
        pair=physical_weights(torch.from_numpy(data.coordinates[members]),torch.from_numpy(np.asarray(data.weights[centers]).copy()),20.,torch.from_numpy(members)).numpy()
        center_position=(members==centers[:,None])
        if not (center_position.sum(1)==1).all():raise ValueError('Center identity missing/duplicated in physical neighborhood')
        incoming=(pair*center_position[:,None,:]).sum(2)
        for spec in program_specs(base_config):
            j=spec['MI']-1;links=lr[lr.mi==j+1]
            ligand=raw[members][:,:,[lookup[g] for g in links.ligand]]
            receptor=raw[centers][:,[lookup[g] for g in links.receptor]]
            loading=links.loading.to_numpy(np.float32);loading=loading/loading.sum()
            coexpression=(np.sqrt(ligand*receptor[:,None,:])*loading[None,None,:]).sum(2)
            weights=incoming*(data.types[members]==TYPES.index(spec['sender']))
            opportunity=weights.sum(1)
            signal=(weights*coexpression).sum(1)/(opportunity+1e-4)
            receiving=data.types[centers]==TYPES.index(spec['receiver'])
            result[centers,j]=receiving*np.clip(signal/base_config.signal_strength,0,2)
    return result


def inject_responses(adata,lr,data,base_config,config):
    """Append held-out synthetic genes; original genes/counts/coordinates stay exact."""
    exposure=receiver_exposure(adata,lr,data,base_config)
    rng=np.random.default_rng(config.seed+1882);records=[]
    specs=program_specs(base_config)
    for spec in specs:
        for j in range(config.response_genes_per_program):
            direction=1 if j<(config.response_genes_per_program+1)//2 else -1
            gene=f'SYN_RESP_MI{spec["MI"]}_{"UP" if direction>0 else "DOWN"}_{j+1}'
            records.append({'gene':gene,'MI':spec['MI'],'receiver':spec['receiver'],'direction':direction,
                            'loading':float(rng.uniform(.7,1.3)),'kind':'synthetic_response','provenance':'Designed synthetic response; NOT a curated biological target'})
    for j in range(config.negative_control_genes):
        records.append({'gene':f'SYN_NULL_{j+1}','MI':0,'receiver':'','direction':0,'loading':0.,
                        'kind':'negative_control','provenance':'Type baseline and independent nuisance; no injected communication response'})
    table=pd.DataFrame(records);n=len(adata);g=len(table)
    if set(table.gene)&set(adata.var_names):raise ValueError('Synthetic response gene identity collision')
    means=rng.uniform(2,6,(len(TYPES),g));rates=means[data.types]
    # Mirror controls share nuisance donors, maintaining the existing pairing.
    pairs=adata.obs.mirror_pair.to_numpy();_,inverse=np.unique(pairs,return_inverse=True)
    nuisance=rng.normal(0,config.nuisance_sd,(inverse.max()+1,2))[inverse]
    nuisance_loading=rng.normal(0,.4,(2,g));rates=rates*np.exp(nuisance@nuisance_loading)
    baseline=rng.poisson(rates).astype(np.float32);after=baseline.copy()
    for j,row in table.iterrows():
        if row.MI:
            effect=config.response_log_fold*row.loading*row.direction*exposure[:,int(row.MI)-1]
            after[:,j]=rng.poisson(rates[:,j]*np.exp(effect)).astype(np.float32)
    original=np.asarray(adata.X,np.float32)
    enriched=ad.AnnData(np.column_stack((original,after)),obs=adata.obs.copy(),var=pd.DataFrame(index=[*adata.var_names,*table.gene]))
    enriched.obsm['spatial']=np.asarray(adata.obsm['spatial']).copy()
    enriched.uns['response_provenance']='Synthetic non-LR receiver response to observed incoming LR signal; background/control genes are not biological pathway annotations'
    assert np.array_equal(enriched.X[:,:original.shape[1]],original)
    assert np.array_equal(enriched.obsm['spatial'],adata.obsm['spatial'])
    return enriched,table,exposure,baseline
