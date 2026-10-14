from dataclasses import dataclass,asdict
from pathlib import Path
import json,hashlib
from ..storage import STORAGE_ROOT
ROOT=Path(__file__).resolve().parents[3]

@dataclass(frozen=True)
class SimulationConfig:
    n_cells:int=8000
    seed:int=40700
    n_genes:int=256
    n_lr:int=75
    mi_lr_count:int=6
    program_design:str='four_programs'
    basal_lr_per_type:int=3
    basal_lr_mean:float=6.
    basal_lr_offtype_mean:float=.03
    core_radius_um:float=250.
    edge_width_um:float=100.
    outer_radius_um:float=600.
    block_um:float=180.
    radial_bins:int=8
    replicates:int=1
    signal_strength:float=20.
    sender_type:str='Tumor'
    receiver_type:str='Fibroblast'
    core_proportions:tuple=(.88,.04,.03,.03,.02)
    edge_proportions:tuple=(.30,.20,.25,.15,.10)
    exterior_proportions:tuple=(.02,.76,.10,.07,.05)
    mi_loadings:list|None=None
    mode:str='expression'
    reference_cells_per_type:int=80
    sc_cube_epochs:int=30
    sc_cube_hidden:int=32
    sc_cube_lr:float=.001
    expression_calibration:str='type_mean'
    split_strategy:str='mirrored_sectors'
    angular_sectors:int=5
    initialize_composition:bool=False
    rescale_activity:bool=True
    ccc_scale_power:float=.5
    ccc_error:str='huber'
    ccc_huber_delta:float=3.
    gnn_epochs:int=40
    dictionary_initialization:str='rms_kmeans'
    nmf_initialization_sample_size:int=8192
    nmf_initialization_max_iter:int=500
    ae_stage1_epochs:int=25
    ae_stage2_epochs:int=8
    patience:int=6
    batch_size:int=128
    programs:int=32
    hidden:int=64
    lambda_comp:float=0.
    lambda_reg:float=.005
    fixed_k:int=4
    k_min:int=4
    k_max:int=15
    device:str='auto'
    figure_dpi:int=600
    lr_atlas:str=str(ROOT/'data/commuspace_human_lr_atlas.tsv')
    output_dir:str=str(STORAGE_ROOT/'outputs/tumor_edge_simulation_four_programs')
    def __post_init__(self):
        if self.dictionary_initialization not in ('rms_kmeans','log1p_active_kmeans','nmf'):
            raise ValueError('Unknown communication dictionary initialization')
        if self.nmf_initialization_sample_size<self.programs or self.nmf_initialization_max_iter<1:
            raise ValueError('Invalid NMF initialization budget')
        if self.n_cells<100 or self.n_cells%2 or self.n_cells//(2*self.replicates)<20:raise ValueError('Use even n_cells >=100 and >=40 cells/replicate')
        if self.replicates not in (1,) and self.replicates<3:raise ValueError('Use one spatially split tissue or >=3 sample-split replicates')
        if not 0<self.core_radius_um<self.core_radius_um+self.edge_width_um<self.outer_radius_um:raise ValueError('Invalid radial geometry')
        if self.program_design not in ('edge_only','four_programs'):raise ValueError('Unknown synthetic program design')
        if self.mode not in ('expression','spatial_pairing_only'):raise ValueError('Unknown injection mode')
        if self.expression_calibration not in ('none','type_mean'):raise ValueError('Unknown expression calibration')
        if self.split_strategy not in ('grid','mirrored_sectors') or self.angular_sectors<5:raise ValueError('Invalid geometry-only split strategy')
        if not 0<=self.ccc_scale_power<=1 or self.ccc_error not in ('huber','mse') or self.ccc_huber_delta<=0:raise ValueError('Invalid CCC supervision scaling/objective')
        if not 50<=self.n_lr<=100 or not 5<=self.mi_lr_count<=8 or self.n_genes<2*self.n_lr:raise ValueError('Use 50–100 LR, 5–8 LR/MI, and enough genes')
        if self.basal_lr_per_type<0 or self.mi_count*self.mi_lr_count+5*self.basal_lr_per_type>self.n_lr or not 0<=self.basal_lr_offtype_mean<self.basal_lr_mean:raise ValueError('Invalid type-specific basal LR profile')
        if self.programs!=32 or min(self.gnn_epochs,self.ae_stage1_epochs,self.ae_stage2_epochs)<2 or self.sc_cube_epochs<1:raise ValueError('Require 32 programs and positive epoch budgets')
        if min(self.lambda_comp,self.lambda_reg)<0 or self.signal_strength<=0:raise ValueError('Invalid loss/signal weights')
        names=('Tumor','Normal epithelial','Fibroblast','Macrophage','T cell')
        if self.sender_type not in names or self.receiver_type not in names or self.sender_type==self.receiver_type:raise ValueError('Choose distinct sender/receiver roles from the five cell types')
        for prop in (self.core_proportions,self.edge_proportions,self.exterior_proportions):
            if len(prop)!=5 or min(prop)<0 or abs(sum(prop)-1)>1e-6:raise ValueError('Type proportions must be five nonnegative entries summing to one')
        if self.mi_loadings is not None:
            if len(self.mi_loadings)!=self.mi_count or any(len(row)!=self.mi_lr_count or min(row)<=0 for row in self.mi_loadings):raise ValueError('mi_loadings must have one positive mi_lr_count-length row per designed MI')
    @property
    def mi_count(self):return 4 if self.program_design=='four_programs' else 2
    def to_dict(self):return json.loads(json.dumps(asdict(self)))
    @classmethod
    def from_json(cls,path):return cls(**json.loads(Path(path).read_text()))
    def signature(self):return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True).encode()).hexdigest()
