"""Audited downstream entry points; legacy models/results are read-only."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score
from ..config import default_config as legacy_config
from ..data import PreparedInput
from ..training import train_all
from .config import GNNConfig
from .audit import audit_study,protect_output
from .data import prepare_graph_data
from .training import fit,infer,fit_alignments
from .model import MicroenvironmentGNN
from .evaluation import discover,diagnostics,plot_spatial,match_programs


def require_audit(config):
    report=audit_study(config)
    if not report['ready_for_training']:
        raise RuntimeError('Real data analysis blocked by audit: '+'; '.join(report['blockers']))
    return report


def run(config):
    require_audit(config)
    output=Path(config.output_dir)
    if (output/'checkpoint.pt').exists(): raise FileExistsError('Existing GNN checkpoint is protected; use infer or a different experiment output')
    data=prepare_graph_data(config)
    observed_reference=data.observations.copy();mask_reference=data.observed_masks.copy()
    mapping_reference={branch:prior.mapping.copy() for branch,prior in data.priors.items()}
    apply_control(data,config)
    model,alignments=fit(data,config)
    # Nulls change the training objective; final biological evaluation always uses
    # actual observations and original priors, never shuffled test labels.
    data.observations=observed_reference;data.observed_masks=mask_reference
    for branch,prior in data.priors.items(): prior.mapping=mapping_reference[branch]
    if config.prior_control!='none': alignments=fit_alignments(data,config)
    arrays=infer(model,data,config.output_dir,config.batch_size);h=model.dictionary.detach().cpu().numpy()
    k,labels,_=discover(data,arrays,h,config)
    summary=diagnostics(data,arrays,h,labels,alignments,config)
    plot_spatial(data,labels,k,config.output_dir)
    summary.update(selected_K=k,mode=config.mode,phenotype_supervision=False,legacy_results_modified=False,prior_control=config.prior_control)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


class IndexedRows:
    """Lazy permutation of observed supervision, preserving immutable cache files."""
    def __init__(self,base,indices): self.base,self.indices,self.shape=base,indices,base.shape
    def __getitem__(self,rows): return self.base[self.indices[rows]]


def apply_control(data,config):
    control=getattr(config,'prior_control','none')
    if control=='none': return
    rng=np.random.default_rng(config.seed)
    if control=='shuffle_links':
        for prior in data.priors.values():
            for _,indices in data.features.groupby(['sender','receiver']).groups.items():
                indices=np.asarray(indices);prior.mapping[indices]=prior.mapping[rng.permutation(indices)]
    elif control=='shuffle_activity':
        # Preserve fold, block and center type; use an index view, not an N×M copy.
        groups={}
        for i,key in enumerate(zip(data.folds,data.blocks,data.types)): groups.setdefault(key,[]).append(i)
        permutation=np.arange(len(data.cells))
        for indices in groups.values():
            indices=np.asarray(indices);permutation[indices]=rng.permutation(indices)
        if np.array_equal(permutation,np.arange(len(permutation))): raise ValueError('Activity negative control has no exchangeable rows')
        for branch in data.observations:
            data.observations[branch]=IndexedRows(data.observations[branch],permutation)
            data.observed_masks[branch]=IndexedRows(data.observed_masks[branch],permutation)
    else: raise ValueError('Unknown biological negative control')
    data.audit['prior_control']=control


def inference(config):
    protect_output(config)
    payload=torch.load(Path(config.output_dir)/'checkpoint.pt',map_location='cpu',weights_only=False)
    saved=GNNConfig(**payload['config'])
    if saved.output_dir!=config.output_dir: saved=replace(saved,output_dir=config.output_dir)
    require_audit(saved)
    data=prepare_graph_data(saved)
    for key in ('source_signature','cell_order_sha256'):
        if data.audit.get(key)!=payload['data_audit'].get(key): raise ValueError('Inference source identity changed: '+key)
    for branch in ('upstream','downstream'):
        if data.audit[branch]['provenance']!=payload['data_audit'][branch]['provenance']: raise ValueError('Inference prior provenance changed: '+branch)
    if data.genes!=tuple(payload['genes']) or data.categories!=tuple(payload['categories']): raise ValueError('Inference cell-gene/type vocabulary differs from training')
    for key in ('encoder_indices','gene_mean','gene_scale','feature_scale','feature_mask','folds'):
        if not np.array_equal(getattr(data,key),payload[key]): raise ValueError('Inference preprocessing changed: '+key)
    if data.features.to_dict('list')!=payload['features']: raise ValueError('CCC feature order changed')
    model=MicroenvironmentGNN(len(data.encoder_indices)+1,len(data.categories),len(data.features),saved.programs,saved.hidden,saved.sigma_um,data.feature_mask)
    model.load_state_dict(payload['model_state'])
    from ..data import resolve_device
    model.to(resolve_device(saved.device))
    return infer(model,data,saved.output_dir,saved.batch_size)


def retrain_vector_baseline(config):
    """Replay the original AE training recipe on the same disjoint spatial split.

    Archival AE metrics cannot establish generalization because its random split overlaps.
    """
    saved=Path(config.output_dir)/'config.json'
    if saved.is_file(): config=replace(GNNConfig.from_json(saved),output_dir=config.output_dir)
    require_audit(config);data=prepare_graph_data(config)
    output=Path(config.output_dir)/'vector_ae';output.mkdir(exist_ok=True)
    if (output/'stage1_best_checkpoint.pt').exists(): raise FileExistsError('Existing baseline is protected')
    prepared=PreparedInput(data.targets.matrix.filename,data.targets.matrix.shape,data.targets.columns,data.feature_scale,data.rows[0],data.rows[1],data.features,data.cells,data.coordinates,output)
    original=legacy_config();budget=config.max_epochs
    stage1=max(2,budget//2);stage2a=max(2,budget//4);stage2b=max(2,budget-stage1-stage2a)
    cfg=replace(original,output_dir=str(output),device=config.device,batch_size=config.batch_size,
                stage1_max_epochs=stage1,stage2a_max_epochs=stage2a,stage2b_max_epochs=stage2b)
    _,_,arrays,h,checkpoints=train_all(prepared,cfg,output)
    native_h=h*data.feature_scale;mass=native_h.sum(1);native_h/=np.maximum(mass[:,None],1e-8)
    np.save(output/'H_native.npy',native_h);np.save(output/'Z_native.npy',np.asarray(arrays['Z32'])*mass)
    np.save(output/'folds.npy',data.folds);data.features.to_csv(output/'features.csv',index=False)
    (output/'baseline_metadata.json').write_text(json.dumps({'recipe':'original dual-decoder AE; same spatial train/validation/test split; epochs apportioned to three original stages','epoch_budgets':[stage1,stage2a,stage2b],'seed':cfg.seed},indent=2)+'\n')
    # Test only after all original stages have selected checkpoints on validation.
    from .losses import ccc_loss
    loss_sum=0.;count=0
    z=np.load(output/'Z_native.npy',mmap_mode='r')
    for start in range(0,len(data.rows[2]),config.batch_size):
        rows=data.rows[2][start:start+config.batch_size]
        loss=ccc_loss(torch.from_numpy(np.asarray(z[rows])@native_h),torch.from_numpy(data.targets[rows]),data.feature_scale,data.feature_mask)
        loss_sum+=float(loss)*len(rows);count+=len(rows)
    (output/'heldout_metrics.json').write_text(json.dumps({'test_ccc':loss_sum/count},indent=2)+'\n')
    native_z=np.load(output/'Z_native.npy',mmap_mode='r')
    np.save(output/'H.npy',native_h)
    baseline_arrays={'Z':native_z,'P':arrays['P32'],'activity':np.asarray(native_z).sum(1)}
    baseline_config=replace(config,output_dir=str(output))
    k,labels,_=discover(data,baseline_arrays,native_h,baseline_config)
    alignments=fit_alignments(data,config)
    diagnostics(data,baseline_arrays,native_h,labels,alignments,baseline_config)
    plot_spatial(data,labels,k,output)
    return {'test_ccc':loss_sum/count,'selected_K':k}


def compare_outputs(config,other_output=None):
    """Compare dictionaries, niches, composition and held-out losses when available."""
    output=Path(config.output_dir);h=np.load(output/'H.npy')
    left_features=__import__('pandas').read_csv(output/'cache/features.csv')
    if other_output:
        other=Path(other_output);right_features=__import__('pandas').read_csv(other/'cache/features.csv');right=np.load(other/'H.npy')
        if not left_features.equals(right_features): raise ValueError('Cannot compare different CCC feature orders')
        import pandas as pd
        l=pd.read_csv(output/'selected_model/niche_assignments.csv.gz');r=pd.read_csv(other/'selected_model/niche_assignments.csv.gz')
        if not np.array_equal(l.cell_id,r.cell_id) or not np.array_equal(l['split'],r['split']): raise ValueError('Comparison cells/spatial folds differ')
        result=match_programs(h,right)
        rows=l['split'].to_numpy()==2
        result['test_niche_ARI']=float(adjusted_rand_score(l.niche.to_numpy()[rows],r.niche.to_numpy()[rows]))
        result['left_diagnostics']=json.loads((output/'diagnostics.json').read_text())
        result['right_diagnostics']=json.loads((other/'diagnostics.json').read_text())
        result['comparison']='paired held-out block evaluation; ablation/seed comparison'
    else:
        baseline=output/'vector_ae'
        if (baseline/'H_native.npy').is_file():
            right=np.load(baseline/'H_native.npy');features=__import__('pandas').read_csv(baseline/'features.csv')
            if not left_features.equals(features): raise ValueError('Baseline feature ordering differs')
            prep=np.load(output/'cache/preprocessing.npz')
            if not np.array_equal(prep['folds'],np.load(baseline/'folds.npy')): raise ValueError('Baseline folds differ')
            result=match_programs(h,right);result['baseline_test']=json.loads((baseline/'heldout_metrics.json').read_text())
            result['comparison']='original AE retrained on identical spatial folds'
            result['baseline_diagnostics']=json.loads((baseline/'diagnostics.json').read_text())
            import pandas as pd
            left=pd.read_csv(output/'selected_model/niche_assignments.csv.gz');right_assignment=pd.read_csv(baseline/'selected_model/niche_assignments.csv.gz')
            if not np.array_equal(left.cell_id,right_assignment.cell_id): raise ValueError('Baseline cell order differs')
            test=left['split'].to_numpy()==2
            result['test_niche_ARI']=float(adjusted_rand_score(left.niche.to_numpy()[test],right_assignment.niche.to_numpy()[test]))
        else:
            legacy=Path(config.legacy_output);right=np.load(legacy/'H32.npy')*np.load(legacy/'cache/feature_scale.npy')
            right/=np.maximum(right.sum(1,keepdims=True),1e-8)
            features=__import__('pandas').read_csv(legacy/'cache/retained_features.csv')
            if not np.array_equal(left_features.ccc,features.ccc): raise ValueError('Legacy CCC feature ordering differs')
            result=match_programs(h,right)
            result['comparison']='ARCHIVAL DESCRIPTIVE ONLY: random overlapping legacy split; no fair performance claim'
        result['gnn_test']=json.loads((output/'heldout_metrics.json').read_text())
        result['gnn_diagnostics']=json.loads((output/'diagnostics.json').read_text())
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
