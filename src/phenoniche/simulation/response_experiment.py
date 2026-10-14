"""Frozen CCC anchor + expression decoder experiment, isolated from clinical work."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import anndata as ad
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score,roc_auc_score
from ..storage import STORAGE_ROOT
from .config import SimulationConfig
from .workflow import run_experiment,dataset
from .adapter import adapt
from .response_simulation import ExpressionExperimentConfig,inject_responses,checksum
from .expression_decoder import weighted_expression_targets,fit_expression,response_split_rows

FROZEN=('gnn/checkpoint.pt','gnn/Z.npy','gnn/H.npy','gnn/P.npy','gnn/activity.npy','raw_clusters.csv','metrics.csv','injected.h5ad','lr.csv','input/ccc.npy','ground_truth/MI_dictionary.npy')
REQUIRED=('response_injected.h5ad','response_gene_design.csv','expression_target.npy','expression_present.npy','expression_normalization.npy','expression/checkpoint.pt','expression/prediction.npy','expression/baseline_prediction.npy','expression/dictionary.npy','expression/preprocessing.npz','expression/training.csv','expression_gene_metrics.csv','synthetic_module_recovery.csv','program_gene_ranking.csv','program_top_genes.csv','ground_truth/receiver_exposure.npy','ground_truth/response_baseline_counts.npy','expression_split_rows.npz','summary.json','complete.json')


def metrics(target,prediction,baseline,present,mask,rows,genes,table,categories,output):
    new_genes=table.set_index('gene');records=[]
    for t,kind in enumerate(categories):
        chosen=rows[present[rows,t]]
        for g,gene in enumerate(genes):
            if not len(chosen) or not mask[t,g]:continue
            truth=target[chosen,t,g];pred=prediction[chosen,t,g];null=baseline[chosen,t,g]
            sst=float(((truth-truth.mean())**2).sum());sse=float(((truth-pred)**2).sum());null_sse=float(((truth-null)**2).sum())
            annotated=new_genes.loc[gene] if gene in new_genes.index else None
            records.append({'cell_type':kind,'gene':gene,'n':len(chosen),
                'kind':annotated['kind'] if annotated is not None else 'original_panel',
                'MI':int(annotated['MI']) if annotated is not None else 0,
                'designed_receiver':annotated['receiver'] if annotated is not None else '',
                'R2':1-sse/sst if sst>1e-10 else np.nan,'baseline_R2':1-null_sse/sst if sst>1e-10 else np.nan,
                'gain_over_type_library_baseline':1-sse/null_sse if null_sse>1e-10 else np.nan,
                'RMSE':np.sqrt(sse/len(chosen)),'baseline_RMSE':np.sqrt(null_sse/len(chosen))})
    frame=pd.DataFrame(records);frame.to_csv(Path(output)/'expression_gene_metrics.csv',index=False)
    return frame


def loading_evaluation(h,true_h,dictionary,genes,table,categories,output):
    # Ground truth is used here only, never in decoder fitting/model selection.
    cosine=(true_h@h.T)/np.maximum(np.linalg.norm(true_h,axis=1)[:,None]*np.linalg.norm(h,axis=1)[None],1e-12)
    mi,program=linear_sum_assignment(-cosine);records=[];rankings=[]
    positions={g:i for i,g in enumerate(genes)}
    for j,p in zip(mi,program):
        definition=table[table.MI==j+1];kind=definition.receiver.iloc[0];t=categories.index(kind)
        scores=np.abs(dictionary[t,p,[positions[g] for g in table.gene]])
        labels=(table.MI.to_numpy()==j+1).astype(int)
        records.append({'MI':j+1,'receiver':kind,'program':p+1,'CCC_dictionary_cosine':float(cosine[j,p]),
            'synthetic_module_AUROC':float(roc_auc_score(labels,scores)),
            'synthetic_module_AUPRC':float(average_precision_score(labels,scores)),
            'positive_fraction':float(labels.mean()),'interpretation':'Synthetic module recovery on appended genes; no GO/biological pathway claim'})
        for index in np.argsort(-scores):
            g=positions[table.gene.iloc[index]]
            rankings.append({'MI':j+1,'program':p+1,'receiver':kind,'gene':genes[g],
                             'loading':float(dictionary[t,p,g]),'kind':table.kind.iloc[index],
                             'designed_gene_MI':int(table.MI.iloc[index]),'designed_direction':int(table.direction.iloc[index])})
    pd.DataFrame(records).to_csv(Path(output)/'synthetic_module_recovery.csv',index=False)
    pd.DataFrame(rankings).to_csv(Path(output)/'program_gene_ranking.csv',index=False)


def run_expression_experiment(config,base_config=None):
    if config.training_mode=='joint':
        from .joint_expression import run_joint_experiment
        return run_joint_experiment(config,base_config)
    base=base_config or SimulationConfig.from_json(config.base_simulation_config)
    source=Path(base.output_dir);out=Path(config.output_dir)
    if out.is_symlink() or not out.resolve().is_relative_to(STORAGE_ROOT.resolve()):raise ValueError('Direct NAS workspace output required')
    if out.resolve()==source.resolve() or out.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(out.resolve()):
        raise ValueError('Expression experiment must not overwrite or contain its CCC anchor')
    out.mkdir(parents=True,exist_ok=True)
    manifest={'expression_config':config.to_dict(),'base_config':base.to_dict(),
              'implementation':{p.name:checksum(p) for p in [Path(__file__),Path(__file__).with_name('response_simulation.py'),Path(__file__).with_name('expression_decoder.py')]}}
    previous=out/'config.json'
    if previous.exists() and json.loads(previous.read_text())!=manifest:raise ValueError('Expression experiment configuration/code changed; preserve existing output and select another directory')
    previous.write_text(json.dumps(manifest,indent=2)+'\n')
    # User Run All may train the existing CCC module if missing. Completed
    # matching source results are reused; no response genes enter that module.
    run_experiment(base)
    snapshots={name:checksum(source/name) for name in FROZEN}
    if (out/'frozen_anchor.json').exists() and json.loads((out/'frozen_anchor.json').read_text())!=snapshots:
        raise ValueError('Frozen CCC anchor changed; refusing to reuse expression outputs')
    (out/'frozen_anchor.json').write_text(json.dumps(snapshots,indent=2)+'\n')
    if all((out/name).is_file() for name in REQUIRED):
        completion=json.loads((out/'complete.json').read_text())
        if completion.get('manifest')==manifest and completion.get('frozen_anchor')==snapshots:return out
    adata=ad.read_h5ad(source/'injected.h5ad');lr=pd.read_csv(source/'lr.csv')
    adapter_dir=out/'anchor_inputs';adapter_dir.mkdir(exist_ok=True)
    cache=adapter_dir/'input';cache.mkdir(exist_ok=True)
    if not (cache/'ccc.npy').exists():shutil.copy2(source/'input/ccc.npy',cache/'ccc.npy')
    data=adapt(adata,lr,base,adapter_dir)
    if checksum(cache/'ccc.npy')!=checksum(source/'input/ccc.npy'):raise ValueError('CCC supervision changed')
    expression_rows=response_split_rows(data)
    np.savez(out/'expression_split_rows.npz',train=expression_rows[0],validation=expression_rows[1],test=expression_rows[2])
    target_path=out/'expression_target.npy'
    input_files=('response_injected.h5ad','response_gene_design.csv','expression_target.npy','expression_present.npy','expression_normalization.npy','ground_truth/receiver_exposure.npy','ground_truth/response_baseline_counts.npy')
    if all((out/name).exists() for name in input_files):
        enriched=ad.read_h5ad(out/'response_injected.h5ad');table=pd.read_csv(out/'response_gene_design.csv',keep_default_na=False)
        target=np.load(target_path,mmap_mode='r');present=np.load(out/'expression_present.npy');normalization=np.load(out/'expression_normalization.npy')
        if not np.array_equal(enriched.X[:,:adata.n_vars],adata.X):raise ValueError('Original panel expression changed')
        if target.shape!=(len(adata),len(data.categories),enriched.n_vars) or present.shape!=target.shape[:2] or normalization.shape!=present.shape:raise ValueError('Expression cache shape mismatch')
    else:
        enriched,table,exposure,before=inject_responses(adata,lr,data,base,config)
        enriched.write_h5ad(out/'response_injected.h5ad');table.to_csv(out/'response_gene_design.csv',index=False)
        truth=out/'ground_truth';truth.mkdir(exist_ok=True)
        np.save(truth/'receiver_exposure.npy',exposure);np.save(truth/'response_baseline_counts.npy',before)
        target,present,normalization=weighted_expression_targets(enriched,data,adata.n_vars)
        np.save(target_path,target);np.save(out/'expression_present.npy',present);np.save(out/'expression_normalization.npy',normalization)
    z=np.load(source/'gnn/Z.npy',mmap_mode='r');h=np.load(source/'gnn/H.npy')
    if z.shape!=(len(adata),base.programs) or h.shape!=(base.programs,len(data.features)):raise ValueError('Frozen CCC program/feature alignment mismatch')
    prediction,baseline,mask,best_epoch=fit_expression(z,target,present,normalization,expression_rows,config,out/'expression')
    gene_metrics=metrics(target,prediction,baseline,present,mask,expression_rows[2],list(enriched.var_names),table,data.categories,out)
    loading_evaluation(h,np.load(source/'ground_truth/MI_dictionary.npy'),np.load(out/'expression/dictionary.npy'),list(enriched.var_names),table,data.categories,out)
    # Every learned program has an inspectable gene dictionary by type.
    dictionary=np.load(out/'expression/dictionary.npy');top=[]
    for t,kind in enumerate(data.categories):
        for p in range(len(h)):
            for g in np.argsort(-np.abs(dictionary[t,p]))[:10]:
                top.append({'program':p+1,'cell_type':kind,'gene':enriched.var_names[g],'loading':float(dictionary[t,p,g])})
    pd.DataFrame(top).to_csv(out/'program_top_genes.csv',index=False)
    for name,digest in snapshots.items():
        if checksum(source/name)!=digest:raise RuntimeError('CCC model/result changed during expression fitting: '+name)
    null=gene_metrics[gene_metrics.kind=='negative_control'];response=gene_metrics[(gene_metrics.kind=='synthetic_response')&(gene_metrics.cell_type==gene_metrics.designed_receiver)]
    summary={'source_CCC_results':str(source),'cells':len(adata),'base_genes':adata.n_vars,'expression_genes':enriched.n_vars,
        'programs':z.shape[1],'CCC_epochs':base.gnn_epochs,'expression_best_epoch':best_epoch,
        'expression_training':'Frozen Z; type-conditioned signed gene dictionary; library offset; masked Huber + sparse dictionary',
        'targets':'Log1p of physical-pair-degree-weighted normalized expression within each cell type',
        'response_genes_excluded_from_encoder':True,'CCC_weights_Z_H_niches_unchanged':True,
        'source_CCC_split_sizes':[len(r) for r in data.rows],'expression_split_sizes':[len(r) for r in expression_rows],
        'response_target_split':'Conservative two-hop physical-cell purge; no incoming-response source cells shared between folds',
        'median_receiver_response_R2':float(response.R2.median()),
        'median_receiver_response_gain_over_baseline':float(response.gain_over_type_library_baseline.median()),
        'median_negative_control_gain_over_baseline':float(null.gain_over_type_library_baseline.median()),
        'synthetic_targets_are_not_curated_biological_priors':True,'clinical_Cox_modified':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'complete.json').write_text(json.dumps({'manifest':manifest,'frozen_anchor':snapshots},indent=2)+'\n')
    # Release NFS mappings before temporary smoke-test directories are removed.
    for value in [data.targets.matrix,*vars(data).values(),target,z]:
        if isinstance(value,np.memmap):value._mmap.close()
    return out
