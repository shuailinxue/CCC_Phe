"""Prepare independent controls for pathway-reaction filtering and deduplication."""
from pathlib import Path
import pandas as pd
base=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/pathway_prior_expansion')
aud=base/'reactome_v1';root=base/'v3_comparison';cache=root/'reaction_in_pathway_dedup/reactome_pathway_events.json'
import json
known=json.loads(cache.read_text())
for mode in ['reaction_in_pathway','reaction_anywhere_dedup']:
 out=root/mode;out.mkdir(parents=True,exist_ok=True)
 for name in ['tumor_simulation','nichecompass_simulation_lr10']:
  d=pd.read_csv(aud/f'{name}_reactome_candidates_with_reactions.csv').fillna({'shared_reaction_ids':''});d=d[d.same_reaction_anywhere].copy()
  if mode=='reaction_in_pathway':
   keep=[bool(set(r.split('|')) & set(known[p])) for p,r in zip(d.pathway_id,d.shared_reaction_ids)];d=d[keep].copy()
  else:
   grouped=d.groupby('pathway_id').apply(lambda x:tuple(sorted(set(zip(x.ligand,x.receptor)))),include_groups=False)
   seen={};rep={}
   for pid in sorted(grouped.index):
    k=grouped[pid];seen.setdefault(k,pid);rep[pid]=seen[k]
   d=d[d.pathway_id.map(rep).eq(d.pathway_id)]
  d.to_csv(out/f'{name}_filtered.csv',index=False)
  print(mode,name,'rows',len(d),'pathways',d.pathway_id.nunique(),'pairs',len(d[['ligand','receptor']].drop_duplicates()),flush=True)
