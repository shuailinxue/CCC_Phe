"""Audit Reactome mappings and retain reactions belonging to the nominated pathway.
Identical LR membership sets are collapsed, without consulting niche labels.
"""
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import json, re, requests, concurrent.futures, time
import pandas as pd
base=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace')
aud=base/'outputs/pathway_prior_expansion/reactome_v1'
out=base/'outputs/pathway_prior_expansion/v3_comparison/reaction_in_pathway_dedup';out.mkdir(parents=True,exist_ok=True)
cache=out/'reactome_pathway_events.json'
all_paths=set()
datasets=['tumor_simulation','nichecompass_simulation_lr10','starmap_legacy_human_ortholog','merfish_legacy_human_ortholog']
for name in datasets:
 d=pd.read_csv(aud/f'{name}_reactome_candidates_with_reactions.csv')
 all_paths.update(d.loc[d.same_reaction_anywhere, 'pathway_id'])
known=json.loads(cache.read_text()) if cache.exists() else {}
def fetch(pid):
 url=f'https://reactome.org/ContentService/data/pathway/{pid}/containedEvents/stId'
 for attempt in range(3):
  try:
   r=requests.get(url,timeout=35);r.raise_for_status();return pid,re.findall(r'R-HSA-\d+',r.text)
  except requests.RequestException:
   if attempt==2: raise
   time.sleep(2**attempt)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
 for pid,ids in pool.map(fetch, sorted(all_paths-set(known))):
  known[pid]=ids;cache.write_text(json.dumps(known))
def descendants(pid): return set(known[pid])
summary=[]
for dataset in datasets:
 d=pd.read_csv(aud/f'{dataset}_reactome_candidates_with_reactions.csv').fillna({'shared_reaction_ids':''})
 anyr=d[d.same_reaction_anywhere].copy()
 anyr['pathway_reaction_ids']=[ '|'.join(sorted(set(rs.split('|')) & (descendants(pid)|{pid}))) for pid,rs in zip(anyr.pathway_id,anyr.shared_reaction_ids)]
 supported=anyr[anyr.pathway_reaction_ids.ne('')].copy()
 groups=supported.groupby('pathway_id').apply(lambda x:tuple(sorted(set(zip(x.ligand,x.receptor)))),include_groups=False)
 # Identical feature membership implies identical scores for every sender-receiver pair.
 canonical={};mapping={}
 for pid in sorted(groups.index):
  key=groups[pid];canonical.setdefault(key,pid);mapping[pid]=canonical[key]
 supported['representative_pathway']=supported.pathway_id.map(mapping)
 filtered=supported[supported.pathway_id.eq(supported.representative_pathway)].copy()
 filtered.to_csv(out/f'{dataset}_filtered.csv',index=False)
 pd.DataFrame([{'pathway_id':k,'representative_pathway':v} for k,v in mapping.items()]).to_csv(out/f'{dataset}_collapsed_pathways.csv',index=False)
 summary.append(dict(dataset=dataset,co_membership_rows=len(d),reaction_anywhere_rows=len(anyr),reaction_within_pathway_rows=len(supported),after_dedup_rows=len(filtered),reaction_anywhere_pairs=anyr[['ligand','receptor']].drop_duplicates().shape[0],within_pathway_pairs=supported[['ligand','receptor']].drop_duplicates().shape[0],anywhere_pathways=anyr.pathway_id.nunique(),within_pathway_pathways=supported.pathway_id.nunique(),dedup_pathways=filtered.pathway_id.nunique()))
pd.DataFrame(summary).to_csv(out/'audit_summary.csv',index=False)
print(json.dumps(summary,indent=2))
