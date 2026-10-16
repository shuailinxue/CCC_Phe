"""Small transparent table of tissue-pair and clustering-distance alternatives."""
from pathlib import Path
import pandas as pd,numpy as np

ROOT=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs')


def main():
    rows=[]
    def add(pair,mode,path):
        f=pd.read_csv(path,index_col=0)
        h=f.loc['HC'].to_numpy(float);d=f.loc['DM'].to_numpy(float);shared=(h>=.01)&(d>=.01)
        rows.append({'Pair':pair,'Distance':mode,'K':len(h),'Shared niches':int(shared.sum()),
            'HC in shared (%)':round(float(h[shared].sum()*100),1),
            'DM in shared (%)':round(float(d[shared].sum()*100),1),
            'Overlap mass (%)':round(float(np.minimum(h,d).sum()*100),1),
            'DM-enriched niches':int(((h<=.01)&(d>=.05)).sum())})
    add('H6 / DM25','Program coordinates',ROOT/'dermatomyositis_5k_pair/condition_niche_fractions.csv')
    exp=ROOT/'dermatomyositis_shared_structure'
    for pair in ('H7_DM19','H8_DM27'):
        out=exp/pair
        control=out/'clustering_controls/program_mixture_activity/condition_niche_fractions.csv'
        if control.exists():add(pair.replace('_',' / '),'Program coordinates',control)
        if (out/'condition_niche_fractions.csv').exists():add(pair.replace('_',' / '),'CCC dictionary',out/'condition_niche_fractions.csv')
    table=pd.DataFrame(rows);table.to_csv(exp/'diagnostics/pair_and_metric_controls.csv',index=False)
    for pair in ('H7_DM19','H8_DM27'):
        if (exp/pair).exists():table.to_csv(exp/pair/'selection_and_metric_comparison.csv',index=False)
    print(table.to_string(index=False))


if __name__=='__main__':main()
