import pandas as pd
from phenoniche.v1005.niche_graph import select_resolution

def test_resolution_selection_respects_range_tiny_and_plateau():
    table=pd.DataFrame({'resolution':[.001,.005,.01,.02],'n_niches':[3,8,10,40],
        'fraction_cells_niches_lt20':[0,0,.02,0],
        'adjacent_count_relative_change':[.1,.2,.1,.1],
        'adjacent_stability':[.9,.8,.99,.9],
        'modularity':[.4,.6,.9,.8],'spatial_agreement':[.5,.7,.9,.8]})
    index,reason=select_resolution(table)
    assert index==1 and 'stability' in reason.lower()

def test_resolution_selection_reports_no_eligible_result():
    table=pd.DataFrame({'resolution':[.001,.01],'n_niches':[2,30],'fraction_cells_niches_lt20':[0,0],
        'adjacent_count_relative_change':[0,0],'adjacent_stability':[1,1],
        'modularity':[.5,.6],'spatial_agreement':[.5,.6]})
    index,reason=select_resolution(table)
    assert index is None and 'No resolution' in reason
