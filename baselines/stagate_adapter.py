"""Official STAGATE_pyG model/training, isolated training graph and frozen inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from threadpoolctl import threadpool_limits
from .adapters import log_expression


def _spatial_net(adata, graph):
    rows, cols = graph.nonzero()
    adata.uns['Spatial_Net'] = pd.DataFrame({'Cell1':adata.obs_names[rows],
                                             'Cell2':adata.obs_names[cols], 'Distance':1.})


def stagate(a, graph, train_cells, config, output):
    from .vendor.stagate.Train_STAGATE import train_STAGATE
    from .vendor.stagate.utils import Transfer_pytorch_Data
    output = Path(output); device = torch.device(config['device'])
    full = a.copy(); full.X = sp.csr_matrix(log_expression(a))
    symmetric = graph.maximum(graph.T).tocsr()
    tr = full[train_cells].copy(); _spatial_net(tr, symmetric[train_cells][:,train_cells])
    with threadpool_limits(limits=config.get('cpu_threads',6)):
        tr, model = train_STAGATE(tr, hidden_dims=config['stagate_hidden_dims'],
                                  n_epochs=config['stagate_epochs'], lr=config['stagate_lr'],
                                  gradient_clipping=config['stagate_gradient_clipping'],
                                  weight_decay=config['stagate_weight_decay'],
                                  random_seed=config['seed'], save_loss=True,verbose=False,
                                  device=device,return_model=True)
        training = np.asarray(tr.obsm['STAGATE'],np.float32)
        _spatial_net(full,symmetric)
        data = Transfer_pytorch_Data(full).to(device); model.eval()
        with torch.no_grad():
            z, reconstructed = model(data.x,data.edge_index)
            embedding = z.cpu().numpy().astype(np.float32)
            row_mse = ((reconstructed-data.x)**2).mean(1).cpu().numpy()
    torch.save({'model_state':model.state_dict(),'config':config,'training_cells':train_cells,
                'genes':list(a.var_names),'normalization':'log1p(count/total*10000)'},output/'checkpoint.pt')
    pd.DataFrame({'epoch':np.arange(1,len(tr.uns['STAGATE_loss'])+1),
                  'expression_MSE':tr.uns['STAGATE_loss']}).to_csv(output/'training.csv',index=False)
    np.save(output/'expression_reconstruction_row_MSE.npy',row_mse)
    (output/'adapter_audit.json').write_text(json.dumps({
        'variant':'Official STAGATE_pyG, all base-panel genes, shared KNN spatial graph',
        'native_model_loss_optimizer_unchanged':True,
        'spatial_graph':'Symmetrized shared KNN30; native self loops; induced training-cell graph only',
        'training_physical_cells_only':True,'cell_type_or_CCC_input':False,
        'truth_MI_labels_or_loading_used':False,'hidden_dims':config['stagate_hidden_dims'],
        'epochs':config['stagate_epochs'],'training_expression_MSE':float(tr.uns['STAGATE_loss'][-1]),
        'inference':'Frozen node-independent native weights on full graph; common two-hop spatial test purge',
        'interface_changes':'Return model for checkpoint/frozen inference; detach scalar loss history only'
    },indent=2)+'\n')
    return embedding,training
