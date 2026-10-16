"""Separate encoder underfitting from rank-32 dictionary capacity on observed CCC."""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from phenoniche.gnn.losses import ccc_loss


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(8);s=args.source;o=args.output;o.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(52173);c=np.load(s/'input/ccc.npy',mmap_mode='r');pre=np.load(s/'input/preprocessing.npz');mask=pre['feature_mask'];scale=pre['feature_scale'][mask]
    rows=np.sort(rng.choice(len(c),min(3072,len(c)),replace=False));nfit=2048;fit=np.arange(len(rows))[:nfit];val=np.arange(len(rows))[nfit:]
    device='cuda:2';native=torch.as_tensor(np.asarray(c[rows][:,mask]),device=device);sc=torch.as_tensor(scale,device=device);target=native/sc
    h0=np.load(s/'gnn/H.npy')[:,mask];h=torch.as_tensor(h0,device=device)/sc;z0=np.load(s/'gnn/Z.npy',mmap_mode='r')[rows];z=torch.as_tensor(z0,device=device).clone()
    records=[]
    def record(name,w,d):
        pred=w@d;error=torch.nn.functional.huber_loss(pred,target,delta=3.,reduction='none')
        for part,ids in [('factorization_fit',fit),('held_out_centers',val)]:
            e=error[ids];positive=target[ids]>0
            records.append({'method':name,'subset':part,'scaled_CCC_Huber':float(e.mean()),'positive_scaled_Huber':float(e[positive].mean()),'zero_scaled_Huber':float(e[~positive].mean())})
    record('Existing GNN',z,h)
    # Fixed learned dictionary, solve only per-center activations (encoder ceiling).
    gram=h@h.T;numerator=target@h.T
    for _ in range(250):z*=numerator/(z@gram).clamp_min(1e-12)
    record('NNLS with learned H',z,h);np.save(o/'fixed_H_NNLS_Z.npy',z.cpu().numpy())
    # MU solves squared error, whereas the real objective is Huber. Use the
    # actual objective for an encoder ceiling rather than conflating losses.
    norm=h.norm(dim=1).clamp_min(1e-12);basis=h/norm[:,None]
    active=torch.nn.Parameter(torch.as_tensor(z0,device=device)*norm)
    optimizer=torch.optim.Adam([active],lr=.1)
    for _ in range(350):
        optimizer.zero_grad();error=torch.nn.functional.huber_loss(active@basis,target,delta=3.)
        error.backward();optimizer.step()
        with torch.no_grad():active.clamp_(min=0)
    record('Huber activation optimum with learned H',active.detach(),basis)
    # Dictionary uses FIT centers only; evaluate held-out rows with NNLS.
    fresh_h=h.clone();fresh_z=z.clone()
    for _ in range(250):
        fresh_z[fit]*=(target[fit]@fresh_h.T)/(fresh_z[fit]@(fresh_h@fresh_h.T)).clamp_min(1e-12)
        fresh_h*=(fresh_z[fit].T@target[fit])/((fresh_z[fit].T@fresh_z[fit])@fresh_h).clamp_min(1e-12)
        mass=fresh_h.norm(dim=1).clamp_min(1e-12);fresh_h/=mass[:,None];fresh_z*=mass
    for _ in range(250):fresh_z[val]*=(target[val]@fresh_h.T)/(fresh_z[val]@(fresh_h@fresh_h.T)).clamp_min(1e-12)
    record('Rank32 NMF refined on 2048 centers',fresh_z,fresh_h)
    pd.DataFrame(records).to_csv(o/'encoder_dictionary_bottleneck.csv',index=False)
    np.save(o/'probe_rows.npy',rows);np.save(o/'rank32_NMF_Z.npy',fresh_z.cpu().numpy());np.save(o/'rank32_NMF_H_scaled.npy',fresh_h.cpu().numpy())
    (o/'bottleneck_audit.json').write_text(json.dumps({'device':device,'condition_labels_used':False,'original_model_modified':False,'NMF_dictionary_fit_centers':len(fit),'NMF_dictionary_holdout_centers':len(val),'scope':'Reconstruction mechanism diagnostic; random centers, not an independent generalization estimate'},indent=2))
    print(pd.DataFrame(records).to_string(index=False),flush=True)

if __name__=='__main__':main()
