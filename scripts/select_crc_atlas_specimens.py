"""Audit author CRC specimens for a small joint-analysis case, without fitting niches.

Expression comes from the author's native count layer. Downloaded file segments
are used only after their recorded SHA256 is checked; other reads use validated
HTTP ranges. This never reads unfilled parts of the in-progress sparse file.
"""
from pathlib import Path
import hashlib
import json
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
import requests

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phenoniche.cohorts.crc_neutrophil_atlas import ROOT, OUTPUT, RemoteH5, _column

OUT = OUTPUT / 'selection'
COUNTS = ROOT / 'processed/selection_counts'
NEIGHBORS = ROOT / 'processed/selection_neighborhoods'


def spatial_screen():
    """Audit all37 author specimens without using expression or known niches."""
    from scipy.spatial import cKDTree
    OUT.mkdir(exist_ok=True,parents=True)
    NEIGHBORS.mkdir(exist_ok=True,parents=True)
    frame=pd.read_parquet(ROOT/'processed/display_cells.parquet');rows=[];infiltration=[]
    for sid,g in frame.groupby('specimen_id',observed=True):
        ty=g.celltype.astype(str).to_numpy();xy=g[['x_um','y_um']].to_numpy()
        ds,ix=cKDTree(xy).query(xy,k=31,workers=6);ix=ix[:,1:];ds=ds[:,1:]
        c=ty=='Cancer cell';t=ty=='T cell'
        im=np.isin(ty,['T cell','B cell','Plasma cell','Macrophage','Dendritic cell','Neutrophil','Mast cell'])
        tc=t[ix].mean(1);ic=im[ix].mean(1);cc=c[ix].mean(1)
        edge=(cc>=.2)&(cc<=.8)&(ds[:,-1]<=150)
        row=dict(specimen_id=sid,cells=len(g),patient=str(g.patient_id.iloc[0]),region=str(g.tissue_region.iloc[0]),
                 radius30_median=float(np.median(ds[:,-1])),radius30_p90=float(np.quantile(ds[:,-1],.9)),
                 interface_cells=int(edge.sum()),tumor_neighbor_T=float(tc[c].mean()) if c.any() else np.nan,
                 tumor_neighbor_immune=float(ic[c].mean()) if c.any() else np.nan,
                 tumor_T_ge3_fraction=float(np.mean(tc[c]>=.1)) if c.any() else np.nan)
        for v in sorted(set(ty)):
            row['fraction_'+v]=float(np.mean(ty==v));row['edge_'+v]=float(np.mean(ty[edge]==v)) if edge.any() else 0.
        rows.append(row)
        np.savez_compressed(NEIGHBORS/f'{sid}_neighborhoods.npz',neighbors=ix.astype(np.int32),distance=ds.astype(np.float32),
                            interface=edge,tumor_fraction=cc.astype(np.float32))
        if row['region']=='core':
            infiltration.append(dict(specimen=sid,T_cells=int(t.sum()),T_in_cancermajority=int((t&(cc>=.5)).sum()),
                                     T_in_cancermajority_fraction=float(np.mean(cc[t]>=.5)),T_in_densecancer=int((t&(cc>=.8)).sum())))
        print('Spatial audit:',sid,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'specimen_spatial_screen.csv',index=False)
    pd.DataFrame(infiltration).to_csv(OUT/'core_T_infiltration.csv',index=False)


class VerifiedDownloadReader(RemoteH5):
    def __init__(self):
        super().__init__(block_size=2 << 20)
        self.partial = ROOT / 'crca_xenium.h5ad.partial'
        self.verified = set()
        self.local_bytes = 0
        self.pool=ThreadPoolExecutor(max_workers=8)
        self.pending={}
        self.block_dir=ROOT.parent.parent/'tmp/crc_selection_http_blocks'
        self.block_dir.mkdir(parents=True,exist_ok=True)

    def fetch_remote(self, number):
        path=self.block_dir/f'{number}.bin'
        if path.exists():return path.read_bytes()
        start=number*self.block_size;end=min(start+self.block_size,self.size)-1
        for attempt in range(5):
            try:
                reply=requests.get(self.url,headers={'Range':f'bytes={start}-{end}'},timeout=(15,60))
                reply.raise_for_status()
                if reply.status_code!=206 or reply.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}' or len(reply.content)!=end-start+1:
                    raise IOError('Unexpected remote expression range')
                temp=path.with_suffix('.tmp');temp.write_bytes(reply.content);temp.replace(path)
                return reply.content
            except Exception:
                if attempt==4:raise

    def _block(self, number):
        if number in self.cache:
            self.cache.move_to_end(number)
            return self.cache[number]
        start = number * self.block_size
        segment = start // (64 << 20)
        state = json.loads((ROOT / 'metadata/download_chunks.json').read_text())
        receipt = state.get('completed', {}).get(str(segment))
        if receipt is not None and self.partial.exists():
            with self.partial.open('rb') as f:
                if segment not in self.verified:
                    f.seek(segment * (64 << 20))
                    buf = f.read(receipt['bytes'])
                    if hashlib.sha256(buf).hexdigest() != receipt['sha256']:
                        raise IOError('Incomplete/corrupt local segment; refusing partial-file data')
                    self.verified.add(segment)
                f.seek(start)
                block = f.read(min(self.block_size, self.size - start))
            self.local_bytes += len(block)
            self.cache[number] = block
            while len(self.cache) > 16:
                self.cache.popitem(last=False)
            return block
        # Read-ahead makes sequential native-count reads bounded and concurrent.
        if number not in self.pending:
            print('Read remote block',number,'offset_GB',round(start/1e9,3),flush=True)
        for nxt in range(number,min(number+8,(self.size+self.block_size-1)//self.block_size)):
            if nxt not in self.pending:
                self.pending[nxt]=self.pool.submit(self.fetch_remote,nxt)
        result=self.pending[number].result()
        self.fetched_bytes+=len(result)
        self.cache[number]=result
        while len(self.cache)>16:self.cache.popitem(last=False)
        return result


def prepare_expression_audit():
    OUT.mkdir(exist_ok=True, parents=True)
    COUNTS.mkdir(exist_ok=True, parents=True)
    done = OUT / 'expression_audit_complete.json'
    if done.exists():
        print('Read completed expression audit', flush=True)
        return
    frame = pd.read_parquet(ROOT / 'processed/display_cells.parquet')
    source = ROOT / 'crca_xenium.h5ad'
    reader = None if source.exists() else VerifiedDownloadReader()
    with h5py.File(source if source.exists() else reader, 'r') as h:
        genes = h['var/_index'].asstr()[:]
        (OUT / 'genes.json').write_text(json.dumps(genes.tolist()))
        qc_path=OUT/'cell_qc.parquet'
        qc = pd.read_parquet(qc_path).drop(columns='source_row').to_dict('series') if qc_path.exists() else {}
        for key in ['transcript_counts', 'total_counts', 'control_probe_counts',
                    'control_codeword_counts', 'cell_area', 'nucleus_area', 'nucleus_count']:
            if key in qc:continue
            if key in h['obs']:
                print('Read author QC:', key, flush=True)
                qc[key] = _column(h['obs'], key)
        pd.DataFrame(qc).assign(source_row=np.arange(len(frame))).to_parquet(OUT / 'cell_qc.parquet', index=False)
        group = h['layers/counts']
        print('Read native CSR pointer', flush=True)
        ptr = group['indptr'][:]
        print('Read verified native count buffers', flush=True)
        data = group['data'][:]
        indices = group['indices'][:]
        matrix = sparse.csr_matrix((data, indices, ptr), shape=(len(frame), len(genes)))
        if matrix.data.min() < 0 or not np.allclose(matrix.data, np.rint(matrix.data)):
            raise ValueError('Author counts are not nonnegative integer counts')
        sums = np.asarray(matrix.sum(1)).ravel()
        # Keep aligned, small specimen-specific sparse inputs; no normalized/X copies.
        for sid, rows in frame.groupby('specimen_id', observed=True).indices.items():
            rows = np.asarray(rows)
            sparse.save_npz(COUNTS / f'{sid}_counts.npz', matrix[rows])
            print('Native counts cached:', sid, len(rows), flush=True)
        qc['native_panel_counts'] = sums
        qc['detected_genes'] = np.diff(matrix.indptr)
        pd.DataFrame(qc).assign(source_row=np.arange(len(frame))).to_parquet(OUT / 'cell_qc.parquet', index=False)
        out = []
        for sid, rows in frame.groupby('specimen_id', observed=True).indices.items():
            row = dict(specimen_id=sid, cells=len(rows))
            for key, values in qc.items():
                v = np.asarray(values)[rows]
                row[key + '_median'] = float(np.median(v))
                row[key + '_p10'] = float(np.quantile(v, .1))
            row['zero_count_fraction'] = float(np.mean(sums[rows] == 0))
            out.append(row)
        pd.DataFrame(out).to_csv(OUT / 'specimen_expression_qc.csv', index=False)
    done.write_text(json.dumps(dict(source='Author layers/counts', cells=len(frame), genes=len(genes),
                                   remote_bytes=reader.fetched_bytes if reader else 0,
                                   verified_local_bytes=reader.local_bytes if reader else 0,
                                   no_niche_labels_or_outcomes=True), indent=2))


def boundary_screen(specimens=None):
    """Observed directed CCC in spatially distributed, mixed tumor-boundary graphs.

    Exactly 30 physical members, sigma=20um, anchor and pair-distance decay,
    distinct ordered pairs and opportunity+1e-4, as in the locked target kernel.
    No GNN, published niche, stage, genotype, or outcome is used here.
    """
    from phenoniche.ccc.lr_atlas import load_lr_atlas, LRAtlas
    from phenoniche.ccc.pairwise import _side_expression
    frame = pd.read_parquet(ROOT / 'processed/display_cells.parquet')
    genes = tuple(json.loads((OUT / 'genes.json').read_text()))
    atlas = load_lr_atlas(Path(__file__).resolve().parents[1] / 'data/commuspace_human_lr_atlas.tsv')
    links = tuple(x for x in atlas.interactions if set(x.ligand_components+x.receptor_components) <= set(genes))
    atlas = LRAtlas(links, {}, atlas.source_path)
    pairs = [('Cancer cell','T cell'), ('T cell','Cancer cell'),
             ('Cancer cell','Fibroblast'), ('Fibroblast','Cancer cell'),
             ('Macrophage','Fibroblast'), ('Neutrophil','Fibroblast'),
             ('Fibroblast','T cell'), ('Macrophage','T cell')]
    specs = sorted(frame.loc[frame.tissue_region=='margin','specimen_id'].unique())
    if specimens is not None:
        if not set(specimens)<=set(specs):raise ValueError('Expected author margin specimen IDs')
        specs=list(specimens)
    allrows=[]
    for sid in specs:
        target = OUT / f'{sid}_boundary_ccc.npz'
        if target.exists():
            print('Use observed boundary CCC cache:', sid, flush=True)
            continue
        group=frame.loc[frame.specimen_id==sid].reset_index(drop=True)
        x=sparse.load_npz(COUNTS/f'{sid}_counts.npz').toarray().astype(np.float32)
        ty=group.celltype.astype(str).to_numpy()
        xy=group[['x_um','y_um']].to_numpy()
        nei=np.load(NEIGHBORS/f'{sid}_neighborhoods.npz')
        eligible=np.flatnonzero(nei['interface'])
        tile=np.floor((xy-xy.min(0))/100).astype(int)
        # One mixed-boundary center per100um tile; deterministic geometry-only choice.
        chosen={}
        for i in eligible:
            k=tuple(tile[i]); center=xy.min(0)+(tile[i]+.5)*100
            score=float(np.sum((xy[i]-center)**2))
            if k not in chosen or score<chosen[k][0]:chosen[k]=(score,int(i))
        anchors=np.array([v[1] for k,v in sorted(chosen.items())],dtype=int)
        members=np.c_[anchors,nei['neighbors'][anchors,:29]]
        cx=xy[members]; ct=ty[members]; delta=cx-cx[:,:1]
        aw=np.exp(-np.sum(delta*delta,axis=2)/(2*20.**2))
        ds=np.sum((cx[:,:,None]-cx[:,None,:])**2,axis=3)
        w=aw[:,:,None]*aw[:,None,:]*np.exp(-ds/(2*20.**2))
        w[:,np.arange(30),np.arange(30)]=0
        raw_results=[]; norm_results=[]; opportunities=[]; cover=[]
        # This second representation is only a library-depth sensitivity audit.
        total=x.sum(1); norm=x/np.maximum(total[:,None],1)*1e4
        print('Boundary expression screen:',sid,len(anchors),'spatial tiles',flush=True)
        for mode,expr,results in [('native',x,raw_results),('panel_depth',norm,norm_results)]:
            l,r=_side_expression(expr,genes,atlas)
            if mode=='native':
                cover=[((l[ty==src]>0).mean(0)>=.1)&((r[ty==dst]>0).mean(0)>=.1) for src,dst in pairs]
            l=np.sqrt(l[members]);r=np.sqrt(r[members])
            for src,dst in pairs:
                weights=(w*(ct==src)[:,:,None]*(ct==dst)[:,None,:]).astype(np.float32)
                op=weights.sum((1,2))
                result=np.einsum('biq,bij,bjq->bq',l,weights,r,optimize=True)/(op[:,None]+1e-4)
                results.append(result.astype(np.float32))
                if mode=='native':
                    opportunities.append(op)
        np.savez_compressed(target,anchors=anchors,tile=tile[anchors],members=members,
                            native=np.stack(raw_results,axis=1),normalized=np.stack(norm_results,axis=1),
                            opportunity=np.stack(opportunities,axis=1),coverage=np.array(cover),
                            pairs=np.array(pairs),ligand=np.array([v.ligand for v in links]),
                            receptor=np.array([v.receptor for v in links]))
    (OUT/'boundary_screen_protocol.json').write_text(json.dumps(dict(
        measured_complete_LR=len(links),members=30,sigma_um=20,
        evaluated_specimens=specs,
        boundary='20-80% Cancer cells among30 nearest others; neighbor radius <=150um',
        sampling='One geometry-selected boundary center per100um square',
        expression='Native author counts; panel-depth-normalized sensitivity check',
        target='sum(w_anchor_i*w_anchor_j*w_pair_ij*sqrt(L_i*R_j))/(opportunity+1e-4)',
        screening_only=True,model_training=False,cell_type_expression_coverage=.1,
        interpretation='Exploratory between-specimen comparison; not an independent validation'),indent=2))


def compare_margins(a='crc_a_m', b='crc_b_m'):
    """Report effect sizes and spatial-half/depth consistency, without model labels."""
    x=np.load(OUT/f'{a}_boundary_ccc.npz');y=np.load(OUT/f'{b}_boundary_ccc.npz')
    assert np.array_equal(x['pairs'],y['pairs']) and np.array_equal(x['ligand'],y['ligand'])
    rows=[]
    for p,(src,dst) in enumerate(x['pairs']):
        sx=x['opportunity'][:,p]>1.;sy=y['opportunity'][:,p]>1.
        if min(sx.sum(),sy.sum())<20:continue
        for q in np.flatnonzero(x['coverage'][p]&y['coverage'][p]):
            xv=x['native'][sx,p,q];yv=y['native'][sy,p,q]
            u=x['normalized'][sx,p,q];v=y['normalized'][sy,p,q]
            av,bv=xv.mean(),yv.mean();an,bn=u.mean(),v.mean()
            # Absence in one boundary is allowed; requiring both to be active
            # would remove precisely the communication contrasts being sought.
            if max(np.mean(xv>0),np.mean(yv>0))<.2:continue
            if max(np.sum(xv>0),np.sum(yv>0))<30:continue
            if max(av,bv)<.01:continue
            ratio=(bv+.02)/(av+.02);nr=(bn+.02)/(an+.02)
            half_ratios=[]
            for half in [0,1]:
                hx=(x['tile'].sum(1)%2==half)&sx;hy=(y['tile'].sum(1)%2==half)&sy
                if min(hx.sum(),hy.sum())<10:break
                half_ratios.append(float((y['native'][hy,p,q].mean()+.02)/(x['native'][hx,p,q].mean()+.02)))
            stable=len(half_ratios)==2 and all((z>1)==(ratio>1) for z in half_ratios) and ((nr>1)==(ratio>1))
            rows.append(dict(specimen_a=a,specimen_b=b,sender=src,receiver=dst,
                ligand=x['ligand'][q],receptor=x['receptor'][q],mean_a=av,mean_b=bv,
                normalized_a=an,normalized_b=bn,ratio_b_over_a=ratio,
                normalized_ratio_b_over_a=nr,tiles_a=int(sx.sum()),tiles_b=int(sy.sum()),
                positive_fraction_a=float(np.mean(xv>0)),positive_fraction_b=float(np.mean(yv>0)),
                half1_ratio=half_ratios[0] if half_ratios else np.nan,
                half2_ratio=half_ratios[1] if len(half_ratios)>1 else np.nan,
                depth_and_spatial_direction_consistent=stable))
    table=pd.DataFrame(rows)
    table['abs_log2_ratio']=np.abs(np.log2(table.ratio_b_over_a))
    table=table.sort_values(['depth_and_spatial_direction_consistent','abs_log2_ratio'],ascending=False)
    table.to_csv(OUT/f'{a}_vs_{b}_directed_LR.csv',index=False)
    print(table.head(30).to_string(index=False),flush=True)
    return table


def finalize_selection():
    """Present five author specimens and the observed evidence for their selection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from phenoniche.cohorts.crc_neutrophil_atlas import COLORS
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none',
                         'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    ids=['crc_a_c','crc_l_c','crc_a_m','crc_b_m','crc_j_n']
    roles=['Immune-infiltrated core','Immune-poor core','Margin A','Margin B','Normal mucosa']
    frame=pd.read_parquet(ROOT/'processed/display_cells.parquet')
    spatial=pd.read_csv(OUT/'specimen_spatial_screen.csv').set_index('specimen_id')
    qc=pd.read_csv(OUT/'specimen_expression_qc.csv').set_index('specimen_id')
    selected=[]
    for sid,role in zip(ids,roles):
        z=spatial.loc[sid];q=qc.loc[sid]
        selected.append(dict(specimen_id=sid,role=role,patient_id=z.patient,region=z.region,
                             cells=int(z.cells),median_native_counts=float(q.native_panel_counts_median),
                             median_detected_genes=float(q.detected_genes_median),
                             T_fraction=float(z['fraction_T cell']),
                             tumor_neighbor_T=float(z.tumor_neighbor_T)))
    pd.DataFrame(selected).to_csv(OUT/'selected_specimens.csv',index=False)
    # Large, geometry-defined boundary windows; LR values do not choose the windows.
    rois={}
    for sid in ['crc_a_m','crc_b_m']:
        g=frame.loc[frame.specimen_id==sid];xy=g[['x_um','y_um']].to_numpy();ty=g.celltype.astype(str).to_numpy()
        z=np.load(OUT/f'{sid}_boundary_ccc.npz');centers=xy[z['anchors']]
        best=None
        for center in centers:
            mask=(np.abs(xy-center)<=450).all(1);ct=ty[mask];cx=xy[mask]
            c=ct=='Cancer cell';t=ct=='T cell';fib=ct=='Fibroblast';mac=ct=='Macrophage'
            if len(ct)<1800 or not .3<=c.mean()<=.7 or min(t.sum(),fib.sum(),mac.sum())<50:continue
            separation=np.linalg.norm(cx[c].mean(0)-cx[~c].mean(0))/900
            score=separation*min(t.sum()/150,1)*min(fib.sum()/150,1)*min(mac.sum()/150,1)
            if best is None or score>best[0]:best=(score,center,mask)
        if best is None:raise ValueError('No large coherent, mixed boundary window')
        _,center,mask=best
        rois[sid]=dict(x_min=float(center[0]-450),x_max=float(center[0]+450),
                       y_min=float(center[1]-450),y_max=float(center[1]+450),cells=int(mask.sum()),
                       cancer=int(np.sum(ty[mask]=='Cancer cell')),T=int(np.sum(ty[mask]=='T cell')),
                       fibroblast=int(np.sum(ty[mask]=='Fibroblast')),macrophage=int(np.sum(ty[mask]=='Macrophage')))
    (OUT/'boundary_rois.json').write_text(json.dumps(rois,indent=2))

    def draw(ax,sid,roi=None,core_T=False):
        g=frame.loc[frame.specimen_id==sid];xy=g[['x_um','y_um']].to_numpy();ty=g.celltype.astype(str).to_numpy()
        origin=xy.min(0)
        if roi is not None:
            mask=(xy[:,0]>=roi['x_min'])&(xy[:,0]<=roi['x_max'])&(xy[:,1]>=roi['y_min'])&(xy[:,1]<=roi['y_max'])
            xy=xy[mask];ty=ty[mask];origin=np.array([roi['x_min'],roi['y_min']])
        xy=xy-origin
        order=np.random.default_rng(40700).permutation(len(xy))
        if core_T:
            col=np.where(ty=='Cancer cell','#F1B8BD',np.where(ty=='T cell','#008C91','#D6D6D6'))
            order=np.r_[np.where(ty!='T cell')[0],np.where(ty=='T cell')[0]]
        else:col=np.array([COLORS[t] for t in ty])
        ax.scatter(xy[order,0],xy[order,1],c=col[order],s=7 if roi else 1.8,lw=0,rasterized=True)
        ax.set_aspect('equal');ax.invert_yaxis();ax.axis('off')
        length=200 if roi else 500;ymax=900 if roi else xy[:,1].max();offset=22 if roi else 65
        ax.plot([25,25+length],[ymax+offset]*2,c='#333',lw=2)
        ax.text(25+length/2,ymax+offset*1.2,f'{length} µm',ha='center',va='top',fontsize=9)
        if roi:
            ax.set_xlim(0,900);ax.set_ylim(960,0)
        elif sid in rois:
            r=rois[sid];ax.add_patch(Rectangle((r['x_min']-origin[0],r['y_min']-origin[1]),900,900,fill=False,ec='#333',lw=1.5))

    fig,axs=plt.subplots(2,3,figsize=(15,10.7))
    positions=[(0,0),(0,1),(1,0),(1,1),(0,2)]
    for entry,pos in zip(selected,positions):
        ax=axs[pos];draw(ax,entry['specimen_id'])
        ax.set_title(f"{entry['role']} · {entry['specimen_id']}\n{entry['cells']:,} cells",fontsize=12)
    axs[1,2].axis('off')
    axs[1,2].legend(handles=[Line2D([],[],marker='o',lw=0,color=v,label=k,ms=6) for k,v in COLORS.items() if k!='Unassigned'],loc='center',frameon=False,fontsize=11)
    fig.suptitle('Selected CRC Xenium specimens · author cell types',fontsize=16,y=.99)
    fig.subplots_adjust(hspace=.2,wspace=.05,top=.92,bottom=.04)
    for ext in ['png','pdf']:fig.savefig(OUT/f'selected_specimens.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5.8))
    for ax,sid in zip(axs,['crc_a_m','crc_b_m']):
        draw(ax,sid,rois[sid]);ax.set_title(f'{sid} · tumor boundary\n{rois[sid]["cells"]:,} cells',fontsize=12)
    fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=v,label=k,ms=5) for k,v in COLORS.items() if k!='Unassigned'],loc='lower center',ncol=5,frameon=False,fontsize=8.5)
    fig.subplots_adjust(bottom=.16,wspace=.08)
    fig.savefig(OUT/'selected_margin_windows.png',dpi=220,bbox_inches='tight');plt.close(fig)

    # Global boundary effects, explicitly separate from the geometry-selected ROIs.
    table=compare_margins()
    keys=[('Fibroblast','T cell','CXCL12','CXCR4'),('Cancer cell','T cell','CXCL16','CXCR6'),
          ('Fibroblast','Cancer cell','FN1','CD44')]
    effect=[]
    for src,dst,l,r in keys:
        row=table.loc[(table.sender==src)&(table.receiver==dst)&(table.ligand==l)&(table.receptor==r)].iloc[0]
        if not row.depth_and_spatial_direction_consistent:raise ValueError('Depth/spatial stability check failed')
        effect.append(dict(sender=src,receiver=dst,ligand=l,receptor=r,
                           native_ratio=float(row.mean_b/row.mean_a),
                           normalized_ratio=float(row.normalized_b/row.normalized_a),
                           native_A=float(row.mean_a),native_B=float(row.mean_b),
                           positive_fraction_A=float(row.positive_fraction_a),positive_fraction_B=float(row.positive_fraction_b),
                           tiles_A=int(row.tiles_a),tiles_B=int(row.tiles_b)))
    pd.DataFrame(effect).to_csv(OUT/'selected_margin_evidence.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.3,4.2));yy=np.arange(len(effect))
    for shift,key,color,label in [(-.14,'native_ratio','#6B91AE','Native counts'),(.14,'normalized_ratio','#C4995B','Panel-depth sensitivity')]:
        vv=np.log2([v[key] for v in effect]);ax.barh(yy+shift,vv,height=.26,color=color,label=label)
        for y,v,ratio in zip(yy+shift,vv,[v[key] for v in effect]):ax.text(v+.05,y,f'{ratio:.1f}×',va='center',fontsize=9)
    ax.set_yticks(yy,[f"{e['ligand']}–{e['receptor']}\n{e['sender']} → {e['receiver']}" for e in effect]);ax.invert_yaxis()
    ax.set_xlabel('log₂(B / A), mean observed directed CCC at tumor boundaries')
    ax.set_xlim(0,max(np.log2([v['native_ratio'] for v in effect]))+.8);ax.legend(frameon=False,loc='lower right',fontsize=9)
    ax.set_title('Margin B versus A · distributed boundary locations',fontsize=12)
    fig.tight_layout();fig.savefig(OUT/'selected_margin_communication.png',dpi=220,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,5.5))
    for ax,sid,title in zip(axs,['crc_a_c','crc_l_c'],['Immune-infiltrated core','Immune-poor core']):
        draw(ax,sid,core_T=True);ax.set_title(f'{title} · {sid}')
    fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=v,label=k,ms=5) for k,v in {'Cancer cell':'#F1B8BD','T cell':'#008C91','Other cells':'#D6D6D6'}.items()],loc='lower center',ncol=3,frameon=False)
    fig.subplots_adjust(bottom=.1,wspace=.08);fig.savefig(OUT/'selected_core_T_cells.png',dpi=220,bbox_inches='tight');plt.close(fig)
    receipt=dict(status='selected_for_exploratory_joint_analysis',specimens=selected,total_cells=sum(v['cells'] for v in selected),
                 selected_margins=['crc_a_m','crc_b_m'],boundary_ROIs=rois,communication_evidence=effect,
                 core_interpretation='Observed immune infiltration contrast; not author gold-standard cold/hot diagnoses',
                 core_selection='A has more intratumoral T cells and higher depth-normalized T-cell GZMB/PRF1 than L; M retained as an audited alternative',
                 selection_uses='Author cell types, expression QC, physical architecture and observed directed LR; no known niche labels, clinical stage, genotype, response or outcome',
                 technical_limits='Between-patient comparison, not biological replication or an independent validation; depth check does not remove all technical effects',
                 counts_cache='Author counts only; specimen rows follow display_cells source_row order',
                 training_started=False,model_changed=False,existing_notebook_changed=False)
    (OUT/'selection.json').write_text(json.dumps(receipt,indent=2))
    config=Path(__file__).resolve().parents[1]/'config/crc_neutrophil_selection.json'
    config.write_text(json.dumps(dict(specimens=ids,data_root=str(ROOT),counts_root=str(COUNTS),selection_receipt=str(OUT/'selection.json'),
                                    physical_group='specimen_id',coordinate_unit='um',native_layer='counts',
                                    author_cell_type='celltype',training_started=False),indent=2)+'\n')
    print('Selection ready:',config,receipt['total_cells'],flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--boundaries',action='store_true')
    parser.add_argument('--compare',nargs=2)
    parser.add_argument('--specimens',nargs='+')
    parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--spatial',action='store_true')
    args=parser.parse_args()
    if args.spatial:spatial_screen()
    elif args.finalize:finalize_selection()
    elif args.compare:compare_margins(*args.compare)
    elif args.boundaries:boundary_screen(args.specimens)
    else:prepare_expression_audit()
