"""Observed-target geometry shared by independent GNN and AE backends."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F


def distances(x, y=None):
    y = x if y is None else y
    return (x.square().sum(1)[:, None]+y.square().sum(1)[None, :]-2*x@y.T).clamp_min(0)


class JointNeighborhood:
    def __init__(self, data, cfg, folder):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.mask = torch.as_tensor(data.feature_mask, device=self.device)
        self.feature_scale = torch.as_tensor(data.feature_scale, device=self.device)[self.mask]
        self.rows = np.asarray(data.rows[0], dtype=np.int64)
        self.data = data
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder/'joint_neighborhood.pt'
        spec = dict(rows_sha256=hashlib.sha256(self.rows.tobytes()).hexdigest(),
            mask_sha256=hashlib.sha256(np.asarray(data.feature_mask).tobytes()).hexdigest(),
            scales_sha256=hashlib.sha256(np.asarray(data.feature_scale).tobytes()).hexdigest(),
            data_audit=data.audit, seed=cfg.seed, reference_cells=cfg.geometry_reference_cells,
            geometry_objective='multiscale_relative_distance_huber',
            metric='relative feature-scaled observed CCC + .25 standardized observed total; sqrt composition; each task RMS',
            calibration_rows='all training rows only', labels_or_coordinates_used=False)
        if self.path.exists():
            saved = torch.load(self.path, map_location='cpu', weights_only=False)
            if saved['spec'] != spec: raise ValueError('Incompatible neighborhood calibration; existing cache preserved')
            self.state = {k: v.to(self.device) for k, v in saved['state'].items()}
            self.groups = saved['groups'].numpy()
            print('Read compatible observed-target neighborhood calibration', flush=True)
        else:
            self._prepare()
            torch.save(dict(spec=spec, state={k: v.cpu() for k, v in self.state.items()},
                groups=torch.as_tensor(self.groups)), self.path.with_suffix('.tmp.pt'))
            self.path.with_suffix('.tmp.pt').replace(self.path)
        (folder/'joint_neighborhood.json').write_text(json.dumps(dict(spec=spec,
            training_rows=len(self.rows), reference_groups=int(len(np.unique(self.groups))),
            group_size=cfg.geometry_group_size, pairs_per_scale=cfg.geometry_pairs_per_scale,
            relative_error_floor=cfg.geometry_distance_floor,lambda_geometry=cfg.lambda_geometry,
            objective='equal mean of near/middle/far relative squared-distance Huber errors',
            shared_normalization='one fixed scalar calibrated on initial training h; no per-cell unit normalization',
            self_pairs_excluded=True, no_posthoc_mapper=True), indent=2)+'\n')

    def calibrate_shared(self, model):
        """One frozen training-only length unit, reused throughout optimization.

        The initial shared RMS radius is matched to the joint target RMS radius
        sqrt(2). This is a global affine transform, not per-cell normalization.
        It cannot absorb a change in any particular pair's distance during fit.
        """
        if 'shared_scale' in self.state:return
        rng=np.random.default_rng(self.cfg.seed+19)
        rows=rng.choice(self.rows,min(self.cfg.geometry_calibration_cells,len(self.rows)),replace=False)
        before=model.training;model.eval();values=[]
        with torch.no_grad():
            for start in range(0,len(rows),self.cfg.batch_size):
                batch=self.data.batch(rows[start:start+self.cfg.batch_size],self.cfg.device,include_target=False)
                values.append(self.data.forward(model,batch,reconstruct=False)['shared'].detach())
        self.set_shared_calibration(torch.cat(values))
        model.train(before)
        saved=torch.load(self.path,map_location='cpu',weights_only=False)
        saved['state']={key:value.cpu() for key,value in self.state.items()}
        saved['shared_calibration_rows']=rows
        temporary=self.path.with_suffix('.tmp.pt');torch.save(saved,temporary);temporary.replace(self.path)
        print(f'Fix one shared-h distance scale using {len(rows):,} training cells; no row normalization',flush=True)

    def set_shared_calibration(self, shared):
        """Fit once on a supplied training sample; also used by geometry tests."""
        with torch.no_grad():
            center=shared.mean(0)
            scale=((shared-center).square().sum(1).mean()/2).clamp_min(1e-8).sqrt()
            self.state.update(shared_center=center.detach(),shared_scale=scale.detach())

    def _read(self, rows):
        target = torch.as_tensor(self.data.targets[rows], device=self.device)
        composition = torch.as_tensor(np.asarray(self.data.composition_target[rows]).copy(), device=self.device)
        return target, composition

    def _raw(self, target, composition):
        native = target[:, self.mask]
        activity = native.sum(1)
        profile = native/self.feature_scale/activity.clamp_min(1e-12)[:, None]
        mix = composition.clamp_min(0)
        mix = (mix/mix.sum(1, keepdim=True).clamp_min(1e-12)).sqrt()
        return profile, activity.log1p(), mix

    @torch.no_grad()
    def _prepare(self):
        print('Calibrate joint geometry on observed training CCC/composition', flush=True)
        # Streaming moments: bounded chunks, no new full-width matrix/cache.
        f = int(self.mask.sum())
        sp = torch.zeros(f, dtype=torch.float64, device=self.device)
        sm = torch.zeros(len(self.data.categories), dtype=torch.float64, device=self.device)
        pp = torch.zeros((), dtype=torch.float64, device=self.device)
        mm = pp.clone(); sl = pp.clone(); ll = pp.clone(); n = 0
        for start in range(0, len(self.rows), 2048):
            p, log, m = self._raw(*self._read(self.rows[start:start+2048]))
            sp += p.double().sum(0); pp += p.double().square().sum()
            sm += m.double().sum(0); mm += m.double().square().sum()
            sl += log.double().sum(); ll += log.double().square().sum(); n += len(p)
        pc, mc, lc = sp/n, sm/n, sl/n
        ps = (pp/n-pc.square().sum()).clamp_min(1e-24).sqrt()
        ms = (mm/n-mc.square().sum()).clamp_min(1e-24).sqrt()
        lv = (ll/n-lc.square()).clamp_min(0)
        ls = lv.sqrt().clamp_min(.1)
        cs = (1+.25**2*lv/ls.square()).sqrt()
        self.state = dict(profile_center=pc.float(), profile_scale=ps.float(),
            activity_center=lc.float(), activity_scale=ls.float(), communication_scale=cs.float(),
            composition_center=mc.float(), composition_scale=ms.float())
        rng = np.random.default_rng(self.cfg.seed)
        ref_ids = rng.choice(self.rows, min(self.cfg.geometry_reference_cells, len(self.rows)), replace=False)
        rc, rm = self.transform(*self._read(ref_ids))
        self.groups = np.empty(len(self.rows), np.int64)
        print(f'Arrange mixed near/far batches with {len(ref_ids)} training-only reference cells (not niche classes)', flush=True)
        for start in range(0, len(self.rows), 1024):
            c, m = self.transform(*self._read(self.rows[start:start+1024]))
            self.groups[start:start+len(c)] = (distances(c, rc)+distances(m, rm)).argmin(1).cpu().numpy()

    def transform(self, target, composition):
        p, log, m = self._raw(target, composition)
        s = self.state
        c = torch.cat(((p-s['profile_center'])/s['profile_scale'],
            (.25*(log-s['activity_center'])/s['activity_scale'])[:, None]), 1)/s['communication_scale']
        return c, (m-s['composition_center'])/s['composition_scale']

    def batches(self, epoch=0, evaluation=False):
        # Every training row exactly once. Small similar-target groups are
        # shuffled together, so a batch also contains unrelated groups.
        rng = np.random.default_rng(self.cfg.seed if evaluation else self.cfg.seed+epoch)
        chunks = []
        for group in np.unique(self.groups):
            rows = self.rows[self.groups == group].copy()
            rng.shuffle(rows)
            chunks.extend(rows[i:i+self.cfg.geometry_group_size] for i in range(0, len(rows), self.cfg.geometry_group_size))
        rng.shuffle(chunks)
        order = np.concatenate(chunks)
        return [order[i:i+self.cfg.batch_size] for i in range(0, len(order), self.cfg.batch_size)]

    def loss(self, shared, target, composition):
        n = len(shared)
        if n < 2:
            zero=shared.sum()*0
            self.last_components={key:zero for key in ('near','middle','far')}
            return zero
        if 'shared_scale' not in self.state:raise RuntimeError('Calibrate the global shared-h scale before training')
        eye = torch.eye(n, dtype=torch.bool, device=shared.device)
        with torch.no_grad():
            c, m = self.transform(target, composition)
            d = distances(c)+distances(m)
            # Three disjoint rank bands per anchor; selection is target-only.
            # Every band has equal influence, regardless of its distances.
            order=d.masked_fill(eye,float('inf')).argsort(1)[:,:n-1]
            k=min(self.cfg.geometry_pairs_per_scale,max((n-1)//3,1))
            middle=max(((n-1)-k)//2,0)
            indices=dict(near=order[:,:k],middle=order[:,middle:middle+k],far=order[:,-k:])
        # The fixed global scalar is shared with final KMeans/UMAP. Mean
        # subtraction is immaterial to distances, but improves numerics.
        embedding=(shared-self.state['shared_center'])/self.state['shared_scale']
        prediction=distances(embedding)
        row=torch.arange(n,device=shared.device)[:,None]
        components={}
        # Each normalized task has RMS radius1, so typical joint squared pair
        # distance is4. The floor prevents nearly identical pairs dominating.
        floor=4*self.cfg.geometry_distance_floor
        for name,cols in indices.items():
            reference=d[row,cols]
            relative=(prediction[row,cols]-reference)/(reference+floor)
            components[name]=F.smooth_l1_loss(relative,torch.zeros_like(relative),beta=1.,reduction='mean')
        self.last_components={key:value.detach() for key,value in components.items()}
        return torch.stack(list(components.values())).mean()
