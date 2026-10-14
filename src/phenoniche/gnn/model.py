"""One directed local graph -> one intensity-preserving CCC program vector."""
import torch
from torch import nn
from torch.nn import functional as F


def physical_weights(coordinates, anchor_weights, sigma, cell_ids=None):
    """Same w_anchor(s) w_anchor(r) exp(-d_sr²/2σ²) as ccc.pairwise."""
    distance2 = (coordinates[:, :, None] - coordinates[:, None, :]).square().sum(-1)
    weight = anchor_weights[:, :, None] * anchor_weights[:, None, :] * torch.exp(-distance2 / (2 * sigma**2))
    k = coordinates.shape[1]
    distinct = ~torch.eye(k, dtype=torch.bool, device=coordinates.device)[None]
    if cell_ids is not None:
        distinct = distinct & (cell_ids[:, :, None] != cell_ids[:, None, :])
    return weight * distinct


class DirectedResidualLayer(nn.Module):
    def __init__(self,hidden,dropout=0.):
        super().__init__();self.hidden=hidden
        self.sender=nn.Linear(hidden,hidden,bias=False);self.receiver=nn.Linear(hidden,hidden,bias=False)
        self.sender_value=nn.Linear(hidden,hidden,bias=False);self.receiver_value=nn.Linear(hidden,hidden,bias=False)
        self.update=nn.Sequential(nn.Linear(3*hidden,hidden),nn.GELU(),nn.Dropout(dropout))
        self.norm=nn.LayerNorm(hidden)

    def forward(self,h,weight):
        valid=weight>0
        scores=self.sender(h)@self.receiver(h).transpose(1,2)/self.hidden**.5+weight.clamp_min(1e-30).log()
        scores=scores.masked_fill(~valid,-1e9)
        incoming=scores.softmax(1)*valid;outgoing=scores.softmax(2)*valid
        received=incoming.transpose(1,2)@self.sender_value(h);sent=outgoing@self.receiver_value(h)
        nodes=self.norm(h+self.update(torch.cat((h,sent,received),dim=-1)))
        return nodes,sent,received


class MicroenvironmentGNN(nn.Module):
    def __init__(self, genes, types, features, programs=32, hidden=64, sigma=20., feature_mask=None,message_layers=1,encoder_depth=1,dropout=0.,dictionary_parameterization='softplus',dictionary_scale=None,program_pair_pooling=False,activity_parameterization='separated',lr_components=None):
        super().__init__()
        self.genes, self.types, self.features = genes, types, features
        self.programs, self.hidden, self.sigma = programs, hidden, sigma
        self.dictionary_parameterization=dictionary_parameterization
        if dictionary_parameterization not in ('softplus','projected'):raise ValueError('Invalid dictionary parameterization')
        self.enhanced=message_layers>1 or encoder_depth>1 or dropout>0
        widths=[genes]+[2*hidden]*(encoder_depth-1)+[hidden];layers=[]
        for first,last in zip(widths[:-1],widths[1:]):
            layers.extend([nn.Linear(first,last),nn.GELU(),nn.LayerNorm(last)])
            # Keep parameter names stable when disabling stochastic masking for
            # a controlled warm-start (p=0 has no parameters and is identity).
            layers.append(nn.Dropout(dropout))
        self.cell_encoder=nn.Sequential(*layers)
        if self.enhanced:
            self.message_blocks=nn.ModuleList([DirectedResidualLayer(hidden,dropout) for _ in range(message_layers)])
        else:
            # Identical state layout and computation for archived expression-only checkpoints.
            self.sender = nn.Linear(hidden, hidden, bias=False)
            self.receiver = nn.Linear(hidden, hidden, bias=False)
            self.sender_value = nn.Linear(hidden, hidden, bias=False)
            self.receiver_value = nn.Linear(hidden, hidden, bias=False)
            self.node_update = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU())
        self.pool = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU())
        self.composition_head = nn.Linear(hidden, programs)
        self.activity_head = nn.Linear(hidden, 1)
        self.raw_composition = nn.Parameter(torch.zeros(programs, types))
        self.raw_dictionary = nn.Parameter(torch.randn(programs, features) * .15 - 2)
        mask = torch.ones(features, dtype=torch.bool) if feature_mask is None else torch.as_tensor(feature_mask, dtype=torch.bool)
        if mask.shape != (features,) or not mask.any(): raise ValueError("Empty/invalid CCC feature mask")
        self.register_buffer("feature_mask", mask)
        if dictionary_scale is not None:
            scale=torch.as_tensor(dictionary_scale,dtype=torch.float32)
            if scale.shape!=(features,) or not torch.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid dictionary feature scale')
            self.register_buffer('dictionary_scale',scale)
        if dictionary_parameterization=='projected':
            with torch.no_grad():self.raw_dictionary.copy_(F.softplus(self.raw_dictionary)*mask)
            self.project_dictionary()
        self.program_pair_pooling=program_pair_pooling
        self.lr_pair_pooling=lr_components is not None
        if self.lr_pair_pooling:
            feature_index,sender_components,receiver_components=lr_components
            self.register_buffer('lr_feature_index',torch.as_tensor(feature_index,dtype=torch.long))
            self.n_lr=len(sender_components)
            for prefix,components in [('lr_sender',sender_components),('lr_receiver',receiver_components)]:
                width=max(map(len,components));positions=torch.zeros(len(components),width,dtype=torch.long);valid=torch.zeros_like(positions,dtype=torch.bool)
                for row,indices in enumerate(components):positions[row,:len(indices)]=torch.as_tensor(indices);valid[row,:len(indices)]=True
                self.register_buffer(prefix+'_index',positions);self.register_buffer(prefix+'_valid',valid)
        self.activity_parameterization=activity_parameterization
        if activity_parameterization=='independent':self.program_head=nn.Linear(hidden,programs)
        if program_pair_pooling:
            self.pair_sender_role=nn.Linear(hidden,programs)
            self.pair_receiver_role=nn.Linear(hidden,programs)
            self.pair_sender_signal=nn.Linear(hidden,programs)
            self.pair_receiver_signal=nn.Linear(hidden,programs)
            self.pair_mixture=nn.Linear(programs,programs,bias=False)
            self.pair_activity=nn.Linear(programs,1,bias=False)
            nn.init.zeros_(self.pair_mixture.weight);nn.init.zeros_(self.pair_activity.weight)

    @property
    def dictionary(self):
        h = (F.softplus(self.raw_dictionary) if self.dictionary_parameterization=='softplus' else self.raw_dictionary.clamp_min(0)) * self.feature_mask
        if hasattr(self,'dictionary_scale'):h=h*self.dictionary_scale
        return h / h.sum(1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def project_dictionary(self):
        """Keep direct dictionary parameters nonnegative, masked and at unit row mass."""
        if self.dictionary_parameterization!='projected':return
        self.raw_dictionary.clamp_(min=0).mul_(self.feature_mask)
        mass=self.raw_dictionary.sum(1,keepdim=True)
        empty=mass.squeeze(1)<=1e-12
        if empty.any():self.raw_dictionary[empty]=self.feature_mask.float()/self.feature_mask.sum()
        self.raw_dictionary.div_(self.raw_dictionary.sum(1,keepdim=True))

    @property
    def composition_dictionary(self):
        return self.raw_composition.softmax(1)

    def forward(self, expression, local_index, coordinates, anchor_weights, cell_ids=None, pair_override=None, reconstruct=True,raw_lr_expression=None):
        # Encode each unique physical cell once in the batch, reuse overlapping members.
        unique_h = self.cell_encoder(expression)
        h = unique_h[local_index]
        weight = physical_weights(coordinates, anchor_weights, self.sigma, cell_ids)
        if pair_override is not None: weight = pair_override
        if self.enhanced:
            nodes=h
            for block in self.message_blocks:nodes,sent,received=block(nodes,weight)
        else:
            valid = weight > 0
            scores = self.sender(h) @ self.receiver(h).transpose(1, 2) / self.hidden**.5
            scores = scores + weight.clamp_min(1e-30).log()
            bounded = scores.masked_fill(~valid, -1e9)
            incoming = torch.softmax(bounded, dim=1) * valid
            outgoing = torch.softmax(bounded, dim=2) * valid
            received = incoming.transpose(1, 2) @ self.sender_value(h)
            sent = outgoing @ self.receiver_value(h)
            nodes = self.node_update(torch.cat((h, sent, received), dim=-1))
        a = anchor_weights / anchor_weights.sum(1, keepdim=True).clamp_min(1e-8)
        sender_mass = weight.sum(2); receiver_mass = weight.sum(1)
        sender_mass = sender_mass / sender_mass.sum(1, keepdim=True).clamp_min(1e-8)
        receiver_mass = receiver_mass / receiver_mass.sum(1, keepdim=True).clamp_min(1e-8)
        pooled = self.pool(torch.cat(((nodes * a[..., None]).sum(1),
                                     (sent * sender_mass[..., None]).sum(1),
                                     (received * receiver_mass[..., None]).sum(1)), dim=1))
        logits=self.composition_head(pooled);intensity=self.activity_head(pooled)
        if self.program_pair_pooling:
            # Learned expression-only sender/receiver roles. Normalize each
            # role's physical pair opportunity, rather than averaging rare roles
            # into the global cell mixture. This is an encoder summary; the CCC
            # supervision and its observed cell-type definition are unchanged.
            sender_role=self.pair_sender_role(h).sigmoid();receiver_role=self.pair_receiver_role(h).sigmoid()
            if self.lr_pair_pooling:
                if raw_lr_expression is None:raise ValueError('LR message pooling requires observed raw gene expression')
                def sides(prefix):
                    valid=getattr(self,prefix+'_valid');values=raw_lr_expression[:,getattr(self,prefix+'_index')]
                    logs=(values.clamp_min(1e-30).log()*valid).sum(-1)/valid.sum(-1)
                    positive=((values>0)|~valid).all(-1)
                    return (logs.mul(.5).exp()*positive)[local_index]
                ligand=sides('lr_sender');receptor=sides('lr_receiver')
                h_dictionary=self.dictionary
                lr_loading=h_dictionary.new_zeros(self.programs,self.n_lr).scatter_add(1,self.lr_feature_index[None].expand(self.programs,-1),h_dictionary)
                numerator=sender_role.new_zeros(len(h),self.programs)
                for start in range(0,self.n_lr,32):
                    stop=min(start+32,self.n_lr)
                    receiving=receiver_role[...,None]*receptor[:,:,None,start:stop]
                    message=(weight@receiving.flatten(2)).reshape(len(h),h.shape[1],self.programs,stop-start)
                    per_lr=(sender_role[...,None]*ligand[:,:,None,start:stop]*message).sum(1)
                    numerator=numerator+(per_lr*lr_loading[None,:,start:stop]).sum(2)
            else:
                sender=sender_role*F.softplus(self.pair_sender_signal(h))
                receiver=receiver_role*F.softplus(self.pair_receiver_signal(h))
                numerator=(sender*(weight@receiver)).sum(1)
            opportunity=(sender_role*(weight@receiver_role)).sum(1)
            signal=torch.log1p(numerator/opportunity.clamp_min(1e-12))
            logits=logits+self.pair_mixture(signal);intensity=intensity+self.pair_activity(signal)
        mixture = logits.softmax(1)
        activity = F.softplus(intensity) + 1e-8
        z = activity * mixture
        if self.activity_parameterization=='independent':
            z=F.softplus(self.program_head(pooled))+1e-8
            activity=z.sum(1,keepdim=True)
            mixture=z/activity
        result = {"Z": z, "P": mixture, "activity": activity.squeeze(1)}
        if reconstruct:
            result['B'] = self.composition_dictionary
            result['composition'] = mixture @ result['B']
            result['H'] = self.dictionary
            result['reconstruction'] = z @ result['H']
        return result


def model_from_config(data,config):
    lr_components=None
    if config.lr_pair_pooling:
        import numpy as np
        from ..ccc.lr_atlas import canonicalize_complex
        catalog=data.features[['lr_id','ligand','receptor']].drop_duplicates().reset_index(drop=True)
        if catalog.lr_id.duplicated().any():raise ValueError('Ambiguous LR catalogue')
        genes=sorted({g for side in ('ligand','receptor') for value in catalog[side] for g in canonicalize_complex(value)})
        positions={g:i for i,g in enumerate(genes)};panel={g:i for i,g in enumerate(data.genes)}
        indices=np.asarray([panel[g] for g in genes]);selected=set(data.encoder_indices)
        if not set(indices)<=selected:raise ValueError('LR pooling genes must already be included in encoder expression inputs')
        data.enable_lr_inputs(indices)
        lr_ids={value:i for i,value in enumerate(catalog.lr_id)}
        lr_components=([lr_ids[value] for value in data.features.lr_id],
            [[positions[g] for g in canonicalize_complex(value)] for value in catalog.ligand],
            [[positions[g] for g in canonicalize_complex(value)] for value in catalog.receptor])
    if config.cell_state_components and (data.cell_state_scores is None or data.cell_state_scores.shape[1]!=config.cell_state_components):raise ValueError('Missing expression-only cell-state PCA features')
    return MicroenvironmentGNN(len(data.encoder_indices)+1+config.cell_state_components,len(data.categories),len(data.features),
        config.programs,config.hidden,config.sigma_um,data.feature_mask,
        message_layers=config.message_layers,encoder_depth=config.encoder_depth,dropout=config.dropout,
        dictionary_parameterization=config.dictionary_parameterization,
        dictionary_scale=data.feature_scale if config.dictionary_coordinates=='feature_scaled' else None,
        program_pair_pooling=config.program_pair_pooling,activity_parameterization=config.activity_parameterization,lr_components=lr_components)
