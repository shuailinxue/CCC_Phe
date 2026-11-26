"""Checkpointed train/infer workflows for content-addressed V1 networks."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile

import numpy as np
import pandas as pd
import torch

from . import evaluate, fit, model
from .storage import Network, digest, key, lock, save_npz, versions


def runtime(device):
    dev = torch.device(device)
    return dict(versions=versions(["torch","numpy","scipy","scikit-learn","pandas"]),
                device=str(dev),gpu=torch.cuda.get_device_name(dev) if dev.type=="cuda" else None,
                cuda=torch.version.cuda,
                code={Path(p).name:digest(p) for p in [__file__,fit.__file__,model.__file__,evaluate.__file__]})


def checkpoint(run,device):
    record = torch.load(Path(run)/"model.pt",map_location=device,weights_only=True)
    cfg = model.Settings(**record["settings"])
    net = model.PhenoMap(cfg,record["nt"],record["nl"]).to(device)
    net.load_state_dict(record["state"])
    return net.requires_grad_(False),record


def write_cells(path,data,output):
    save_npz(path/"cells.npz",ids=data.ids,xyz=data.xyz,specimens=data.samples,mass=data.mass,
             types=data.names,train=data.train,test=data.test,**output)


def execute(network,root,cfg,device):
    network = Path(network).resolve()
    upstream = json.loads((network/"manifest.json").read_text())
    meta = dict(network_key=key(upstream),settings=asdict(cfg),**runtime(device))
    path = Path(root)/("fit-"+key(meta))
    path.mkdir(parents=True,exist_ok=True)
    with lock(path):
        if (path/"complete").exists():
            if json.loads((path/"manifest.json").read_text()) != meta:
                raise ValueError("Fit provenance mismatch")
            return path
        (path/"manifest.json").write_text(json.dumps(meta,indent=2))
        (path/"execution.json").write_text(json.dumps(dict(host=socket.gethostname(),
            visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"), network=str(network)),indent=2))
        print(f"Loading {network.name} for {cfg.mode} seed {cfg.seed}",flush=True)
        data = Network(network)
        if (path/"model.pt").is_file():
            net,record = checkpoint(path,device)
            tau = record["tau"]
        else:
            net,tau,history,visits = fit.train(data,cfg,device)
            pd.DataFrame(history).to_csv(path/"loss.tsv",sep="\t",index=False)
            save_npz(path/"sampling.npz",ids=data.ids,visits=visits)
            record = dict(settings=asdict(cfg),nt=data.nt,nl=data.nl,state=net.state_dict(),tau=tau)
            torch.save(record,path/"checkpoint.partial")
            os.replace(path/"checkpoint.partial",path/"model.pt")
        print(f"Full inference {path.name}",flush=True)
        if (path/"cells.npz").is_file():
            with np.load(path/"cells.npz") as z:
                if not np.array_equal(z["ids"],data.ids):
                    raise ValueError("Saved inference IDs disagree")
                out = {k:z[k] for k in ("u","activity","soft","niche","composition","latent")}
        else:
            out = fit.infer(net,data,device)
            write_cells(path,data,out)
        metrics = evaluate.export(net,data,out,path)
        metrics.update(tau=tau,heldout=evaluate.heldout(net,data,device))
        (path/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False))
        shutil.copyfile(network/"schema.npz",path/"training_schema.npz")
        (path/"complete").touch()
    return path


def predict(run,network,root,device):
    run,network = Path(run),Path(network)
    net,record = checkpoint(run,device)
    meta = dict(model=digest(run/"model.pt"),network=digest(network/"manifest.json"),**runtime(device))
    path = Path(root)/("predict-"+key(meta))
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_dir():
        if json.loads((path/"manifest.json").read_text()) != meta:
            raise ValueError("Prediction provenance mismatch")
        return path
    data = Network(network)
    with np.load(run/"training_schema.npz") as ref:
        for k in ("genes","scales","names","lr","radii","group","window","signaling"):
            if not np.array_equal(ref[k],getattr(data,k)):
                raise ValueError(f"Training/prediction schema mismatch: {k}")
    out = fit.infer(net,data,device)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        staging = Path(tmp)/"prediction"
        staging.mkdir()
        write_cells(staging,data,out)
        (staging/"manifest.json").write_text(json.dumps(meta,indent=2))
        staging.rename(path)
    return path
