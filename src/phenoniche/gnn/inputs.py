"""Resolve a historical feature universe without changing its source cache."""
from dataclasses import replace,fields
from pathlib import Path
import json


def resolve_input(config, current):
    if not config.use_legacy_feature_mask: return current
    manifest=Path(config.legacy_output)/'cache/input_manifest.json'
    if not manifest.is_file(): return current
    source=Path(json.loads(manifest.read_text())['signature']['source'])
    frozen=json.loads((source.parent/'input_manifest.json').read_text())['signature']
    allowed={f.name for f in fields(current)}
    return replace(current,**{k:v for k,v in frozen.items() if k in allowed},output_dir=str(source.parent.parent))
