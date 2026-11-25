"""Resolve cached physical inputs for read-only historical analyses."""
from pathlib import Path
import json

def input_root(source):
    source=Path(source);c=json.loads((source/'cohort_config.json').read_text())
    return Path(c.get('prepared_data_output',source))/'input'
