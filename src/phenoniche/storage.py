"""Shared NAS locations for generated results and large project files."""

from pathlib import Path

STORAGE_ROOT = Path("/home/nas3/biod/xueshuailin/CCC_Phe/workspace")
INPUT_DIR = STORAGE_ROOT / "data/ccc"
PROCESSED_DIR = STORAGE_ROOT / "data/processed"
OUTPUT_DIR = STORAGE_ROOT / "outputs/xenium_prime_5k"
ARTIFACT_DIR = STORAGE_ROOT / "artifacts"
TEMP_DIR = STORAGE_ROOT / "tmp"
