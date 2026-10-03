"""V1005 namespace with read-only imports of validated V1002 utilities."""
from pathlib import Path
__path__.append(str(Path(__file__).resolve().parents[3] / 'V1002/src/phenoniche'))
