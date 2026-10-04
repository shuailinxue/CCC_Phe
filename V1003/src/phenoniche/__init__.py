"""V1003 package namespace with read-only access to V1002 modules."""
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

