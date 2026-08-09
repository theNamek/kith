"""kith — social memory for multi-agent systems.

Who remembers what, about whom, visible to whom.
"""

from .model import KithError, Observation, Scope
from .store import BoundPrincipal, Store
from .view import RelationshipView

__version__ = "0.2.0"

__all__ = [
    "Store",
    "BoundPrincipal",
    "Observation",
    "Scope",
    "RelationshipView",
    "KithError",
]
