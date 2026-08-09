"""hermes-agent plugin shim for kith.

Install:
    pip install kith-ai
    mkdir -p ~/.hermes/plugins/memory/kith
    cp plugin.yaml __init__.py ~/.hermes/plugins/memory/kith/
    # then in ~/.hermes/config.yaml:  memory: { provider: kith }

The provider subclasses the real MemoryProvider ABC when hermes is on the
path (isinstance checks pass), and duck-types otherwise.
"""

from kith.integrations.hermes import KithProvider as _KithProvider

try:
    from agent.memory_provider import MemoryProvider as _ABC

    class KithProvider(_KithProvider, _ABC):
        pass
except Exception:                                    # pragma: no cover
    KithProvider = _KithProvider


def register(collector):
    collector.register_memory_provider(KithProvider())
