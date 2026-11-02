"""One immutable asset binding shared by a receiver's worker threads."""
from contextlib import contextmanager
import threading

_active=None
_lock=threading.RLock()


def current():
    if _active is None:raise RuntimeError('Use Runtime with verified assets before inference.')
    return _active


@contextmanager
def use(assets):
    global _active
    with _lock:
        previous=_active;_active=assets
        try:yield
        finally:_active=previous


class AssetDirectory:
    def __init__(self,relative):self.relative=relative
    def __truediv__(self,name):return current().root/self.relative/name
