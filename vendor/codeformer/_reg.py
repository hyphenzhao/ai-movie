"""Minimal shims so the vendored CodeFormer arch files need no basicsr."""
import logging

def get_root_logger(*args, **kwargs):
    return logging.getLogger("codeformer")

class _NoopRegistry:
    def register(self, *args, **kwargs):
        def _decorator(cls):
            return cls
        return _decorator

ARCH_REGISTRY = _NoopRegistry()
