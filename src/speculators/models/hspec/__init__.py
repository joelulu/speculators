"""Research reference implementation of the H-Spec block drafter."""

from .reference import HSpecReference, HSpecReferenceConfig, TargetContext
from .vllm_bridge import context_from_vllm_pages

__all__ = [
    "HSpecReference",
    "HSpecReferenceConfig",
    "TargetContext",
    "context_from_vllm_pages",
]
