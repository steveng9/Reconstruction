"""
Enhancement wrappers for reconstruction attacks.

Enhancements are composable wrappers that augment base attack methods
without modifying their core implementation. Examples include:
- Chaining: Sequential feature prediction
- Ensembling: Combining multiple attack methods (TODO)
- Auxiliary data: Augmenting training data (TODO)
"""

from .chaining_wrapper import apply_chaining

__all__ = ['apply_chaining']
