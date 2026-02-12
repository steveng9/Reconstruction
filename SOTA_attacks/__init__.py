"""
State-of-the-Art (SOTA) reconstruction attacks from published papers.

This module provides wrappers for published attack implementations,
linking to external repositories while conforming to our standard attack interface.
"""

from .linear_reconstruction import linear_reconstruction_attack

__all__ = [
    'linear_reconstruction_attack',
]
