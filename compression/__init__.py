"""Explicit experimental masks; no automatic pruning or training integration."""
from .weight_masks import MaskedWeights
from .masked_checkpoint import save_masked_training, load_masked_training
