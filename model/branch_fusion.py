"""Parameter-free fixed branch fusion, shared by backbone and head interfaces."""
import math


def validate_alph(alph):
    if not isinstance(alph, (int, float)) or not math.isfinite(alph) or not 0 <= alph <= 1:
        raise ValueError('alph must be a finite fixed coefficient in [0,1]')


def fixed_fusion(cnn_features, graph_features, alph=0.8):
    validate_alph(alph)
    if cnn_features.ndim != 2 or cnn_features.shape != graph_features.shape:
        raise ValueError('Fusion expects matching [pixels,channels] branch features')
    return alph * cnn_features + (1.0 - alph) * graph_features
