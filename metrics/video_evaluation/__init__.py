# Lazy imports to avoid dependency errors for unused evaluators
try:
    from .dynamics import DynamicsEvaluator
except ImportError:
    DynamicsEvaluator = None

try:
    from .shadow import ShadowEvaluator
except ImportError:
    ShadowEvaluator = None

try:
    from .perspective import PerspectiveEvaluator
except ImportError:
    PerspectiveEvaluator = None

try:
    from .lines import LinesEvaluator
except ImportError:
    LinesEvaluator = None

try:
    from .meter import MEt3REvaluator
except ImportError:
    MEt3REvaluator = None

from .epipolar import EpipolarEvaluator

__all__ = ["DynamicsEvaluator", "ShadowEvaluator", "PerspectiveEvaluator", "LinesEvaluator", "MEt3REvaluator", "EpipolarEvaluator"]
