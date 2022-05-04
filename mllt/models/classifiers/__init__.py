from .base import BaseClassifier
from .simple import SimpleClassifier
from .dualsimple import DualSimpleClassifier
from .contrast_simple import ContrastSimpleClassifier
from .contrast import ContrastClassifier
from .gcn_simple import GCNSimpleClassifier
from .CSRA_simple import CSRASimpleClassifier
from .gcn_kd import GCNKDClassifier
from .dualkd import DualKDClassifier
from .gcntest import GCNTestClassifier
from .contrast_gcn import ContrastGCNClassifier


__all__ = [
    'BaseClassifier', 'SimpleClassifier','DualSimpleClassifier','ContrastSimpleClassifier','ContrastClassifier','GCNSimpleClassifier','CSRASimpleClassifier','GCNKDClassifier','DualKDClassifier','GCNTestClassifier','ContrastGCNClassifier'
]
