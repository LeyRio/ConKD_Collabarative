from .cls_head import ClsHead
from .weldon_head import WeldonHead
from .gcn_head import GcnHead
from .cos_head import CosHead
from .dual_head import DualHead
from .dual_head_random import DualHeadRandom
from .contrast import ContrastHead
from .csra_head import MulCSRAHead


__all__ = ['ClsHead', 'WeldonHead', 'GcnHead','CosHead','DualHead','DualHeadRandom','ContrastHead','MulCSRAHead']
