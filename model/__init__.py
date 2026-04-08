from .frozen_ssl import FrozenSSLDownscaler
from .ssl_downscaler import SSLDownscaler
from .casd import CASD
from .fgd import FGD, fgd_loss

__all__ = ["FrozenSSLDownscaler", "SSLDownscaler", "CASD", "FGD", "fgd_loss"]
