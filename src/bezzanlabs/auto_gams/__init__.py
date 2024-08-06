# flake8: noqa
"""
API entrypoint for the package.
"""

__package_name__ = "bezzanlabs.auto_gams"
__version__ = "0.1.0"


from .gams.base import BaseGAM
from .gams.gams import ClassifierGAM, RegressionGAM
