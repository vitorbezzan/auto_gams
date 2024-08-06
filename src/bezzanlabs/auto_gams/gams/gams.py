"""
Base class for a regression generalized additive model.
"""

from pydantic import NonNegativeInt
from sklearn.base import ClassifierMixin, RegressorMixin
from statsmodels.genmod.families.family import Binomial, Gaussian

from .base import BaseGAM


class ClassifierGAM(BaseGAM, ClassifierMixin):
    """
    Defines a classifier GAM. Limited to just two (0/1) categories.
    """

    def __init__(
        self,
        n_trials: NonNegativeInt = 100,
        timeout: NonNegativeInt = 180,
    ) -> None:
        """
        Constructor for ClassifierGAM.

        Args:
            n_trials: Number of optimization trials to use when finding a model.
            timeout: Timeout in seconds to stop the optimization.
        """
        super().__init__(Binomial(), n_trials, timeout)


class RegressionGAM(BaseGAM, RegressorMixin):
    """
    Defines a regression GAM.
    """

    def __init__(
        self,
        n_trials: NonNegativeInt = 100,
        timeout: NonNegativeInt = 180,
    ) -> None:
        """
        Constructor for RegressionGAM.

        Args:
            n_trials: Number of optimization trials to use when finding a model.
            timeout: Timeout in seconds to stop the optimization.
        """
        super().__init__(Gaussian(), n_trials, timeout)
