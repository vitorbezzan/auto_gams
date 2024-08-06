"""
Base class for auto gams models.
"""
import typing as tp
from functools import partial

import numpy as np
import optuna
from optuna.trial import Trial
from pydantic import NonNegativeInt, validate_call
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _check_y, check_array
from statsmodels.gam.api import BSplines, GLMGam
from statsmodels.gam.generalized_additive_model import GLMGamResults
from statsmodels.genmod.families.family import Family

from .types import Actuals, Inputs


class GAMOptions(tp.TypedDict, total=False):
    """
    Available options to use when fitting a GAM model.

    Args:
        max_df: Maximum degree of freedom to use when fitting GAMs. Minimum is
            automatically set for 4 since we use BSplines as basis functions.
        minmax_alpha: Tuple with minimum and maximum exponents for alpha penalization
            when fitting GAMs.
    """

    max_df: int
    minmax_alpha: tuple[int, int]


class BaseGAM(BaseEstimator):
    """
    Implements base behavior for all GAM models.
    """
    _feature_names: list[str]
    study_: optuna.Study
    model_: GLMGamResults

    def __new__(cls, *args, **kwargs):
        if cls is BaseGAM:
            raise TypeError(
                "BaseGAM is not directly instantiable.",
            )  # pragma: no cover
        return super(BaseGAM, cls).__new__(cls)

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        family: Family,
        n_trials: NonNegativeInt = 100,
        timeout: NonNegativeInt = 180,
    ) -> None:
        """
        Constructor for BaseGAM.

        Args:
            family: Family of distribution to use for target variable.
            n_trials: Number of optimization trials to use when finding a model.
            timeout: Timeout in seconds to stop the optimization.
        """
        self._family = family
        self._n_trials = n_trials
        self._timeout = timeout

    @property
    def study(self) -> optuna.Study | None:
        """
        Returns study post-optimization.
        """
        return getattr(self, "study_", None)

    def fit(
        self,
        X: Inputs,
        y: Actuals,
        **fit_params: GAMOptions,
    ) -> "BaseGAM":
        """
        Fits model based in configurations.

        Args:
            X: input data to use in fitting model.
            y: actual targets for fitting.
            fit_params: dictionary containing specific parameters to pass in fit
                process.
        """
        self.study_ = optuna.create_study(name="gam_model", direction="minimize")
        self.study_.optimize(
            func=partial(
                _to_optimize,
                family=self._family,
                X=self._treat_x(X),
                y=self._treat_y(y),
                max_df=fit_params.get("max_df", 12),
                minmax_alpha=fit_params.get("minmax_alpha", (-6, 3)),
            ),
            n_trials=self._n_trials,
            timeout=self._timeout,
        )

        self.model_ = GLMGam(
            family=self._family,
            exog=self._treat_x(X),
            endog=self._treat_y(y),
            smoother=BSplines(
                X,
                df=[self.study_.best_params[f"d_{i}"] for i in range(X.shape[1])],
                degree=[3 for _ in range(X.shape[1])],
            ),
            alpha=[self.study_.best_params[f"a_{i}"] for i in range(X.shape[1])],
        ).fit()

        return self

    @validate_call(config={"arbitrary_types_allowed": True})
    def _treat_x(self, X: Inputs) -> Inputs:
        """
        Checks and treats X inputs for model consumption.
        """
        if not hasattr(self, "_feature_names"):
            self._feature_names = list(X.columns)

        check_array(  # type: ignore
            np.array(X[self._feature_names]),
            dtype="numeric",
            force_all_finite=True,
        )

        return X[self._feature_names]

    @staticmethod
    @validate_call(config={"arbitrary_types_allowed": True})
    def _treat_y(y: Actuals) -> Actuals:
        """
        Checks and treats y inputs for model consumption.
        """
        return _check_y(np.array(y), multi_output=False, y_numeric=True)


def _to_optimize(
    trial: Trial,
    family: Family,
    X: Inputs,
    y: Actuals,
    max_df: int,
    minmax_alpha: tuple[int, int],
):
    """
    Returns BIC for model with parameters selected by optuna trial.

    Args:
        family: Family of GLM to use. Defined inside GAM definition.
        trial: Optuna trial.
        X: Input data for model.
        y: Target data for model.
        max_df: Maximum number of degrees of freedom to use in optimization search for
            each spline.
        minmax_alpha: Tuple with exponents for the regularization factor to be used for
            each spline.
    """
    df_ = [trial.suggest_int(f"d_{i}", 4, max_df) for i in range(X.shape[1])]
    alpha_ = [
        10 ** trial.suggest_float(f"a_{i}", minmax_alpha[0], minmax_alpha[1])
        for i in range(X.shape[1])
    ]

    return (
        GLMGam(
            family=family,
            exog=X,
            endog=y,
            smoother=BSplines(X, df=df_, degree=[3 for _ in range(X.shape[1])]),
            alpha=alpha_,
        )
        .fit()
        .bic
    )
