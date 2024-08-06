# fmt: off

"""
Some type definitions for the auto_gams library.
"""
import numpy as np
import pandas as pd
from numpy.typing import NDArray

Inputs = pd.DataFrame
Actuals = NDArray[np.float64] | pd.Series
Predictions = NDArray[np.float64]

# fmt: on
