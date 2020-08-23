# -*- coding: utf-8 -*-
#
"""Init for tsetlin_tk."""

from __future__ import absolute_import

# This is extracted automatically by the top-level setup.py.
__version__ = '0.0.8'

# add any imports here, if you wish to bring things into the library's
# top-level namespace when the library is imported.

from .sklearn_estimator import TsetlinMachineClassifier
from .sklearn_estimator import TsetlinMachineRegressor
from .box_scaler import BoxScaler
