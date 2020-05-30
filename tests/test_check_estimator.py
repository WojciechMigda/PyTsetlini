# -*- coding: utf-8 -*-

import pytest

import numpy as np
from pytsetlini import TsetlinMachineClassifier


def _unpack_bits(a):
    a = np.clip(a, 0, 255)
    return np.unpackbits(a.astype(np.uint8), axis=1)


def check_finite_X_y(X, y):
    # With this test setup, where we preprocess input
    # before passing it to the actual estimator,
    # we lose ability to perform some checks ourselves.
    # With a newer scikit-learn its check_estimator()
    # passes input X with NaN and Inf to see if our
    # estimator will throw, but MinMaxScaler silently
    # swallows such input. Hence, we work around this
    # calling check_X_y here just as our native estimator
    # does, but we will re-throw to satisfy
    # check_estimator()
    from sklearn.utils.validation import check_X_y
    if y is None and isinstance(X, np.ndarray):
        y = np.zeros((X.shape[0],))
        y[0] = 1.
    try:
        check_X_y(X, y, force_all_finite=True)
    except ValueError as e:
        if 'inf' in repr(e) or 'NaN' in repr(e):
            raise e
        else:
            # silently discard
            pass
    except:
        # silently discard
        pass


class XTsetlinMachineClassifier(TsetlinMachineClassifier):
    """Wrapped estimator

    Pipeline doesn't work well with check_estimator
    (https://github.com/scikit-learn/scikit-learn/issues/9768).
    This wrapper provides embedded input X transformation, also
    ensuring that all exceptions at the transformation step are caught
    so that they can be raised when the checks are run by the wrapped
    type.
    """
    def fit(self, X, y, n_iter=500):
        check_finite_X_y(X, y)
        X = self._fit_transform(X)

        super().fit(X, y, n_iter)
        return self


    def partial_fit(self, X, y, classes=None, n_iter=500):
        check_finite_X_y(X, y)
        X = self._fit_transform(X)

        super().partial_fit(X, y, classes=classes, n_iter=n_iter)
        return self


    def predict(self, X):
        check_finite_X_y(X, None)
        X = self._transform(X)

        return super().predict(X)


    def predict_proba(self, X):
        check_finite_X_y(X, None)
        X = self._transform(X)

        return super().predict_proba(X)


    def _transform(self, X):
        if hasattr(self, 'xformer_'):
            return self.xformer_.transform(X)
        else:
            return self._fit_transform(X)


    def _fit_transform(self, X):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import FunctionTransformer

        xformer = Pipeline(steps=[
            ('scaler', MinMaxScaler(feature_range=(0, 255))),
            ('unpacker', FunctionTransformer(_unpack_bits)),
        ])

        try:
            X = xformer.fit_transform(X)
            self.xformer_ = xformer
        except:
            pass

        return X


def test_classifier_passes_check_estimator():
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(XTsetlinMachineClassifier)
