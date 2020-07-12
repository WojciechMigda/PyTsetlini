# coding: utf-8

import numpy as np
from scipy import sparse
from sklearn.base import (
    BaseEstimator, TransformerMixin)
from sklearn.utils.validation import (
    FLOAT_DTYPES, check_is_fitted, check_array)


def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.'''
    """
    Copied from https://github.com/scikit-learn/scikit-learn/blob/3017ea8c30e82931895f4ec46c4a5a9d00ed67c2/sklearn/preprocessing/_data.py
    """


    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


class BoxScaler(BaseEstimator, TransformerMixin):
    """Transform features by scaling each feature to a given range.
    This estimator scales and translates all features together such
    that they are in the given range on the training set, e.g. between
    zero and one.
    The transformation is given by::
        X_std = (X - imin) / (imax - imin)
        X_scaled = X_std * (omax - omin) + omin
    where imin, imax = in_feature_range
    and omin, oman = out_feature_range.
    In absence of in_feature_range (imin, imax) = (X.max, X.min).
    In absence of out_feature_range (omin, omax) = (0, 1).
    Parameters
    ----------
    in_feature_range : tuple (min, max), default=(X.max, X.min)
        Desired range of transformed data.
    out_feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).
    clip: bool, default=False
        Set to True to clip transformed values of held-out data to
        provided `feature range`.
    Attributes
    ----------
    min_ : float
        All features adjustment for minimum. Equivalent to
        ``omin - imin * self.scale_``
    scale_ : float
        All features relative scaling of the data. Equivalent to
        ``(omax - omin) / (imax - imin)``
    data_min_ : float
        All features minimum seen in the data
    data_max_ : float
        All features maximum seen in the data
    data_range_ : float
        All features range ``(data_max_ - data_min_)`` seen in the data
    n_samples_seen_ : int
        The number of samples processed by the estimator.
        It will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.
    Examples
    --------
    >>> import BoxScaler
    >>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    >>> scaler = BoxScaler()
    >>> print(scaler.fit(data))
    BoxScaler()
    >>> print(scaler.data_max_)
    18.0
    >>> print(scaler.transform(data))
    [[0.         0.15789474]
     [0.02631579 0.36842105]
     [0.05263158 0.57894737]
     [0.10526316 1.        ]]
    >>> print(scaler.transform([[2, 2]]))
    [[0.15789474 0.15789474]]
    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.
    Code heavily based on sklearn's MinMaxScaler
    """
    def __init__(self, in_feature_range=None, out_feature_range=(0, 1), *, copy=True, clip=False):
        self.in_feature_range = in_feature_range
        self.out_feature_range = out_feature_range
        self.copy = copy
        self.clip = clip

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Online computation of min and max on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """
        in_feature_range = self.in_feature_range
        if in_feature_range and in_feature_range[0] >= in_feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(in_feature_range))
        out_feature_range = self.out_feature_range
        if out_feature_range[0] >= out_feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(out_feature_range))

        if sparse.issparse(X):
            raise TypeError("BoxScaler does not support sparse input.")

        first_pass = not hasattr(self, 'n_samples_seen_')
        X = self._validate_data(X, reset=first_pass,
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite="allow-nan")

        data_min, data_max = in_feature_range or (np.nanmin(X), np.nanmax(X))

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            data_min = np.minimum(self.data_min_, data_min)
            data_max = np.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = ((out_feature_range[1] - out_feature_range[0]) /
                       _handle_zeros_in_scale(data_range))
        self.min_ = out_feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        """Scale features of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
                        force_all_finite="allow-nan")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X.shape[1] should be {0:d}, not {1:d}.".format(
                self.n_features_in_, X.shape[1]))

        X *= self.scale_
        X += self.min_
        if self.clip:
            np.clip(X, self.out_feature_range[0], self.out_feature_range[1], out=X)
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
                        force_all_finite="allow-nan")

        X -= self.min_
        X /= self.scale_
        return X

    def _more_tags(self):
        return {'allow_nan': True}
