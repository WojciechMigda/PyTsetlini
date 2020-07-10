import json

import scipy.sparse as sp
from sklearn.utils.multiclass import type_of_target

from . import libpytsetlini


def _validate_params(params):
    rv = dict(params)

    for k, v in params.items():
        if k == "s":
            v = float(v)
            assert(v > 0.)
        elif k == "boost_true_positive_feedback":
            v = int(v)
            assert(v in [0, 1])
        elif k == "n_jobs":
            v = int(v)
            assert(v == -1 or v > 0)
        elif k == "number_of_pos_neg_clauses_per_label":
            v = int(v)
            assert(v > 0)
        elif k == "number_regressor_clauses":
            v = int(v)
            assert(v > 0)
        elif k == "number_of_states":
            v = int(v)
            assert(v > 0)
        elif k == "random_state":
            if v is not None:
                v = int(v)
        elif k == "threshold":
            v = int(v)
            assert(v > 0)
        elif k == "verbose":
            v = bool(v)
        elif k == "counting_type":
            v = str(v)
            assert(v in ["auto", "int8", "int16", "int32"])
        elif k == "clause_output_tile_size":
            v = int(v)
            assert(v in [16, 32, 64, 128])

        rv[k] = v

    return rv


def _check_regression_targets(y):
    """Ensure that target y is of a regression type.
    Only the following target types (as defined in type_of_target) are allowed:
        'continuous', 'binary', 'multiclass'
    Parameters
    ----------
    y : array-like
    """
    y_type = type_of_target(y)
    if y_type not in ['continuous', 'binary', 'multiclass']:
        raise ValueError("Unknown response value type: %r" % y_type)


def _params_as_json_bytes(params):
    return json.dumps(params).encode('UTF-8')


def _classifier_fit(X, y, params, number_of_labels, n_iter):
    """
    "number_of_labels" and "number_of_features" will be derived from X and y
    """

    X_is_sparse = sp.issparse(X)
    y_is_sparse = sp.issparse(y)

    js_state = libpytsetlini.classifier_fit(
        X, X_is_sparse,
        y, y_is_sparse,
        _params_as_json_bytes(params),
        number_of_labels,
        n_iter)

    return js_state


def _classifier_partial_fit(X, y, js_model, n_iter):
    """
    "number_of_labels" and "number_of_features" will be derived from X and y
    """

    X_is_sparse = sp.issparse(X)
    y_is_sparse = sp.issparse(y)

    js_state = libpytsetlini.classifier_partial_fit(
        X, X_is_sparse,
        y, y_is_sparse,
        js_model,
        n_iter)

    return js_state


def _classifier_predict(X, js_model):

    X_is_sparse = sp.issparse(X)

    y_hat = libpytsetlini.classifier_predict(X, X_is_sparse, js_model)

    return y_hat


def _classifier_predict_proba(X, js_model, threshold):
    X_is_sparse = sp.issparse(X)

    probas = libpytsetlini.classifier_predict_proba(
        X, X_is_sparse, js_model, threshold)

    return probas


def _regressor_fit(X, y, params, n_iter):
    """
    "number_of_features" will be derived from X and y
    """

    X_is_sparse = sp.issparse(X)
    y_is_sparse = sp.issparse(y)

    js_state = libpytsetlini.regressor_fit(
        X, X_is_sparse,
        y, y_is_sparse,
        _params_as_json_bytes(params),
        n_iter)

    return js_state


def _regressor_predict(X, js_model):

    X_is_sparse = sp.issparse(X)

    y_hat = libpytsetlini.regressor_predict(X, X_is_sparse, js_model)

    return y_hat


def _regressor_partial_fit(X, y, js_model, n_iter):
    """
    "number_of_features" will be derived from X and y
    """

    X_is_sparse = sp.issparse(X)
    y_is_sparse = sp.issparse(y)

    js_state = libpytsetlini.regressor_partial_fit(
        X, X_is_sparse,
        y, y_is_sparse,
        js_model,
        n_iter)

    return js_state
