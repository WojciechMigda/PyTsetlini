# -*- coding: utf-8 -*-

import pytest

import numpy as np
from pytsetlini import BoxScaler


def test_box_scaler_passes_check_estimator():
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(BoxScaler())

def test_default_box_scaler_transforms_data():
    scaler = BoxScaler()
    X = np.arange(10, 21).reshape(-1, 1)
    """
    array([[10],
       [11],
       [12],
       [13],
       [14],
       [15],
       [16],
       [17],
       [18],
       [19],
       [20]])
    """

    Xt = scaler.fit_transform(X)
    """
    array([[0. ],
       [0.1],
       [0.2],
       [0.3],
       [0.4],
       [0.5],
       [0.6],
       [0.7],
       [0.8],
       [0.9],
       [1. ]])
    """
    expected = np.arange(0.0, 1.0 + 0.01, 0.1).reshape(-1, 1)
    assert(np.allclose(Xt, expected))

def test_box_scaler_transforms_data_into_output_range():
    scaler = BoxScaler(out_feature_range=(1, 3))
    X = np.arange(10, 21).reshape(-1, 1)
    """
    array([[10],
       [11],
       [12],
       [13],
       [14],
       [15],
       [16],
       [17],
       [18],
       [19],
       [20]])
    """

    Xt = scaler.fit_transform(X)
    """
    array([[1. ],
       [1.2],
       [1.4],
       [1.6],
       [1.8],
       [2. ],
       [2.2],
       [2.4],
       [2.6],
       [2.8],
       [3. ]])
    """
    expected = np.arange(1.0, 3.0 + 0.01, 0.2).reshape(-1, 1)
    assert(np.allclose(Xt, expected))

def test_box_scaler_transforms_data_from_input_range():
    scaler = BoxScaler(in_feature_range=(0, 40))
    X = np.arange(10, 31, 2).reshape(-1, 1)
    """
    array([[10],
       [12],
       [14],
       [16],
       [18],
       [20],
       [22],
       [24],
       [26],
       [28],
       [30]])
    """

    Xt = scaler.fit_transform(X)
    """
    array([[0.25],
       [0.3 ],
       [0.35],
       [0.4 ],
       [0.45],
       [0.5 ],
       [0.55],
       [0.6 ],
       [0.65],
       [0.7 ],
       [0.75]])
    """
    expected = np.arange(0.25, 0.75 + 0.01, 0.05).reshape(-1, 1)
    assert(np.allclose(Xt, expected))

def test_box_scaler_transforms_data_from_input_range_into_output_range():
    scaler = BoxScaler(in_feature_range=(0, 40), out_feature_range=(1, 3))
    X = np.arange(10, 31, 2).reshape(-1, 1)
    """
    array([[10],
       [12],
       [14],
       [16],
       [18],
       [20],
       [22],
       [24],
       [26],
       [28],
       [30]])
    """

    Xt = scaler.fit_transform(X)
    """
    array([[1.5],
       [1.6],
       [1.7],
       [1.8],
       [1.9],
       [2. ],
       [2.1],
       [2.2],
       [2.3],
       [2.4],
       [2.5]])
    """
    expected = np.arange(1.5, 2.5 + 0.01, 0.1).reshape(-1, 1)
    assert(np.allclose(Xt, expected))

def test_box_scaler_inverse_transforms_data_from_input_range_into_output_range_and_back():
    scaler = BoxScaler(in_feature_range=(0, 40), out_feature_range=(1, 3))
    X = np.arange(10, 31, 2).reshape(-1, 1)
    """
    array([[10],
       [12],
       [14],
       [16],
       [18],
       [20],
       [22],
       [24],
       [26],
       [28],
       [30]])
    """

    Xt = scaler.fit_transform(X)
    """
    array([[1.5],
       [1.6],
       [1.7],
       [1.8],
       [1.9],
       [2. ],
       [2.1],
       [2.2],
       [2.3],
       [2.4],
       [2.5]])
    """
    Xit = scaler.inverse_transform(Xt)
    """
    array([[10.],
       [12.],
       [14.],
       [16.],
       [18.],
       [20.],
       [22.],
       [24.],
       [26.],
       [28.],
       [30.]])
    """
    assert(np.allclose(Xit, X))
