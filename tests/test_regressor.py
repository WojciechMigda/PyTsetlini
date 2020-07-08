# -*- coding: utf-8 -*-

import pytest
from pytsetlini import TsetlinMachineRegressor


def test_regressor_can_be_created():
    clf = TsetlinMachineRegressor()


def test_regressor_can_be_created_with_named_params():
    params = dict(
        s=7.5,
        number_of_states=256,
        threshold=27,
        number_of_clauses=30,
        boost_true_positive_feedback=1,
        n_jobs=2,
        verbose=True,
        random_state=42
    )
    clf = TsetlinMachineRegressor(**params)


def test_regressor_throws_when_constructed_with_unknown_param():
    params = dict(
        this_should_throw=True
    )

    with pytest.raises(TypeError):
        clf = TsetlinMachineRegressor(**params)
