# -*- coding: utf-8 -*-
# distutils: language = c++

from pytsetlini.tsetlini_estimator_state cimport (
    ClassifierStateClassic, RegressorStateClassic)

from libcpp.string cimport string


cdef extern from "tsetlini_state_json.hpp" namespace "Tsetlini" nogil:
    cdef string to_json_string(ClassifierStateClassic state)
    cdef string to_json_string(RegressorStateClassic state)
