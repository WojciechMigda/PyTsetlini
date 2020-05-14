# -*- coding: utf-8 -*-
# distutils: language = c++

from pytsetlini.tsetlini_classifier_state cimport ClassifierState

from libcpp.string cimport string


cdef extern from "tsetlini_state_json.hpp" namespace "Tsetlini" nogil:
    cdef string to_json_string(ClassifierState state)
