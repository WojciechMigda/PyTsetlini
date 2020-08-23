# -*- coding: utf-8 -*-
# distutils: language = c++


cdef extern from "estimator_state.hpp" namespace "Tsetlini":
    cdef cppclass ClassifierStateClassic
    cdef cppclass RegressorStateClassic
