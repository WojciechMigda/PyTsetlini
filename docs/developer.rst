Development notes
=================

Execution paths
---------------

* ``self.fit(X, y, n_iter=500)``

  Parameters as privided by the user. Initial input ``X`` and ``y`` verification takes place.

  * ``self.fit_(X, y, classes, n_iter)``

    ``n_iter`` is verified. ``y`` is encoded numerically. Number of labels is verified. Estimator parameters are validated. ``self.model`` is initialized from call to ``_classifier_fit``.

    * ``_classifier_fit(X, y, params, number_of_labels, n_iter)``

      ``params`` is a python dictionary, e.g. {'boost_true_positive_feedback': 0, 'clause_output_tile_size': 16, 'counting_type': 'auto', 'n_jobs': -1, 'number_of_pos_neg_clauses_per_label': 5, 'number_of_states': 100, 'random_state': 1, 's': 2.0, 'threshold': 15, 'verbose': False}. Call to ``libpytsetlini.classifier_fit`` returns ``bytes`` with ``json``-ized classifier state.

      * ``libpytsetlini.classifier_fit(np.ndarray npX, bint X_is_sparse, np.ndarray npy, bint y_is_sparse, bytes js_params, int number_of_labels, unsigned int n_epochs)``

        Translates ``numpy`` arrays to C++ vectors.

        * ``train_lambda(std::string const & params, std::vector<Tsetlini::aligned_vector_char> const & X, Tsetlini::label_vector_type const & y, int number_of_labels, unsigned int n_epochs)``

          ``params`` are deserialized with ``Tsetlini::make_params_from_json(params)`` and passed to create new classifier state instance ``Tsetlini::ClassifierState(params)``
          
          * ``Tsetlini::fit_impl(state, X, y, number_of_labels, n_epochs)``

          Return value is prepared with ``Tsetlini::to_json_string(state)``.
