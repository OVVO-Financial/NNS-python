# Python 54c98418 repaired parity failure inventory

All rows remain category F because executable R 54c98418 fixture artifacts have not been imported in this environment. Do not update expectations from this inventory alone.

- Total failures: 55
- R commit: 54c98418c2a11499ebb1c456570d2b66c37eb817

| # | test | function | fixture case | classification | action |
|---:|---|---|---|---|---|
| 1 | `tests/parity/test_boost.py::test_nns_boost_numeric_matches_r[None]` | NNS.boost | `boost_numeric` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 2 | `tests/parity/test_boost.py::test_nns_boost_numeric_matches_r[1]` | NNS.boost | `boost_numeric` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 3 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_matches_r[50-2-linear-None-None-None-False-off]` | multivariate regression | `numeric_l2_default` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 4 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_matches_r[50-3-nonlinear-1-1-point_est1-False-off]` | multivariate regression | `numeric_l2_default` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 5 | `tests/parity/test_boost.py::test_nns_boost_numeric_matches_r[2]` | NNS.boost | `boost_numeric` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 6 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_matches_r[200-3-mixed-2-2-None-False-mean]` | multivariate regression | `numeric_l2_default` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 7 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_matches_r[200-5-linear-max-None-None-False-median]` | multivariate regression | `numeric_order_max` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 8 | `tests/parity/test_boost.py::test_nns_boost_ivs_test_none_matches_r` | NNS.boost | `boost_class_pred_int` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 9 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_matches_r[50-2-nonlinear-1-1-point_est4-True-off]` | multivariate regression | `numeric_l2_default` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 10 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_confidence_interval_matches_r[2-0.8-None-None-None]` | multivariate regression | `numeric_l2_default` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 11 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_confidence_interval_matches_r[3-0.95-None-2-None]` | multivariate regression | `numeric_l2_default` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 12 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_confidence_interval_matches_r[2-0.95-1-1-point_est2]` | multivariate regression | `numeric_l2_default` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 13 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_confidence_interval_matches_r[3-0.8-2-2-point_est3]` | multivariate regression | `numeric_l2_default` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 14 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_classification_matches_r[2-classes0-point_est0-1-1]` | multivariate regression | `multiclass` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 15 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_classification_matches_r[2-classes0-point_est0-1-2]` | multivariate regression | `multiclass` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 16 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_classification_matches_r[3-classes1-point_est1-2-1]` | multivariate regression | `multiclass` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 17 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_classification_matches_r[3-classes1-point_est1-2-2]` | multivariate regression | `multiclass` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 18 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_class_confidence_interval_matches_r[2-classes0-point_est0-1-1]` | multivariate regression | `multiclass` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 19 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_class_confidence_interval_matches_r[2-classes0-point_est0-1-2]` | multivariate regression | `multiclass` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 20 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_class_confidence_interval_matches_r[3-classes1-point_est1-2-1]` | multivariate regression | `multiclass` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 21 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_class_confidence_interval_matches_r[3-classes1-point_est1-2-2]` | multivariate regression | `multiclass` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 22 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_factor_levels_return_numeric_codes` | multivariate regression | `multiclass` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 23 | `tests/parity/test_multivariate_regression.py::test_nns_m_reg_factor_levels_class_confidence_interval_matches_r` | multivariate regression | `multiclass` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 24 | `tests/parity/test_multivariate_regression.py::test_nns_reg_matrix_classification_dispatches_to_m_reg` | multivariate regression | `multiclass` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 25 | `tests/parity/test_boost.py::test_nns_boost_ts_test_deterministic_matches_r[3]` | NNS.boost | `boost_ts` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 26 | `tests/parity/test_boost.py::test_nns_boost_ts_test_deterministic_matches_r[5]` | NNS.boost | `boost_ts` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 27 | `tests/parity/test_boost.py::test_nns_boost_ts_test_deterministic_matches_r[8]` | NNS.boost | `boost_ts` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 28 | `tests/parity/test_r13_smoke.py::test_r_nns_13_seeded_stack_smoke_sample` | NNS.stack | `stack_method1_regression` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 29 | `tests/parity/test_regression.py::test_nns_reg_factor_predictor_matches_r_full_rank_dummy_path` | NNS.reg | `reg_default` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 30 | `tests/parity/test_boost.py::test_nns_boost_numeric_pred_int_matches_r[1-0.95]` | NNS.boost | `boost_numeric` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 31 | `tests/parity/test_boost.py::test_nns_boost_numeric_pred_int_matches_r[2-0.8]` | NNS.boost | `boost_numeric` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 32 | `tests/parity/test_stack.py::test_nns_stack_ts_test_matches_r[method2-5]` | NNS.stack | `stack_method12_ts` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 33 | `tests/parity/test_stack.py::test_nns_stack_ts_test_matches_r[method3-10]` | NNS.stack | `stack_method12_ts` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 34 | `tests/parity/test_stack.py::test_nns_stack_numeric_matches_r[True-method0]` | NNS.stack | `stack_method1_regression` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 35 | `tests/parity/test_stack.py::test_nns_stack_ts_test_matches_r[method4-10]` | NNS.stack | `stack_method12_ts` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 36 | `tests/parity/test_stack.py::test_nns_stack_numeric_matches_r[True-method2]` | NNS.stack | `stack_method12_ts` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 37 | `tests/parity/test_stack.py::test_nns_stack_var_like_ts_test_matches_r` | NNS.stack | `stack_method12_ts` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 38 | `tests/parity/test_boost.py::test_nns_boost_binary_class_pred_int_matches_r[1]` | NNS.boost | `boost_class_pred_int` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 39 | `tests/parity/test_stack.py::test_nns_stack_pred_int_matches_r[method0]` | NNS.stack | `stack_pred_int` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 40 | `tests/parity/test_stack.py::test_nns_stack_numeric_matches_r[False-method0]` | NNS.stack | `stack_method1_regression` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 41 | `tests/parity/test_boost.py::test_nns_boost_binary_class_pred_int_matches_r[2]` | NNS.boost | `boost_class_pred_int` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 42 | `tests/parity/test_stack.py::test_nns_stack_pred_int_matches_r[method2]` | NNS.stack | `stack_method12_ts` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 43 | `tests/parity/test_stack.py::test_nns_stack_numeric_matches_r[False-method2]` | NNS.stack | `stack_method12_ts` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 44 | `tests/parity/test_stack.py::test_nns_stack_binary_class_matches_r[method0]` | NNS.stack | `stack_classification` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 45 | `tests/parity/test_stack.py::test_nns_stack_mixed_factor_predictor_method12_matches_r` | NNS.stack | `stack_method1_regression` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 46 | `tests/parity/test_stack.py::test_nns_stack_ts_test_matches_r[method0-5]` | NNS.stack | `stack_method12_ts` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 47 | `tests/parity/test_stack.py::test_nns_stack_binary_class_pred_int_matches_r[method0]` | NNS.stack | `stack_classification` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 48 | `tests/parity/test_stack.py::test_nns_stack_ts_test_matches_r[method1-10]` | NNS.stack | `stack_method12_ts` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 49 | `tests/parity/test_stack.py::test_nns_stack_multiclass_matches_r[method2]` | NNS.stack | `stack_method12_ts` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 50 | `tests/parity/test_stack.py::test_nns_stack_factor_like_class_pred_int_matches_r` | NNS.stack | `stack_classification` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 51 | `tests/parity/test_stack.py::test_nns_stack_factor_like_class_matches_r` | NNS.stack | `stack_classification` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 52 | `tests/parity/test_var.py::test_var_interpolate_and_extrapolate_matches_r[trailing_na-3]` | NNS.VAR | `var_cor_missing` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 53 | `tests/parity/test_var.py::test_var_multivariate_stack_stage_matches_r[tau1-1-cor]` | NNS.VAR | `var_cor_tau1` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 54 | `tests/parity/test_var.py::test_public_nns_var_cor_handles_missing_values_like_r` | NNS.VAR | `var_cor_missing` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
| 55 | `tests/parity/test_var.py::test_public_nns_var_cor_matches_r[scalar_tau-1]` | NNS.VAR | `var_cor_tau1` | F | Import 54c98418 R artifacts, compare actuals, then reclassify. |
