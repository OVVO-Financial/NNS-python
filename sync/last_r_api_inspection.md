# R API parity review plan

## Changed files

- `R/Multivariate_Regression.R`
- `R/RcppExports.R`
- `R/Regression.R`
- `R/Stack.R`
- `benchmarks/benchmark_stack_native.R`
- `benchmarks/stack_baseline_report.md`
- `benchmarks/stack_baseline_results.csv`
- `benchmarks/stack_native_report.md`
- `benchmarks/stack_native_results.csv`
- `src/NNS_mreg_predict.cpp`
- `src/NNS_mreg_setup.cpp`
- `src/NNS_stack_fast.cpp`
- `src/RcppExports.cpp`
- `tests/testthat/helper-stack-parity.R`
- `tests/testthat/test-mreg-path-native.R`
- `tests/testthat/test-mreg-setup-native.R`
- `tests/testthat/test-stack-method1-native-parity.R`
- `tests/testthat/test-stack-method2-native-parity.R`
- `tests/testthat/test-stack-native-parity.R`
- `tests/testthat/test-univariate-fast-parity.R`
- `tests/testthat/test-xstar-path-native.R`

## Affected Python modules

- `src/nns/multivariate_regression.py`
- `src/nns/regression.py`
- `src/nns/stack.py`

## Parity tests to run

- `tests/docs/test_vignette_examples.py`
- `tests/parity/test_practical_examples.py`
- `tests/parity/test_r13_smoke.py`

## Cache scope

- `NNS.M.reg`
- `NNS.reg`
- `NNS.stack`

## Required actions

- Fresh cache required: `False`
- Export review required: `False`
- Unmapped R files present: `True`

## Unmapped R files

- `R/Multivariate_Regression.R`
- `R/RcppExports.R`
