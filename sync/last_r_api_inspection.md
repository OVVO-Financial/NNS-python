# R API parity review plan

## Changed files

- `R/Boost.R`
- `R/Partition_Map.R`
- `R/Stack.R`
- `src/NNS_distance.cpp`

## Affected Python modules

- `src/nns/boost.py`
- `src/nns/stack.py`

## Parity tests to run

- `tests/docs/test_vignette_examples.py`

## Cache scope

- `NNS.boost`
- `NNS.stack`

## Required actions

- Fresh cache required: `False`
- Export review required: `False`
- Unmapped R files present: `True`

## Unmapped R files

- `R/Partition_Map.R`
