# Live R parity report

- Plan: `sync/last_r_api_plan.json`
- R checkout: `upstream/NNS`
- Fresh cache requested: `False`
- Skip install: `False`
- Live R recompute: `True`

## Result: live R parity diverged

Mapped parity tests recomputed every R value from the freshly installed live R NNS and the Python implementation did not match. Public Python behavior differs from live R at the recorded commit.

Failing command: `/opt/hostedtoolcache/Python/3.11.15/x64/bin/python -m pytest -q -n 0 tests/parity/test_practical_examples.py tests/parity/test_r13_smoke.py`
Exit status: `1`
