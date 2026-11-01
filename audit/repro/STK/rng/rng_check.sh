#!/bin/sh
# R vs nns._rrng: weighted sampling without replacement (unsorted probs) diverges; plain streams match
Rscript -e 'set.seed(42); cat("R  unsorted prob:", sample.int(5,3,prob=c(.1,.5,.2,.15,.05)),"\n"); set.seed(42); cat("R  sorted prob:  ", sample.int(5,3,prob=c(.5,.2,.15,.1,.05)),"\n"); set.seed(7); cat("R  sample(10):", sample(10),"\n")'
/home/user/NNS-python/.venv/bin/python -c "
from nns._rrng import RRNG; import numpy as np
r=RRNG(42); print('PY unsorted prob:', r.sample_int_prob_noreplace(5,3,np.array([.1,.5,.2,.15,.05])))
r=RRNG(42); print('PY sorted prob:  ', r.sample_int_prob_noreplace(5,3,np.array([.5,.2,.15,.1,.05])))
r=RRNG(7); print('PY sample(10):', r.sample_int(10))"
