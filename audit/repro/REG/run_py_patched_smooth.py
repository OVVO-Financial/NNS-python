import sys; sys.path.insert(0, "/home/user/audit/work/REG/smooth"); sys.path.insert(0, "/home/user/audit/work/REG")
import patched_ss, nns._reg_engine as eng
eng._smooth_spline_predictor = patched_ss.predictor
import run_py
run_py.OUT = "/home/user/audit/work/REG/out_py_patched"
import os; os.makedirs(run_py.OUT, exist_ok=True)
run_py.main(sys.argv[1:])
