import time, io, contextlib, importlib.util, cProfile, pstats, sys
import matplotlib; matplotlib.use("Agg")
spec = importlib.util.spec_from_file_location("v07", "/home/user/wt/native/examples/vignettes/07_clustering_and_regression.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
buf = io.StringIO()
for i in range(3):
    t = time.perf_counter()
    with contextlib.redirect_stdout(buf): mod.main()
    print(f"run {i}: {time.perf_counter() - t:.2f}s")
if len(sys.argv) > 1:
    with contextlib.redirect_stdout(buf): cProfile.run("mod.main()", sys.argv[1])
    pstats.Stats(sys.argv[1]).sort_stats("tottime").print_stats(12)
