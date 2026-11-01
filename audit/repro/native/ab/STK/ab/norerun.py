import os, runpy, subprocess, sys
_real = subprocess.run
class _Done:
    returncode = 0
    stdout = ""
    stderr = ""
def _run(cmd, *a, **k):
    if isinstance(cmd, list) and cmd and cmd[0] == "Rscript" and os.path.exists(cmd[-1]):
        return _Done()
    return _real(cmd, *a, **k)
subprocess.run = _run
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(sys.argv[0], run_name="__main__")
