"""Candidate speedups for nns._reg_engine (verified to give identical outputs)."""
import numpy as np
import nns._reg_engine as eng

_orig_dist = eng._mreg_distances
_orig_rpm = eng._mreg_build_rpm

def fast_distances(rpm_x, xtest, dist, mins, maxs):
    if dist == "FACTOR":
        out = np.zeros((xtest.shape[0], rpm_x.shape[0]))
        for j in range(rpm_x.shape[1]):
            out += xtest[:, j, None] != rpm_x[None, :, j]
        return out / rpm_x.shape[1]
    ranges = maxs - mins
    active = np.flatnonzero(np.isfinite(ranges) & (ranges > 0))
    out = np.zeros((xtest.shape[0], rpm_x.shape[0]))
    if active.size == 0:
        return out
    tmp = np.empty_like(out); sq = np.empty_like(out)
    for j in active:
        np.subtract(xtest[:, j, None], rpm_x[None, :, j], out=tmp); tmp /= ranges[j]
        if dist == "NNS":
            np.multiply(tmp, tmp, out=sq); np.abs(tmp, out=tmp); tmp += sq; out += tmp
        elif dist == "L1":
            np.abs(tmp, out=tmp); out += tmp
        else:
            np.multiply(tmp, tmp, out=tmp); out += tmp
    return np.sqrt(out) if dist == "L2" else out

def fast_build_rpm(x, y, ids, noise_reduction, is_class):
    order = np.argsort(ids, kind="stable")
    ids_sorted = ids[order]
    uniq, starts, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
    if uniq.size == ids.size:
        return x[order].astype(np.float64), y[order].astype(np.float64)
    xs, ys = x[order], y[order]
    rpm_x = xs[starts].astype(np.float64).copy(); rpm_y = ys[starts].astype(np.float64).copy()
    for g in np.flatnonzero(counts > 1):           # only multi-member cells need a reducer
        s, e = starts[g], starts[g] + counts[g]
        rpm_x[g] = [eng._reduce_value(xs[s:e, j], noise_reduction, False) for j in range(x.shape[1])]
        rpm_y[g] = eng._reduce_value(ys[s:e], noise_reduction, is_class)
    return rpm_x, rpm_y

def install():
    eng._mreg_distances = fast_distances
    eng._mreg_build_rpm = fast_build_rpm
