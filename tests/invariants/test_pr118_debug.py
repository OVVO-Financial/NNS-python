import numpy as np

from nns import nns_causation
from nns.seasonality import nns_seas


def test_pr118_debug_seasonal_lags():
    t = np.arange(1, 71, dtype=np.float64)
    x = np.sin(2.0 * np.pi * t / 7.0)
    y = np.roll(x, 1) + 0.05 * np.cos(t / 3.0)
    rows = {
        'period7_x_periods': nns_seas(x, plot=False)['periods'].tolist(),
        'period7_y_periods': nns_seas(y, plot=False)['periods'].tolist(),
        'period7_by_tau': {tau: list(nns_causation(x, y, tau=tau).values()) for tau in range(1, 9)},
    }
    t2 = np.arange(1, 81, dtype=np.float64)
    z = [
        np.sin(2.0 * np.pi * t2 / 5.0),
        np.sin(2.0 * np.pi * t2 / 6.0 + 0.2),
        np.sin(2.0 * np.pi * t2 / 7.0 + 0.5),
    ]
    rows['matrix_periods'] = [nns_seas(v, plot=False)['periods'].tolist() for v in z]
    raise AssertionError(repr(rows))
