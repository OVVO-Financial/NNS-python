from __future__ import annotations

import numpy as np

from pynns import causal_matrix, nns_causation, nns_copula, nns_cor, nns_dep


def main() -> None:
    x = np.linspace(-2.0, 2.0, 101, dtype=np.float64)
    linear_y = 2.0 * x
    nonlinear_y = x**2
    cyclic_y = np.sin(np.pi * x)

    linear = nns_dep(x, linear_y)
    nonlinear = nns_dep(x, nonlinear_y)
    cyclic = nns_dep(x, cyclic_y)
    copula_value = nns_copula(x, nonlinear_y)
    causation = nns_causation(x[:-1], nonlinear_y[1:], tau=1)
    causes = causal_matrix(np.column_stack((x, linear_y, nonlinear_y)), tau=0)

    np.testing.assert_allclose(nns_cor(x, linear_y), linear["Correlation"])
    assert linear["Dependence"] > 0.95
    assert nonlinear["Dependence"] > abs(nonlinear["Correlation"])
    assert cyclic["Dependence"] > abs(cyclic["Correlation"])
    assert 0.0 <= copula_value <= 1.0
    assert any(key.startswith("C(") for key in causation)
    np.testing.assert_allclose(causes, -causes.T)

    print("linear relationship:", linear)
    print("nonlinear relationship:", nonlinear)
    print("cyclic relationship:", cyclic)
    print("copula dependence:", copula_value)
    print("lagged causation summary:", causation)
    print("causal matrix:")
    print(causes)


if __name__ == "__main__":
    main()
