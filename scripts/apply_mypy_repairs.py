from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected exactly one match, found {count}\nOLD:\n{old}"
        )
    file.write_text(text.replace(old, new, 1))


replace_once(
    "src/nns/_reg_engine.py",
    '    is_xonly = type_value == "xonly"\n\n    if y.dtype.kind == "b":\n',
    '    is_xonly = type_value == "xonly"\n\n'
    '    class_values: NDArray[np.float64] | None\n'
    '    class_levels: list[str] | None\n'
    '    if y.dtype.kind == "b":\n',
)
replace_once(
    "src/nns/_reg_engine.py",
    "    from scipy import stats  # type: ignore[import-untyped]\n",
    "    from scipy import stats\n",
)
replace_once(
    "src/nns/_reg_engine.py",
    '''    if dist == "FACTOR":
        return np.mean(
            rpm_x[None, :, :] != xtest[:, None, :], axis=2, dtype=np.float64
        )
''',
    '''    if dist == "FACTOR":
        return np.asarray(
            np.mean(
                rpm_x[None, :, :] != xtest[:, None, :], axis=2, dtype=np.float64
            ),
            dtype=np.float64,
        )
''',
)
replace_once(
    "src/nns/_reg_engine.py",
    '''    if dist == "L1":
        return np.sum(np.abs(z), axis=2)
    return np.sqrt(np.sum(z * z, axis=2))
''',
    '''    if dist == "L1":
        return np.asarray(np.sum(np.abs(z), axis=2), dtype=np.float64)
    return np.asarray(np.sqrt(np.sum(z * z, axis=2)), dtype=np.float64)
''',
)

for path in ("src/nns/stack.py", "src/nns/boost.py"):
    replace_once(
        path,
        '''    seed_value = (
        _scalar_integer(seed, "seed", minimum=0)
        if seed is not None
        else int(np.random.SeedSequence().generate_state(1)[0])
    )
''',
        '''    seed_value = (
        cast(int, _scalar_integer(seed, "seed", minimum=0))
        if seed is not None
        else int(np.random.SeedSequence().generate_state(1)[0])
    )
''',
    )

replace_once(
    "src/nns/stack.py",
    '''            final_train: NDArray[np.float64] = np.column_stack(
                (dim_full_xstar_train, dim_full_xstar_train)
            )
            final_test: NDArray[np.float64] = np.column_stack(
                (dim_full_xstar_test, dim_full_xstar_test)
            )
''',
    '''            dim_train = cast(NDArray[np.float64], dim_full_xstar_train)
            dim_test = cast(NDArray[np.float64], dim_full_xstar_test)
            final_train: NDArray[np.float64] = np.column_stack((dim_train, dim_train))
            final_test: NDArray[np.float64] = np.column_stack((dim_test, dim_test))
''',
)

replace_once(
    "src/nns/boost.py",
    "        cols = list(feature_index)\n",
    "        cols = np.asarray(feature_index, dtype=np.int64)\n",
)

replace_once(
    "tests/parity/test_regression.py",
    "        np.testing.assert_allclose(actual, _array(expected), atol=atol)\n",
    "        np.testing.assert_allclose(\n"
    "            np.asarray(actual, dtype=np.float64), _array(expected), atol=atol\n"
    "        )\n",
)

replace_once(
    "tests/parity/test_multivariate_regression.py",
    "from typing import Any, cast\n",
    "from collections.abc import Iterable\nfrom typing import Any, cast\n",
)
replace_once(
    "tests/parity/test_multivariate_regression.py",
    '''    a = list(actual)  # type: ignore[arg-type]
    e = list(expected)  # type: ignore[arg-type]
''',
    '''    a = list(cast(Iterable[object], actual))
    e = list(cast(Iterable[object], expected))
''',
)

replace_once(
    "src/nns/_public_reg.py",
    "from typing import Any\n",
    "from typing import Any, cast\n",
)
replace_once(
    "src/nns/_public_reg.py",
    "    result = _nns_reg(\n",
    "    result = cast(Any, _nns_reg)(\n",
)
