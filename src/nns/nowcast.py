from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime

import numpy as np
from numpy.typing import NDArray

from nns.var import nns_var

_DEFAULT_NOWCAST_SERIES = (
    "PAYEMS",
    "JTSJOL",
    "CPIAUCSL",
    "DGORDER",
    "RSAFS",
    "UNRATE",
    "HOUST",
    "INDPRO",
    "DSPIC96",
    "BOPTEXP",
    "BOPTIMP",
    "TTLCONS",
    "IR",
    "CPILFESL",
    "PCEPILFE",
    "PCEPI",
    "PERMIT",
    "TCU",
    "BUSINV",
    "ULCNFB",
    "IQ",
    "GACDISA066MSFRBNY",
    "GACDFSA066MSFRBPHI",
    "PCEC96",
    "GDPC1",
    "ICSA",
    "DGS10",
    "T10Y2Y",
    "WALCL",
    "PALLFNFINDEXM",
    "FEDFUNDS",
    "PPIACO",
    "CIVPART",
    "M2NS",
    "ADPMNUSNERNSA",
)


def nns_nowcast_panel(
    panel: object,
    *,
    h: int = 0,
    tau: int | list[int] | list[list[int]] = 12,
    dim_red_method: str = "cor",
    naive_weights: bool = False,
    dates: Sequence[object] | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Deterministic nowcast core for user-supplied monthly panels."""
    if h < 0:
        raise ValueError("h must be non-negative.")

    matrix, panel_names = _panel_matrix_and_names(panel, names)
    observed_dates, forecast_dates = _normalize_nowcast_dates(dates, matrix.shape[0], h)

    result = nns_var(
        matrix,
        h,
        tau=tau,
        dim_red_method=dim_red_method,
        naive_weights=naive_weights,
    )
    output: dict[str, object] = dict(result)
    output["names"] = panel_names
    if "relevant_variables" in output:
        output["relevant_variables"] = _rename_relevant_variables(
            output["relevant_variables"],
            panel_names,
        )
    output["dates"] = {
        "observed": observed_dates,
        "forecast": forecast_dates,
        "interpolated_and_extrapolated": observed_dates,
    }
    output["metadata"] = {
        "source": "user_panel",
        "freq": "monthly",
        "tau": tau,
        "dim_red_method": dim_red_method,
        "naive_weights": naive_weights,
    }
    return output


def _panel_matrix_and_names(
    panel: object,
    names: Sequence[str] | None,
) -> tuple[NDArray[np.float64], list[str]]:
    if isinstance(panel, Mapping):
        if names is not None:
            raise ValueError("names cannot be provided when panel is a mapping.")
        panel_names = [str(key) for key in panel]
        columns = [np.asarray(values, dtype=np.float64).reshape(-1) for values in panel.values()]
        if not columns:
            raise ValueError("panel must contain at least one column.")
        row_count = columns[0].size
        if any(column.size != row_count for column in columns):
            raise ValueError("mapping panel columns must have equal lengths.")
        matrix = np.column_stack(columns)
    else:
        matrix = np.asarray(panel, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("panel must be a 2-D numeric matrix or an ordered mapping of columns.")
        panel_names = [f"x{i + 1}" for i in range(matrix.shape[1])]

    if matrix.ndim != 2:
        raise ValueError("panel must be a 2-D numeric matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("panel must be non-empty.")

    if names is not None:
        if len(names) != matrix.shape[1]:
            raise ValueError("names length must match panel column count.")
        panel_names = [str(name) for name in names]

    return matrix.astype(np.float64, copy=False), panel_names


def _normalize_nowcast_dates(
    dates: Sequence[object] | None,
    row_count: int,
    h: int,
) -> tuple[list[str] | None, list[str]]:
    if dates is None:
        return None, [f"t+{step}" for step in range(1, h + 1)]
    if len(dates) != row_count:
        raise ValueError("dates length must match panel row count.")

    observed = [_normalize_month_label(value) for value in dates]
    if len(set(observed)) != len(observed):
        raise ValueError("dates must not contain duplicate months.")
    if observed != sorted(observed):
        raise ValueError("dates must be sorted in ascending monthly order.")
    return observed, _forecast_month_labels(observed[-1], h)


def _normalize_month_label(value: object) -> str:
    if isinstance(value, np.datetime64):
        return str(value.astype("datetime64[M]"))
    if isinstance(value, datetime | date):
        return f"{value.year:04d}-{value.month:02d}"
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text)
        return f"{parsed.year:04d}-{parsed.month:02d}"
    except ValueError:
        pass
    try:
        parsed_month = np.datetime64(text, "M")
    except ValueError as exc:
        raise ValueError("dates must be parseable as monthly date labels.") from exc
    return str(parsed_month)


def _forecast_month_labels(last_observed: str, h: int) -> list[str]:
    year_text, month_text = last_observed.split("-")
    year = int(year_text)
    month = int(month_text)
    labels: list[str] = []
    for _ in range(h):
        month += 1
        if month > 12:
            year += 1
            month = 1
        labels.append(f"{year:04d}-{month:02d}")
    return labels


def _rename_relevant_variables(values: object, names: Sequence[str]) -> object:
    mapping = {f"x{i + 1}": name for i, name in enumerate(names)}
    array = np.asarray(values, dtype=object).copy()
    for index, item in np.ndenumerate(array):
        if item is None:
            continue
        text = str(item)
        for old, new in mapping.items():
            if text == old:
                text = new
            elif text.startswith(f"{old}_tau_"):
                text = f"{new}{text[len(old) :]}"
        array[index] = text
    return array
