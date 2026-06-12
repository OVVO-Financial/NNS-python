from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pytest

from pynns import nns_nowcast_panel, nns_var
from pynns.providers import CsvNowcastProvider


def _panel() -> np.ndarray:
    idx = np.arange(1, 40, dtype=np.float64)
    return np.column_stack(
        (
            np.sin(idx / 3.0) + 2.0,
            np.cos(idx / 5.0) + 3.0,
        )
    )


def test_nns_nowcast_panel_array_h0_matches_var_core() -> None:
    panel = _panel()

    actual = nns_nowcast_panel(panel, h=0, tau=2)
    expected = nns_var(panel, h=0, tau=2)

    assert set(actual) == {
        "interpolated_and_extrapolated",
        "names",
        "dates",
        "metadata",
    }
    np.testing.assert_allclose(
        actual["interpolated_and_extrapolated"],
        expected["interpolated_and_extrapolated"],
    )
    assert actual["names"] == ["x1", "x2"]
    assert actual["dates"] == {
        "observed": None,
        "forecast": [],
        "interpolated_and_extrapolated": None,
    }
    assert actual["metadata"] == {
        "source": "user_panel",
        "freq": "monthly",
        "tau": 2,
        "dim_red_method": "cor",
        "naive_weights": False,
    }


def test_nns_nowcast_panel_array_h3_matches_var_core() -> None:
    panel = _panel()

    actual = nns_nowcast_panel(panel, h=3, tau=2, dim_red_method="NNS.dep")
    expected = nns_var(panel, h=3, tau=2, dim_red_method="NNS.dep", naive_weights=False)

    assert set(actual) == {
        "interpolated_and_extrapolated",
        "relevant_variables",
        "univariate",
        "multivariate",
        "ensemble",
        "names",
        "dates",
        "metadata",
    }
    for key in ("interpolated_and_extrapolated", "univariate", "multivariate", "ensemble"):
        np.testing.assert_allclose(actual[key], expected[key])
    assert np.array_equal(actual["relevant_variables"], expected["relevant_variables"])
    assert actual["names"] == ["x1", "x2"]
    assert actual["dates"]["observed"] is None
    assert actual["dates"]["forecast"] == ["t+1", "t+2", "t+3"]
    assert actual["dates"]["interpolated_and_extrapolated"] is None


def test_nns_nowcast_panel_mapping_preserves_column_order_and_names() -> None:
    panel = OrderedDict(
        (
            ("PAYEMS", [1.0, 2.0, 3.0, 4.0, 5.0]),
            ("GDPC1", [2.0, 3.0, 4.0, 5.0, 6.0]),
        )
    )

    actual = nns_nowcast_panel(panel, h=0, tau=1)

    assert actual["names"] == ["PAYEMS", "GDPC1"]
    np.testing.assert_allclose(
        actual["interpolated_and_extrapolated"],
        np.column_stack((panel["PAYEMS"], panel["GDPC1"])),
    )


def test_nns_nowcast_panel_rejects_mismatched_names() -> None:
    with pytest.raises(ValueError, match="names length"):
        nns_nowcast_panel(_panel(), h=0, names=["only_one"])


def test_nns_nowcast_panel_normalizes_dates_and_forecast_months() -> None:
    panel = _panel()
    dates = ["2020-01-15", "2020-02", np.datetime64("2020-03-31")]
    dates.extend(f"2020-{month:02d}" for month in range(4, 13))
    dates.extend(f"2021-{month:02d}" for month in range(1, 13))
    dates.extend(f"2022-{month:02d}" for month in range(1, 13))
    dates.extend(f"2023-{month:02d}" for month in range(1, 4))

    actual = nns_nowcast_panel(
        panel,
        h=2,
        tau=1,
        dates=dates,
    )

    assert actual["dates"]["observed"][:3] == ["2020-01", "2020-02", "2020-03"]
    assert actual["dates"]["forecast"] == ["2023-04", "2023-05"]
    assert actual["dates"]["interpolated_and_extrapolated"] == actual["dates"]["observed"]


def test_nns_nowcast_panel_rejects_invalid_dates() -> None:
    panel = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="dates length"):
        nns_nowcast_panel(panel, h=0, dates=["2020-01"])
    with pytest.raises(ValueError, match="duplicate"):
        nns_nowcast_panel(panel, h=0, dates=["2020-01", "2020-01", "2020-02"])
    with pytest.raises(ValueError, match="sorted"):
        nns_nowcast_panel(panel, h=0, dates=["2020-02", "2020-01", "2020-03"])


def test_nns_nowcast_panel_missing_values_delegate_to_var() -> None:
    panel = _panel()
    panel[4, 0] = np.nan
    panel[-1, 1] = np.nan

    actual = nns_nowcast_panel(panel, h=3, tau=2)

    assert np.all(np.isfinite(actual["interpolated_and_extrapolated"]))
    assert np.all(np.isfinite(actual["univariate"]))
    assert np.all(np.isfinite(actual["multivariate"]))
    assert np.all(np.isfinite(actual["ensemble"]))


def _provider_payload() -> dict[str, Any]:
    panel = _panel()
    return {
        "dates": [f"2020-{month:02d}" for month in range(1, 13)]
        + [f"2021-{month:02d}" for month in range(1, 13)]
        + [f"2022-{month:02d}" for month in range(1, 13)]
        + [f"2023-{month:02d}" for month in range(1, 4)],
        "series": OrderedDict(
            (
                ("PAYEMS", panel[:, 0].tolist()),
                ("UNRATE", panel[:, 1].tolist()),
            )
        ),
        "metadata": {"provider": "fixture"},
    }


def test_provider_payload_feeds_nowcast_panel_core() -> None:
    payload = _provider_payload()

    actual = nns_nowcast_panel(
        payload["series"],
        h=2,
        tau=12,
        dates=payload["dates"],
        naive_weights=False,
    )
    expected = nns_nowcast_panel(
        payload["series"],
        h=2,
        tau=12,
        dates=payload["dates"],
        naive_weights=False,
    )

    assert actual["names"] == ["PAYEMS", "UNRATE"]
    assert actual["dates"]["forecast"] == ["2023-04", "2023-05"]
    for key in ("interpolated_and_extrapolated", "univariate", "multivariate", "ensemble"):
        np.testing.assert_allclose(actual[key], expected[key])
    assert np.array_equal(actual["relevant_variables"], expected["relevant_variables"])


def test_csv_nowcast_provider_returns_payload(tmp_path: Any) -> None:
    csv_path = tmp_path / "macro.csv"
    csv_path.write_text(
        "date,PAYEMS,UNRATE\n2020-01-15,1.0,4.0\n2020-02,2.0,5.0\n2020-03-31,3.0,6.0\n",
        encoding="utf-8",
    )

    payload = CsvNowcastProvider(csv_path).fetch(("PAYEMS",), "2020-01-01")

    assert payload["dates"] == ["2020-01", "2020-02", "2020-03"]
    series_payload = cast(Mapping[str, object], payload["series"])
    assert list(series_payload) == ["PAYEMS", "UNRATE"]
    assert payload["series"] == OrderedDict(
        (
            ("PAYEMS", [1.0, 2.0, 3.0]),
            ("UNRATE", [4.0, 5.0, 6.0]),
        )
    )
    assert payload["metadata"] == {
        "provider": "csv",
        "path": str(csv_path),
        "date_column": "date",
        "series_columns": ["PAYEMS", "UNRATE"],
    }


def test_csv_provider_payload_matches_panel_core(tmp_path: Any) -> None:
    csv_path = tmp_path / "macro.csv"
    panel = _panel()
    rows = ["date,PAYEMS,UNRATE"]
    for index, month in enumerate(
        [f"2020-{month:02d}" for month in range(1, 13)]
        + [f"2021-{month:02d}" for month in range(1, 13)]
        + [f"2022-{month:02d}" for month in range(1, 13)]
        + [f"2023-{month:02d}" for month in range(1, 4)]
    ):
        rows.append(f"{month},{panel[index, 0]},{panel[index, 1]}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    payload = CsvNowcastProvider(csv_path).fetch((), "2000-01-03")
    actual = nns_nowcast_panel(
        payload["series"],
        h=2,
        tau=12,
        dates=cast(Sequence[object], payload["dates"]),
    )
    expected = nns_nowcast_panel(
        OrderedDict(
            (
                ("PAYEMS", panel[:, 0].tolist()),
                ("UNRATE", panel[:, 1].tolist()),
            )
        ),
        h=2,
        tau=12,
        dates=[row.split(",", maxsplit=1)[0] for row in rows[1:]],
    )

    assert actual["names"] == ["PAYEMS", "UNRATE"]
    assert actual["dates"]["forecast"] == ["2023-04", "2023-05"]
    for key in ("interpolated_and_extrapolated", "univariate", "multivariate", "ensemble"):
        np.testing.assert_allclose(actual[key], expected[key])
    assert np.array_equal(actual["relevant_variables"], expected["relevant_variables"])


def test_csv_nowcast_provider_selects_and_orders_series_columns(tmp_path: Any) -> None:
    csv_path = tmp_path / "macro.csv"
    csv_path.write_text(
        "date,PAYEMS,UNRATE,GDPC1\n2020-01,1.0,4.0,7.0\n2020-02,2.0,5.0,8.0\n",
        encoding="utf-8",
    )

    payload = CsvNowcastProvider(csv_path, series_columns=["GDPC1", "PAYEMS"]).fetch((), "2020-01")

    series_payload = cast(Mapping[str, object], payload["series"])
    assert list(series_payload) == ["GDPC1", "PAYEMS"]
    assert payload["series"] == OrderedDict((("GDPC1", [7.0, 8.0]), ("PAYEMS", [1.0, 2.0])))


def test_csv_nowcast_provider_parses_missing_values(tmp_path: Any) -> None:
    csv_path = tmp_path / "macro.csv"
    csv_path.write_text(
        "date,PAYEMS,UNRATE\n2020-01,1.0,\n2020-02,NA,5.0\n2020-03,nan,null\n",
        encoding="utf-8",
    )

    payload = CsvNowcastProvider(csv_path).fetch((), "2020-01")

    assert payload["series"] == OrderedDict(
        (
            ("PAYEMS", [1.0, None, None]),
            ("UNRATE", [None, 5.0, None]),
        )
    )


def test_csv_nowcast_provider_filters_start_date(tmp_path: Any) -> None:
    csv_path = tmp_path / "macro.csv"
    csv_path.write_text(
        "date,PAYEMS\n2020-01,1.0\n2020-02,2.0\n2020-03,3.0\n",
        encoding="utf-8",
    )

    payload = CsvNowcastProvider(csv_path).fetch((), "2020-02-15")

    assert payload["dates"] == ["2020-02", "2020-03"]
    assert payload["series"] == OrderedDict((("PAYEMS", [2.0, 3.0]),))


def test_csv_nowcast_provider_rejects_bad_dates(tmp_path: Any) -> None:
    duplicate_path = tmp_path / "duplicate.csv"
    duplicate_path.write_text(
        "date,PAYEMS\n2020-01,1.0\n2020-01,2.0\n",
        encoding="utf-8",
    )
    unsorted_path = tmp_path / "unsorted.csv"
    unsorted_path.write_text(
        "date,PAYEMS\n2020-02,2.0\n2020-01,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        CsvNowcastProvider(duplicate_path).fetch((), "2020-01")
    with pytest.raises(ValueError, match="sorted"):
        CsvNowcastProvider(unsorted_path).fetch((), "2020-01")


def test_csv_nowcast_provider_rejects_missing_columns(tmp_path: Any) -> None:
    csv_path = tmp_path / "macro.csv"
    csv_path.write_text(
        "month,PAYEMS\n2020-01,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing date column"):
        CsvNowcastProvider(csv_path).fetch((), "2020-01")
    with pytest.raises(ValueError, match="missing selected series column"):
        CsvNowcastProvider(csv_path, date_column="month", series_columns=["UNRATE"]).fetch(
            (), "2020-01"
        )


def test_csv_nowcast_provider_rejects_nonnumeric_values(tmp_path: Any) -> None:
    csv_path = tmp_path / "macro.csv"
    csv_path.write_text(
        "date,PAYEMS\n2020-01,bad\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="nonnumeric"):
        CsvNowcastProvider(csv_path).fetch((), "2020-01")


def test_csv_nowcast_provider_rejects_empty_or_no_series_csv(tmp_path: Any) -> None:
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")
    header_only_path = tmp_path / "header_only.csv"
    header_only_path.write_text("date,PAYEMS\n", encoding="utf-8")
    no_series_path = tmp_path / "no_series.csv"
    no_series_path.write_text(
        "date\n2020-01\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="empty"):
        CsvNowcastProvider(empty_path).fetch((), "2020-01")
    with pytest.raises(ValueError, match="no data rows"):
        CsvNowcastProvider(header_only_path).fetch((), "2020-01")
    with pytest.raises(ValueError, match="at least one usable series"):
        CsvNowcastProvider(no_series_path).fetch((), "2020-01")
