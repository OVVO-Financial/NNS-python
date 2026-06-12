from __future__ import annotations

import csv
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from pynns.nowcast import _normalize_month_label


class CsvNowcastProvider:
    """Local CSV provider for deterministic nowcast panels."""

    def __init__(
        self,
        path: str | Path,
        *,
        date_column: str = "date",
        series_columns: Sequence[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.date_column = date_column
        self.series_columns = (
            None if series_columns is None else [str(name) for name in series_columns]
        )

    def fetch(self, series: Sequence[str], start_date: str) -> dict[str, object]:
        del series
        rows, fieldnames = self._read_rows()
        selected_columns = self._selected_columns(fieldnames)
        dates, values = self._parse_rows(rows, selected_columns, start_date)
        return {
            "dates": dates,
            "series": values,
            "metadata": {
                "provider": "csv",
                "path": str(self.path),
                "date_column": self.date_column,
                "series_columns": selected_columns,
            },
        }

    def _read_rows(self) -> tuple[list[Mapping[str, str]], list[str]]:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV nowcast provider file does not exist: {self.path}")
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("CSV nowcast provider file is empty.")
            fieldnames = [str(name) for name in reader.fieldnames]
            rows = cast(list[Mapping[str, str]], list(reader))
        if not rows:
            raise ValueError("CSV nowcast provider file has no data rows.")
        if self.date_column not in fieldnames:
            raise ValueError(f"CSV nowcast provider missing date column: {self.date_column}")
        return rows, fieldnames

    def _selected_columns(self, fieldnames: Sequence[str]) -> list[str]:
        if self.series_columns is None:
            selected = [name for name in fieldnames if name != self.date_column]
        else:
            selected = list(self.series_columns)
            missing = [name for name in selected if name not in fieldnames]
            if missing:
                raise ValueError(
                    f"CSV nowcast provider missing selected series column: {missing[0]}"
                )
        if not selected:
            raise ValueError("CSV nowcast provider requires at least one usable series column.")
        return selected

    def _parse_rows(
        self,
        rows: Sequence[Mapping[str, str]],
        selected_columns: Sequence[str],
        start_date: str,
    ) -> tuple[list[str], OrderedDict[str, list[float | None]]]:
        start_month = _normalize_month_label(start_date)
        dates: list[str] = []
        values: OrderedDict[str, list[float | None]] = OrderedDict(
            (name, []) for name in selected_columns
        )
        for row_number, row in enumerate(rows, start=2):
            raw_date = row.get(self.date_column)
            if raw_date is None:
                raise ValueError(f"CSV nowcast provider row {row_number} is missing a date value.")
            month = _normalize_month_label(raw_date)
            if month < start_month:
                continue
            dates.append(month)
            for column in selected_columns:
                values[column].append(_parse_optional_float(row.get(column), column, row_number))

        if not dates:
            raise ValueError("CSV nowcast provider has no rows on or after start_date.")
        if len(set(dates)) != len(dates):
            raise ValueError("CSV nowcast provider dates must not contain duplicate months.")
        if dates != sorted(dates):
            raise ValueError(
                "CSV nowcast provider dates must be sorted in ascending monthly order."
            )
        lengths = {len(column_values) for column_values in values.values()}
        if lengths != {len(dates)}:
            raise ValueError("CSV nowcast provider series columns must have equal lengths.")
        return dates, values


def _parse_optional_float(value: Any, column: str, row_number: int) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"na", "nan", "null", "none"}:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"CSV nowcast provider column {column!r} row {row_number} "
            f"contains a nonnumeric value: {value!r}"
        ) from exc
