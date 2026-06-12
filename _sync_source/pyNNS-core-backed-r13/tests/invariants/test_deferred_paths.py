from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "pynns"
API_STATUS = ROOT / "docs" / "api_status.md"


EXPECTED_DEFERRED_FRAGMENTS = {
    "threshold on the n_features > 10 stochastic epoch path": (
        "`threshold` on the `n_features > 10` stochastic path"
    ),
    "direct nns_m_reg factor_2_dummy=True": "direct `factor_2_dummy=True` raw predictor path",
}


def test_production_notimplemented_guards_are_documented() -> None:
    messages = _production_notimplemented_messages()
    docs = API_STATUS.read_text(encoding="utf-8")

    assert messages
    stale_mappings = [
        message_fragment
        for message_fragment in EXPECTED_DEFERRED_FRAGMENTS
        if not any(message_fragment in message for message in messages)
    ]
    assert stale_mappings == []

    unmapped = [
        message
        for message in messages
        if not any(fragment in message for fragment in EXPECTED_DEFERRED_FRAGMENTS)
    ]
    assert unmapped == []

    missing_docs = [
        docs_fragment
        for message_fragment, docs_fragment in EXPECTED_DEFERRED_FRAGMENTS.items()
        if any(message_fragment in message for message in messages) and docs_fragment not in docs
    ]
    assert missing_docs == []


def _production_notimplemented_messages() -> set[str]:
    messages: set[str] = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise):
                message = _notimplemented_message(node.exc)
                if message is not None:
                    messages.add(message)
    return messages


def _notimplemented_message(expr: ast.expr | None) -> str | None:
    if not isinstance(expr, ast.Call):
        return None
    if not isinstance(expr.func, ast.Name) or expr.func.id != "NotImplementedError":
        return None
    if not expr.args:
        return ""
    return _literal_message(expr.args[0])


def _literal_message(expr: ast.expr) -> str:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.JoinedStr):
        return "".join(
            part.value
            for part in expr.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return _literal_message(expr.left) + _literal_message(expr.right)
    raise AssertionError(f"NotImplementedError message must be a static string: {ast.dump(expr)}")
