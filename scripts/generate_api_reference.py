from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "docs" / "api_reference.md"

sys.path.insert(0, str(SRC))


@dataclass(frozen=True)
class ApiMeta:
    area: str
    r_name: str
    status: str
    summary: str


META: dict[str, ApiMeta] = {
    "lpm": ApiMeta("Partial moments", "LPM", "implemented", "Lower partial moment."),
    "upm": ApiMeta("Partial moments", "UPM", "implemented", "Upper partial moment."),
    "lpm_ratio": ApiMeta("Partial moments", "LPM.ratio", "implemented", "Lower partial moment ratio."),
    "upm_ratio": ApiMeta("Partial moments", "UPM.ratio", "implemented", "Upper partial moment ratio."),
    "pm_matrix": ApiMeta("Partial moments", "PM.matrix", "implemented", "Partial moment matrix helper."),
    "co_lpm": ApiMeta("Co-moments", "Co.LPM", "implemented", "Pairwise co-lower partial moment."),
    "co_upm": ApiMeta("Co-moments", "Co.UPM", "implemented", "Pairwise co-upper partial moment."),
    "d_lpm": ApiMeta("Co-moments", "D.LPM", "implemented", "Pairwise lower directional partial moment."),
    "d_upm": ApiMeta("Co-moments", "D.UPM", "implemented", "Pairwise upper directional partial moment."),
    "co_lpm_nd": ApiMeta("Co-moments", "N-dimensional co-LPM path", "implemented", "N-dimensional co-lower partial moment wrapper."),
    "co_upm_nd": ApiMeta("Co-moments", "N-dimensional co-UPM path", "implemented", "N-dimensional co-upper partial moment wrapper."),
    "dpm_nd": ApiMeta("Co-moments", "N-dimensional DPM path", "implemented", "N-dimensional directional partial moment wrapper."),
    "ecdf_pm": ApiMeta("Classical moments", "ECDF partial-moment helper", "implemented", "Empirical distribution helper used by moment routines."),
    "mean_pm": ApiMeta("Classical moments", "Mean partial-moment helper", "implemented", "Mean via partial-moment conventions."),
    "var_pm": ApiMeta("Classical moments", "Variance partial-moment helper", "implemented", "Variance via partial-moment conventions."),
    "skew_pm": ApiMeta("Classical moments", "Skewness partial-moment helper", "implemented", "Skewness via partial-moment conventions."),
    "kurt_pm": ApiMeta("Classical moments", "Kurtosis partial-moment helper", "implemented", "Kurtosis via partial-moment conventions."),
    "nns_moments": ApiMeta("Classical moments", "NNS.moments", "implemented", "Bundled NNS moment diagnostics."),
    "nns_dep": ApiMeta("Dependence, correlation, copula, and causation", "NNS.dep", "implemented", "Nonlinear dependence and correlation."),
    "nns_cor": ApiMeta("Dependence, correlation, copula, and causation", "NNS.cor", "implemented", "Nonlinear correlation convenience API."),
    "nns_copula": ApiMeta("Dependence, correlation, copula, and causation", "NNS.copula", "implemented", "Bivariate copula surface."),
    "nns_causation": ApiMeta("Dependence, correlation, copula, and causation", "NNS.caus", "implemented", "Directional causation."),
    "causal_matrix": ApiMeta("Dependence, correlation, copula, and causation", "Causal.matrix", "implemented", "Matrix form of causation relationships."),
    "nns_reg": ApiMeta("Regression, classification, and ensembles", "NNS.reg", "implemented", "Bivariate regression, classification, and interval surfaces."),
    "nns_m_reg": ApiMeta("Regression, classification, and ensembles", "NNS.M.reg", "partial", "Multivariate regression. Raw factor expansion is guarded."),
    "nns_stack": ApiMeta("Regression, classification, and ensembles", "NNS.stack", "implemented", "Stacked ensemble path."),
    "nns_boost": ApiMeta("Regression, classification, and ensembles", "NNS.boost", "partial", "Boosted ensemble path with one high-feature stochastic threshold guard."),
    "FactorDesign": ApiMeta("Categorical and factor helpers", "Python helper", "implemented", "Dataclass returned by factor-preparation helpers."),
    "prepare_factor_predictors": ApiMeta("Categorical and factor helpers", "Factor preparation path", "implemented", "Builds regression-ready factor design matrices."),
    "encode_factor_codes": ApiMeta("Categorical and factor helpers", "Factor-code helper", "implemented", "Encodes categorical labels into deterministic numeric codes."),
    "factor_2_dummy": ApiMeta("Categorical and factor helpers", "factor.2.dummy", "implemented", "Dummy expansion helper."),
    "factor_2_dummy_fr": ApiMeta("Categorical and factor helpers", "full-rank dummy helper", "implemented", "Full-rank dummy expansion helper."),
    "nns_seas": ApiMeta("Forecasting", "NNS.seas", "implemented", "Seasonality detection."),
    "nns_arma": ApiMeta("Forecasting", "NNS.ARMA", "partial", "Univariate ARMA-style forecast surface."),
    "nns_arma_optim": ApiMeta("Forecasting", "NNS.ARMA.optim", "partial", "ARMA optimizer surface."),
    "nns_var": ApiMeta("Forecasting", "NNS.VAR", "partial", "Multivariate VAR-style forecast surface."),
    "nns_cdf": ApiMeta("Distribution, ANOVA, normalization, and distance", "NNS.cdf", "implemented", "NNS empirical CDF path."),
    "nns_anova": ApiMeta("Distribution, ANOVA, normalization, and distance", "NNS.ANOVA", "implemented", "Binary, multi-group, and pairwise ANOVA-style comparisons."),
    "nns_norm": ApiMeta("Distribution, ANOVA, normalization, and distance", "NNS.norm", "implemented", "NNS normalization helper."),
    "nns_distance": ApiMeta("Distribution, ANOVA, normalization, and distance", "NNS.distance", "implemented", "Single distance computation."),
    "nns_distance_bulk": ApiMeta("Distribution, ANOVA, normalization, and distance", "NNS.distance.bulk", "implemented", "Bulk distance computation."),
    "nns_part": ApiMeta("Distribution, ANOVA, normalization, and distance", "NNS.part", "implemented", "Partitioning helper returning Python-native structures."),
    "nns_gravity": ApiMeta("Central tendencies", "NNS.gravity", "implemented", "NNS gravity center helper."),
    "nns_mode": ApiMeta("Central tendencies", "NNS.mode", "implemented", "NNS mode helper."),
    "nns_rescale": ApiMeta("Central tendencies", "NNS.rescale", "implemented", "Rescaling helper."),
    "fsd": ApiMeta("Stochastic dominance, superiority, and simulation", "FSD", "implemented", "First-order stochastic dominance."),
    "ssd": ApiMeta("Stochastic dominance, superiority, and simulation", "SSD", "implemented", "Second-order stochastic dominance."),
    "tsd": ApiMeta("Stochastic dominance, superiority, and simulation", "TSD", "implemented", "Third-order stochastic dominance."),
    "fsd_uni": ApiMeta("Stochastic dominance, superiority, and simulation", "FSD.uni", "implemented", "Univariate FSD wrapper."),
    "ssd_uni": ApiMeta("Stochastic dominance, superiority, and simulation", "SSD.uni", "implemented", "Univariate SSD wrapper."),
    "tsd_uni": ApiMeta("Stochastic dominance, superiority, and simulation", "TSD.uni", "implemented", "Univariate TSD wrapper."),
    "nns_sd_cluster": ApiMeta("Stochastic dominance, superiority, and simulation", "NNS.SD.cluster", "implemented", "Stochastic-dominance clustering."),
    "sd_efficient_set": ApiMeta("Stochastic dominance, superiority, and simulation", "SD.efficient.set", "implemented", "Stochastic-dominance efficient set."),
    "nns_ss": ApiMeta("Stochastic dominance, superiority, and simulation", "NNS.SS", "implemented", "Stochastic superiority."),
    "nns_mc": ApiMeta("Stochastic dominance, superiority, and simulation", "NNS.MC", "implemented", "Monte Carlo helper."),
    "nns_meboot": ApiMeta("Stochastic dominance, superiority, and simulation", "NNS.meboot", "implemented", "Maximum-entropy bootstrap helper."),
    "nns_diff": ApiMeta("Differentiation", "NNS.diff", "implemented", "Numerical differentiation."),
    "dy_dx": ApiMeta("Differentiation", "dy.dx", "implemented", "Scalar derivative helper."),
    "dy_d": ApiMeta("Differentiation", "dy.d", "partial", "Multivariate derivative helper."),
    "lpm_var": ApiMeta("VaR helpers", "LPM.VaR", "implemented", "Lower partial-moment VaR helper."),
    "upm_var": ApiMeta("VaR helpers", "UPM.VaR", "implemented", "Upper partial-moment VaR helper."),
}

AREA_ORDER = [
    "Partial moments",
    "Co-moments",
    "Classical moments",
    "Dependence, correlation, copula, and causation",
    "Regression, classification, and ensembles",
    "Categorical and factor helpers",
    "Forecasting",
    "Distribution, ANOVA, normalization, and distance",
    "Central tendencies",
    "Stochastic dominance, superiority, and simulation",
    "Differentiation",
    "VaR helpers",
    "Other public exports",
]


def object_for_export(nns_module: object, name: str) -> object:
    return getattr(nns_module, name)


def signature_for(obj: object) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"


def first_doc_sentence(obj: object) -> str:
    doc = inspect.getdoc(obj) or ""
    if not doc:
        return ""
    paragraph = doc.strip().split("\n\n", maxsplit=1)[0]
    return " ".join(paragraph.split())


def render() -> str:
    nns = import_module("nns")
    exports = sorted(nns.__all__)

    by_area: dict[str, list[str]] = {area: [] for area in AREA_ORDER}
    for name in exports:
        meta = META.get(name, ApiMeta("Other public exports", "", "", ""))
        by_area.setdefault(meta.area, []).append(name)

    lines: list[str] = []
    lines.append("# NNS Python API Reference Manual")
    lines.append("")
    lines.append("This file is generated by `scripts/generate_api_reference.py` from `nns.__all__` plus curated R API crosswalk metadata.")
    lines.append("")
    lines.append("Run:")
    lines.append("")
    lines.append("```bash")
    lines.append("uv run python scripts/generate_api_reference.py")
    lines.append("```")
    lines.append("")
    lines.append("## Public API crosswalk")
    lines.append("")
    lines.append("| Area | Python API | Closest R NNS API | Status | Notes |")
    lines.append("|---|---|---|---|---|")
    for area in AREA_ORDER:
        for name in by_area.get(area, []):
            meta = META.get(name, ApiMeta(area, "", "", ""))
            lines.append(
                f"| {meta.area} | `{name}` | {meta.r_name or '-'} | {meta.status or '-'} | {meta.summary or '-'} |"
            )
    lines.append("")
    lines.append("## Reference by API area")
    lines.append("")
    lines.append("NNS Python returns Python-native values: NumPy arrays, Python scalars, dataclasses, and plain dictionaries rather than R `data.table` objects.")
    lines.append("")

    for area in AREA_ORDER:
        names = by_area.get(area, [])
        if not names:
            continue
        lines.append(f"### {area}")
        lines.append("")
        for name in names:
            obj = object_for_export(nns, name)
            meta = META.get(name, ApiMeta(area, "", "", ""))
            summary = first_doc_sentence(obj) or meta.summary or "Public NNS Python API."
            lines.append(f"#### `{name}`")
            lines.append("")
            if meta.r_name:
                lines.append(f"Closest R API: `{meta.r_name}`.")
                lines.append("")
            if meta.status:
                lines.append(f"Status: {meta.status}.")
                lines.append("")
            lines.append("Signature:")
            lines.append("")
            lines.append("```python")
            lines.append(f"nns.{name}{signature_for(obj)}")
            lines.append("```")
            lines.append("")
            lines.append(summary)
            lines.append("")
    lines.append("## Documentation maintenance checklist")
    lines.append("")
    lines.append(dedent("""
        When a public API changes:

        1. Update or add the function docstring in `src/nns`.
        2. Update implementation status in `docs/api_status.md` if parity or support changed.
        3. Run `uv run python scripts/generate_api_reference.py`.
        4. Review examples in `docs/examples` if the signature or return shape changed.
        5. Confirm `README.md` still points to this manual and the API status page.
        """).strip())
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
