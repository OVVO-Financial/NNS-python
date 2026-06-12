# Native original C++ source coverage audit

This document audits the vendored C++ core under `extern/NNS-core/include/nns` and
`extern/NNS-core/src`. The goal is targeted native coverage for original C++ core
source files, not binding the whole Python package and not changing PyPI packaging.

Status values used below:

- `bound-public`: exposed through `_nnscore` and routed from an existing public Python API.
- `bound-private`: exposed through private `_nnscore` bindings for backend support/smoke tests.
- `cxx-exists-unbound`: present in C++ but intentionally not bound in this PR.
- `python-only`: Python implementation exists without a direct C++ binding in this PR.
- `no-python-wrapper`: no existing public Python wrapper was found.
- `internal-helper`: helper intentionally treated as private backend support.
- `unclear`: semantics/shape mapping need more audit before binding.

## Audited C++ files

- `extern/NNS-core/src/partial_moments.cpp`
- `extern/NNS-core/src/central_tendencies.cpp`
- `extern/NNS-core/src/fast_lm.cpp`
- `extern/NNS-core/src/internal_functions.cpp`
- `extern/NNS-core/src/dependence.cpp`
- `extern/NNS-core/src/distance.cpp`
- `extern/NNS-core/src/partition.cpp`
- `extern/NNS-core/src/seasonality.cpp`
- `extern/NNS-core/src/stochastic_dominance.cpp`

## Coverage table

| C++ header | C++ function or type | C++ source file | Existing Python public function | Existing Python module | Currently bound in `_nnscore` | Should be public Python API | Should be private backend helper only | Binding priority | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `partial_moments.hpp` | `PMMatrixResult` | `partial_moments.cpp` | `pm_matrix` result dict | `nns.pm_matrix` | bound-public | yes | no | Phase 1 | Bound as dict preserving Python `cov.matrix` key. |
| `partial_moments.hpp` | `lpm` | `partial_moments.cpp` | `lpm` | `nns.core` | bound-public | yes | no | Phase 1 | Existing binding confirmed and routed. |
| `partial_moments.hpp` | `upm` | `partial_moments.cpp` | `upm` | `nns.core` | bound-public | yes | no | Phase 1 | Existing binding confirmed and routed. |
| `partial_moments.hpp` | `lpm_v` | `partial_moments.cpp` | `lpm` vector target path | `nns.core` | bound-public | yes | no | Phase 1 | Also exposed as private explicit `_nnscore.lpm_v`. |
| `partial_moments.hpp` | `upm_v` | `partial_moments.cpp` | `upm` vector target path | `nns.core` | bound-public | yes | no | Phase 1 | Also exposed as private explicit `_nnscore.upm_v`. |
| `partial_moments.hpp` | `lpm_ratio_v` | `partial_moments.cpp` | `lpm_ratio` | `nns.core` | bound-public | yes | no | Phase 1 | Routed through native when available. |
| `partial_moments.hpp` | `upm_ratio_v` | `partial_moments.cpp` | `upm_ratio` | `nns.core` | bound-public | yes | no | Phase 1 | Routed through native when available. |
| `partial_moments.hpp` | `co_lpm` | `partial_moments.cpp` | `co_lpm` | `nns.co_moments` | bound-public | yes | no | Phase 1 | Scalar smoke binding plus vector route. |
| `partial_moments.hpp` | `co_upm` | `partial_moments.cpp` | `co_upm` | `nns.co_moments` | bound-public | yes | no | Phase 1 | Scalar smoke binding plus vector route. |
| `partial_moments.hpp` | `d_lpm` | `partial_moments.cpp` | `d_lpm` | `nns.co_moments` | bound-public | yes | no | Phase 1 | Scalar smoke binding plus vector route. |
| `partial_moments.hpp` | `d_upm` | `partial_moments.cpp` | `d_upm` | `nns.co_moments` | bound-public | yes | no | Phase 1 | Scalar smoke binding plus vector route. |
| `partial_moments.hpp` | `co_lpm_v` | `partial_moments.cpp` | `co_lpm` vector target path | `nns.co_moments` | bound-public | yes | no | Phase 1 | Preserves recycled target vector behavior. |
| `partial_moments.hpp` | `co_upm_v` | `partial_moments.cpp` | `co_upm` vector target path | `nns.co_moments` | bound-public | yes | no | Phase 1 | Preserves recycled target vector behavior. |
| `partial_moments.hpp` | `d_lpm_v` | `partial_moments.cpp` | `d_lpm` vector target path | `nns.co_moments` | bound-public | yes | no | Phase 1 | Preserves recycled target vector behavior. |
| `partial_moments.hpp` | `d_upm_v` | `partial_moments.cpp` | `d_upm` vector target path | `nns.co_moments` | bound-public | yes | no | Phase 1 | Preserves recycled target vector behavior. |
| `partial_moments.hpp` | `clpm_nd` | `partial_moments.cpp` | `co_lpm_nd` | `nns.dependence` | bound-private | yes | no | Phase 1 | Bound for native coverage; public routing deferred because current Python shape semantics need separate parity work. |
| `partial_moments.hpp` | `cupm_nd` | `partial_moments.cpp` | `co_upm_nd` | `nns.dependence` | bound-private | yes | no | Phase 1 | Bound for native coverage; public routing deferred. |
| `partial_moments.hpp` | `dpm_nd` | `partial_moments.cpp` | `dpm_nd` | `nns.dependence` | bound-private | yes | no | Phase 1 | Bound for native coverage; public routing deferred. |
| `partial_moments.hpp` | `clpm_nd_batch` | `partial_moments.cpp` | none | none | bound-private | no | yes | Phase 1 | Backend vectorized helper only. |
| `partial_moments.hpp` | `pm_matrix` | `partial_moments.cpp` | `pm_matrix` | `nns.pm_matrix` | bound-public | yes | no | Phase 1 | Routed through native with column-major flattening. |
| `central_tendencies.hpp` | `gravity` | `central_tendencies.cpp` | `nns_gravity` | `nns.central_tendencies` | bound-private | yes | no | Phase 5 | Already bound before this PR; public routing was already present/available through module behavior. |
| `central_tendencies.hpp` | `rescale` | `central_tendencies.cpp` | `nns_rescale` | `nns.central_tendencies` | cxx-exists-unbound | yes | no | Phase 5 | Left unbound to avoid changing risk-neutral/min-max edge behavior without parity tests. |
| `central_tendencies.hpp` | `mode` | `central_tendencies.cpp` | `nns_mode` | `nns.central_tendencies` | bound-private | yes | no | Phase 5 | Already bound before this PR. |
| `fast_lm.hpp` | `FastLmResult` | `fast_lm.cpp` | `_fast_lm` result dict | `nns.multivariate_regression` | bound-private | no | yes | Phase 2 | Existing `fast_lm` binding confirmed. |
| `fast_lm.hpp` | `FastLmMultResult` | `fast_lm.cpp` | none found | none | bound-private | no | yes | Phase 2 | Added native binding; no public route because no existing public wrapper uses it directly. |
| `fast_lm.hpp` | `fast_lm` | `fast_lm.cpp` | `_fast_lm` helper | `nns.multivariate_regression` | bound-private | no | yes | Phase 2 | Existing binding confirmed; remains backend-only. |
| `fast_lm.hpp` | `fast_lm_mult` | `fast_lm.cpp` | none found | none | bound-private | no | yes | Phase 2 | Added smoke-tested backend binding. |
| `internal_functions.hpp` | `ValueKind` | `internal_functions.cpp` | none | none | cxx-exists-unbound | no | yes | Phase 3 | Enum is only useful if `is_fcl` is exposed; Python has no direct type mapping need. |
| `internal_functions.hpp` | `is_fcl` | `internal_functions.cpp` | `_is_fcl` internal equivalent | `nns.regression` | cxx-exists-unbound | no | yes | Phase 3 | Not bound; Python object dtype/factor detection is richer than the C++ enum boundary. |
| `internal_functions.hpp` | `Factor` | `internal_functions.cpp` | factor helpers | `nns.categorical` | bound-private | no | yes | Phase 3 | Mapped to `(codes, levels)` arguments, not exposed as a C++ class. |
| `internal_functions.hpp` | `DummyMatrix` | `internal_functions.cpp` | factor helpers | `nns.categorical` | bound-private | no | yes | Phase 3 | Returned as dict with flat column-major data, names, nrow, ncol. |
| `internal_functions.hpp` | `factor_2_dummy` | `internal_functions.cpp` | `factor_2_dummy` | `nns.categorical` | bound-private | yes | yes | Phase 3 | Bound only as private backend helper; public routing deferred. |
| `internal_functions.hpp` | `factor_2_dummy_fr` | `internal_functions.cpp` | `factor_2_dummy_fr` | `nns.categorical` | bound-private | yes | yes | Phase 3 | Bound only as private backend helper; public routing deferred. |
| `internal_functions.hpp` | `vec_sd` | `internal_functions.cpp` | none public | none | bound-private | no | yes | Phase 3 | Safe numeric helper bound for backend use. |
| `internal_functions.hpp` | `col_sd` | `internal_functions.cpp` | none public | none | bound-private | no | yes | Phase 3 | Safe numeric helper bound for backend use with explicit dimensions. |
| `internal_functions.hpp` | `is_discrete` | `internal_functions.cpp` | internal checks | multiple | bound-private | no | yes | Phase 3 | Safe numeric helper bound for backend use. |
| `internal_functions.hpp` | `TimeSeriesVectors` | `internal_functions.cpp` | none public | none | bound-private | no | yes | Phase 3 | Dict result for private backend support. |
| `internal_functions.hpp` | `ForecastVectors` | `internal_functions.cpp` | none public | none | bound-private | no | yes | Phase 3 | Dict result for private backend support. |
| `internal_functions.hpp` | `generate_vectors` | `internal_functions.cpp` | none public | none | bound-private | no | yes | Phase 3 | Safe explicit vector/list conversion. |
| `internal_functions.hpp` | `generate_lin_vectors` | `internal_functions.cpp` | none public | none | bound-private | no | yes | Phase 3 | Safe explicit vector/list conversion. |
| `internal_functions.hpp` | `ARMAWeights` | `internal_functions.cpp` | ARMA internals | `nns.arma` | cxx-exists-unbound | no | yes | Phase 4 | Left unbound; structured ARMA weighting semantics need parity tests. |
| `internal_functions.hpp` | `arma_seas_weighting` | `internal_functions.cpp` | ARMA internals | `nns.arma` | cxx-exists-unbound | no | yes | Phase 4 | Left unbound because period/covariance frame semantics need separate validation. |
| `internal_functions.hpp` | `meboot_part` | `internal_functions.cpp` | `nns_meboot` internals | `nns.meboot` | cxx-exists-unbound | no | yes | Phase 4 | Left unbound because it has random seed and boundary semantics requiring dedicated parity tests. |
| `internal_functions.hpp` | `meboot_expand_sd` | `internal_functions.cpp` | `nns_meboot` internals | `nns.meboot` | cxx-exists-unbound | no | yes | Phase 4 | Left unbound because it mutates column-major ensemble buffers in place. |
| `internal_functions.hpp` | `force_clt` | `internal_functions.cpp` | `nns_meboot` internals | `nns.meboot` | cxx-exists-unbound | no | yes | Phase 4 | Left unbound because it mutates buffers and affects stochastic bootstrap distributions. |
| `internal_functions.hpp` | `SampleResult` | `internal_functions.cpp` | sampling internals | none | cxx-exists-unbound | no | yes | Phase 4 | Structured output; no current public API route. |
| `internal_functions.hpp` | `up_sample` | `internal_functions.cpp` | none public | none | cxx-exists-unbound | no | yes | Phase 4 | Left unbound because class balancing and seed semantics need a public contract first. |
| `internal_functions.hpp` | `down_sample` | `internal_functions.cpp` | none public | none | cxx-exists-unbound | no | yes | Phase 4 | Left unbound because class balancing and seed semantics need a public contract first. |
| `dependence.hpp` | `DepResult` | `dependence.cpp` | `nns_dep`/`nns_cor` result pieces | `nns.dependence` | cxx-exists-unbound | yes | no | Phase 5 | Requires pre-hashed partition labels for `dep_pair`; leave unbound pending wrapper design. |
| `dependence.hpp` | `DepMatrixResult` | `dependence.cpp` | matrix results | `nns.dependence` | cxx-exists-unbound | yes | no | Phase 5 | Structured matrix result; leave until parity for matrix orientation is added. |
| `dependence.hpp` | `dep_pair` | `dependence.cpp` | `nns_dep`, `nns_cor` | `nns.dependence` | cxx-exists-unbound | yes | no | Phase 5 | Needs partition hash inputs not exposed by current Python public API. |
| `dependence.hpp` | `dep_matrix` | `dependence.cpp` | dependence matrix APIs | `nns.dependence` | cxx-exists-unbound | yes | no | Phase 5 | Safe candidate later; not bound in this PR to avoid output shape changes. |
| `distance.hpp` | `distance` | `distance.cpp` | `nns_distance` | `nns.distance` | cxx-exists-unbound | yes | no | Phase 5 | Left unbound; current Python code includes rescaling/weighting behavior requiring parity comparison. |
| `distance.hpp` | `distance_path` | `distance.cpp` | distance path behavior | `nns.distance` | cxx-exists-unbound | yes | no | Phase 5 | Left unbound pending k/path output contract tests. |
| `distance.hpp` | `distance_bulk` | `distance.cpp` | `nns_distance_bulk` | `nns.distance` | cxx-exists-unbound | yes | no | Phase 5 | Left unbound pending row/column-major parity tests. |
| `distance.hpp` | `distance_path_parallel` | `distance.cpp` | none direct | none | cxx-exists-unbound | no | yes | Phase 5 | Parallel helper; no public wrapper. |
| `distance.hpp` | `distance_path_single_parallel` | `distance.cpp` | none direct | none | cxx-exists-unbound | no | yes | Phase 5 | Parallel helper; no public wrapper. |
| `partition.hpp` | `PartitionRow` | `partition.cpp` | partition result rows | `nns.part` | cxx-exists-unbound | yes | no | Phase 5 | Structured object mapping deferred. |
| `partition.hpp` | `RegressionPoint` | `partition.cpp` | regression points | `nns.part` | cxx-exists-unbound | yes | no | Phase 5 | Structured object mapping deferred. |
| `partition.hpp` | `SegmentH` | `partition.cpp` | `segments_h` | `nns.part` | cxx-exists-unbound | yes | no | Phase 5 | Structured object mapping deferred. |
| `partition.hpp` | `SegmentV` | `partition.cpp` | `segments_v` | `nns.part` | cxx-exists-unbound | yes | no | Phase 5 | Structured object mapping deferred. |
| `partition.hpp` | `PartitionResult` | `partition.cpp` | `nns_part` result dict | `nns.part` | cxx-exists-unbound | yes | no | Phase 5 | Complex R-compatible payload; not changed in this PR. |
| `partition.hpp` | `partition` | `partition.cpp` | `nns_part` | `nns.part` | cxx-exists-unbound | yes | no | Phase 5 | Safe candidate later, but output shape/labels must remain exact. |
| `seasonality.hpp` | `SeasonalityResult` | `seasonality.cpp` | `nns_seas` result pieces | `nns.seasonality` | cxx-exists-unbound | yes | no | Phase 5 | Structured result left unbound pending parity tests. |
| `seasonality.hpp` | `seasonality` | `seasonality.cpp` | `nns_seas` | `nns.seasonality` | cxx-exists-unbound | yes | no | Phase 5 | Left unbound because modulo and result-shape semantics need public parity tests. |
| `stochastic_dominance.hpp` | `fsd_uni` | `stochastic_dominance.cpp` | `fsd_uni` | `nns.stochastic_dominance` | cxx-exists-unbound | yes | no | Phase 5 | Candidate for future; not required by current native routing tests. |
| `stochastic_dominance.hpp` | `ssd_uni` | `stochastic_dominance.cpp` | `ssd_uni` | `nns.stochastic_dominance` | cxx-exists-unbound | yes | no | Phase 5 | Candidate for future. |
| `stochastic_dominance.hpp` | `tsd_uni` | `stochastic_dominance.cpp` | `tsd_uni` | `nns.stochastic_dominance` | cxx-exists-unbound | yes | no | Phase 5 | Candidate for future. |
| `stochastic_dominance.hpp` | `fsd` | `stochastic_dominance.cpp` | `fsd` | `nns.stochastic_dominance` | cxx-exists-unbound | yes | no | Phase 5 | Matrix orientation and index base must be validated before routing. |
| `stochastic_dominance.hpp` | `ssd` | `stochastic_dominance.cpp` | `ssd` | `nns.stochastic_dominance` | cxx-exists-unbound | yes | no | Phase 5 | Matrix orientation and index base must be validated before routing. |
| `stochastic_dominance.hpp` | `tsd` | `stochastic_dominance.cpp` | `tsd` | `nns.stochastic_dominance` | cxx-exists-unbound | yes | no | Phase 5 | Matrix orientation and index base must be validated before routing. |
| `stochastic_dominance.hpp` | `StochSupResult` | `stochastic_dominance.cpp` | `nns_ss` result dict | `nns.stochastic_superiority` | bound-private | yes | no | Existing | Already bound before this PR. |
| `stochastic_dominance.hpp` | `stochastic_superiority` | `stochastic_dominance.cpp` | `nns_ss` | `nns.stochastic_superiority` | bound-private | yes | no | Existing | Already bound before this PR. |

## Python APIs routed through native in this PR

- `nns.core.lpm`
- `nns.core.upm`
- `nns.core.lpm_ratio`
- `nns.core.upm_ratio`
- `nns.co_moments.co_lpm`
- `nns.co_moments.co_upm`
- `nns.co_moments.d_lpm`
- `nns.co_moments.d_upm`
- `nns.pm_matrix.pm_matrix`

## Functions newly bound in `_nnscore`

- Partial moment vector and ratio helpers: `lpm_v`, `upm_v`, `lpm_ratio_v`, `upm_ratio_v`.
- Co-partial moment helpers: `co_lpm`, `co_upm`, `d_lpm`, `d_upm`, `co_lpm_v`, `co_upm_v`, `d_lpm_v`, `d_upm_v`.
- N-dimensional/backend helpers: `clpm_nd`, `cupm_nd`, `dpm_nd`, `clpm_nd_batch`, `pm_matrix`.
- Fast linear model helper: `fast_lm_mult` (`fast_lm` was already bound).
- Private internal helpers: `is_discrete`, `vec_sd`, `col_sd`, `factor_2_dummy`, `factor_2_dummy_fr`, `generate_vectors`, `generate_lin_vectors`.

## Functions already bound before this PR

- `lpm`
- `upm`
- `gravity`
- `mode`
- `fast_lm`
- `stochastic_superiority`

## Intentionally left unbound or Python-only

- `central_tendencies::rescale`: Python remains authoritative until min-max/risk-neutral edge cases have direct parity tests.
- `dependence::{dep_pair, dep_matrix}` and result types: `dep_pair` needs pre-hashed partition labels, and matrix orientation/routing needs a dedicated test suite.
- `distance::*`: existing Python wrappers include public rescaling, class, weighting, and k-path behavior. They remain Python-only until shape and parity tests are added.
- `partition::*`: complex R-compatible result payload is left Python-only to avoid changing dictionary/list shapes.
- `seasonality::*`: structured result and modulo behavior need separate parity coverage.
- `stochastic_dominance::{fsd_uni, ssd_uni, tsd_uni, fsd, ssd, tsd}`: public Python implementations remain in place; matrix output index conventions need explicit tests before native routing.
- `internal_functions::{is_fcl, arma_seas_weighting, meboot_part, meboot_expand_sd, force_clt, up_sample, down_sample}`: intentionally not bound in this PR. The ARMA and meboot helpers involve structured outputs, mutation, random seeds, or statistical distribution semantics. Sampling helpers need a public class-balancing contract before exposure.

## Additional notes

- Regression is not treated as a direct C++ binding unless a C++ equivalent exists. The `fast_lm` and `fast_lm_mult` helpers are private backend utilities, not replacements for the Python NNS regression API.
- `internal_functions.cpp` is treated mostly as private backend support. Its bindings are not public top-level Python exports.
- Public APIs call `from nns._native import nnscore`; if `nnscore()` returns a module they use native C++, and if it returns `None` they fall back to the existing Python implementation.
- Windows local MinGW builds may fail to load `_nnscore`; official Windows wheels should be built with MSVC.

## Non-source-support headers in `extern/NNS-core/include/nns`

| C++ header | C++ function or type | C++ source file | Existing Python public function | Existing Python module | Currently bound in `_nnscore` | Should be public Python API | Should be private backend helper only | Binding priority | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `nns.hpp` | umbrella header includes component modules | none | none | none | internal-helper | no | yes | none | Include-only aggregator; no functions or result types to bind. |
| `parallel.hpp` | parallel execution helpers | header/internal support | none | none | internal-helper | no | yes | none | Build/runtime support for C++ core parallel loops; no public Python API. |
| `version.hpp` | `NNS_CORE_VERSION_MAJOR`, `NNS_CORE_VERSION_MINOR`, `NNS_CORE_VERSION_PATCH`, `NNS_CORE_VERSION` | none | none | none | cxx-exists-unbound | no | yes | none | Compile-time version macros; not bound in this PR. |
