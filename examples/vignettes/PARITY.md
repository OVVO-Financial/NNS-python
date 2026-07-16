# Canonical R-to-Python vignette parity

The R package is the source of truth. This ledger records what each Python
instructional script reproduces and names remaining API differences explicitly.
A missing Python facility is never replaced silently with an unrelated example.

## 01 — `01_overview.py`

Preserves the overview curriculum: partial-moment foundations, descriptive and
distributional tools, nonlinear dependence, normalization/rescaling, ANOVA and
stochastic superiority, regression/boosting/stacking/causality, forecasting,
maximum-entropy simulation, and stochastic dominance. Figures accompany the
major statistical demonstrations.

**Known gaps:** interactive R plotting devices are rendered as saved Matplotlib
figures. This changes presentation, not the statistical calculation.

## 02 — `02_partial_moments.py`

Reproduces the mean, variance, standard deviation, higher-moment, mode,
covariance, covariance-matrix, correlation, empirical/continuous CDF, copula,
numerical-integration, and Bayes demonstrations in R order. Co-partial matrices
and reconstructed covariance are printed, and CDF/copula figures are saved.

**Known gaps:** R's interactive 3-D `rgl` views are translated to static
Matplotlib 3-D figures.

## 03 — `03_correlation_and_dependence.py`

Reproduces linear, quadratic, cubic, sinusoidal, circular, and discrete examples;
contrasts Pearson correlation with NNS correlation/dependence; evaluates
asymmetry, copula dependence, and causation; and saves relationship and matrix
figures.

**Known gaps:** Python's public dependence API returns pairwise scalar results;
matrix displays are assembled transparently from those public pair calls.

## 04 — `04_normalization_and_rescaling.py`

Preserves linear and nonlinear normalization, affine min/max rescaling,
risk-neutral terminal and discounted transformations, and the R interpretation
of shape-preserving transformations. Input/output distributions are plotted.

**Known gaps:** R chart-type side effects are replaced with explicit saved
figures generated from returned arrays.

## 05 — `05_sampling_and_simulation.py`

Preserves empirical-CDF inversion across partial-moment degrees,
maximum-entropy bootstrap sampling, dependence-targeted Monte Carlo sampling,
and ensemble diagnostics. Distribution and path figures are generated.

**Known gaps:** stochastic draws use NumPy's local generator, so simulations are
distributionally equivalent but not generally bit-for-bit equal to R draws.

## 06 — `06_comparing_distributions.py`

Preserves two-sample and multi-group NNS ANOVA, stochastic superiority,
first/second/third-order stochastic dominance, efficient-set extraction, and
stochastic-dominance clustering. CDF, superiority, and cluster figures are
saved, while the full returned structures are printed.

**Known gaps:** R's ANOVA plotting side effect is reconstructed from the public
returned statistics and empirical CDFs.

## 07 — `07_clustering_and_regression.py`

Preserves partition orders 1–4, X-only partitions, clusters used in regression,
univariate and full-grid multivariate regression, interpolation/extrapolation,
dimension reduction, thresholding, classification, `NNS.stack`, duplicated
predictor dimensions, smoothing, and univariate/multivariate imputation. The
exact iris data are committed locally. Six figure groups and important model
objects are produced.

**Known gaps:** Python does not yet expose R's Voronoi tessellation side effect or
`NNS.reg(... )$rhs.partitions`; both are labeled in the relevant sections and
supported public partition diagnostics are shown instead.

## 08 — `08_classification.py`

Preserves the splits-versus-partitions narrative, exact iris holdout rows
141–150, `NNS.boost` with balancing and the R parameterization,
`NNS.stack` cross-validation, and the depth/nearest-neighbor/extreme controls.
Feature diagnostics, holdout paths, and returned model structures are displayed.

**Known gaps:** Python does not yet expose `rhs.partitions`. Balanced boosting
has a documented bit-for-bit RNG ordering gap from R's interleaved class draws;
the procedure and diagnostics remain equivalent.

## 09 — `09_forecasting.py`

Preserves the exact AirPassengers series, 100-observation training set,
44-observation validation set, linear/nonlinear seasonal-factor-12 comparison,
periods 1–25 validation, `NNS.seas`, `NNS.ARMA.optim` over periods 12–60 by 6,
and the 50-period extension with prediction intervals. Every forecast stage is
plotted and its returned structure displayed.

**Known gaps:** Python does not separately expose R's `seasonal.plot` side
effect, so the equivalent diagnostic is generated from `NNS.seas`. The R
vignette mentions `NNS.VAR` but contains no executable VAR demonstration; the
Python canonical port therefore does not invent one.
