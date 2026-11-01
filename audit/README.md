# Overnight audit (2026-10-31 / 11-01)

Working material behind the commits on this branch. Nothing here is imported by the package;
the folder can be dropped once the PRs land.

| Path | What |
|---|---|
| `findings/{PM,DEP,REG,STK,TS,SDA,CORE,X}.md` | 177 findings (bug, evidence, which side is right, proposed fix), one file per area |
| `findings/REVIEW.md` | Independent review of the branch diff after the fixes |
| `R_PROPOSALS.md` | Method questions for the R author (no behaviour changed for these) |
| `r-patches/r-bugs/*.patch` | 61 R-level fixes for OVVO-Financial/NNS (`git am` on NNS-Beta-Version @ 82a92eb) |
| `r-patches/cpp-core-bugs/*.patch` | 7 C++ fixes for the same repo (apply after r-bugs) |
| `repro/` | Repro and A/B scripts referenced by the findings (`work/<GROUP>/...` paths) |
| `journal.md` | Append-only log of what was examined, decided and committed |

Data used by the repro scripts lived in `/home/user/audit/data`: Yahoo Finance closes 2010-2026 for
SPY QQQ IWM TLT GLD USO AAPL MSFT NVDA XOM JPM BTC-USD ^VIX EURUSD=X (`yfinance.download(..., auto_adjust=True)`),
plus R's Boston, iris, mtcars, airquality and AirPassengers written with `write.csv`. They are not committed.

Reference builds: R 4.3.3, NNS 13.3 from OVVO-Financial/NNS@82a92eb; "fixed R" = the same source with both patch
series applied. The parity cache in `tests/_r_cache` was regenerated from fixed R (see `docs/parity.md`).
