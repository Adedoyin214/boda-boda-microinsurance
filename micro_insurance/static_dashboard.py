"""Export a self-contained, server-free HTML dashboard (for GitHub Pages).

    python -m micro_insurance.static_dashboard                 # -> docs/index.html
    python -m micro_insurance.static_dashboard --data my_data --out site/index.html

Everything estimated from data (distribution fits, EBCT, sigma*, the
per-claim stop-loss curve) is computed here in Python and embedded as JSON.
The page then re-prices in the browser: Black-Scholes and its Greeks are
closed-form, so the K / sigma / r controls work without a Python server.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from . import comparison
from .pipeline import PipelineConfig, run

TEMPLATE = Path(__file__).parent / "templates" / "static_dashboard.html"
PLACEHOLDER = "/*__DASHBOARD_DATA__*/null"
HEAD_SPLIT = "<!--BODY-->"


def _r(x, nd=6):
    """Round floats (and arrays) so the embedded JSON stays small."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, nd) for v in x]
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return float(np.round(float(x), nd))


def build_data(data_dir: str | None = None, seed: int = 2026) -> dict:
    res = run(PipelineConfig(data_dir=data_dir, seed=seed))
    claims = res.claims
    fleets = res.ebct2.fleets
    freq = comparison.frequency_by_fleet(claims)

    def cred(c):
        return {"model": c.model, "grand_mean": _r(c.grand_mean), "within": _r(c.within_variance),
                "between": _r(c.between_variance), "k": _r(c.k), "risk_means": _r(c.risk_means),
                "z": _r(c.z), "premium": _r(c.premium_per_unit)}

    # Per-claim stop-loss curve E[(X-K)+] on a K grid (family-agnostic, interpolated in JS)
    s_max = float(max(res.ebct1.premium_per_unit.max(), res.ebct2.premium_per_unit.max()))
    k_max = float(np.ceil(2 * s_max / 500) * 500)
    k_grid = np.linspace(0, k_max, 241)
    sev = res.best_severity

    # Severity histogram with both fitted densities scaled to counts
    amounts = res.portfolio.individual_claims["amount"].to_numpy()
    hi = float(np.quantile(amounts, 0.99))
    counts, edges = np.histogram(amounts[amounts <= hi], bins=40, range=(0, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    xs = np.linspace(width / 4, hi, 160)
    dens = {name: _r(f.dist.pdf(xs) * amounts.size * width, 3) for name, f in res.severity.items()}

    # Weekly treatment-cost index (enough resolution for the chart)
    ci = res.portfolio.cost_index.copy()
    ci["date"] = ci["date"].astype("datetime64[ns]")
    weekly = ci.set_index("date")["cost_index"].resample("W").last().dropna()

    panel = claims.pivot(index="fleet", columns="year", values="claims_per_unit").loc[fleets]
    vol = claims.groupby("fleet")["policyholders"].sum().loc[fleets]

    return {
        "meta": {
            "generated": date.today().isoformat(),
            "synthetic": data_dir is None,
            "source": "synthetic portfolio (seed %d)" % seed if data_dir is None else str(data_dir),
            "n_fleets": len(fleets),
            "years": [int(claims["year"].min()), int(claims["year"].max())],
            "n_claims": int(amounts.size),
            "exposure": _r(claims["policyholders"].sum(), 0),
            "currency": "KES",
        },
        "defaults": {"k": res.config.deductible, "r": res.config.risk_free_rate,
                     "sigma": _r(res.gbm.sigma, 4), "term": res.config.term, "k_max": k_max},
        "fleets": fleets,
        "volume": _r(vol.to_numpy(), 0),
        "frequency": _r([freq[f] for f in fleets]),
        "ebct1": cred(res.ebct1),
        "ebct2": cred(res.ebct2),
        "panel": {"years": [int(y) for y in panel.columns], "x": [_r(row, 2) for row in panel.to_numpy()]},
        "poisson": {"rate": _r(res.frequency.rate), "dispersion_ratio": _r(res.frequency.dispersion_ratio, 3),
                    "p_value": _r(res.frequency.dispersion_p_value, 12)},
        "severity": {
            "best": sev.family,
            "fits": {name: {"params": {k: _r(v) for k, v in f.params.items()}, "aic": _r(f.aic, 1),
                            "ks": _r(f.ks_statistic, 4), "ks_p": _r(f.ks_p_value, 6), "mean": _r(f.mean, 1)}
                     for name, f in res.severity.items()},
            "hist": {"centers": _r(centers, 1), "counts": counts.tolist(), "width": _r(width, 1)},
            "density_x": _r(xs, 1), "density": dens, "p99": _r(hi, 0),
        },
        "stop_loss": {"k": _r(k_grid, 2), "per_claim": _r([sev.stop_loss(k) for k in k_grid], 4)},
        "bootstrap": {k: _r(getattr(res.bootstrap, k), 3) for k in ("estimate", "std_error", "ci_low", "ci_high")},
        "gbm": {"daily": {"sigma": _r(res.gbm.sigma, 5), "mu": _r(res.gbm.mu, 5), "n": res.gbm.n_returns},
                "monthly": {"sigma": _r(res.gbm_alt.sigma, 5), "mu": _r(res.gbm_alt.mu, 5), "n": res.gbm_alt.n_returns}},
        "cost_index": {"dates": [d.strftime("%Y-%m-%d") for d in weekly.index], "values": _r(weekly.to_numpy(), 3)},
    }


def render(data: dict, full_document: bool = True) -> str:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in tpl or HEAD_SPLIT not in tpl:
        raise RuntimeError("template is missing its data placeholder or body marker")
    page = tpl.replace(PLACEHOLDER, json.dumps(data, separators=(",", ":")))
    head, body = page.split(HEAD_SPLIT, 1)
    if not full_document:  # fragment (head content + body) for hosts that supply the skeleton
        return head + body
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"{head}</head>\n<body>\n{body}</body>\n</html>\n"
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Export the static HTML dashboard")
    ap.add_argument("--data", default=None, help="folder with claims CSVs (default: synthetic)")
    ap.add_argument("--out", default="docs/index.html")
    ap.add_argument("--fragment", action="store_true", help="omit the <html>/<head>/<body> skeleton")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(build_data(args.data, args.seed), not args.fragment), encoding="utf-8")
    print(f"static dashboard written to {out.resolve()} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
