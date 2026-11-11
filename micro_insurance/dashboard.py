"""Week 4 - interactive sensitivity dashboard (Dash).

    python -m micro_insurance.dashboard                # synthetic portfolio
    python -m micro_insurance.dashboard --data folder  # your own CSVs
    python -m micro_insurance.dashboard --port 8051

The distribution fits, EBCT and volatility estimates are computed once at
start-up by :func:`micro_insurance.pipeline.run`; the controls then re-price
with :mod:`micro_insurance.pricing` / :mod:`micro_insurance.comparison`, so
the dashboard holds no pricing logic of its own.
"""

from __future__ import annotations

import argparse

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

from . import comparison
from .pipeline import PipelineConfig, run
from .pricing import call_greeks

COLORS = {"ebct": "#2f6db5", "bs": "#d9822b", "stop": "#4b9a6a", "grid": "#e4e7eb"}
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"


def _layout(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15}},
        template="plotly_white", height=height, font={"family": FONT, "size": 12},
        margin={"l": 60, "r": 20, "t": 50, "b": 50}, legend={"orientation": "h", "y": -0.2},
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])
    return fig


def _card(label: str, value: str, note: str = "") -> html.Div:
    return html.Div(
        [html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value"),
         html.Div(note, className="kpi-note")],
        className="kpi",
    )


def build_app(data_dir: str | None = None) -> Dash:
    base = run(PipelineConfig(data_dir=data_dir))
    fleets = base.ebct2.fleets
    freq = comparison.frequency_by_fleet(base.claims)
    sev = base.best_severity
    s_max = float(max(base.ebct1.premium_per_unit.max(), base.ebct2.premium_per_unit.max()))
    k_max = float(np.ceil(2 * s_max / 500) * 500)
    sigma0 = round(base.gbm.sigma, 3)

    app = Dash(__name__, title="Micro-Insurance Pricing")
    app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")

    def slider(id_, lo, hi, step, value, marks):
        return dcc.Slider(id=id_, min=lo, max=hi, step=step, value=value, marks=marks,
                          tooltip={"placement": "bottom", "always_visible": True})

    app.layout = html.Div(
        [
            html.H1("Micro-insurance pricing: EBCT vs Black-Scholes"),
            html.P(
                f"Boda-boda accident/health cover. {len(fleets)} fleets, "
                f"{base.claims['year'].min()}-{base.claims['year'].max()}, "
                f"{'synthetic data' if data_dir is None else data_dir}. Estimated sigma* = {base.gbm.sigma:.3f} "
                f"({base.config.vol_frequency}), best severity fit: {sev.family}. Amounts in KES per operator-year.",
                className="sub",
            ),
            html.Div(
                [
                    html.Div([html.Label("Deductible K"),
                              slider("k", 0, k_max, 50, base.config.deductible,
                                     {int(v): f"{int(v):,}" for v in np.linspace(0, k_max, 5)})], className="ctl"),
                    html.Div([html.Label("Volatility sigma"),
                              slider("sigma", 0.05, 1.0, 0.01, sigma0,
                                     {v: f"{v:.0%}" for v in (0.05, 0.25, 0.5, 0.75, 1.0)})], className="ctl"),
                    html.Div([html.Label("Risk-free rate r"),
                              slider("r", 0.0, 0.25, 0.005, base.config.risk_free_rate,
                                     {v: f"{v:.0%}" for v in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25)})], className="ctl"),
                    html.Div([html.Label("EBCT model"),
                              dcc.RadioItems(id="model", options=[{"label": " Model I", "value": 1},
                                                                  {"label": " Model II", "value": 2}],
                                             value=2, inline=True)], className="ctl small"),
                    html.Div([html.Label("Fleet (detail charts)"),
                              dcc.Dropdown(id="fleet", options=fleets, value=fleets[0], clearable=False)],
                             className="ctl small"),
                ],
                className="controls",
            ),
            html.Div(id="kpis", className="kpis"),
            html.Div([dcc.Graph(id="by-fleet"), dcc.Graph(id="vs-k")], className="row"),
            html.Div([dcc.Graph(id="heatmap"), dcc.Graph(id="greeks")], className="row"),
            html.Div([dcc.Graph(id="credibility"), html.Div(id="stress")], className="row"),
            html.P(
                "Option underlying S = the fleet's EBCT credibility premium (expected annual claim cost per operator), "
                "deductible applied to the annual aggregate. Stop-loss benchmark = claim frequency x E[(X-K)+] "
                "under the fitted severity, deductible applied per claim.",
                className="foot",
            ),
        ],
        className="page",
    )

    @app.callback(
        Output("kpis", "children"), Output("by-fleet", "figure"), Output("vs-k", "figure"),
        Output("heatmap", "figure"), Output("greeks", "figure"), Output("credibility", "figure"),
        Output("stress", "children"),
        Input("k", "value"), Input("sigma", "value"), Input("r", "value"),
        Input("model", "value"), Input("fleet", "value"),
    )
    def update(k, sigma, r, model, fleet):
        cred = base.ebct1 if model == 1 else base.ebct2
        comp = comparison.compare_premiums(cred, k, sigma, r, base.config.term,
                                           frequency_by_fleet=freq, severity=sev)
        i = fleets.index(fleet)
        s = float(cred.premium_per_unit[i])

        kpis = [
            _card("Avg EBCT premium", f"{comp['ebct_premium'].mean():,.0f}", f"Model {'I' if model == 1 else 'II'}"),
            _card("Avg option premium", f"{comp['bs_premium'].mean():,.0f}",
                  f"{comp['bs_premium'].mean() / comp['ebct_premium'].mean():.0%} of EBCT"),
            _card("Avg stop-loss premium", f"{comp['stop_loss_premium'].mean():,.0f}", "per-claim deductible"),
            _card(f"{fleet} delta / vega", f"{comp['delta'].iloc[i]:.2f} / {comp['vega'].iloc[i]:,.0f}",
                  "vega per 1.00 of vol"),
        ]

        bars = go.Figure()
        for col, name, c in (("ebct_premium", "EBCT", "ebct"), ("bs_premium", "Black-Scholes", "bs"),
                             ("stop_loss_premium", "Stop-loss", "stop")):
            bars.add_bar(x=comp["fleet"], y=comp[col], name=name, marker_color=COLORS[c])
        _layout(bars, f"Premium by fleet (K = {k:,.0f}, sigma = {sigma:.0%}, r = {r:.1%})")
        bars.update_layout(barmode="group", yaxis_title="KES per operator-year")

        ks = np.linspace(0, k_max, 121)
        g_k = call_greeks(s, ks, base.config.term, r, sigma)
        curve = go.Figure()
        curve.add_scatter(x=ks, y=g_k.price, name="Black-Scholes", line={"color": COLORS["bs"], "width": 2.5})
        curve.add_scatter(x=ks, y=[freq[fleet] * sev.stop_loss(x) for x in ks], name="Stop-loss",
                          line={"color": COLORS["stop"], "width": 2.5})
        curve.add_scatter(x=ks, y=np.maximum(s - ks, 0), name="Intrinsic max(S-K,0)",
                          line={"color": "#9aa3ad", "dash": "dot"})
        curve.add_hline(y=s, line={"color": COLORS["ebct"], "dash": "dash"},
                        annotation_text=f"EBCT S = {s:,.0f}", annotation_position="top right")
        curve.add_vline(x=k, line={"color": "#555", "width": 1})
        _layout(curve, f"{fleet}: premium vs deductible")
        curve.update_layout(xaxis_title="Deductible K", yaxis_title="Premium")

        grid = comparison.sensitivity_grid(s, np.linspace(0, k_max, 41), np.linspace(0.05, 1.0, 20), r,
                                           base.config.term)
        z = grid.pivot(index="volatility", columns="deductible", values="premium")
        heat = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, colorscale="Blues",
                                    colorbar={"title": "Premium"},
                                    hovertemplate="K=%{x:,.0f}<br>sigma=%{y:.0%}<br>C=%{z:,.0f}<extra></extra>"))
        heat.add_scatter(x=[k], y=[sigma], mode="markers", marker={"color": COLORS["bs"], "size": 11,
                                                                   "line": {"color": "white", "width": 2}},
                         name="current", hoverinfo="skip")
        _layout(heat, f"{fleet}: option premium over K x sigma")
        heat.update_layout(xaxis_title="Deductible K", yaxis_title="Volatility", yaxis_tickformat=".0%",
                           hovermode="closest", showlegend=False)

        gr = make_subplots(rows=2, cols=2, subplot_titles=("Delta dC/dS", "Gamma d2C/dS2", "Vega dC/dsigma",
                                                           "Rho dC/dr"), vertical_spacing=0.18)
        for (row, col), name in zip(((1, 1), (1, 2), (2, 1), (2, 2)), ("delta", "gamma", "vega", "rho")):
            gr.add_scatter(x=ks, y=getattr(g_k, name), row=row, col=col, showlegend=False,
                           line={"color": COLORS["bs"], "width": 2})
            gr.add_vline(x=k, line={"color": "#555", "width": 1}, row=row, col=col)
        _layout(gr, f"{fleet}: Greeks vs deductible", height=360)
        gr.update_annotations(font_size=12)

        credfig = go.Figure()
        credfig.add_bar(x=cred.fleets, y=cred.z, marker_color=COLORS["ebct"], name="Z",
                        text=[f"{v:.2f}" for v in cred.z], textposition="outside")
        _layout(credfig, f"Credibility factor Z ({cred.model}); k = E[s2]/Var[m] = {cred.k:,.1f}")
        credfig.update_layout(yaxis_range=[0, 1.1], yaxis_title="Z", showlegend=False)

        st = comparison.stress_test(s, k, sigma, r, base.config.term)
        header = html.Tr([html.Th(h) for h in ("Factor", "Shock", "Premium", "Change", "% change")])
        body = [
            html.Tr([html.Td(row.factor),
                     html.Td(f"{row.shock:+.0%}" if row.factor != "rate" else f"{row.shock * 1e4:+.0f} bp"),
                     html.Td(f"{row.premium:,.0f}"), html.Td(f"{row.change:+,.0f}"),
                     html.Td(f"{row.pct_change:+.1%}" if np.isfinite(row.pct_change) else "n/a")])
            for row in st.itertuples()
        ]
        stress = html.Div([html.H3(f"{fleet}: stress test (one factor at a time)"),
                           html.Table([html.Thead(header), html.Tbody(body)], className="tbl")],
                          className="stress")
        return kpis, bars, curve, heat, gr, credfig, stress

    return app


CSS = """
body{margin:0;background:#f6f7f9;font-family:Inter,Segoe UI,Helvetica,Arial,sans-serif;color:#1f2933}
.page{max-width:1400px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:22px;margin:4px 0}.sub{color:#52606d;margin:0 0 16px;font-size:13px}
.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px 24px;background:#fff;
 padding:16px;border-radius:8px;border:1px solid #e4e7eb}
.ctl label{font-weight:600;font-size:13px;display:block;margin-bottom:6px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}
.kpi{background:#fff;border:1px solid #e4e7eb;border-radius:8px;padding:12px 14px}
.kpi-label{font-size:12px;color:#52606d}.kpi-value{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}
.kpi-note{font-size:12px;color:#7b8794}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px;margin-bottom:12px}
.row>div{background:#fff;border:1px solid #e4e7eb;border-radius:8px;min-width:0}
.stress{padding:12px 16px}.stress h3{font-size:15px;margin:4px 0 10px}
.tbl{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
.tbl th,.tbl td{padding:5px 8px;border-bottom:1px solid #eef0f2;text-align:right}
.tbl th:first-child,.tbl td:first-child{text-align:left}
.foot{font-size:12px;color:#7b8794}
@media(max-width:520px){.row{grid-template-columns:1fr}}
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Micro-insurance pricing dashboard")
    ap.add_argument("--data", default=None)
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    build_app(args.data).run(port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
