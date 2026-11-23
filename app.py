"""Step 5: Streamlit app. Run: .venv/bin/streamlit run app.py"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (C_ALERT, C_MODEL, C_MUTED, DATA, FOODS, LABELS, REPORTS, SPIKE_THRESHOLD,
                    TREND_WINDOW)

MODEL_NAMES = {"naive": "Naive", "sarima": "SARIMA", "lgbm": "LightGBM"}

st.set_page_config(page_title="Indonesia food prices", layout="wide")


@st.cache_data
def load():
    p = pd.read_parquet(DATA / "monthly.parquet")
    fc = pd.read_parquet(DATA / "forecasts.parquet")
    m = pd.read_csv(REPORTS / "backtest_metrics.csv")
    al = pd.read_csv(REPORTS / "backtest_alerts.csv")
    return p, fc, m, al


p, fc, metrics, alerts_bt = load()
last = p.month.max()

# ---------------------------------------------------------------- controls
st.sidebar.title("Indonesia food prices")
food = st.sidebar.selectbox("Food", list(FOODS), format_func=LABELS.get)
best = fc[(fc.food == food) & fc.best].model.iat[0]
model = st.sidebar.selectbox("Model", list(MODEL_NAMES), index=list(MODEL_NAMES).index(best),
                             format_func=lambda k: MODEL_NAMES[k] + (" (best in backtest)" if k == best else ""))
thr = st.sidebar.slider("Spike threshold above trend (%)", 5, 40, int(SPIKE_THRESHOLD * 100), 1) / 100
f_food = fc[(fc.food == food) & (fc.model == model)].copy()
f_food["vs_trend"] = f_food.yhat / f_food.trend - 1
f_food["alert"] = f_food.vs_trend > thr

provinces = sorted(p[p.food == food].province.dropna().unique())
prov = st.sidebar.selectbox("Province", ["All"] + provinces, format_func=str.title)
mk = p[(p.food == food) & ((p.province == prov) | (prov == "All"))].groupby("market_id").agg(
    market=("market", "first"), province=("province", "first"), district=("district", "first"))
flagged = set(f_food[f_food.alert].market_id)
mk["label"] = [f"{'! ' if i in flagged else ''}{r.market} ({str(r.district).title()})" for i, r in mk.iterrows()]
mk = mk.sort_values("label")
mid = st.sidebar.selectbox("Market (! = alert)", mk.index, format_func=lambda i: mk.at[i, "label"])
st.sidebar.caption(f"Data: WFP retail prices to {last:%B %Y}. Trend = trailing {TREND_WINDOW}-month median.")

tab_m, tab_a, tab_p = st.tabs(["Market forecast", "All alerts", "Model performance"])

# ---------------------------------------------------------------- market view
with tab_m:
    s = p[(p.food == food) & (p.market_id == mid)].set_index("month").sort_index()
    f = f_food[f_food.market_id == mid].sort_values("h")
    st.subheader(f"{LABELS[food]}, {mk.at[mid, 'market']}, {str(mk.at[mid, 'district']).title()}, "
                 f"{str(mk.at[mid, 'province']).title()}")

    if f.empty:
        st.warning(f"No forecast: this market did not report a price for {last:%B %Y}. "
                   f"Last report: {s.price.last_valid_index():%B %Y}.")
    else:
        latest = s.price_clean.dropna().iat[-1]
        tr = f.trend.iat[0]
        peak = f.loc[f.yhat.idxmax()]
        c = st.columns(4)
        c[0].metric(f"Price {last:%b %Y}", f"Rp {latest:,.0f}/kg")
        c[1].metric("Normal trend", f"Rp {tr:,.0f}/kg", f"{latest / tr - 1:+.1%} now vs trend", delta_color="off")
        c[2].metric(f"Forecast {f.target.iat[-1]:%b %Y}", f"Rp {f.yhat.iat[-1]:,.0f}/kg",
                    f"{f.yhat.iat[-1] / latest - 1:+.1%} vs now", delta_color="inverse")
        if f.alert.any():
            c[3].error(f"Spike alert: {peak.target:%b %Y} forecast is {peak.yhat / tr - 1:+.0%} above trend "
                       f"(limit {thr:.0%}).")
        else:
            c[3].success(f"No alert: forecasts stay within {thr:.0%} of trend "
                         f"(max {f.yhat.max() / tr - 1:+.0%}).")

    trend_hist = s.price_clean.rolling(TREND_WINDOW, min_periods=6).median()
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s.price, mode="lines", name="Reported price",
                    line=dict(color="#0b0b0b", width=2), connectgaps=False)
    imp = s[s.imputed]
    if len(imp):
        fig.add_scatter(x=imp.index, y=imp.price_clean, mode="markers", name="Filled gap",
                        marker=dict(size=8, color="white", line=dict(color="#0b0b0b", width=1.5)))
    out = s[s.outlier]
    if len(out):
        fig.add_scatter(x=out.index, y=out.price, mode="markers", name="Outlier (ignored)",
                        marker=dict(size=10, symbol="x", color=C_ALERT))
    fig.add_scatter(x=trend_hist.index, y=trend_hist, mode="lines", name=f"Trend ({TREND_WINDOW}-mo median)",
                    line=dict(color=C_MUTED, width=2))
    if not f.empty:
        fx = [last] + list(f.target)
        fig.add_scatter(x=[last, f.target.iat[-1]], y=[tr * (1 + thr)] * 2, mode="lines",
                        name=f"Alert limit (trend +{thr:.0%})", line=dict(color=C_ALERT, width=1.5, dash="dash"))
        allm = fc[(fc.food == food) & (fc.market_id == mid)]
        for k in MODEL_NAMES:
            g = allm[allm.model == k].sort_values("h")
            if g.empty:
                continue
            on = k == model
            if k == "sarima" and on and g.lo.notna().all():
                fig.add_scatter(x=fx + fx[::-1], y=[latest] + list(g.hi) + list(g.lo[::-1]) + [latest],
                                fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                                name="SARIMA 80% interval", hoverinfo="skip")
            fig.add_scatter(x=fx, y=[latest] + list(g.yhat), mode="lines+markers",
                            name=f"{MODEL_NAMES[k]} forecast", opacity=1 if on else 0.5,
                            line=dict(color=C_MODEL[k], width=2.5 if on else 1.5, dash=None if on else "dot"),
                            marker=dict(size=8 if on else 5))
    fig.update_layout(height=480, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.15), yaxis_title="IDR per kg",
                      yaxis_tickformat=",.0f", plot_bgcolor="white")
    fig.update_xaxes(range=["2020-01-01", last + pd.DateOffset(months=4)], gridcolor="#ecebe7")
    fig.update_yaxes(gridcolor="#ecebe7")
    st.plotly_chart(fig, width="stretch")

    if not f.empty:
        t = fc[(fc.food == food) & (fc.market_id == mid)].pivot_table(index="target", columns="model", values="yhat")
        t = t.rename(columns=MODEL_NAMES)
        t.index = t.index.strftime("%b %Y")
        t.index.name = "Month"
        st.dataframe(t.style.format("Rp {:,.0f}"), width="stretch")

# ---------------------------------------------------------------- alerts
with tab_a:
    a = f_food[f_food.alert]
    per = a.groupby("market_id").agg(market=("market", "first"), province=("province", "first"),
                                     district=("district", "first"), first_month=("target", "min"),
                                     forecast=("yhat", "max"), trend=("trend", "first"),
                                     above=("vs_trend", "max")).sort_values("above", ascending=False)
    st.subheader(f"{LABELS[food]}: {len(per)} of {f_food.market_id.nunique()} markets flagged "
                 f"({MODEL_NAMES[model]}, limit +{thr:.0%})")
    if per.empty:
        st.info("No market forecast is above the alert limit.")
    else:
        geo = p.groupby("market_id")[["latitude", "longitude"]].first()
        mp = per.join(geo).dropna(subset=["latitude"])
        st.map(mp.rename(columns={"latitude": "lat", "longitude": "lon"}), color=C_ALERT, size=9000)
        show = per.assign(first_month=per.first_month.dt.strftime("%b %Y"),
                          province=per.province.str.title(), district=per.district.str.title())
        st.dataframe(show.style.format({"forecast": "Rp {:,.0f}", "trend": "Rp {:,.0f}", "above": "{:+.0%}"}),
                     width="stretch", hide_index=True,
                     column_config={"first_month": "First alert month", "forecast": "Peak forecast",
                                    "above": "Above trend"})

# ---------------------------------------------------------------- performance
with tab_p:
    st.subheader("Backtest: forecast error by model and horizon")
    st.caption("Rolling-origin backtest: each forecast uses only data up to its origin month. "
               "MAPE = mean absolute percentage error; skill = error reduction vs naive.")
    mm = metrics[metrics.food == food].copy()
    mm["model"] = mm.model.map(MODEL_NAMES)
    st.dataframe(mm.drop(columns="food").style.format(
        {"mape": "{:.2%}", "median_ape": "{:.2%}", "within_5pct": "{:.0%}", "bias": "{:+.2%}",
         "skill_vs_naive": "{:+.0%}"}), width="stretch", hide_index=True)
    st.subheader(f"Spike alert in the backtest (limit +{SPIKE_THRESHOLD:.0%})")
    aa = alerts_bt[alerts_bt.food == food].copy()
    aa["model"] = aa.model.map(MODEL_NAMES)
    st.dataframe(aa.drop(columns="food").style.format({"precision": "{:.0%}", "recall": "{:.0%}"}, na_rep="-"),
                 width="stretch", hide_index=True)
    st.caption(f"Real spike = reported price more than {SPIKE_THRESHOLD:.0%} above the trend at the origin month. "
               "Precision = share of alerts that were real spikes; recall = share of real spikes that got an alert.")
