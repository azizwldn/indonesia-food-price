"""Step 5: Streamlit app. Run: streamlit run app.py"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (C_ALERT, C_MODEL, C_MUTED, DATA, FOODS, LABELS, REPORTS, SPIKE_THRESHOLD,
                    TREND_WINDOW)

MODEL_NAMES = {"naive": "Naive", "sarima": "SARIMA", "lgbm": "LightGBM"}
FILES = [DATA / "monthly.parquet", DATA / "forecasts.parquet",
         REPORTS / "backtest_metrics.csv", REPORTS / "backtest_alerts.csv"]
INK, GRID = "#0b0b0b", "#ecebe7"
C_FC = "#2a78d6"                       # chosen forecast, always blue
C_OTHER = ["#eb6834", "#1baf7a"]         # other models when comparing

st.set_page_config(page_title="Indonesia food prices", layout="wide")


@st.cache_data
def load(stamp):
    """stamp = file modification times, so new data pushed to the repo resets the cache."""
    p = pd.read_parquet(FILES[0])
    fc = pd.read_parquet(FILES[1])
    return p, fc, pd.read_csv(FILES[2]), pd.read_csv(FILES[3])


def rp(v):
    return f"Rp {v:,.0f}"


def rp_short(v):
    return f"Rp {v / 1000:,.1f}k"


p, fc, metrics, alerts_bt = load(tuple(f.stat().st_mtime for f in FILES))
last = p.month.max()

# ---------------------------------------------------------------- controls
st.sidebar.title("Indonesia food prices")
food = st.sidebar.selectbox("Food", list(FOODS), format_func=LABELS.get)
best = fc[(fc.food == food) & fc.best].model.iat[0]
model = st.sidebar.selectbox("Model", list(MODEL_NAMES), index=list(MODEL_NAMES).index(best),
                             format_func=lambda k: MODEL_NAMES[k] + (" (best in backtest)" if k == best else ""))
thr = st.sidebar.slider("Spike alert: % above normal price", 5, 40, int(SPIKE_THRESHOLD * 100), 1) / 100
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
st.sidebar.caption(f"Data: WFP retail prices to {last:%B %Y}. Normal price = median of the last "
                   f"{TREND_WINDOW} months.")

tab_m, tab_a, tab_p = st.tabs(["Market forecast", "All alerts", "Model performance"])


# ---------------------------------------------------------------- market chart
def end_labels(items, y_span):
    """Right-edge labels, pushed apart so they do not overlap. items: (y, text, colour)."""
    items = sorted(items, key=lambda t: t[0])
    gap = y_span * 0.06
    out, prev = [], None
    for y, text, color in items:
        ypos = y if prev is None else max(y, prev + gap)
        out.append((ypos, text, color))
        prev = ypos
    return out


def market_chart(s, f, allm, compare, name):
    end = last + pd.DateOffset(months=3)
    trend_hist = s.price_clean.rolling(TREND_WINDOW, min_periods=6).median()
    fig = go.Figure()

    # history
    fig.add_scatter(x=s.index, y=trend_hist, mode="lines", name="Normal price (12-month median)",
                    line=dict(color=C_MUTED, width=2), showlegend=False,
                    hovertemplate="Normal price: Rp %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=s.index, y=s.price, mode="lines", name="Price", showlegend=False,
                    line=dict(color=INK, width=2.5), customdata=s.price / trend_hist - 1,
                    hovertemplate="Price: Rp %{y:,.0f} (%{customdata:+.0%} vs normal)<extra></extra>")
    imp = s[s.imputed]
    if len(imp):
        fig.add_scatter(x=imp.index, y=imp.price_clean, mode="markers", name="Missing month (filled)",
                        marker=dict(size=8, color="white", line=dict(color=INK, width=1.5)),
                        hovertemplate="Missing, filled: Rp %{y:,.0f}<extra></extra>")
    out = s[s.outlier]
    if len(out):
        fig.add_scatter(x=out.index, y=out.price, mode="markers", name="Price outlier (ignored)",
                        marker=dict(size=10, symbol="x", color=C_ALERT),
                        hovertemplate="Outlier, ignored: Rp %{y:,.0f}<extra></extra>")

    labels = []
    if not f.empty:
        latest = s.price_clean.dropna().iat[-1]
        tr = f.trend.iat[0]
        limit = tr * (1 + thr)
        fx = [last] + list(f.target)
        # forecast period, "now" line and red zone above the alert line
        fig.add_vrect(x0=last, x1=end, fillcolor="#f3f2ee", line_width=0, layer="below")
        fig.add_vline(x=last, line=dict(color=C_MUTED, width=1))
        fig.add_annotation(x=last, y=1, yref="paper", text=f"Now ({last:%b %Y})", showarrow=False,
                           xanchor="right", yanchor="bottom", font=dict(size=11, color="#52514e"))
        fig.add_annotation(x=end, y=1, yref="paper", text="Forecast", showarrow=False,
                           xanchor="right", yanchor="bottom", font=dict(size=11, color="#52514e"))
        top = max(limit, f.yhat.max(), f.hi.max() if f.hi.notna().any() else 0) * 1.08
        fig.add_shape(type="rect", x0=last, x1=end, y0=limit, y1=top, fillcolor=C_ALERT, opacity=0.08,
                      line_width=0, layer="below")
        fig.add_scatter(x=[last, end], y=[limit, limit], mode="lines", showlegend=False,
                        line=dict(color=C_ALERT, width=1.5, dash="dash"),
                        hovertemplate=f"Alert line: {rp(limit)}<extra></extra>")
        fig.add_scatter(x=[last, end], y=[tr, tr], mode="lines", showlegend=False,
                        line=dict(color=C_MUTED, width=2, dash="dot"), hoverinfo="skip")

        if model == "sarima" and f.lo.notna().all():
            fig.add_scatter(x=fx + fx[::-1], y=[latest] + list(f.hi) + list(f.lo[::-1]) + [latest],
                            fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                            name="Likely range (80%)", hoverinfo="skip")
        if compare:
            for k, col in zip([k for k in MODEL_NAMES if k != model], C_OTHER):
                g = allm[allm.model == k].sort_values("h")
                if g.empty:
                    continue
                fig.add_scatter(x=fx, y=[latest] + list(g.yhat), mode="lines+markers",
                                name=f"{MODEL_NAMES[k]} forecast", opacity=0.6,
                                line=dict(color=col, width=1.5, dash="dot"), marker=dict(size=5),
                                hovertemplate=f"{MODEL_NAMES[k]}: Rp %{{y:,.0f}}<extra></extra>")
        fig.add_scatter(x=fx, y=[latest] + list(f.yhat), mode="lines+markers", showlegend=False,
                        name="Forecast", line=dict(color=C_FC, width=3),
                        marker=dict(size=8), customdata=[0] + list(f.yhat / tr - 1),
                        hovertemplate=f"{MODEL_NAMES[model]} forecast: Rp %{{y:,.0f}} "
                                      "(%{customdata:+.0%} vs normal)<extra></extra>")
        # y range of the whole chart (plotly autoranges over all history, not only the visible part)
        span = max(top, s.price.max()) - min(s.price.min(), f.yhat.min())
        labels = end_labels([
            (f.yhat.iat[-1], f"<b>Forecast {f.target.iat[-1]:%b}: {rp_short(f.yhat.iat[-1])}</b>", C_FC),
            (limit, f"Alert above {rp_short(limit)}", C_ALERT),
            (tr, f"Normal {rp_short(tr)}", "#8a8984"),
        ], span)
    for y, text, color in labels:
        fig.add_annotation(x=end, y=y, text=text, showarrow=False, xanchor="left", xshift=6,
                           font=dict(size=12, color=color))

    fig.update_layout(
        height=500, hovermode="x unified", plot_bgcolor="white",
        margin=dict(l=10, r=150, t=30, b=10),
        legend=dict(orientation="h", x=0, y=1.08, font=dict(size=11)),
        yaxis=dict(tickprefix="Rp ", tickformat="~s", gridcolor=GRID, zeroline=False),
        xaxis=dict(gridcolor=GRID, range=[last - pd.DateOffset(months=24), end + pd.DateOffset(days=10)],
                   rangeselector=dict(x=1, xanchor="right", y=-0.12, buttons=[
                       dict(count=1, label="1Y", step="year", stepmode="backward"),
                       dict(count=2, label="2Y", step="year", stepmode="backward"),
                       dict(step="all", label="All")])))
    return fig


def s_last_price(f):
    return p[(p.food == food) & (p.market_id == f.market_id.iat[0])].dropna(subset=["price_clean"]) \
        .sort_values("month").price_clean.iat[-1]


def headline(f, name):
    if f.empty:
        return f"{LABELS[food]} in {name}: no price reported for {last:%B %Y}, so no forecast"
    tr, limit = f.trend.iat[0], f.trend.iat[0] * (1 + thr)
    now = s_last_price(f) / tr - 1
    if f.alert.any() and now > thr:
        return (f"{LABELS[food]} in {name}: price already {now:.0%} above normal, "
                f"forecast stays above the alert line")
    if f.alert.any():
        first = f[f.alert].iloc[0]
        return (f"{LABELS[food]} in {name}: spike likely by {first.target:%B %Y}, "
                f"{first.yhat / tr - 1:.0%} above the normal price")
    return (f"{LABELS[food]} in {name}: no spike expected, forecast stays "
            f"{1 - f.yhat.max() / limit:.0%} below the alert line")


with tab_m:
    s = p[(p.food == food) & (p.market_id == mid)].set_index("month").sort_index()
    f = f_food[f_food.market_id == mid].sort_values("h")
    name = mk.at[mid, "market"]
    st.subheader(headline(f, name))
    st.caption(f"{str(mk.at[mid, 'district']).title()}, {str(mk.at[mid, 'province']).title()}. "
               f"Model: {MODEL_NAMES[model]}. Alert when the forecast is more than {thr:.0%} above the "
               f"normal price (median of the last {TREND_WINDOW} months).")

    if not f.empty:
        latest = s.price_clean.dropna().iat[-1]
        tr = f.trend.iat[0]
        c = st.columns(3)
        c[0].metric(f"Price now ({last:%b %Y})", f"{rp(latest)}/kg",
                    f"{latest / tr - 1:+.1%} vs normal", delta_color="inverse")
        c[1].metric("Normal price", f"{rp(tr)}/kg")
        c[2].metric(f"Forecast {f.target.iat[-1]:%b %Y}", f"{rp(f.yhat.iat[-1])}/kg",
                    f"{f.yhat.iat[-1] / latest - 1:+.1%} vs now", delta_color="inverse")

    compare = st.toggle("Compare models", value=False)
    allm = fc[(fc.food == food) & (fc.market_id == mid)]
    st.plotly_chart(market_chart(s, f, allm, compare, name), width="stretch")
    st.caption("Black: reported price. Grey: normal price. Blue: forecast. Shaded right part: the next "
               "3 months; red area: above the alert line.")

    if not f.empty:
        t = allm.pivot_table(index="target", columns="model", values="yhat").rename(columns=MODEL_NAMES)
        t.index = t.index.strftime("%b %Y")
        t.index.name = "Month"
        with st.expander("Forecast numbers, all models"):
            st.dataframe(t.style.format("Rp {:,.0f}"), width="stretch")

# ---------------------------------------------------------------- alerts
with tab_a:
    a = f_food[f_food.alert]
    per = a.groupby("market_id").agg(market=("market", "first"), province=("province", "first"),
                                     district=("district", "first"), first_month=("target", "min"),
                                     forecast=("yhat", "max"), trend=("trend", "first"),
                                     above=("vs_trend", "max")).sort_values("above", ascending=False)
    st.subheader(f"{LABELS[food]}: {len(per)} of {f_food.market_id.nunique()} markets flagged "
                 f"({MODEL_NAMES[model]}, alert at +{thr:.0%})")
    if per.empty:
        st.info("No market forecast is above the alert line.")
    else:
        geo = p.groupby("market_id")[["latitude", "longitude"]].first()
        mp = per.join(geo).dropna(subset=["latitude"])
        st.map(mp.rename(columns={"latitude": "lat", "longitude": "lon"}), color=C_ALERT, size=9000)
        show = per.assign(first_month=per.first_month.dt.strftime("%b %Y"),
                          province=per.province.str.title(), district=per.district.str.title())
        st.dataframe(show.style.format({"forecast": "Rp {:,.0f}", "trend": "Rp {:,.0f}", "above": "{:+.0%}"}),
                     width="stretch", hide_index=True,
                     column_config={"first_month": "First alert month", "forecast": "Peak forecast",
                                    "trend": "Normal price", "above": "Above normal"})

# ---------------------------------------------------------------- performance
with tab_p:
    st.subheader("How good are the forecasts?")
    mm = metrics[metrics.food == food]
    piv = mm.pivot_table(index="h", columns="model", values="mape")
    rel = (piv[["sarima", "lgbm"]].div(piv.naive, axis=0) - 1) * 100
    bar = go.Figure()
    for k in ("sarima", "lgbm"):
        bar.add_bar(x=[f"{h} month{'s' if h > 1 else ''} ahead" for h in rel.index], y=rel[k],
                    name=MODEL_NAMES[k], marker_color=C_MODEL[k],
                    text=[f"{v:+.0f}%" for v in rel[k]], textposition="outside",
                    hovertemplate=f"{MODEL_NAMES[k]}: %{{y:+.1f}}% error vs naive<extra></extra>")
    bar.add_hline(y=0, line=dict(color=INK, width=1))
    bar.update_layout(height=340, barmode="group", plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(title="Error vs naive (%)", ticksuffix="%", gridcolor=GRID, zeroline=False),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(bar, width="stretch")
    st.caption("0 = as good as the naive forecast (next month = this month). Above 0 = worse, below 0 = better. "
               "Rolling-origin backtest: each forecast uses only data up to its origin month.")
    with st.expander("Full error table"):
        t = mm.copy()
        t["model"] = t.model.map(MODEL_NAMES)
        st.dataframe(t.drop(columns="food").style.format(
            {"mape": "{:.2%}", "median_ape": "{:.2%}", "within_5pct": "{:.0%}", "bias": "{:+.2%}",
             "skill_vs_naive": "{:+.0%}"}), width="stretch", hide_index=True)
    st.subheader(f"Did the spike alert work? (alert at +{SPIKE_THRESHOLD:.0%})")
    aa = alerts_bt[alerts_bt.food == food].copy()
    aa["model"] = aa.model.map(MODEL_NAMES)
    aa = aa.rename(columns={"h": "months ahead", "actual_spikes": "real spikes", "hits": "caught"})
    st.dataframe(aa.drop(columns="food").style.format({"precision": "{:.0%}", "recall": "{:.0%}"}, na_rep="-"),
                 width="stretch", hide_index=True)
    st.caption("Recall = share of real spikes that got an alert. Precision = share of alerts that were real.")
