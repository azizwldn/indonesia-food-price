"""Step 4: forecast staple prices 1-3 months ahead per market and flag spikes.

Models
  naive   next months equal this month
  sarima  SARIMA(1,1,1)x(1,0,0,12) on log price, one fit per market series
  lgbm    LightGBM on lag features, one global model per food and horizon (all markets pooled),
          predicting the log change from this month

Alert: forecast > (1 + SPIKE_THRESHOLD) x trend, where trend is the trailing 12-month median price.

Backtest: rolling origin. At each origin only data up to that month is used (also for the
LightGBM training rows and the context features), then the forecast is compared with the price
actually reported 1, 2 and 3 months later.

Outputs
  data/backtest.parquet        every backtest prediction
  data/forecasts.parquet       forecasts from the latest month, all models, with trend and alert
  reports/backtest_metrics.csv  MAPE and more, per food x model x horizon
  reports/backtest_alerts.csv   spike alert precision / recall
  reports/model.md              summary tables
  figures/backtest_mape.png

Run: python model.py          full backtest, 22 origins (about 12 min on 8 cores)
     python model.py --quick  last 6 origins only (about 3 min), same outputs
"""
import os
import sys

# One BLAS thread per process: the SARIMA pool runs one fit per core, and 8 workers x 8 BLAS
# threads each overloads the CPU (load 65 on 8 cores, run many times slower).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import warnings
from multiprocessing import Pool

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (C_MODEL, DATA, FIG, FOODS, HORIZONS, LABELS, MODEL_START, REPORTS, SPIKE_THRESHOLD,
                    TREND_WINDOW, style_axes)

MIN_OBS = 24            # SARIMA needs at least two years of points
MIN_MARKETS = 150       # skip origins where most markets did not report (2024-02..05, 2026-01)
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL = (1, 0, 0, 12)
MODELS = ("naive", "sarima", "lgbm")
LGB_PARAMS = dict(objective="l1", n_estimators=500, learning_rate=0.03, num_leaves=15,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, verbose=-1, n_jobs=os.cpu_count())


# ---------------------------------------------------------------- data
def load():
    p = pd.read_parquet(DATA / "monthly.parquet")
    p = p[p.month >= MODEL_START]
    ctx = pd.read_parquet(DATA / "context.parquet")
    return p, ctx


def wide(p, food):
    """month x market matrix of log clean price, plus the observed (not imputed) mask."""
    g = p[p.food == food]
    months = pd.date_range(MODEL_START, p.month.max(), freq="MS")
    ly = np.log(g.pivot(index="month", columns="market_id", values="price_clean")).reindex(months)
    obs = g.assign(o=g.price_clean.notna() & ~g.imputed).pivot(
        index="month", columns="market_id", values="o").reindex(months).fillna(False).astype(bool)
    prov = g.groupby("market_id").province.first()
    return ly, obs, prov


def trend(ly):
    """Trailing 12-month median price ('normal' level) at each month."""
    return np.exp(ly).rolling(TREND_WINDOW, min_periods=6).median()


# ---------------------------------------------------------------- features for LightGBM
def features(ly, prov, ctx):
    """Long table: one row per (month t, market) with features known at t."""
    nat = ly.median(axis=1)
    fx = np.log(ctx.idr_usd.reindex(ly.index)).ffill()
    infl_prev = ctx.inflation_pct.groupby(ctx.index.year).first().shift(1)  # last full year's rate
    prov_codes = prov.astype("category").cat.codes
    pmed = ly.T.groupby(prov).median().T  # province median per month
    pmed_m = pmed.reindex(columns=prov.reindex(ly.columns).values)
    pmed_m.columns = ly.columns

    f = {}
    for k in (1, 2, 3, 6, 12):
        f[f"d{k}"] = ly - ly.shift(k)
    f["vs_nat"] = ly.sub(nat, axis=0)
    f["vs_nat_d12"] = f["vs_nat"] - f["vs_nat"].shift(12)
    f["vs_trend"] = ly - np.log(trend(ly))
    f["vol12"] = (ly - ly.shift(1)).rolling(12, min_periods=6).std()
    f["vs_prov"] = ly - pmed_m
    f["prov_d3"] = pmed_m - pmed_m.shift(3)
    long = pd.concat({k: v.stack(future_stack=True) for k, v in f.items()}, axis=1)
    long.index.names = ["month", "market_id"]
    long = long.reset_index()
    m = long.month
    for k in (1, 3, 12):
        long[f"nat_d{k}"] = m.map(nat - nat.shift(k))
        long[f"fx_d{k}"] = m.map(fx - fx.shift(k))
    long["infl_prev"] = m.dt.year.map(infl_prev)
    long["moy"] = m.dt.month
    long["province"] = long.market_id.map(prov_codes).astype("category")
    long["ly"] = ly.stack(future_stack=True).values
    return long


FEATS = None  # filled on first use


def lgbm_forecast(feat, ly, obs, origin, h):
    """Train on rows whose target month <= origin, predict from origin."""
    global FEATS
    if FEATS is None:
        FEATS = [c for c in feat.columns if c not in ("month", "market_id", "ly")] + ["tmoy"]
    tgt = (ly.shift(-h) - ly).where(obs.shift(-h, fill_value=False) & obs)
    t = feat.copy()
    t["y"] = tgt.stack(future_stack=True).values
    t["tmoy"] = (t.month + pd.DateOffset(months=h)).dt.month
    train = t[(t.month + pd.DateOffset(months=h) <= origin) & t.y.notna()]
    test = t[(t.month == origin) & t.ly.notna()]
    mdl = lgb.LGBMRegressor(**LGB_PARAMS).fit(train[FEATS], train.y)
    pred = pd.Series(test.ly.values + mdl.predict(test[FEATS]), index=test.market_id.values)
    return pred, mdl


# ---------------------------------------------------------------- SARIMA
def _sarima_one(args):
    key, y = args
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    warnings.filterwarnings("ignore")
    y = y[np.argmax(~np.isnan(y)):]  # drop leading NaNs, keep inner gaps (Kalman filter handles them)
    if np.isfinite(y).sum() < MIN_OBS or np.isnan(y[-1]):
        return key, None
    try:
        res = SARIMAX(y, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL).fit(disp=False, maxiter=60)
        fc = res.get_forecast(max(HORIZONS))
        ci = fc.conf_int(alpha=0.2)
        out = np.column_stack([fc.predicted_mean, ci])
        if not np.all(np.isfinite(out)) or np.any(np.abs(out[:, 0] - y[-1]) > 1.0):
            return key, None  # diverged
        return key, out
    except Exception:
        return key, None


def sarima_batch(jobs):
    out, t0 = {}, time.time()
    with Pool(os.cpu_count()) as pool:
        for i, (k, v) in enumerate(pool.imap_unordered(_sarima_one, jobs, chunksize=8), 1):
            out[k] = v
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)} fits, {time.time() - t0:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------- one origin
def forecast_at(food, origin, ly, obs, feat, sarima_res):
    """All models from one origin; returns long rows (market, h, model, yhat, lo, hi)."""
    rows = []
    ly0 = ly.loc[origin]
    tr = trend(ly).loc[origin]
    lg = {h: lgbm_forecast(feat, ly, obs, origin, h)[0] for h in HORIZONS}
    for mid in ly.columns[obs.loc[origin]]:
        base = dict(food=food, origin=origin, market_id=mid, trend=tr[mid])
        sr = sarima_res.get((food, origin, mid))
        for h in HORIZONS:
            b = dict(base, h=h, target=origin + pd.DateOffset(months=h))
            rows.append(dict(b, model="naive", yhat=np.exp(ly0[mid])))
            if sr is not None:
                rows.append(dict(b, model="sarima", yhat=np.exp(sr[h - 1, 0]),
                                 lo=np.exp(sr[h - 1, 1]), hi=np.exp(sr[h - 1, 2])))
            else:  # failed fit: fall back to naive, marked
                rows.append(dict(b, model="sarima", yhat=np.exp(ly0[mid]), fallback=True))
            if mid in lg[h].index:
                rows.append(dict(b, model="lgbm", yhat=np.exp(lg[h][mid])))
    return rows


def sarima_jobs(food, origin, ly, obs):
    cols = ly.columns[obs.loc[origin]]
    hist = ly.loc[:origin]
    return [((food, origin, mid), hist[mid].to_numpy()) for mid in cols]


# ---------------------------------------------------------------- metrics
def metrics(bt):
    bt = bt.dropna(subset=["actual"]).copy()
    bt["ape"] = (bt.yhat - bt.actual).abs() / bt.actual
    bt["err"] = (bt.yhat - bt.actual) / bt.actual
    g = bt.groupby(["food", "model", "h"])
    m = g.agg(n=("ape", "size"), mape=("ape", "mean"), median_ape=("ape", "median"),
              within_5pct=("ape", lambda s: (s <= 0.05).mean()), bias=("err", "mean")).reset_index()
    naive = m[m.model == "naive"].set_index(["food", "h"]).mape
    m["skill_vs_naive"] = 1 - m.mape / m.set_index(["food", "h"]).index.map(naive)
    return m, bt


def alert_metrics(bt):
    bt = bt.dropna(subset=["actual", "trend"]).copy()
    lim = bt.trend * (1 + SPIKE_THRESHOLD)
    bt["pred_spike"] = bt.yhat > lim
    bt["true_spike"] = bt.actual > lim
    rows = []
    for (food, model, h), g in bt.groupby(["food", "model", "h"]):
        tp = (g.pred_spike & g.true_spike).sum()
        rows.append(dict(food=food, model=model, h=h, actual_spikes=int(g.true_spike.sum()),
                         alerts=int(g.pred_spike.sum()), hits=int(tp),
                         precision=tp / g.pred_spike.sum() if g.pred_spike.any() else np.nan,
                         recall=tp / g.true_spike.sum() if g.true_spike.any() else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    p, ctx = load()
    last = p.month.max()
    grid = list(pd.date_range("2021-03-01", last - pd.DateOffset(months=max(HORIZONS)), freq="3MS"))
    grid.append(last - pd.DateOffset(months=max(HORIZONS)))
    per_food = {}
    jobs = []
    for food in FOODS:
        ly, obs, prov = wide(p, food)
        feat = features(ly, prov, ctx)
        origins = [o for o in dict.fromkeys(grid) if obs.loc[o].sum() >= MIN_MARKETS]
        if "--quick" in sys.argv:
            origins = origins[-6:]
        per_food[food] = (ly, obs, feat, origins)
        for o in origins + [last]:
            jobs += sarima_jobs(food, o, ly, obs)
    print(f"SARIMA fits: {len(jobs)} on {os.cpu_count()} cores", flush=True)
    sres = sarima_batch(jobs)
    print(f"SARIMA failed or diverged: {sum(v is None for v in sres.values())}", flush=True)

    bt_rows, fc_rows, importances = [], [], {}
    for food, (ly, obs, feat, origins) in per_food.items():
        print(f"{food}: origins {[f'{o:%Y-%m}' for o in origins]}", flush=True)
        actual = np.exp(ly.where(obs))
        for o in origins:
            for r in forecast_at(food, o, ly, obs, feat, sres):
                r["actual"] = actual.at[r["target"], r["market_id"]] if r["target"] <= last else np.nan
                bt_rows.append(r)
        fc_rows += forecast_at(food, last, ly, obs, feat, sres)
        mdl = lgbm_forecast(feat, ly, obs, last, 1)[1]
        importances[food] = pd.Series(mdl.booster_.feature_importance("gain"), index=FEATS)

    bt = pd.DataFrame(bt_rows)
    bt["fallback"] = bt.get("fallback", False)
    bt["fallback"] = bt["fallback"].fillna(False).astype(bool)
    m, bt = metrics(bt)
    al = alert_metrics(bt)
    bt.to_parquet(DATA / "backtest.parquet", index=False)
    m.to_csv(REPORTS / "backtest_metrics.csv", index=False)
    al.to_csv(REPORTS / "backtest_alerts.csv", index=False)

    # Best model per food: lowest MAPE averaged over the three horizons.
    best = m.groupby(["food", "model"]).mape.mean().unstack().idxmin(axis=1)
    fc = pd.DataFrame(fc_rows)
    fc["fallback"] = fc.get("fallback", False)
    fc["fallback"] = fc["fallback"].fillna(False).astype(bool)
    fc["threshold"] = fc.trend * (1 + SPIKE_THRESHOLD)
    fc["vs_trend"] = fc.yhat / fc.trend - 1
    fc["alert"] = fc.yhat > fc.threshold
    fc["best"] = fc.model == fc.food.map(best)
    mk = p.groupby("market_id")[["market", "province", "district"]].first()
    fc = fc.join(mk, on="market_id")
    fc.to_parquet(DATA / "forecasts.parquet", index=False)

    # Recent-period MAPE too (origins from 2025), the period closest to how the app is used.
    recent, _ = metrics(bt[bt.origin >= "2025-01-01"].drop(columns=["ape", "err"]))
    report(m, recent, al, best, fc, importances, last, bt)
    fig_mape(m)


def report(m, recent, al, best, fc, imp, last, bt):
    L = [f"# Model results (data to {last:%Y-%m})", ""]
    L += [f"Backtest origins: {bt.origin.nunique()} ({bt.origin.min():%Y-%m} to {bt.origin.max():%Y-%m}), "
          f"{len(bt[bt.model == 'naive']):,} forecasts per model. SARIMA fallbacks to naive: "
          f"{bt[bt.model == 'sarima'].fallback.mean():.1%}.", ""]
    for title, t in (("All origins", m), ("Origins from 2025", recent)):
        L += [f"## MAPE, {title.lower()}", "", "| Food | h | Naive | SARIMA | LightGBM | Best vs naive |",
              "|---|---|---|---|---|---|"]
        piv = t.pivot_table(index=["food", "h"], columns="model", values="mape")
        for (food, h), r in piv.iterrows():
            bm = r[["sarima", "lgbm"]].idxmin()
            L.append(f"| {LABELS[food]} | {h} | {r.naive:.2%} | {r.sarima:.2%} | {r.lgbm:.2%} | "
                     f"{bm}: {1 - r[bm] / r.naive:+.0%} |")
        L.append("")
    L += ["Best vs naive = error reduction of the better model (positive = better than naive).", "",
          "## Spike alert (forecast > 15% above trailing 12-month median), all origins", "",
          "| Food | Model | h | Real spikes | Alerts | Hits | Precision | Recall |", "|---|---|---|---|---|---|---|---|"]
    for _, r in al.iterrows():
        L.append(f"| {LABELS[r.food]} | {r.model} | {r.h} | {r.actual_spikes} | {r.alerts} | {r.hits} | "
                 f"{r.precision:.0%} | {r.recall:.0%} |".replace("nan%", "-"))
    L += ["", "## Model used in the app per food (lowest mean MAPE over h = 1-3)", ""]
    L += [f"- {LABELS[f]}: {b}" for f, b in best.items()]
    cur = fc[fc.best & fc.alert]
    L += ["", f"## Current alerts from {last:%Y-%m} (best model)", "",
          f"{cur.market_id.nunique()} market-food pairs flagged: " +
          ", ".join(f"{LABELS[f]} {n}" for f, n in cur.groupby("food").market_id.nunique().items()), ""]
    L += ["## LightGBM top features (gain, h = 1)", ""]
    for f, s in imp.items():
        s = (s / s.sum()).sort_values(ascending=False).head(6)
        L.append(f"- {LABELS[f]}: " + ", ".join(f"{k} {v:.0%}" for k, v in s.items()))
    (REPORTS / "model.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def fig_mape(m):
    """Error of each model relative to naive: 0 = as good as naive, above 0 = worse."""
    piv = m.pivot_table(index=["food", "h"], columns="model", values="mape")
    rel = (piv[["sarima", "lgbm"]].div(piv.naive, axis=0) - 1) * 100
    fig, axes = plt.subplots(1, len(FOODS), figsize=(12, 3.8), sharey=True)
    w = 0.36
    for ax, food in zip(axes, FOODS):
        r = rel.loc[food]
        for i, model in enumerate(("sarima", "lgbm")):
            x = np.array(r.index) + (i - 0.5) * w
            bars = ax.bar(x, r[model], width=w - 0.04, color=C_MODEL[model],
                          label={"sarima": "SARIMA", "lgbm": "LightGBM"}[model])
            for xi, v in zip(x, r[model]):
                ax.text(xi, v + (1 if v >= 0 else -1), f"{v:+.0f}%", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8, color="#52514e")
        ax.axhline(0, color="#0b0b0b", lw=1)
        ax.set_xticks(list(HORIZONS), [f"{h} mo ahead" for h in HORIZONS])
        ax.set_title(LABELS[food], loc="left", fontsize=10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
        style_axes(ax)
    axes[0].set_ylabel("Error vs naive")
    axes[0].text(0.02, 0.98, "0 = naive (next month = this month)", transform=axes[0].transAxes,
                 fontsize=8, va="top", color="#52514e")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Neither model beats the naive forecast: bars above 0 mean more error than naive",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "backtest_mape.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
