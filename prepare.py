"""Steps 2 and 3: clean, explore, and add context data.

Outputs
  data/monthly.parquet      one row per food x market x month (2016-07 .. latest), raw and clean price
  data/context.parquet      monthly IDR/USD rate, CPI index, annual inflation
  data/worldbank_idn.csv    cached World Bank download
  reports/gaps.csv          per series: span, missing months, longest gap
  reports/outliers.csv      every flagged outlier
  reports/explore.md        key numbers
  figures/*.png
"""
import json
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (C_ALERT, C_MUTED, C_SERIES, C_TEXT, DATA, FIG, FOODS, LABELS, MODEL_START,
                    RAW_MARKETS, RAW_PRICES, REPORTS, style_axes)

OUT_WIN = 5          # centred rolling median window for outlier check
OUT_Z = 5.0          # robust z-score limit
OUT_MIN_DEV = 0.15   # and at least 15% away from the local median
FILL_LIMIT = 2       # fill gaps up to 2 months (log-linear), longer gaps stay empty


# ---------------------------------------------------------------- load
def load_raw():
    d = pd.read_csv(RAW_PRICES, skiprows=[1], parse_dates=["date"])  # row 1 holds HXL tags
    return d


def staple_panel(raw):
    name_to_key = {v: k for k, v in FOODS.items()}
    d = raw[raw.commodity.isin(FOODS.values()) & (raw.unit == "KG") & (raw.market != "National Average")].copy()
    d["food"] = d.commodity.map(name_to_key)
    d["month"] = d.date.dt.to_period("M").dt.to_timestamp()
    # Monthly average per market and food. WFP already reports one value per month (dated the 15th),
    # so this mostly merges duplicates; it also makes the step correct if weekly data is added.
    m = (d.groupby(["food", "market_id", "month"], as_index=False)
           .agg(price=("price", "mean"), n_obs=("price", "size")))
    mk = pd.read_csv(RAW_MARKETS, skiprows=[1])[["market_id", "market", "admin1", "admin2", "latitude", "longitude"]]
    mk = mk.rename(columns={"admin1": "province", "admin2": "district"})
    return m.merge(mk, on="market_id", how="left")


# ---------------------------------------------------------------- gaps + outliers
def complete_months(m):
    """Reindex each series to every month from its first to its last report, so gaps become NaN rows."""
    out = []
    for (food, mid), g in m.groupby(["food", "market_id"]):
        idx = pd.date_range(g.month.min(), g.month.max(), freq="MS")
        g = g.set_index("month").reindex(idx)
        g.index.name = "month"
        g["food"], g["market_id"] = food, mid
        for c in ("market", "province", "district", "latitude", "longitude"):
            g[c] = g[c].ffill().bfill()
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def flag_outliers(p):
    """Robust check of each market against its own recent level, after removing the all-market move.

    rel = log price - log median of all markets that month. A national shock (the 2022 cooking-oil
    crisis) moves every market and leaves rel flat, so it is not flagged; a single market jumping
    away from its neighbours and from its own history is.
    """
    lp = np.log(p["price"])
    nat = lp.groupby([p.food, p.month]).transform("median")
    rel = lp - nat
    med = rel.groupby([p.food, p.market_id]).transform(
        lambda s: s.rolling(OUT_WIN, center=True, min_periods=3).median())
    resid = rel - med
    mad = resid.abs().groupby([p.food, p.market_id]).transform("median") * 1.4826
    scale = mad.clip(lower=0.02)  # flat series: do not flag tiny moves
    p["z"] = resid / scale
    p["outlier"] = (p.z.abs() > OUT_Z) & (resid.abs() > OUT_MIN_DEV)
    return p


def clean_series(p):
    p["price_clean"] = p.price.where(~p.outlier)
    lp = np.log(p.price_clean)
    filled = lp.groupby([p.food, p.market_id]).transform(
        lambda s: s.interpolate(limit=FILL_LIMIT, limit_area="inside"))
    p["imputed"] = p.price_clean.isna() & filled.notna()
    p["price_clean"] = np.exp(filled)
    return p


def gap_report(p, last_month):
    rows = []
    for (food, mid), g in p[p.month >= MODEL_START].groupby(["food", "market_id"]):
        miss = g.price.isna().to_numpy()
        longest = run = 0
        for x in miss:
            run = run + 1 if x else 0
            longest = max(longest, run)
        rows.append(dict(food=food, market_id=mid, market=g.market.iat[0], province=g.province.iat[0],
                         first=g.month.min().date(), last=g.month.max().date(),
                         months=len(g), missing=int(miss.sum()), longest_gap=longest,
                         outliers=int(g.outlier.sum()),
                         stale=g.month.max() < last_month - pd.DateOffset(months=2)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- context data
WB = {"FP.CPI.TOTL": "cpi_2010", "FP.CPI.TOTL.ZG": "inflation_pct", "PA.NUS.FCRF": "idr_usd_wb"}


def world_bank():
    cache = DATA / "worldbank_idn.csv"
    try:
        frames = []
        for code, col in WB.items():
            url = f"https://api.worldbank.org/v2/country/IDN/indicator/{code}?format=json&per_page=200"
            with urllib.request.urlopen(url, timeout=30) as r:
                js = json.load(r)[1]
            frames.append(pd.DataFrame({"year": [int(x["date"]) for x in js],
                                        col: [x["value"] for x in js]}).set_index("year"))
        wb = pd.concat(frames, axis=1).sort_index().dropna(how="all")
        wb.to_csv(cache)
    except Exception as e:  # offline: use the cached copy
        print(f"World Bank download failed ({e}); using cache")
        wb = pd.read_csv(cache, index_col="year")
    return wb


def context(raw, months):
    # Monthly exchange rate implied by WFP's own conversion (price / usdprice). usdprice is rounded
    # to cents, so only rows with usdprice >= 5 are used, and the median is taken across them.
    r = raw[raw.usdprice >= 5]
    fx = (r.price / r.usdprice).groupby(r.date.dt.to_period("M").dt.to_timestamp()).median().rename("idr_usd")
    wb = world_bank()
    # Annual CPI placed at mid-year, interpolated log-linearly by month; months after the last
    # published year are carried forward with the last annual inflation rate.
    cpi = wb.cpi_2010.dropna()
    anchors = pd.Series(np.log(cpi.values), index=pd.to_datetime([f"{y}-07-01" for y in cpi.index]))
    idx = pd.date_range(min(anchors.index.min(), months.min()), months.max(), freq="MS")
    lc = anchors.reindex(anchors.index.union(idx)).interpolate(method="time", limit_area="inside")
    last_rate = np.log1p(wb.inflation_pct.dropna().iat[-1] / 100) / 12
    tail = lc.index > anchors.index.max()
    lc[tail] = anchors.iat[-1] + last_rate * np.arange(1, tail.sum() + 1)
    ctx = pd.DataFrame({"cpi": np.exp(lc.reindex(idx))}, index=idx)
    ctx["idr_usd"] = fx.reindex(idx).interpolate(limit_area="inside")
    ctx["year"] = ctx.index.year
    ctx = ctx.join(wb[["inflation_pct", "idr_usd_wb"]], on="year").drop(columns="year")
    ctx.index.name = "month"
    return ctx, wb


# ---------------------------------------------------------------- figures
def fig_prices(p, food):
    g = p[p.food == food]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for _, s in g.groupby("market_id"):
        ax.plot(s.month, s.price_clean, color=C_MUTED, lw=0.5, alpha=0.5)
    nat = g.groupby("month").price_clean.median()
    ax.plot(nat.index, nat.values, color=C_SERIES[food], lw=2, label="Median of all markets")
    ax.axvspan(pd.Timestamp("2017-04-01"), pd.Timestamp("2019-12-01"), color="#f3f2ee", zorder=0)
    y0, y1 = ax.get_ylim()
    ax.text(pd.Timestamp("2018-08-01"), (y0 + y1) / 2, "no market data\n2017-04 to 2019-12", ha="center",
            va="center", color=C_TEXT, fontsize=9)
    ax.plot([], [], color=C_MUTED, lw=1, label="One market")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title(f"{LABELS[food]}: retail price per kg, {g.market_id.nunique()} markets (IDR)",
                 loc="left", fontsize=11, color="#0b0b0b")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG / f"prices_{food}.png", dpi=130)
    plt.close(fig)


def fig_coverage(p):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for food in FOODS:
        c = p[(p.food == food)].groupby("month").price.count()
        c = c.reindex(pd.date_range(c.index.min(), c.index.max(), freq="MS"), fill_value=0)
        ax.plot(c.index, c.values, color=C_SERIES[food], lw=2, label=LABELS[food])
    ax.set_title("Markets reporting each month (gaps: 2017-01 to 2017-02, 2017-04 to 2019-12, 2024-02 to 2024-05, 2026-01)",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG / "coverage.png", dpi=130)
    plt.close(fig)


def fig_outliers(p, out):
    out = out[out.month >= MODEL_START]
    worst = out.groupby(["food", "market_id"]).size().sort_values(ascending=False)
    picks = [worst.xs(f, level="food").index[0] for f in FOODS if f in worst.index.get_level_values(0)]
    fig, axes = plt.subplots(1, len(picks), figsize=(12, 3.6), sharey=False)
    for ax, food, mid in zip(np.atleast_1d(axes), FOODS, picks):
        s = p[(p.food == food) & (p.market_id == mid) & (p.month >= MODEL_START)]
        ax.plot(s.month, s.price, color=C_SERIES[food], lw=1.5)
        o = s[s.outlier]
        ax.scatter(o.month, o.price, s=40, facecolor="white", edgecolor=C_ALERT, lw=2, zorder=3,
                   label="Flagged outlier")
        ax.set_title(f"{LABELS[food]}: {s.market.iat[0]}", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.tick_params(axis="x", rotation=30)
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG / "outliers_examples.png", dpi=130)
    plt.close(fig)


def fig_real(p, ctx):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    base = ctx.cpi.iat[-1]
    for ax, food in zip(axes, FOODS):
        nat = p[(p.food == food) & (p.month >= MODEL_START)].groupby("month").price_clean.median()
        real = nat * base / ctx.cpi.reindex(nat.index)
        ax.plot(nat.index, nat.values, color=C_SERIES[food], lw=2, label="Nominal")
        ax.plot(real.index, real.values, color=C_SERIES[food], lw=2, ls=(0, (3, 2)),
                label=f"Real ({ctx.index[-1]:%b %Y} prices)")
        ax.set_title(LABELS[food], loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.tick_params(axis="x", rotation=30)
        style_axes(ax)
    fig.suptitle("Median market price per kg (IDR): nominal vs inflation-adjusted", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "nominal_vs_real.png", dpi=130)
    plt.close(fig)


def fig_fx(ctx):
    c = ctx[ctx.index >= "2016-01-01"]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(c.index, c.idr_usd, color=C_SERIES["rice"], lw=2, label="Monthly, implied by WFP (price / usdprice)")
    wb = c.groupby(c.index.year).idr_usd_wb.first().dropna()
    ax.scatter(pd.to_datetime([f"{y}-07-01" for y in wb.index]), wb.values, s=40, color=C_SERIES["sugar"],
               zorder=3, label="Annual average, World Bank")
    ax.set_title("IDR per USD", loc="left", fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fx.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    raw = load_raw()
    m = staple_panel(raw)
    p = complete_months(m)
    p = flag_outliers(p)
    p = clean_series(p)
    last = p.month.max()
    p = p.sort_values(["food", "market_id", "month"]).reset_index(drop=True)
    p.to_parquet(DATA / "monthly.parquet", index=False)

    gaps = gap_report(p, last)
    gaps.to_csv(REPORTS / "gaps.csv", index=False)
    out = p[p.outlier][["food", "market_id", "market", "province", "month", "price", "z"]]
    out.to_csv(REPORTS / "outliers.csv", index=False)

    ctx, wb = context(raw, p.month)
    ctx.to_parquet(DATA / "context.parquet")

    for food in FOODS:
        fig_prices(p, food)
    fig_coverage(p)
    fig_outliers(p, out)
    fig_real(p, ctx)
    fig_fx(ctx)

    # key numbers
    lines = ["# Explore: key numbers", "", f"Latest month: {last:%Y-%m}", ""]
    lines.append("| Food | Markets | Rows (2020+) | Missing months (2020+) | Series with gap > 2 months | Outliers | Stale series |")
    lines.append("|---|---|---|---|---|---|---|")
    for food in FOODS:
        g = gaps[gaps.food == food]
        rows = p[(p.food == food) & (p.month >= MODEL_START)]
        lines.append(f"| {LABELS[food]} | {len(g)} | {rows.price.notna().sum():,} | {g.missing.sum():,} "
                     f"({g.missing.sum() / g.months.sum():.1%}) | {(g.longest_gap > 2).sum()} | {g.outliers.sum()} | {g.stale.sum()} |")
    lines += ["", "Median price per kg (IDR), all markets:", "",
              "| Food | 2020-01 | latest | change |", "|---|---|---|---|"]
    for food in FOODS:
        nat = p[p.food == food].groupby("month").price_clean.median()
        a, b = nat.loc["2020-01-01"], nat.iat[-1]
        lines.append(f"| {LABELS[food]} | {a:,.0f} | {b:,.0f} | {b / a - 1:+.0%} |")
    infl = wb.inflation_pct.dropna()
    lines += ["", f"CPI change 2020-01 to {last:%Y-%m}: {ctx.cpi.iat[-1] / ctx.cpi.loc['2020-01-01'] - 1:+.1%} "
              f"(World Bank CPI to {int(wb.cpi_2010.dropna().index.max())}, then {infl.iat[-1]:.1f}%/yr carried forward)",
              f"IDR/USD (WFP implied): 2020-01 {ctx.idr_usd.loc['2020-01-01']:,.0f}, latest {ctx.idr_usd.dropna().iat[-1]:,.0f}"]
    (REPORTS / "explore.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
