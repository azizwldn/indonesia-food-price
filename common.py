"""Shared settings and helpers for the Indonesia food price project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_PRICES = ROOT / "wfp_food_prices_idn.csv"
RAW_MARKETS = ROOT / "wfp_markets_idn.csv"
DATA = ROOT / "data"
FIG = ROOT / "figures"
REPORTS = ROOT / "reports"
for p in (DATA, FIG, REPORTS):
    p.mkdir(exist_ok=True)

# Staple foods: short key -> WFP commodity name. All three are per KG at market level.
# Maize is not in the WFP Indonesia file; wheat flour has only one national series that ends in 2020.
FOODS = {
    "rice": "Rice (medium quality)",
    "sugar": "Sugar (local)",
    "oil": "Oil (vegetable)",
}
LABELS = {"rice": "Rice (medium)", "sugar": "Sugar (local)", "oil": "Vegetable oil"}

# The market panel is continuous from 2020-01; 2017-04 to 2019-12 is missing for almost all markets.
MODEL_START = "2020-01-01"
HORIZONS = (1, 2, 3)
SPIKE_THRESHOLD = 0.15   # alert when forecast > trend * (1 + threshold)
TREND_WINDOW = 12        # trailing months used for the "normal" trend

# Chart colours (validated categorical palette, light mode).
C_SERIES = {"rice": "#2a78d6", "sugar": "#eb6834", "oil": "#1baf7a"}
C_MODEL = {"naive": "#8a8984", "sarima": "#2a78d6", "lgbm": "#eb6834"}
C_MUTED = "#c9c8c2"
C_TEXT = "#52514e"
C_ALERT = "#d03b3b"


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)
