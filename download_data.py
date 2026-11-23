"""Step 1: download the WFP Indonesia food price data from HDX (CC BY-IGO).

Dataset: https://data.humdata.org/dataset/wfp-food-prices-for-indonesia
The file URLs are looked up through the HDX API, so the script keeps working when HDX
publishes a new version of the files.
"""
import json
import urllib.request

from common import RAW_MARKETS, RAW_PRICES

API = "https://data.humdata.org/api/3/action/package_show?id=wfp-food-prices-for-indonesia"
WANTED = {"wfp_food_prices_idn.csv": RAW_PRICES, "wfp_markets_idn.csv": RAW_MARKETS}


def main():
    with urllib.request.urlopen(API, timeout=60) as r:
        pkg = json.load(r)["result"]
    print(f"{pkg['title']} (updated {pkg['last_modified'][:10]}, licence {pkg['license_title']})")
    for res in pkg["resources"]:
        name = res["url"].rsplit("/", 1)[-1]
        if name in WANTED:
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(res["url"], WANTED[name])
    missing = [n for n, p in WANTED.items() if not p.exists()]
    if missing:
        raise SystemExit(f"not found on HDX: {missing}")


if __name__ == "__main__":
    main()
