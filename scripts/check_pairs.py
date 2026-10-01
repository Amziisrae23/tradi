import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import requests
from config.settings import settings

def main():
    r = requests.get('https://fapi.bitunix.com/api/v1/futures/market/trading_pairs').json()
    pairs = {p['symbol']: p for p in r.get('data', [])}
    print("SYMBOL_SPECS = {")
    for s in settings.DEFAULT_SYMBOLS:
        if s in pairs:
            bp = pairs[s]['basePrecision']
            mv = pairs[s]['minTradeVolume']
            qp = pairs[s]['quotePrecision']
            print(f'    "{s}": {{"base_prec": {bp}, "min_vol": {mv}, "quote_prec": {qp}}},')
        else:
            print(f'    # {s} NOT FOUND')
    print("}")

if __name__ == "__main__":
    main()
