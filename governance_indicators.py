# ============================================================
#  Step 2b — Download Governance Indicators (WGI fix)
#  Run this in the SAME Google Colab notebook
#  The 3 WGI indicators need the direct World Bank API
# ============================================================

import requests
import pandas as pd
import time

def fetch_wgi(indicator_code, friendly_name, start=1996, end=2017):
    """Pull one WGI indicator for all countries via direct WB API."""
    print(f"  Fetching {friendly_name} ...")
    all_records = []
    page = 1

    while True:
        url = (
            f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}"
            f"?format=json&per_page=1000&date={start}:{end}&page={page}"
        )
        try:
            r = requests.get(url, timeout=30)
            data = r.json()
        except Exception as e:
            print(f"    Error on page {page}: {e}")
            break

        if len(data) < 2 or data[1] is None:
            break

        for item in data[1]:
            all_records.append({
                'country': item['country']['value'],
                'year':    int(item['date']),
                friendly_name: item['value']
            })

        total_pages = data[0].get('pages', 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.3)   # be polite to the API

    df = pd.DataFrame(all_records)
    print(f"    -> {len(df)} rows retrieved")
    return df

# ── Download the 3 governance indicators ────────────────────────────────────
# Note: WGI data starts from 1996 (not 1980)
gov_eff  = fetch_wgi('GE.EST', 'gov_effectiveness',    1996, 2017)
reg_qual = fetch_wgi('RQ.EST', 'regulatory_quality',   1996, 2017)
ctrl_cor = fetch_wgi('CC.EST', 'control_of_corruption', 1996, 2017)

# ── Merge the three into one file ───────────────────────────────────────────
wgi = (gov_eff
       .merge(reg_qual, on=['country','year'], how='outer')
       .merge(ctrl_cor, on=['country','year'], how='outer'))

wgi['year'] = wgi['year'].astype(int)
wgi = wgi.sort_values(['country','year']).reset_index(drop=True)

# ── Save ─────────────────────────────────────────────────────────────────────
wgi.to_csv('data/governance_panel.csv', index=False)
print(f"\nSaved 'governance_panel.csv'  ->  shape: {wgi.shape}")
print(wgi.head(10))

# ── Download from Colab ──────────────────────────────────────────────────────
from google.colab import files
files.download('governance_panel.csv')
