# VCP Streamlit Dashboard

This package contains:

- `streamlit_vcp_dashboard.py` — the web dashboard
- `vcp_daily_weekly_with_charts.py` — the screener engine
- `requirements.txt` — Python dependencies

## 1. Install dependencies

Create a fresh environment, then run:

```bash
pip install -r requirements.txt
```

## 2. Keep your Nifty 500 universe file ready

Your file should contain these columns:

- Company Name
- Industry
- Symbol
- Series
- ISIN Code

Example file name:
`nifty500.csv`

## 3. Run the dashboard

```bash
streamlit run streamlit_vcp_dashboard.py
```

## 4. In the app sidebar

Enter:
- Universe CSV path, for example: `nifty500.csv`
- Output folder, for example: `outputs`
- Number of top charts to export

Then click **Run / Refresh scan**.

## 5. Output generated

The app creates:

- `outputs/vcp_daily_ranked.csv`
- `outputs/vcp_weekly_ranked.csv`
- `outputs/vcp_combined_ranked.csv`
- `outputs/industry_strength.csv`
- `outputs/market_regime.csv`
- `outputs/charts/daily/*.png`
- `outputs/charts/weekly/*.png`

## 6. What each screen is for

- **Dashboard**: overall summary
- **Combined**: final shortlist using both daily and weekly logic
- **Daily**: entry timing and short-term setup strength
- **Weekly**: structural pattern strength
- **Industries**: industry leadership
- **Stock detail**: one stock with daily and weekly chart images

## Notes

- Keep `streamlit_vcp_dashboard.py` and `vcp_daily_weekly_with_charts.py` in the same folder.
- `yfinance` depends on internet access.
- The first run can take time because it downloads the universe data and exports charts.
