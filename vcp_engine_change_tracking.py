from __future__ import annotations
import argparse
import json
import shutil
import time
import gc
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# Mobile-first chart readability defaults.
# These affect generated PNG chart text, unlike dashboard CSS which cannot resize text inside images.
CHART_DPI = 360
CHART_FIGSIZE_DAILY = (18, 11)
CHART_FIGSIZE_WEEKLY = (18, 11)
plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 19,
    "figure.titlesize": 26,
    "lines.linewidth": 2.8,
})

# Explicit sizes used inside chart functions. Increase these if mobile chart text is still hard to read.
CHART_TITLE_FONTSIZE = 28
CHART_AXIS_FONTSIZE = 22
CHART_TICK_FONTSIZE = 20
CHART_LEGEND_FONTSIZE = 19
CHART_ANNOTATION_FONTSIZE = 20
CHART_SMALL_ANNOTATION_FONTSIZE = 18


import numpy as np
import pandas as pd
try:
    import yfinance as yf
except ImportError:
    yf = None

DEFAULT_CONFIG = {
    "market_index": "^NSEI", "period": "24mo", "min_history": 300, "swing_order_daily": 8, "swing_order_weekly": 3,
    "max_contractions": 4, "pivot_lookback_daily": 30, "pivot_lookback_weekly": 10, "volume_short_window": 10,
    "volume_long_window": 50, "market_ma_fast": 50, "market_ma_slow": 200, "breakout_volume_ratio": 1.8,
    "near_pivot_min_pct": -5.0, "near_pivot_max_pct": 1.5, "recent_range_days": 10, "recent_range_max_pct": 8.0,
    "min_avg_turnover_inr": 5e7, "industry_boost_top": 80.0, "industry_boost_mid": 60.0, "industry_boost_low": 40.0,
    "industry_boost_top_points": 10.0, "industry_boost_mid_points": 5.0, "industry_boost_low_points": 2.0,
    "min_contraction_days_daily": 5, "min_contraction_days_weekly": 2, "min_contraction_depth_pct_daily": 4.0,
    "min_contraction_depth_pct_weekly": 5.0, "min_base_duration_days": 30, "min_base_duration_weeks": 8,
    "max_latest_contraction_pct": 10.0, "min_weekly_strength_score": 0.45,
    "history_init_enabled": False,
    "history_init_lookback_trading_days": 63,
    "max_price_rows": 620,
    # Public stage-state memory. Prevents 1-day Stage 1/2/1 flicker and keeps failed Stage 2 visible briefly.
    "stage2_failed_hold_days": 21,
    # Minimum daily runs before public promotion into a new advancing stage.
    "stage_transition_confirm_days": 3,
    "stage2_entry_confirm_days": 3,
    # A stock cannot publicly move Stage 4 -> Stage 2 without first showing Stage 1/base repair.
    "stage4_to_stage2_min_stage1_days": 3,
    "enforce_no_stage_jumps": True,
}


def _safe_read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    """Read a CSV defensively.

    Streamlit may read the output folder while the engine is running. If a
    previous file is empty/corrupt/half-written, the engine should not crash;
    it should treat the previous snapshot as unavailable.
    """
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return None
        return pd.read_csv(p, **kwargs)
    except Exception as exc:
        print(f"Warning: could not read previous CSV {path}: {exc}")
        return None


def atomic_to_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Write CSV atomically so the Streamlit app never sees a half-written file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    df.to_csv(tmp, index=index)
    tmp.replace(p)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text atomically for small metadata files."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(p)


def ensure_stage_classification_columns(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Add dashboard-friendly stage classification aliases.

    `stage` stays the stable bucket used for counts and filters. `stage_variant`
    carries the detailed classification such as Early Stage 2, Clean Stage 2,
    Stage 1 - Base/Repair, and Failed Stage 2. `stage_classification` is an
    alias for Streamlit/table display so older app code can remain simple.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    if "stage" not in out.columns:
        out["stage"] = "Not Sure"
    out["stage"] = out["stage"].fillna("Not Sure").astype(str).replace({"": "Not Sure", "nan": "Not Sure", "None": "Not Sure"})
    if "stage_variant" not in out.columns:
        out["stage_variant"] = out["stage"]
    out["stage_variant"] = out["stage_variant"].fillna(out["stage"]).astype(str)
    bad_variant = out["stage_variant"].str.strip().isin(["", "nan", "None"])
    out.loc[bad_variant, "stage_variant"] = out.loc[bad_variant, "stage"]
    out["stage_classification"] = out["stage_variant"]
    out["stage_display"] = np.where(
        out["stage_classification"].astype(str).eq(out["stage"].astype(str)),
        out["stage"].astype(str),
        out["stage"].astype(str) + " • " + out["stage_classification"].astype(str),
    )
    return out

@dataclass
class MarketRegime:
    index_symbol: str
    last_close: float
    ma20: float
    ma50: float
    ma200: float
    slope20_pct: float
    slope50_pct: float
    slope200_pct: float
    ret_1m_pct: float
    ret_3m_pct: float
    drawdown_52w_pct: float
    above_20: bool
    above_50: bool
    above_200: bool
    breadth_above_20_pct: float
    breadth_above_50_pct: float
    breadth_above_200_pct: float
    breadth_stage2_pct: float
    trend_score: float
    breadth_score: float
    regime_label: str

@dataclass
class VCPScoreCard:
    ticker: str
    close: float
    ma50: float
    ma150: float
    ma200: float
    stage: str
    stage_variant: str
    stage_confidence: float
    stage_reason: str
    rs_3m_pct: float
    rs_6m_pct: float
    avg_turnover_inr: float
    daily_setup_bucket: str
    daily_score: float
    daily_pivot: float
    daily_breakout_distance_pct: float
    daily_contraction_depths_pct: List[float]
    daily_contraction_durations: List[int]
    daily_contraction_score: float
    daily_base_duration_days: float
    weekly_setup_bucket: str
    weekly_score: float
    weekly_pivot: float
    weekly_breakout_distance_pct: float
    weekly_contraction_depths_pct: List[float]
    weekly_contraction_durations: List[int]
    weekly_contraction_score: float
    weekly_base_duration_weeks: float
    weekly_vcp_quality: str
    combined_bucket: str
    combined_score: float
    volume_dryup_ratio: float
    breakout_volume_ratio: float
    weekly_volume_ratio: float
    volume_is_drying_up: bool
    weekly_volume_is_drying_up: bool
    notes: str


def normalize_yahoo_ticker(value: str, symbol: Optional[str] = None) -> str:
    ticker = str(value or "").strip().upper()
    sym = str(symbol or "").strip().upper()
    if ticker and ticker not in {"NAN", "NONE"}:
        if ticker.startswith("^") or ticker.endswith(".NS"):
            return ticker
        return f"{ticker}.NS"
    if sym and sym not in {"NAN", "NONE"}:
        return f"{sym}.NS"
    return ""


def parse_truthy_flag(value) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "include", "included"}


def parse_fo_flag(value) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if pd.isna(value):
        return False
    text = str(value).strip().lower().replace(" ", "")
    return text in {"1", "true", "yes", "y", "fo", "f&o", "fno", "fando", "f_and_o"}


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]
    return None


def load_nifty500_universe(file_path: str) -> pd.DataFrame:
    """Load old NSE CSVs and the new universe_2026 schema.

    New schema supported:
      company_name, industry, symbol, Series, sector, industry_group, ticker, f&o, Include

    Only rows with Include == 1 are kept when the Include column is present.
    F&O membership is persisted as is_fo_stock / fo_category.
    """
    df = pd.read_csv(file_path, sep=None, engine="python")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    company_col = _find_col(df, ["company_name", "Company Name", "company", "name"])
    industry_col = _find_col(df, ["industry", "Industry"])
    symbol_col = _find_col(df, ["symbol", "Symbol", "SYMBOL"])
    series_col = _find_col(df, ["Series", "series"])
    sector_col = _find_col(df, ["sector", "Sector"])
    industry_group_col = _find_col(df, [
        "industry_group", "Industry Group", "industry group", "industrygroup",
        "industry_group_name", "IndustryGroup", "industry_group_new", "group",
        "sector_group", "Sector Group", "sector group", "broad_sector", "Broad Sector"
    ])
    ticker_col = _find_col(df, ["ticker", "Ticker", "Yahoo Ticker"])
    include_col = _find_col(df, ["Include", "include", "include_signal", "include signal"])
    fo_col = _find_col(df, ["f&o", "F&O", "fo", "fno", "FNO", "FnO", "is_fo_stock"])
    isin_col = _find_col(df, ["ISIN Code", "isin", "isin_code"])

    missing = [name for name, col in [("company_name/Company Name", company_col), ("industry", industry_col), ("symbol", symbol_col)] if col is None]
    if missing:
        raise ValueError(f"Missing required universe columns: {missing}. Available columns: {list(df.columns)}")

    if include_col is not None:
        before = len(df)
        df = df[df[include_col].apply(parse_truthy_flag)].copy()
        print(f"Universe Include filter: {before:,} -> {len(df):,}")

    if series_col is not None:
        df = df[df[series_col].astype(str).str.upper().str.strip().eq("EQ")].copy()

    out = pd.DataFrame()
    out["Company Name"] = df[company_col].astype(str).str.strip()
    out["Industry"] = df[industry_col].astype(str).str.strip().replace("", "Unknown")
    out["Symbol"] = df[symbol_col].astype(str).str.strip().str.upper()
    out["Series"] = df[series_col].astype(str).str.strip() if series_col else "EQ"
    out["sector"] = df[sector_col].astype(str).str.strip().replace("", "Unknown") if sector_col else "Unknown"
    if industry_group_col:
        out["industry_group"] = df[industry_group_col].astype(str).str.strip().replace("", "Unknown")
    elif sector_col:
         # New universe files may use sector as the broader industry group.
        out["industry_group"] = df[sector_col].astype(str).str.strip().replace("", "Unknown")
    else:
        out["industry_group"] = "Unknown"
    if isin_col:
        out["ISIN Code"] = df[isin_col].astype(str).str.strip()
    else:
        out["ISIN Code"] = ""

    if ticker_col:
        out["Ticker"] = [normalize_yahoo_ticker(t, s) for t, s in zip(df[ticker_col], out["Symbol"])]
    else:
        out["Ticker"] = [normalize_yahoo_ticker("", s) for s in out["Symbol"]]

    if fo_col:
        out["is_fo_stock"] = df[fo_col].apply(parse_fo_flag).astype(bool).values
    else:
        out["is_fo_stock"] = False
    out["fo_category"] = np.where(out["is_fo_stock"], "F&O", "Cash")
    out["Include"] = 1

    out = out[(out["Symbol"] != "") & (out["Ticker"] != "")].drop_duplicates(subset=["Symbol"]).reset_index(drop=True)
    return out[["Company Name", "Industry", "Symbol", "Series", "ISIN Code", "Ticker", "sector", "industry_group", "is_fo_stock", "fo_category", "Include"]]

def fetch_prices(tickers: List[str], period: str, interval: str = "1d", batch_size: int = 40) -> Dict[str, pd.DataFrame]:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Pass --wide-price with your local wide CSV folder to avoid downloading prices.")
    out: Dict[str, pd.DataFrame] = {}

    def parse_download(raw: pd.DataFrame, batch: List[str]) -> Dict[str, pd.DataFrame]:
        parsed: Dict[str, pd.DataFrame] = {}
        if len(batch) == 1:
            t = batch[0]
            df = raw.copy().rename(columns=str.title).dropna(how="all")
            if not df.empty:
                parsed[t] = df
            return parsed
        level0 = raw.columns.get_level_values(0)
        for t in batch:
            if t in level0:
                df = raw[t].copy().rename(columns=str.title).dropna(how="all")
                if not df.empty:
                    parsed[t] = df
        return parsed

    failed: List[str] = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            raw = yf.download(batch, period=period, interval=interval, auto_adjust=True, group_by="ticker", threads=False, progress=False)
            parsed = parse_download(raw, batch)
            out.update(parsed)
            failed.extend([t for t in batch if t not in parsed])
        except Exception:
            failed.extend(batch)
    for t in failed:
        try:
            df = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=True)
            df = df.rename(columns=str.title).dropna(how="all")
            if not df.empty:
                out[t] = df
        except Exception:
            pass
    return out


def _candidate_wide_csv_path(root: Path, attr: str) -> Optional[Path]:
    names = [
        f"wide_{attr.lower()}.csv",
        f"{attr.lower()}.csv",
        f"wide_{attr}.csv",
        f"Wide_{attr}.csv",
    ]
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def _read_wide_csv_selected(path: Path, wanted_cols: List[str], max_rows: Optional[int] = None) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    original_cols = list(header.columns)
    col_lookup = {str(c).strip().upper(): c for c in original_cols}
    date_col = None
    for c in original_cols:
        if str(c).strip().lower() in {"date", "datetime", "timestamp"}:
            date_col = c
            break
    if date_col is None:
        date_col = original_cols[0]

    selected = [date_col]
    for t in wanted_cols:
        key = str(t).strip().upper()
        if key in col_lookup:
            selected.append(col_lookup[key])
    selected = list(dict.fromkeys(selected))

    df = pd.read_csv(path, usecols=selected)
    if max_rows and len(df) > max_rows:
        df = df.tail(max_rows).copy()
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    rename_map = {}
    for c in df.columns:
        if c == "date":
            continue
        rename_map[c] = str(c).strip().upper()
    return df.rename(columns=rename_map)


def _read_wide_excel_selected(path: Path, attr: str, wanted_cols: List[str], max_rows: Optional[int] = None) -> pd.DataFrame:
     # Excel is inherently slower than CSV/Parquet. Use the wide CSV folder when possible.
    header = pd.read_excel(path, sheet_name=attr, nrows=0)
    original_cols = list(header.columns)
    col_lookup = {str(c).strip().upper(): c for c in original_cols}
    date_col = None
    for c in original_cols:
        if str(c).strip().lower() in {"date", "datetime", "timestamp"}:
            date_col = c
            break
    if date_col is None:
        date_col = original_cols[0]
    selected = [date_col]
    for t in wanted_cols:
        key = str(t).strip().upper()
        if key in col_lookup:
            selected.append(col_lookup[key])
    selected = list(dict.fromkeys(selected))
    df = pd.read_excel(path, sheet_name=attr, usecols=selected)
    if max_rows and len(df) > max_rows:
        df = df.tail(max_rows).copy()
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df.rename(columns={c: str(c).strip().upper() for c in df.columns if c != "date"})


def load_wide_price_data(wide_price: str, tickers: List[str], market_index: str, max_rows: Optional[int] = 620) -> Dict[str, pd.DataFrame]:
    """Load local Yahoo wide files quickly.

    Preferred input: a folder containing wide_open.csv, wide_high.csv, wide_low.csv,
    wide_close.csv and wide_volume.csv. The loader reads only date + requested tickers.
    Excel is supported, but is slower.
    """
    root = Path(wide_price)
    wanted = list(dict.fromkeys([str(t).strip().upper() for t in tickers + [market_index] if str(t).strip()]))
    attrs = ["Open", "High", "Low", "Close", "Volume"]
    tables: Dict[str, pd.DataFrame] = {}

    t0 = time.perf_counter()
    if root.is_dir():
        for attr in attrs:
            csv_path = _candidate_wide_csv_path(root, attr)
            if csv_path is None:
                raise FileNotFoundError(f"Missing {attr} wide CSV in {root}. Expected wide_{attr.lower()}.csv")
            a0 = time.perf_counter()
            tables[attr] = _read_wide_csv_selected(csv_path, wanted, max_rows=max_rows)
            print(f"Loaded {csv_path.name}: {tables[attr].shape[0]:,} rows x {tables[attr].shape[1]-1:,} tickers in {time.perf_counter()-a0:.2f}s")
    elif root.suffix.lower() in {".xlsx", ".xls"}:
        print("Reading wide Excel workbook. For speed, prefer the folder with wide_open.csv / wide_close.csv files.")
        for attr in attrs:
            a0 = time.perf_counter()
            tables[attr] = _read_wide_excel_selected(root, attr, wanted, max_rows=max_rows)
            print(f"Loaded Excel sheet {attr}: {tables[attr].shape[0]:,} rows x {tables[attr].shape[1]-1:,} tickers in {time.perf_counter()-a0:.2f}s")
    else:
        raise ValueError("--wide-price must be a folder of wide_*.csv files or yahoo_price_data_wide.xlsx")

    close_cols = [c for c in tables["Close"].columns if c != "date"]
    available = [t for t in wanted if t in close_cols]
    if market_index.upper() not in available:
        raise RuntimeError(f"Market index {market_index} not found in wide price file. Available benchmark columns include: {[c for c in close_cols if str(c).startswith('^')][:10]}")

    indexed: Dict[str, pd.DataFrame] = {}
    attr_indexed = {attr: df.set_index("date") for attr, df in tables.items()}
    for ticker in available:
        cols = {}
        ok = True
        for attr in attrs:
            src = attr_indexed[attr]
            if ticker not in src.columns:
                ok = False
                break
            cols[attr] = pd.to_numeric(src[ticker], errors="coerce")
        if not ok:
            continue
        df = pd.DataFrame(cols).dropna(subset=["Close"]).sort_index()
        if max_rows and len(df) > max_rows:
            df = df.tail(max_rows).copy()
        if not df.empty:
            indexed[ticker] = df

    print(f"Wide price load complete: {len(indexed):,}/{len(wanted):,} requested series in {time.perf_counter()-t0:.2f}s")
    return indexed

def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    weekly = pd.DataFrame()
    weekly["Open"] = df["Open"].resample("W-FRI").first()
    weekly["High"] = df["High"].resample("W-FRI").max()
    weekly["Low"] = df["Low"].resample("W-FRI").min()
    weekly["Close"] = df["Close"].resample("W-FRI").last()
    weekly["Volume"] = df["Volume"].resample("W-FRI").sum()
    return weekly.dropna(how="any")

def rolling_slope(series: pd.Series, window: int = 20) -> float:
    s = series.dropna()
    if len(s) < window:
        return np.nan
    y = s.iloc[-window:].values
    x = np.arange(window)
    return float(np.polyfit(x, y, 1)[0])

def pct_return(series: pd.Series, lookback: int) -> float:
    s = series.dropna()
    if len(s) <= lookback:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-lookback] - 1) * 100)

def avg_turnover(close: pd.Series, volume: pd.Series, window: int = 20) -> float:
    if len(close) < window or len(volume) < window:
        return np.nan
    return float((close.iloc[-window:] * volume.iloc[-window:]).mean())

def volume_ratio(volume: pd.Series, short: int, long: int) -> float:
    if len(volume) < long:
        return np.nan
    short_avg = volume.iloc[-short:].mean()
    long_avg = volume.iloc[-long:].mean()
    if long_avg == 0:
        return np.nan
    return float(short_avg / long_avg)

def recent_breakout_volume_ratio(volume: pd.Series, window: int = 30) -> float:
    """Daily volume ratio: current day volume / previous N-day average volume.

    The current day is excluded from the average.
    """
    if len(volume) <= window:
        return np.nan
    baseline = volume.iloc[-window-1:-1].mean()
    if baseline == 0:
        return np.nan
    return float(volume.iloc[-1] / baseline)


def current_week_volume_ratio(daily_volume: pd.Series, weekly_volume: pd.Series, current_days: int = 5, weekly_window: int = 10) -> float:
    """Weekly volume ratio for dashboard cards.

    Numerator: latest up-to-5 trading days of volume.
    Denominator: average weekly volume of the previous `weekly_window` completed weeks,
    excluding the current/partial week.
    """
    dv = daily_volume.dropna().astype(float)
    wv = weekly_volume.dropna().astype(float)
    if len(dv) < 1 or len(wv) <= weekly_window:
        return np.nan
    current_week_like_volume = dv.iloc[-current_days:].sum()
    baseline = wv.iloc[-weekly_window-1:-1].mean()
    if baseline == 0:
        return np.nan
    return float(current_week_like_volume / baseline)


def slope_pct(series: pd.Series, window: int = 20) -> float:
    s = series.dropna()
    if len(s) < window:
        return np.nan
    level = float(np.nanmean(s.iloc[-window:].values))
    if level == 0:
        return np.nan
    return float(rolling_slope(s, window) / level)

def local_peaks_troughs(high: pd.Series, low: pd.Series, order: int) -> Tuple[List[int], List[int]]:
    high_arr = high.values
    low_arr = low.values
    peaks: List[int] = []
    troughs: List[int] = []
    for i in range(order, len(high_arr) - order):
        high_window = high_arr[i-order:i+order+1]
        low_window = low_arr[i-order:i+order+1]
        center_high = high_arr[i]
        center_low = low_arr[i]
        if np.isfinite(center_high) and center_high == np.max(high_window) and np.sum(high_window == center_high) == 1:
            peaks.append(i)
        if np.isfinite(center_low) and center_low == np.min(low_window) and np.sum(low_window == center_low) == 1:
            troughs.append(i)
    return peaks, troughs

def _candidate_contractions(high: pd.Series, low: pd.Series, order: int, min_duration_bars: int, min_depth_pct: float) -> List[Tuple[int, int, float, int]]:
    peaks, troughs = local_peaks_troughs(high, low, order=order)
    if not peaks or not troughs:
        return []

    pairs: List[Tuple[int, int, float, int]] = []
    for peak_idx, p in enumerate(peaks):
        next_peak = peaks[peak_idx + 1] if peak_idx + 1 < len(peaks) else len(high)
        valid_troughs = [t for t in troughs if p + min_duration_bars <= t < next_peak]
        if not valid_troughs:
            valid_troughs = [t for t in troughs if t > p and (t - p) >= min_duration_bars]
        if not valid_troughs:
            continue
        t = min(valid_troughs, key=lambda idx: float(low.iloc[idx]))
        peak_price = float(high.iloc[p])
        trough_price = float(low.iloc[t])
        if peak_price <= 0 or trough_price <= 0:
            continue
        depth = (peak_price - trough_price) / peak_price * 100
        duration = t - p
        if depth >= min_depth_pct:
            pairs.append((p, t, depth, duration))

    filtered: List[Tuple[int, int, float, int]] = []
    for pair in pairs:
        if not filtered:
            filtered.append(pair)
            continue
        prev = filtered[-1]
        if pair[0] <= prev[1]:
            if pair[2] < prev[2]:
                filtered[-1] = pair
            continue
        filtered.append(pair)
    return filtered

def detect_vcp_contractions(high: pd.Series, low: pd.Series, close: pd.Series, order: int, max_pairs: int, min_duration_bars: int, min_depth_pct: float) -> Tuple[List[float], List[int], float]:
    seq = extract_vcp_contraction_pairs(high, low, order=order, max_pairs=max_pairs, min_duration_bars=min_duration_bars, min_depth_pct=min_depth_pct)
    if not seq:
        return [], [], 0.0

    depths = [round(float(x[2]), 2) for x in seq]
    durations = [int(x[3]) for x in seq]
    base_duration = float(seq[-1][1] - seq[0][0])

    if len(seq) >= 2:
        highest_peak = float(high.iloc[seq[0][0]])
        lowest_trough = float(min(float(low.iloc[t]) for _, t, _, _ in seq))
        total_depth = (highest_peak - lowest_trough) / highest_peak * 100 if highest_peak > 0 else np.nan
        if np.isfinite(total_depth) and total_depth < min_depth_pct:
            return [], [], 0.0
    return depths, durations, base_duration

def contraction_score(depths: List[float]) -> float:
    if len(depths) < 2:
        return 0.0
    wins = sum(1 for i in range(1, len(depths)) if depths[i] <= depths[i-1] * 1.05)
    size_bonus = min(1.0, len(depths) / 4)
    return round((wins / (len(depths) - 1)) * 0.8 + size_bonus * 0.2, 4)

def extract_vcp_contraction_pairs(high: pd.Series, low: pd.Series, order: int, max_pairs: int, min_duration_bars: int, min_depth_pct: float) -> List[Tuple[int, int, float, int]]:
    pairs = _candidate_contractions(high, low, order=order, min_duration_bars=min_duration_bars, min_depth_pct=min_depth_pct)
    if not pairs:
        return []

    seq: List[Tuple[int, int, float, int]] = []
    for pair in pairs:
        if not seq:
            seq.append(pair)
            continue
        prev = seq[-1]
        prev_peak = float(high.iloc[prev[0]])
        curr_peak = float(high.iloc[pair[0]])
        depth_contracting = pair[2] <= prev[2] * 1.15
        price_tightening = curr_peak <= prev_peak * 1.10
        if depth_contracting and price_tightening:
            seq.append(pair)
        else:
            seq = [pair]
    return seq[-max_pairs:]


def _local_peak_indices(series: pd.Series, order: int = 3) -> List[int]:
    vals = series.values
    peaks: List[int] = []
    for i in range(order, len(vals) - order):
        window = vals[i - order:i + order + 1]
        center = vals[i]
        if np.isfinite(center) and center == np.max(window) and np.sum(window == center) == 1:
            peaks.append(i)
    return peaks


def compute_pivot_zone(
    high: pd.Series,
    lookback: int,
    base_duration: Optional[float] = None,
    *,
    is_weekly: bool = False,
    tolerance_pct: float = 1.5,
    min_band_pct: float = 0.35,
    max_band_pct: float = 2.0,
) -> Tuple[float, float, float]:
    if high.empty:
        return np.nan, np.nan, np.nan

    dynamic_window = lookback
    if base_duration and np.isfinite(base_duration) and base_duration > 0:
        dynamic_window = max(lookback, int(np.ceil(base_duration)) + (3 if is_weekly else 5))

    s = high.iloc[-dynamic_window:-1].dropna()
    if len(s) < 3:
        return np.nan, np.nan, np.nan

    order = 2 if is_weekly else 4
    peak_idx = _local_peak_indices(s, order=min(order, max(1, len(s) // 8)))
    if peak_idx:
        peak_vals = s.iloc[peak_idx].astype(float)
    else:
        peak_vals = s.nlargest(min(3, len(s))).sort_values()

    pivot_high = float(peak_vals.max())
    cluster_cutoff = pivot_high * (1 - tolerance_pct / 100)
    cluster = peak_vals[peak_vals >= cluster_cutoff]
    if cluster.empty:
        cluster = peak_vals.nlargest(1)

    zone_low = float(cluster.min())
    zone_high = float(cluster.max())

    min_width = pivot_high * (min_band_pct / 100)
    max_width = pivot_high * (max_band_pct / 100)
    width = zone_high - zone_low
    if width < min_width:
        pad = (min_width - width) / 2
        zone_low -= pad
        zone_high += pad
    elif width > max_width:
        zone_low = zone_high - max_width

    zone_low = max(0.0, zone_low)
    return float(zone_low), float(zone_high), float(zone_high)


def compute_pivot(high: pd.Series, lookback: int, base_duration: Optional[float] = None) -> float:
    _, _, pivot = compute_pivot_zone(high, lookback, base_duration=base_duration, is_weekly=False)
    return pivot

def classify_market_regime(score: float) -> str:
    if score >= 14:
        return "strong_risk_on"
    if score >= 8:
        return "risk_on"
    if score >= 3:
        return "mixed"
    if score >= -3:
        return "risk_off"
    return "strong_risk_off"


def compute_market_breadth(
    price_data: Dict[str, pd.DataFrame],
    universe_tickers: List[str],
) -> Dict[str, float]:
    above20 = above50 = above200 = eligible20 = eligible50 = eligible200 = 0
    stage2_count = stage_eligible = 0

    for ticker in universe_tickers:
        df = price_data.get(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            continue

        close = df["Close"].dropna().astype(float)
        if len(close) >= 20:
            ma20 = float(close.rolling(20).mean().iloc[-1])
            if np.isfinite(ma20):
                eligible20 += 1
                if float(close.iloc[-1]) > ma20:
                    above20 += 1

        if len(close) >= 50:
            ma50 = float(close.rolling(50).mean().iloc[-1])
            if np.isfinite(ma50):
                eligible50 += 1
                if float(close.iloc[-1]) > ma50:
                    above50 += 1

        if len(close) >= 200:
            ma200 = float(close.rolling(200).mean().iloc[-1])
            if np.isfinite(ma200):
                eligible200 += 1
                if float(close.iloc[-1]) > ma200:
                    above200 += 1

        if len(close) >= 260:
            ma50 = float(close.rolling(50).mean().iloc[-1])
            ma150 = float(close.rolling(150).mean().iloc[-1])
            ma200 = float(close.rolling(200).mean().iloc[-1])
            stage = determine_stage(close, ma50, ma150, ma200)
            stage_eligible += 1
            if stage == "Stage 2":
                stage2_count += 1

    def pct(n: int, d: int) -> float:
        return round((n / d) * 100, 2) if d else np.nan

    return {
        "breadth_above_20_pct": pct(above20, eligible20),
        "breadth_above_50_pct": pct(above50, eligible50),
        "breadth_above_200_pct": pct(above200, eligible200),
        "breadth_stage2_pct": pct(stage2_count, stage_eligible),
    }


def market_regime(
    index_df: pd.DataFrame,
    index_symbol: str,
    ma_fast: int,
    ma_slow: int,
    price_data: Optional[Dict[str, pd.DataFrame]] = None,
    universe_tickers: Optional[List[str]] = None,
) -> MarketRegime:
    close = index_df["Close"].dropna().astype(float)
    if len(close) < 260:
        raise ValueError("Not enough index history to compute market regime")

    ma20_series = close.rolling(20).mean()
    ma50_series = close.rolling(ma_fast).mean()
    ma200_series = close.rolling(ma_slow).mean()

    last_close = float(close.iloc[-1])
    ma20 = float(ma20_series.iloc[-1])
    ma50 = float(ma50_series.iloc[-1])
    ma200 = float(ma200_series.iloc[-1])

    slope20_pct = slope_pct(ma20_series, 20)
    slope50_pct = slope_pct(ma50_series, 20)
    slope200_pct = slope_pct(ma200_series, 20)

    ret_1m_pct = pct_return(close, 21)
    ret_3m_pct = pct_return(close, 63)

    high_52w = float(close.iloc[-252:].max())
    drawdown_52w_pct = (last_close / high_52w - 1) * 100 if high_52w > 0 else np.nan

    above_20 = last_close > ma20 if pd.notna(ma20) else False
    above_50 = last_close > ma50 if pd.notna(ma50) else False
    above_200 = last_close > ma200 if pd.notna(ma200) else False

    breadth = {
        "breadth_above_20_pct": np.nan,
        "breadth_above_50_pct": np.nan,
        "breadth_above_200_pct": np.nan,
        "breadth_stage2_pct": np.nan,
    }
    if price_data is not None and universe_tickers:
        breadth = compute_market_breadth(price_data, universe_tickers)

    trend_score = 0.0

    if above_20:
        trend_score += 2
    else:
        trend_score -= 2

    if above_50:
        trend_score += 3
    else:
        trend_score -= 3

    if above_200:
        trend_score += 4
    else:
        trend_score -= 4

    if pd.notna(ma20) and pd.notna(ma50) and pd.notna(ma200):
        if ma20 > ma50 > ma200:
            trend_score += 4
        elif ma50 > ma200:
            trend_score += 2
        elif ma20 < ma50 < ma200:
            trend_score -= 4
        elif ma50 < ma200:
            trend_score -= 2

    if pd.notna(slope20_pct):
        if slope20_pct > 0.0010:
            trend_score += 2
        elif slope20_pct < -0.0010:
            trend_score -= 2

    if pd.notna(slope50_pct):
        if slope50_pct > 0.0005:
            trend_score += 2
        elif slope50_pct < -0.0005:
            trend_score -= 2

    if pd.notna(slope200_pct):
        if slope200_pct > 0.0001:
            trend_score += 2
        elif slope200_pct < -0.0001:
            trend_score -= 2

    if pd.notna(ret_1m_pct):
        if ret_1m_pct > 3:
            trend_score += 1
        elif ret_1m_pct < -3:
            trend_score -= 1

    if pd.notna(ret_3m_pct):
        if ret_3m_pct > 8:
            trend_score += 2
        elif ret_3m_pct < -8:
            trend_score -= 2

    if pd.notna(drawdown_52w_pct):
        if drawdown_52w_pct >= -5:
            trend_score += 2
        elif drawdown_52w_pct >= -10:
            trend_score += 1
        elif drawdown_52w_pct <= -30:
            trend_score -= 3
        elif drawdown_52w_pct <= -20:
            trend_score -= 2

    breadth_score = 0.0
    b20 = breadth["breadth_above_20_pct"]
    b50 = breadth["breadth_above_50_pct"]
    b200 = breadth["breadth_above_200_pct"]
    bstage2 = breadth["breadth_stage2_pct"]

    if pd.notna(b20):
        if b20 >= 70:
            breadth_score += 2
        elif b20 >= 55:
            breadth_score += 1
        elif b20 <= 35:
            breadth_score -= 1
        elif b20 <= 25:
            breadth_score -= 2

    if pd.notna(b50):
        if b50 >= 65:
            breadth_score += 3
        elif b50 >= 50:
            breadth_score += 1.5
        elif b50 <= 35:
            breadth_score -= 1.5
        elif b50 <= 25:
            breadth_score -= 3

    if pd.notna(b200):
        if b200 >= 60:
            breadth_score += 3
        elif b200 >= 45:
            breadth_score += 1.5
        elif b200 <= 30:
            breadth_score -= 1.5
        elif b200 <= 20:
            breadth_score -= 3

    if pd.notna(bstage2):
        if bstage2 >= 35:
            breadth_score += 2
        elif bstage2 >= 25:
            breadth_score += 1
        elif bstage2 <= 12:
            breadth_score -= 1
        elif bstage2 <= 7:
            breadth_score -= 2

    final_score = trend_score + breadth_score
    regime_label = classify_market_regime(final_score)

    return MarketRegime(
        index_symbol=index_symbol,
        last_close=round(last_close, 2),
        ma20=round(ma20, 2) if pd.notna(ma20) else np.nan,
        ma50=round(ma50, 2) if pd.notna(ma50) else np.nan,
        ma200=round(ma200, 2) if pd.notna(ma200) else np.nan,
        slope20_pct=round(float(slope20_pct), 6) if pd.notna(slope20_pct) else np.nan,
        slope50_pct=round(float(slope50_pct), 6) if pd.notna(slope50_pct) else np.nan,
        slope200_pct=round(float(slope200_pct), 6) if pd.notna(slope200_pct) else np.nan,
        ret_1m_pct=round(float(ret_1m_pct), 2) if pd.notna(ret_1m_pct) else np.nan,
        ret_3m_pct=round(float(ret_3m_pct), 2) if pd.notna(ret_3m_pct) else np.nan,
        drawdown_52w_pct=round(float(drawdown_52w_pct), 2) if pd.notna(drawdown_52w_pct) else np.nan,
        above_20=bool(above_20),
        above_50=bool(above_50),
        above_200=bool(above_200),
        breadth_above_20_pct=round(float(b20), 2) if pd.notna(b20) else np.nan,
        breadth_above_50_pct=round(float(b50), 2) if pd.notna(b50) else np.nan,
        breadth_above_200_pct=round(float(b200), 2) if pd.notna(b200) else np.nan,
        breadth_stage2_pct=round(float(bstage2), 2) if pd.notna(bstage2) else np.nan,
        trend_score=round(float(trend_score), 2),
        breadth_score=round(float(breadth_score), 2),
        regime_label=regime_label,
    )



def determine_stage(close: pd.Series, ma50: float, ma150: float, ma200: float) -> str:
    """
    Stage classifier tuned for public structure scan.

    Main design change:
    - Stage 3 is no longer the default dustbin.
    - Stage 2 has hard guards for near-52W-high advancing names.
    - Stage 3 now requires actual deterioration/distribution evidence.
    - Messy recoveries that are not clear Stage 2/4 become Stage 1 instead of Stage 3.
    """
    c = close.dropna().astype(float)
    if len(c) < 260:
        return "Not Sure"

    def finite(x) -> bool:
        return pd.notna(x) and np.isfinite(float(x))

    last = float(c.iloc[-1])

    ma50_series = c.rolling(50).mean()
    ma150_series = c.rolling(150).mean()
    ma200_series = c.rolling(200).mean()

    ma50_now = float(ma50_series.iloc[-1]) if pd.notna(ma50_series.iloc[-1]) else float(ma50)
    ma150_now = float(ma150_series.iloc[-1]) if pd.notna(ma150_series.iloc[-1]) else float(ma150)
    ma200_now = float(ma200_series.iloc[-1]) if pd.notna(ma200_series.iloc[-1]) else float(ma200)

    ma50_slope = slope_pct(ma50_series, 20)
    ma150_slope = slope_pct(ma150_series, 20)
    ma200_slope = slope_pct(ma200_series, 20)

    weekly_close = c.resample("W-FRI").last().dropna()
    weekly_ma10 = weekly_close.rolling(10).mean()
    weekly_ma30 = weekly_close.rolling(30).mean()
    weekly_ma10_now = float(weekly_ma10.iloc[-1]) if len(weekly_ma10) and pd.notna(weekly_ma10.iloc[-1]) else np.nan
    weekly_ma30_now = float(weekly_ma30.iloc[-1]) if len(weekly_ma30) and pd.notna(weekly_ma30.iloc[-1]) else np.nan
    weekly_ma10_slope = slope_pct(weekly_ma10, 6)
    weekly_ma30_slope = slope_pct(weekly_ma30, 6)

    high_52w = float(c.iloc[-252:].max())
    low_52w = float(c.iloc[-252:].min())
    dist_from_high = (last / high_52w - 1) * 100 if high_52w > 0 else np.nan
    advance_from_low = (last / low_52w - 1) * 100 if low_52w > 0 else np.nan

    ret_8w = pct_return(c, 42)
    ret_13w = pct_return(c, 63)
    ret_26w = pct_return(c, 126)

    range_13w = ((c.iloc[-63:].max() / c.iloc[-63:].min()) - 1) * 100 if c.iloc[-63:].min() > 0 else np.nan

    def _turning_points(series: pd.Series, order: int = 5) -> Tuple[List[int], List[int]]:
        vals = series.values
        peaks: List[int] = []
        troughs: List[int] = []
        for i in range(order, len(vals) - order):
            window = vals[i - order:i + order + 1]
            center = vals[i]
            if not np.isfinite(center):
                continue
            if center == np.max(window) and np.sum(window == center) == 1:
                peaks.append(i)
            if center == np.min(window) and np.sum(window == center) == 1:
                troughs.append(i)
        return peaks, troughs

    def _recent_structure(series: pd.Series, lookback: int = 90, order: int = 5) -> dict:
        s = series.iloc[-lookback:].copy()
        peaks, troughs = _turning_points(s, order=order)
        recent_peaks = [float(s.iloc[i]) for i in peaks[-3:]]
        recent_troughs = [float(s.iloc[i]) for i in troughs[-3:]]
        return {
            "lower_highs": len(recent_peaks) >= 2 and all(recent_peaks[i] < recent_peaks[i - 1] for i in range(1, len(recent_peaks))),
            "higher_highs": len(recent_peaks) >= 2 and all(recent_peaks[i] > recent_peaks[i - 1] for i in range(1, len(recent_peaks))),
            "lower_lows": len(recent_troughs) >= 2 and all(recent_troughs[i] < recent_troughs[i - 1] for i in range(1, len(recent_troughs))),
            "higher_lows": len(recent_troughs) >= 2 and all(recent_troughs[i] > recent_troughs[i - 1] for i in range(1, len(recent_troughs))),
        }

    structure = _recent_structure(c)
    lower_highs = structure["lower_highs"]
    higher_highs = structure["higher_highs"]
    lower_lows = structure["lower_lows"]
    higher_lows = structure["higher_lows"]

    above_50 = finite(ma50_now) and last > ma50_now
    above_150 = finite(ma150_now) and last > ma150_now
    above_200 = finite(ma200_now) and last > ma200_now
    below_50 = finite(ma50_now) and last < ma50_now
    below_150 = finite(ma150_now) and last < ma150_now
    below_200 = finite(ma200_now) and last < ma200_now

    ma_stack_bull = above_50 and above_150 and above_200 and ma50_now > ma150_now > ma200_now
    ma_stack_bear = below_50 and below_150 and below_200 and ma50_now < ma150_now < ma200_now

    ma50_rising = finite(ma50_slope) and ma50_slope > 0.00030
    ma150_rising = finite(ma150_slope) and ma150_slope > 0.00003
    ma200_rising = finite(ma200_slope) and ma200_slope > 0.00001
    ma50_falling = finite(ma50_slope) and ma50_slope < -0.00030
    ma150_falling = finite(ma150_slope) and ma150_slope < -0.00008
    ma200_falling = finite(ma200_slope) and ma200_slope < -0.00004
    ma200_flat_or_rising = finite(ma200_slope) and ma200_slope >= -0.00012
    ma150_flat_or_rising = finite(ma150_slope) and ma150_slope >= -0.00008

    weekly_bull = finite(weekly_ma10_now) and finite(weekly_ma30_now) and last > weekly_ma10_now > weekly_ma30_now
    weekly_bear = finite(weekly_ma10_now) and finite(weekly_ma30_now) and last < weekly_ma10_now < weekly_ma30_now
    weekly_10_rising = finite(weekly_ma10_slope) and weekly_ma10_slope > 0.00035
    weekly_30_rising = finite(weekly_ma30_slope) and weekly_ma30_slope > 0.00005
    weekly_30_falling = finite(weekly_ma30_slope) and weekly_ma30_slope < -0.00010

    near_high = finite(dist_from_high) and dist_from_high >= -10
    not_far_from_high = finite(dist_from_high) and dist_from_high >= -25
    above_long_term = above_150 and above_200
    positive_medium_momentum = (finite(ret_13w) and ret_13w >= 3) or (finite(ret_26w) and ret_26w >= 8)
    strong_advance_from_low = finite(advance_from_low) and advance_from_low >= 25

     # ---- Stage 2: advancing structure / breakout / continuation ----
     # Hard guard: a stock near 52W high, above 50/200DMA, with rising 50DMA should not become Stage 3.
    stage2_near_high_guard = (
        near_high
        and above_50 and above_200
        and ma50_rising
        and ma200_flat_or_rising
        and (positive_medium_momentum or weekly_bull or weekly_10_rising)
        and not (lower_highs and lower_lows and below_50)
    )

    stage2_breakout_or_recovery = (
        above_long_term
        and ma150_flat_or_rising
        and ma200_flat_or_rising
        and not_far_from_high
        and positive_medium_momentum
        and (ma50_rising or weekly_10_rising or higher_highs or higher_lows)
        and not (lower_highs and lower_lows and finite(ret_13w) and ret_13w < 0)
    )

    stage2_continuation = (
        above_long_term
        and (ma_stack_bull or weekly_bull or (above_50 and ma50_rising))
        and (ma150_rising or weekly_30_rising or ma200_rising)
        and not_far_from_high
        and (positive_medium_momentum or strong_advance_from_low)
        and not lower_lows
    )

    if stage2_near_high_guard or stage2_breakout_or_recovery or stage2_continuation:
        return "Stage 2"

     # ---- Stage 4: persistent downtrend ----
    persistent_decline = (
        below_150 and below_200
        and (ma150_falling or ma200_falling or weekly_30_falling)
        and (lower_highs or lower_lows or weekly_bear)
        and finite(dist_from_high) and dist_from_high <= -22
        and finite(ret_13w) and ret_13w <= -6
    )
    failed_rally_decline = (
        below_150
        and (below_200 or ma200_falling or weekly_30_falling)
        and lower_highs
        and finite(ret_8w) and ret_8w <= 0
        and finite(dist_from_high) and dist_from_high <= -18
        and not higher_lows
    )
    deep_decline = (
        ma_stack_bear
        and (ma50_falling or weekly_bear)
        and finite(ret_26w) and ret_26w <= -12
    )

    if persistent_decline or failed_rally_decline or deep_decline:
        return "Stage 4"

     # ---- Stage 3: true top / distribution / damaged uptrend ----
     # Stage 3 should be a narrow bucket: prior strength + visible deterioration.
    prior_strength = (
        above_200
        and finite(dist_from_high) and dist_from_high >= -30
        and ((finite(ret_26w) and ret_26w >= 0) or strong_advance_from_low)
    )
    distribution_damage = (
        (below_50 or ma50_falling)
        and (lower_highs or lower_lows)
        and finite(ret_8w) and ret_8w <= 0
    )
    weekly_damage = (
        above_200
        and below_50
        and (weekly_bear or weekly_30_falling)
        and finite(dist_from_high) and -30 <= dist_from_high <= -8
    )
    wide_churn_after_advance = (
        prior_strength
        and finite(range_13w) and range_13w >= 22
        and finite(ret_13w) and ret_13w <= 4
        and (lower_highs or lower_lows)
    )

    if prior_strength and (distribution_damage or weekly_damage or wide_churn_after_advance):
        return "Stage 3"

     # ---- Stage 1: repair / base / early turn ----
     # Most non-declining, non-advancing structures should be Stage 1, not Stage 3.
    near_ma150 = finite(ma150_now) and 0.90 * ma150_now <= last <= 1.12 * ma150_now
    near_ma200 = finite(ma200_now) and 0.90 * ma200_now <= last <= 1.12 * ma200_now
    long_ma_not_collapsing = ma200_flat_or_rising or not ma200_falling
    basing_or_repair = (
        (near_ma150 or near_ma200 or above_200)
        and long_ma_not_collapsing
        and not (lower_highs and lower_lows and finite(ret_13w) and ret_13w < -6)
    )
    early_recovery = (
        above_50
        and (above_150 or above_200)
        and (ma50_rising or higher_lows)
        and finite(ret_13w) and ret_13w > 0
        and not (below_150 and below_200)
    )

    if basing_or_repair or early_recovery:
        return "Stage 1"

     # Conservative fallback.
    if below_200 and (ma200_falling or weekly_bear):
        return "Stage 4"
    return "Not Sure"


def determine_stage_details(close: pd.Series, ma50: float, ma150: float, ma200: float) -> Tuple[str, str, float, str]:
    """Return the main Weinstein-style stage plus a human-friendly variant.

    The main `stage` column intentionally remains one of Stage 1/2/3/4/Unknown.
    `stage_variant` adds nuance for the dashboard without breaking filters,
    history, or stage counts.
    """
    stage = determine_stage(close, ma50, ma150, ma200)
    c = close.dropna().astype(float)
    if len(c) < 260 or stage in {"Unknown", "Not Sure"}:
        return "Not Sure", "Not Sure", 0.0, "Not enough price history for reliable stage classification."

    def finite(x) -> bool:
        return pd.notna(x) and np.isfinite(float(x))

    last = float(c.iloc[-1])
    ma50_series = c.rolling(50).mean()
    ma150_series = c.rolling(150).mean()
    ma200_series = c.rolling(200).mean()

    ma50_now = float(ma50_series.iloc[-1]) if finite(ma50_series.iloc[-1]) else float(ma50)
    ma150_now = float(ma150_series.iloc[-1]) if finite(ma150_series.iloc[-1]) else float(ma150)
    ma200_now = float(ma200_series.iloc[-1]) if finite(ma200_series.iloc[-1]) else float(ma200)

    ma50_slope = slope_pct(ma50_series, 20)
    ma150_slope = slope_pct(ma150_series, 20)
    ma200_slope = slope_pct(ma200_series, 20)

    weekly_close = c.resample("W-FRI").last().dropna()
    weekly_ma10 = weekly_close.rolling(10).mean()
    weekly_ma30 = weekly_close.rolling(30).mean()
    weekly_ma10_now = float(weekly_ma10.iloc[-1]) if len(weekly_ma10) and finite(weekly_ma10.iloc[-1]) else np.nan
    weekly_ma30_now = float(weekly_ma30.iloc[-1]) if len(weekly_ma30) and finite(weekly_ma30.iloc[-1]) else np.nan
    weekly_ma10_slope = slope_pct(weekly_ma10, 6)
    weekly_ma30_slope = slope_pct(weekly_ma30, 6)

    high_52w = float(c.iloc[-252:].max())
    low_52w = float(c.iloc[-252:].min())
    dist_from_high = (last / high_52w - 1) * 100 if high_52w > 0 else np.nan
    advance_from_low = (last / low_52w - 1) * 100 if low_52w > 0 else np.nan
    ret_4w = pct_return(c, 21)
    ret_8w = pct_return(c, 42)
    ret_13w = pct_return(c, 63)
    ret_26w = pct_return(c, 126)

    above_50 = finite(ma50_now) and last > ma50_now
    above_150 = finite(ma150_now) and last > ma150_now
    above_200 = finite(ma200_now) and last > ma200_now
    below_50 = finite(ma50_now) and last < ma50_now
    below_150 = finite(ma150_now) and last < ma150_now
    below_200 = finite(ma200_now) and last < ma200_now
    ma_stack_bull = above_50 and above_150 and above_200 and ma50_now > ma150_now > ma200_now
    ma_stack_bear = below_50 and below_150 and below_200 and ma50_now < ma150_now < ma200_now

    ma50_rising = finite(ma50_slope) and ma50_slope > 0.00030
    ma150_rising = finite(ma150_slope) and ma150_slope > 0.00003
    ma200_rising = finite(ma200_slope) and ma200_slope > 0.00001
    ma50_falling = finite(ma50_slope) and ma50_slope < -0.00030
    ma150_falling = finite(ma150_slope) and ma150_slope < -0.00008
    ma200_falling = finite(ma200_slope) and ma200_slope < -0.00004
    weekly_bull = finite(weekly_ma10_now) and finite(weekly_ma30_now) and last > weekly_ma10_now > weekly_ma30_now
    weekly_bear = finite(weekly_ma10_now) and finite(weekly_ma30_now) and last < weekly_ma10_now < weekly_ma30_now
    weekly_10_rising = finite(weekly_ma10_slope) and weekly_ma10_slope > 0.00035
    weekly_30_rising = finite(weekly_ma30_slope) and weekly_ma30_slope > 0.00005
    weekly_30_falling = finite(weekly_ma30_slope) and weekly_ma30_slope < -0.00010

    near_high = finite(dist_from_high) and dist_from_high >= -10
    far_above_50 = finite(ma50_now) and last >= ma50_now * 1.18
    far_above_200 = finite(ma200_now) and last >= ma200_now * 1.45
    near_long_ma = (finite(ma150_now) and 0.94 * ma150_now <= last <= 1.10 * ma150_now) or (finite(ma200_now) and 0.94 * ma200_now <= last <= 1.10 * ma200_now)

     # Stage 2 variants
    if stage == "Stage 2":
        if far_above_50 or far_above_200 or (finite(advance_from_low) and advance_from_low >= 120 and finite(ret_13w) and ret_13w >= 25):
            return stage, "Extended Stage 2", 0.78, "Advancing structure remains intact, but price is stretched versus key moving averages or far above the 52-week low."
        if ma_stack_bull and weekly_bull and ma50_rising and (ma150_rising or weekly_30_rising) and near_high:
            return stage, "Clean Stage 2", 0.90, "Price is above rising key averages, weekly structure is supportive, and the stock is near its 52-week high."
        if above_150 and above_200 and (ma50_rising or weekly_10_rising) and not ma_stack_bull:
            return stage, "Early Stage 2", 0.72, "Price has moved above long-term structure, but the full moving-average stack is not clean yet."
        if (below_50 or not weekly_bull) and above_150 and above_200:
            return stage, "Messy Stage 2", 0.64, "Long-term structure is still constructive, but the short-term/weekly structure is not clean."
        if below_50 and above_150 and above_200 and not ma150_falling and not ma200_falling:
            return stage, "Stage 2 Pullback", 0.68, "Prior advancing structure is pulling back, but long-term moving averages are still supportive."
        return stage, "Stage 2", 0.76, "Advancing structure detected."

     # Stage 1 variants
    if stage == "Stage 1":
        if above_50 and (above_150 or above_200) and (ma50_rising or weekly_10_rising) and finite(ret_13w) and ret_13w > 0:
            return stage, "Stage 1 - Early Turn", 0.66, "The stock is improving from a repair/base area but has not confirmed a clean Stage 2 yet."
        if near_long_ma and not ma200_falling:
            return stage, "Stage 1 - Base/Repair", 0.70, "Price is near long-term moving averages with no strong long-term downtrend signal."
        return stage, "Stage 1 - Unclear/Repair", 0.58, "Not enough evidence for Stage 2 or Stage 4; classified as repair/unclear structure."

     # Stage 3 variants
    if stage == "Stage 3":
        if above_200 and below_50 and (ma50_falling or weekly_bear or weekly_30_falling):
            return stage, "Stage 3 - Distribution/Damage", 0.76, "Prior strength exists, but short-term and/or weekly structure is deteriorating."
        return stage, "Stage 3 - Transition", 0.62, "The structure is transitioning after prior strength, but it is not yet a confirmed Stage 4 decline."

     # Stage 4 variants
    if stage == "Stage 4":
        if ma_stack_bear and ma50_falling and (ma150_falling or ma200_falling or weekly_30_falling):
            return stage, "Stage 4 - Confirmed Downtrend", 0.86, "Price is below a bearish moving-average stack with falling trend signals."
        if below_200 and (ma200_falling or weekly_bear):
            return stage, "Stage 4 - Weak/Declining", 0.72, "Price is below long-term support with weak trend evidence."
        return stage, "Stage 4", 0.66, "Declining structure detected."

    return stage, stage, 0.50, "Stage variant not classified."


def vcp_quality_label(score: float, base_bars: float, depths: List[float], min_base_bars: int) -> str:
    if len(depths) < 2 or base_bars < min_base_bars:
        return "weak"
    return "strong" if score >= 0.66 else ("moderate" if score >= 0.5 else "weak")

def score_daily(stage: str, trend_template_ok: bool, regime_label: str, liquidity_ok: bool, near_pivot_ok: bool, breakout_today: bool, contraction_score_val: float, base_duration: float, dist_from_high: float, volume_dryup_ratio: float, breakout_volume_ratio: float, rs_3m: float, rs_6m: float) -> float:
    score = 0.0
    if trend_template_ok:
        score += 18

    if regime_label == "strong_risk_on":
        score += 12
    elif regime_label == "risk_on":
        score += 8
    elif regime_label == "mixed":
        score += 3
    elif regime_label == "risk_off":
        score -= 6
    elif regime_label == "strong_risk_off":
        score -= 12
    if liquidity_ok:
        score += 8
    if near_pivot_ok:
        score += 10
    if breakout_today:
        score += 8
    if stage == "Stage 2":
        score += 10
    elif stage == "Stage 1":
        score += 3
    score += max(0, min(18, contraction_score_val * 18))
    score += max(0, min(8, base_duration / 8))
    score += max(0, min(5, 15 + dist_from_high))
    if np.isfinite(volume_dryup_ratio):
        score += max(0, min(4, (1 - volume_dryup_ratio) * 10))
    if np.isfinite(breakout_volume_ratio):
        score += max(0, min(5, (breakout_volume_ratio - 1) * 4))
    rs_combo = np.nanmean([rs_3m, rs_6m])
    if np.isfinite(rs_combo):
        score += max(0, min(6, rs_combo / 5))
    return round(float(score), 2)

def score_weekly(stage: str, contraction_score_val: float, base_duration: float, weekly_breakout_distance_pct: float, weekly_quality: str, rs_3m: float, rs_6m: float) -> float:
    score = 0.0
    if stage == "Stage 2":
        score += 12
    elif stage == "Stage 1":
        score += 4
    score += max(0, min(22, contraction_score_val * 22))
    score += max(0, min(14, base_duration * 1.2))
    if np.isfinite(weekly_breakout_distance_pct) and -8 <= weekly_breakout_distance_pct <= 3:
        score += 8
    if weekly_quality == "strong":
        score += 12
    elif weekly_quality == "moderate":
        score += 6
    rs_combo = np.nanmean([rs_3m, rs_6m])
    if np.isfinite(rs_combo):
        score += max(0, min(8, rs_combo / 5))
    return round(float(score), 2)

def classify_daily_bucket(trend_template_ok: bool, daily_vcp_ok: bool, near_pivot_ok: bool, breakout_today: bool, tight_range_ok: bool, market_regime_ok: bool) -> str:
    if breakout_today and trend_template_ok and daily_vcp_ok and market_regime_ok:
        return "breakout_today"
    if trend_template_ok and daily_vcp_ok and near_pivot_ok and tight_range_ok:
        return "near_pivot"
    if trend_template_ok and daily_vcp_ok:
        return "forming_vcp"
    return "watchlist"

def classify_weekly_bucket(stage: str, weekly_vcp_ok: bool, weekly_breakout_distance_pct: float, weekly_quality: str) -> str:
    near_weekly_pivot = pd.notna(weekly_breakout_distance_pct) and -8 <= weekly_breakout_distance_pct <= 3
    if stage == "Stage 2" and weekly_vcp_ok and weekly_quality == "strong" and pd.notna(weekly_breakout_distance_pct) and weekly_breakout_distance_pct > 0:
        return "weekly_breakout"
    if stage == "Stage 2" and weekly_vcp_ok and near_weekly_pivot:
        return "weekly_near_pivot"
    if stage == "Stage 2" and weekly_vcp_ok:
        return "weekly_forming"
    return "weekly_watchlist"

def combined_bucket(daily_bucket: str, weekly_bucket: str) -> str:
    if daily_bucket == "breakout_today" and weekly_bucket in {"weekly_breakout", "weekly_near_pivot", "weekly_forming"}:
        return "high_conviction_breakout"
    if daily_bucket == "near_pivot" and weekly_bucket in {"weekly_near_pivot", "weekly_forming"}:
        return "high_conviction_near_pivot"
    if daily_bucket == "forming_vcp" and weekly_bucket in {"weekly_near_pivot", "weekly_forming"}:
        return "building_setup"
    return "watchlist"

def analyze_symbol(ticker: str, df: pd.DataFrame, benchmark_df: pd.DataFrame, regime: MarketRegime, config: dict) -> Optional[VCPScoreCard]:
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return None

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    if len(df) < config["min_history"]:
        return None

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    weekly_df = resample_weekly(df)
    if len(weekly_df) < 60:
        return None
    weekly_close = weekly_df["Close"].astype(float)
    weekly_high = weekly_df["High"].astype(float)
    weekly_low = weekly_df["Low"].astype(float)

    close_now = float(close.iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma150 = float(close.rolling(150).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    ma50_series = close.rolling(50).mean()
    ma150_series = close.rolling(150).mean()
    ma200_series = close.rolling(200).mean()
    stage, stage_variant, stage_confidence, stage_reason = determine_stage_details(close, ma50, ma150, ma200)

    high_52w = float(close.iloc[-252:].max())
    low_52w = float(close.iloc[-252:].min())
    dist_from_high = (close_now / high_52w - 1) * 100 if high_52w > 0 else np.nan
    advance_from_low = (close_now / low_52w - 1) * 100 if low_52w > 0 else np.nan

    ma50_slope_pct = slope_pct(ma50_series, 20)
    ma150_slope_pct = slope_pct(ma150_series, 20)
    ma200_slope_pct = slope_pct(ma200_series, 20)
    weekly_ma10 = float(weekly_close.rolling(10).mean().iloc[-1]) if len(weekly_close) >= 10 else np.nan
    weekly_ma30 = float(weekly_close.rolling(30).mean().iloc[-1]) if len(weekly_close) >= 30 else np.nan

    market_regime_ok = regime.regime_label in {"strong_risk_on", "risk_on", "mixed"}

    daily_window = df.iloc[-140:]
    daily_depths, daily_durations, daily_base_duration = detect_vcp_contractions(
        daily_window["High"], daily_window["Low"], daily_window["Close"],
        config["swing_order_daily"], config["max_contractions"],
        config["min_contraction_days_daily"], config["min_contraction_depth_pct_daily"]
    )
    daily_contraction_score_val = contraction_score(daily_depths)

    weekly_window = weekly_df.iloc[-52:]
    weekly_depths, weekly_durations, weekly_base_duration = detect_vcp_contractions(
        weekly_window["High"], weekly_window["Low"], weekly_window["Close"],
        config["swing_order_weekly"], config["max_contractions"],
        config["min_contraction_days_weekly"], config["min_contraction_depth_pct_weekly"]
    )
    weekly_contraction_score_val = contraction_score(weekly_depths)
    weekly_quality = vcp_quality_label(
        weekly_contraction_score_val, weekly_base_duration, weekly_depths, config["min_base_duration_weeks"]
    )

    volume_dryup_ratio = volume_ratio(volume, config["volume_short_window"], config["volume_long_window"])
    weekly_volume_dryup_ratio = volume_ratio(weekly_df["Volume"].astype(float), 4, 12) if len(weekly_df) >= 12 else np.nan
     # Daily card volume: current day / previous 30 trading-day average.
    breakout_volume_ratio = recent_breakout_volume_ratio(volume, 30)
     # Weekly card volume: latest 5 trading days / previous 10 completed weekly volumes.
    weekly_volume_ratio = current_week_volume_ratio(volume, weekly_df["Volume"].astype(float), current_days=5, weekly_window=10)
    avg_turnover_inr = avg_turnover(close, volume, 20)
    liquidity_ok = pd.notna(avg_turnover_inr) and avg_turnover_inr >= config["min_avg_turnover_inr"]

    stock_3m = pct_return(close, 63)
    stock_6m = pct_return(close, 126)
    bm_3m = pct_return(benchmark_df["Close"], 63)
    bm_6m = pct_return(benchmark_df["Close"], 126)
    rs_3m = stock_3m - bm_3m if pd.notna(stock_3m) and pd.notna(bm_3m) else np.nan
    rs_6m = stock_6m - bm_6m if pd.notna(stock_6m) and pd.notna(bm_6m) else np.nan
    rs_combo = np.nanmean([rs_3m, rs_6m])

    daily_pivot = compute_pivot(high, config["pivot_lookback_daily"], daily_base_duration)
    daily_breakout_distance = (close_now / daily_pivot - 1) * 100 if pd.notna(daily_pivot) and daily_pivot > 0 else np.nan
    weekly_pivot = compute_pivot(weekly_high, config["pivot_lookback_weekly"], weekly_base_duration)
    weekly_breakout_distance = (float(weekly_close.iloc[-1]) / weekly_pivot - 1) * 100 if pd.notna(weekly_pivot) and weekly_pivot > 0 else np.nan

    recent_range_pct = (
        (close.iloc[-config["recent_range_days"]:].max() - close.iloc[-config["recent_range_days"]:].min()) /
        close.iloc[-config["recent_range_days"]:].max() * 100
    ) if len(close) >= config["recent_range_days"] else np.nan
    tight_range_ok = pd.notna(recent_range_pct) and recent_range_pct <= config["recent_range_max_pct"]

    price_above_ma50 = close_now > ma50
    price_above_ma150 = close_now > ma150
    price_above_ma200 = close_now > ma200
    ma_stack_bull = close_now > ma50 > ma150 > ma200
    ma_stack_bear = close_now < ma50 < ma150 < ma200

    weekly_range_12w = ((weekly_close.iloc[-12:].max() / weekly_close.iloc[-12:].min()) - 1) * 100 if len(weekly_close) >= 12 and weekly_close.iloc[-12:].min() > 0 else np.nan
    weekly_range_20w = ((weekly_close.iloc[-20:].max() / weekly_close.iloc[-20:].min()) - 1) * 100 if len(weekly_close) >= 20 and weekly_close.iloc[-20:].min() > 0 else np.nan
    recent_low_6w = float(low.iloc[-30:].min()) if len(low) >= 30 else np.nan
    no_recent_breakdown = pd.notna(recent_low_6w) and close_now >= recent_low_6w * 1.03

    stage2_trend_template = (
        stage == "Stage 2"
        and ma_stack_bull
        and pd.notna(ma50_slope_pct) and ma50_slope_pct > 0.0005
        and pd.notna(ma150_slope_pct) and ma150_slope_pct >= 0
        and pd.notna(ma200_slope_pct) and ma200_slope_pct >= -0.00015
        and pd.notna(dist_from_high) and dist_from_high >= -18
        and pd.notna(advance_from_low) and advance_from_low >= 30
        and pd.notna(rs_combo) and rs_combo >= 0
    )

    stage1_base_ready = (
        stage == "Stage 1"
        and pd.notna(ma200_slope_pct) and -0.00035 <= ma200_slope_pct <= 0.00035
        and pd.notna(ma150_slope_pct) and ma150_slope_pct >= -0.00035
        and price_above_ma150
        and price_above_ma200
        and pd.notna(dist_from_high) and -30 <= dist_from_high <= -3
        and pd.notna(weekly_range_12w) and weekly_range_12w <= 20
        and pd.notna(weekly_range_20w) and weekly_range_20w <= 35
        and pd.notna(rs_combo) and rs_combo >= -5
        and no_recent_breakdown
        and not ma_stack_bear
    )

    strong_daily_vcp = (
        len(daily_depths) >= 2
        and daily_base_duration >= config["min_base_duration_days"]
        and daily_contraction_score_val >= 0.60
        and daily_depths[-1] <= min(config["max_latest_contraction_pct"], 8.0)
        and pd.notna(volume_dryup_ratio) and volume_dryup_ratio <= 0.90
    )
    strict_stage1_daily_vcp = (
        strong_daily_vcp
        and daily_depths[0] <= 30
        and max(daily_depths) <= 30
        and pd.notna(daily_breakout_distance) and -4.0 <= daily_breakout_distance <= 1.5
        and tight_range_ok
    )
    weekly_vcp_ok = (
        len(weekly_depths) >= 2
        and weekly_base_duration >= config["min_base_duration_weeks"]
        and weekly_contraction_score_val >= max(config["min_weekly_strength_score"], 0.55)
        and weekly_quality in {"strong", "moderate"}
    )

    near_pivot_stage2_ok = (
        pd.notna(daily_breakout_distance)
        and -5.0 <= daily_breakout_distance <= 1.5
        and tight_range_ok
        and pd.notna(breakout_volume_ratio) and breakout_volume_ratio >= 0.85
    )
    near_pivot_stage1_ok = (
        pd.notna(daily_breakout_distance)
        and -3.0 <= daily_breakout_distance <= 1.0
        and tight_range_ok
        and pd.notna(volume_dryup_ratio) and volume_dryup_ratio <= 0.90
        and no_recent_breakdown
    )
    near_pivot_ok = near_pivot_stage2_ok if stage == "Stage 2" else near_pivot_stage1_ok if stage == "Stage 1" else False

    breakout_today = bool(
        pd.notna(daily_breakout_distance)
        and daily_breakout_distance > 0
        and pd.notna(breakout_volume_ratio)
        and breakout_volume_ratio >= config["breakout_volume_ratio"]
        and stage2_trend_template
        and strong_daily_vcp
    )

    daily_vcp_ok = strong_daily_vcp if stage == "Stage 2" else strict_stage1_daily_vcp if stage == "Stage 1" else False
    trend_template_ok = stage2_trend_template

    if stage == "Stage 1" and (not stage1_base_ready or not strict_stage1_daily_vcp):
        daily_bucket = "watchlist"
    else:
        daily_bucket = classify_daily_bucket(
            trend_template_ok if stage == "Stage 2" else False,
            daily_vcp_ok,
            near_pivot_ok,
            breakout_today,
            tight_range_ok,
            market_regime_ok,
        )
        if stage == "Stage 1" and daily_bucket == "building_setup":
            daily_bucket = "watchlist"

    weekly_bucket = classify_weekly_bucket(stage, weekly_vcp_ok, weekly_breakout_distance, weekly_quality)
    if stage == "Stage 1" and (not stage1_base_ready or not weekly_vcp_ok):
        weekly_bucket = "weekly_watchlist"

    daily_score = score_daily(
        stage,
        trend_template_ok,
        regime.regime_label,
        liquidity_ok,
        near_pivot_ok,
        breakout_today,
        daily_contraction_score_val,
        daily_base_duration,
        dist_from_high,
        volume_dryup_ratio,
        breakout_volume_ratio,
        rs_3m,
        rs_6m,
    )
    weekly_score = score_weekly(
        stage,
        weekly_contraction_score_val,
        weekly_base_duration,
        weekly_breakout_distance,
        weekly_quality,
        rs_3m,
        rs_6m,
    )

    if stage == "Stage 1":
        if not stage1_base_ready:
            daily_score -= 12
            weekly_score -= 8
        elif not strict_stage1_daily_vcp:
            daily_score -= 8
            weekly_score -= 5
        if breakout_today:
            daily_score -= 8
        if pd.notna(daily_breakout_distance) and daily_breakout_distance > 0:
            daily_score -= 3

    if stage == "Stage 3":
        daily_score -= 8
        weekly_score -= 6
    elif stage == "Stage 4":
        daily_score -= 12
        weekly_score -= 10

    daily_score = round(float(max(0.0, daily_score)), 2)
    weekly_score = round(float(max(0.0, weekly_score)), 2)

    combo_bucket = combined_bucket(daily_bucket, weekly_bucket)
    combined_score = round(0.55 * daily_score + 0.45 * weekly_score, 2)

    volume_is_drying_up = bool(pd.notna(volume_dryup_ratio) and volume_dryup_ratio <= 0.85)
    weekly_volume_is_drying_up = bool(pd.notna(weekly_volume_ratio) and weekly_volume_ratio <= 0.90)

    notes = [stage, stage_variant]
    if trend_template_ok:
        notes.append("trend_template_ok")
    if stage1_base_ready:
        notes.append("stage1_base_ready")
    if daily_vcp_ok:
        notes.append("daily_vcp_ok")
    if weekly_vcp_ok:
        notes.append("weekly_vcp_ok")
    if volume_is_drying_up:
        notes.append("volume_dryup")
    if weekly_volume_is_drying_up:
        notes.append("weekly_volume_dryup")
    if breakout_today:
        notes.append("daily_breakout_volume")
    if weekly_quality == "strong":
        notes.append("weekly_strong")
    if stage == "Stage 1" and not strict_stage1_daily_vcp:
        notes.append("stage1_not_actionable")
    if stage == "Stage 1" and not stage1_base_ready:
        notes.append("stage1_needs_more_base")
    if stage == "Stage 3":
        notes.append("distribution_risk")
    if stage == "Stage 4":
        notes.append("downtrend")

    return VCPScoreCard(
        ticker, round(close_now, 2), round(ma50, 2), round(ma150, 2), round(ma200, 2), stage,
        stage_variant, round(float(stage_confidence), 2), stage_reason,
        round(float(rs_3m), 2) if pd.notna(rs_3m) else np.nan,
        round(float(rs_6m), 2) if pd.notna(rs_6m) else np.nan,
        round(float(avg_turnover_inr), 2) if pd.notna(avg_turnover_inr) else np.nan,
        daily_bucket, daily_score, round(float(daily_pivot), 2) if pd.notna(daily_pivot) else np.nan,
        round(float(daily_breakout_distance), 2) if pd.notna(daily_breakout_distance) else np.nan,
        daily_depths, daily_durations, round(float(daily_contraction_score_val), 2), round(float(daily_base_duration), 2),
        weekly_bucket, weekly_score, round(float(weekly_pivot), 2) if pd.notna(weekly_pivot) else np.nan,
        round(float(weekly_breakout_distance), 2) if pd.notna(weekly_breakout_distance) else np.nan,
        weekly_depths, weekly_durations, round(float(weekly_contraction_score_val), 2), round(float(weekly_base_duration), 2),
        weekly_quality, combo_bucket, combined_score,
        round(float(volume_dryup_ratio), 2) if pd.notna(volume_dryup_ratio) else np.nan,
        round(float(breakout_volume_ratio), 2) if pd.notna(breakout_volume_ratio) else np.nan,
        round(float(weekly_volume_ratio), 2) if pd.notna(weekly_volume_ratio) else np.nan,
        volume_is_drying_up,
        weekly_volume_is_drying_up,
        ", ".join(notes),
    )


def build_vcp_universe_report(tickers: List[str], config: Optional[dict] = None, price_data: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[pd.DataFrame, MarketRegime, Dict[str, pd.DataFrame]]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    full_tickers = list(dict.fromkeys(tickers + [cfg["market_index"]]))

    if price_data is None:
        print("No --wide-price supplied, falling back to Yahoo download. This will be slower.")
        price_data = fetch_prices(full_tickers, cfg["period"], interval="1d")

    if cfg["market_index"] not in price_data:
        raise RuntimeError(f"Missing market index data for {cfg['market_index']}")
    benchmark_df = price_data[cfg["market_index"]]
    regime = market_regime(
        benchmark_df,
        cfg["market_index"],
        cfg["market_ma_fast"],
        cfg["market_ma_slow"],
        price_data=price_data,
        universe_tickers=tickers,
    )

    rows = []
    t0 = time.perf_counter()
    for idx, ticker in enumerate(tickers, start=1):
        df = price_data.get(str(ticker).upper(), price_data.get(ticker))
        if df is None or df.empty:
            continue
        try:
            result = analyze_symbol(ticker, df, benchmark_df, regime, cfg)
            if result:
                rows.append(asdict(result))
        except Exception as exc:
            rows.append({"ticker": ticker, "combined_score": -1, "combined_bucket": "error", "notes": f"error: {exc}"})
        if idx % 100 == 0:
            print(f"Analyzed {idx:,}/{len(tickers):,} symbols...")

    print(f"Analyzed {len(tickers):,} symbols in {time.perf_counter()-t0:.2f}s")
    out = pd.DataFrame(rows)
    if out.empty:
        return out, regime, price_data
    order = {"high_conviction_breakout": 0, "high_conviction_near_pivot": 1, "building_setup": 2, "watchlist": 3, "error": 4}
    out["bucket_order"] = out["combined_bucket"].map(order).fillna(99)
    out = out.sort_values(["bucket_order", "combined_score", "daily_score", "weekly_score"], ascending=[True, False, False, False]).drop(columns=["bucket_order"])
    return out.reset_index(drop=True), regime, price_data

def build_industry_strength_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("Industry").agg(
        avg_rs_3m=("rs_3m_pct", "mean"),
        avg_rs_6m=("rs_6m_pct", "mean"),
        avg_daily_score=("daily_score", "mean"),
        avg_weekly_score=("weekly_score", "mean"),
        avg_combined_score=("combined_score", "mean"),
        stock_count=("ticker", "count"),
        actionable_daily=("daily_setup_bucket", lambda x: x.isin(["near_pivot", "breakout_today"]).sum()),
        actionable_weekly=("weekly_setup_bucket", lambda x: x.isin(["weekly_near_pivot", "weekly_breakout"]).sum()),
        strong_combined=("combined_bucket", lambda x: x.isin(["high_conviction_breakout", "high_conviction_near_pivot"]).sum()),
    ).reset_index()
    summary["rs_score"] = summary[["avg_rs_3m", "avg_rs_6m"]].mean(axis=1)
    summary["rs_rank"] = summary["rs_score"].rank(pct=True, method="average") * 100
    return summary.sort_values(["avg_combined_score", "rs_rank", "strong_combined"], ascending=[False, False, False]).reset_index(drop=True)

def apply_industry_boost(report_df: pd.DataFrame, industry_df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    df = report_df.merge(industry_df[["Industry", "rs_rank"]], on="Industry", how="left")

    def boost(industry_rank: float) -> float:
        if pd.isna(industry_rank):
            return 0.0
        if industry_rank >= cfg["industry_boost_top"]:
            return cfg["industry_boost_top_points"]
        if industry_rank >= cfg["industry_boost_mid"]:
            return cfg["industry_boost_mid_points"]
        if industry_rank >= cfg["industry_boost_low"]:
            return cfg["industry_boost_low_points"]
        return 0.0

    df["industry_boost"] = df["rs_rank"].apply(boost)
    df["final_daily_score"] = (df["daily_score"] + 0.5 * df["industry_boost"]).round(2)
    df["final_weekly_score"] = (df["weekly_score"] + 0.5 * df["industry_boost"]).round(2)
    df["final_combined_score"] = (df["combined_score"] + df["industry_boost"]).round(2)
    return df.sort_values(["final_combined_score", "final_daily_score", "final_weekly_score"], ascending=[False, False, False]).reset_index(drop=True)

def sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)



def export_chart(
    df: pd.DataFrame,
    symbol: str,
    title: str,
    outfile: Path,
    pivot: Optional[float],
    setup_bucket: str,
    score: float,
    stage: str,
    is_weekly: bool = False,
    *,
    dpi: int = 150,
    skip_existing: bool = False,
) -> None:
    """Export a dashboard-ready chart.

    This is intentionally sized for Streamlit/mobile display instead of print.
    The analytical content is unchanged: close, key MAs, pivot zone, volume and
    VCP contraction labels are still shown. Runtime drops sharply versus the old
    34x22 inch / 240 dpi chart export.
    """
    if df.empty:
        return
    # Always regenerate charts on every run. Existing PNGs are intentionally overwritten
    # because the Watchlist and Charts search pages must reflect the latest price/state.
    # The `skip_existing` argument is kept only for backward compatibility and ignored.

    target_display_bars = 180 if not is_weekly else 104
    fast_window = 10 if is_weekly else 50
    slow_window = 30 if is_weekly else 200
    min_visible_bars = 55 if is_weekly else 120
    history_buffer = target_display_bars + slow_window + (20 if is_weekly else 60)

    working_df = df.copy().tail(history_buffer).copy()
    if working_df.empty:
        return

    close_all = working_df["Close"].astype(float)
    ma_fast_all = close_all.rolling(fast_window).mean()
    ma_slow_all = close_all.rolling(slow_window).mean()

    default_start_idx = max(0, len(working_df) - target_display_bars)

    def _first_full_window_start(series: pd.Series, target_len: int) -> Optional[int]:
        valid = np.where(series.notna().values)[0]
        if len(valid) == 0:
            return None
        first_valid = int(valid[0])
        if len(series) - first_valid >= target_len:
            return first_valid
        return None

    visible_bars = target_display_bars
    start_idx = default_start_idx

    if is_weekly:
        max_bars_with_slow = len(working_df) - slow_window + 1
        if max_bars_with_slow > 0:
            visible_bars = min(target_display_bars, max(max_bars_with_slow, min_visible_bars))
            start_idx = max(0, len(working_df) - visible_bars)
        else:
            visible_bars = min(target_display_bars, len(working_df))
            start_idx = max(0, len(working_df) - visible_bars)
    else:
        slow_full_start = _first_full_window_start(ma_slow_all, target_display_bars)
        fast_full_start = _first_full_window_start(ma_fast_all, target_display_bars)
        if slow_full_start is not None:
            start_idx = max(default_start_idx, slow_full_start)
        elif fast_full_start is not None:
            start_idx = max(default_start_idx, fast_full_start)

    plot_df = working_df.iloc[start_idx:].copy()
    if plot_df.empty:
        return

    close = plot_df["Close"].astype(float)
    high = plot_df["High"].astype(float)
    low = plot_df["Low"].astype(float)
    volume = plot_df["Volume"].astype(float)
    x = plot_df.index

    ma_fast = ma_fast_all.iloc[start_idx:].copy()
    ma_slow = ma_slow_all.iloc[start_idx:].copy()
    if ma_fast.notna().sum() < len(plot_df):
        ma_fast = None
    if ma_slow.notna().sum() < len(plot_df):
        ma_slow = None

    pair_seq = extract_vcp_contraction_pairs(
        high,
        low,
        order=DEFAULT_CONFIG["swing_order_weekly"] if is_weekly else DEFAULT_CONFIG["swing_order_daily"],
        max_pairs=DEFAULT_CONFIG["max_contractions"],
        min_duration_bars=DEFAULT_CONFIG["min_contraction_days_weekly"] if is_weekly else DEFAULT_CONFIG["min_contraction_days_daily"],
        min_depth_pct=DEFAULT_CONFIG["min_contraction_depth_pct_weekly"] if is_weekly else DEFAULT_CONFIG["min_contraction_depth_pct_daily"],
    )
    base_duration = float(pair_seq[-1][1] - pair_seq[0][0]) if pair_seq else np.nan
    pivot_low, pivot_high, _ = compute_pivot_zone(
        high,
        DEFAULT_CONFIG["pivot_lookback_weekly"] if is_weekly else DEFAULT_CONFIG["pivot_lookback_daily"],
        base_duration=base_duration,
        is_weekly=is_weekly,
        min_band_pct=1.6 if is_weekly else 1.15,
        max_band_pct=4.8 if is_weekly else 3.6,
    )

    # Web/dashboard friendly chart dimensions. This keeps quality high in the app
    # while avoiding oversized print-resolution PNGs. v27 keeps the original chart
    # visual style but increases readability on laptop screens.
    base_font = 16 if is_weekly else 15
    plt.rcParams.update({
        "font.size": base_font,
        "axes.titlesize": 30,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
    })

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(15.8, 10.0),
        sharex=True,
        gridspec_kw={"height_ratios": [4.7, 1.0]},
    )
    fig.patch.set_facecolor("#f8fafc")
    ax1.set_facecolor("#ffffff")
    ax2.set_facecolor("#ffffff")

    ax1.plot(x, close.values, label="Close", linewidth=2.2, color="#4F81BD")
    if ma_fast is not None:
        ax1.plot(x, ma_fast.values, label=("10W MA" if is_weekly else "50DMA"), linewidth=2.7, alpha=0.96, color="#339933")
    if ma_slow is not None:
        ax1.plot(x, ma_slow.values, label=("30W MA" if is_weekly else "200DMA"), linewidth=2.7, alpha=0.92, color="#C0504D")

    if pd.notna(pivot_low) and pd.notna(pivot_high):
        ax1.axhspan(float(pivot_low), float(pivot_high), alpha=0.18, label="Pivot zone", color="#f59e0b")
        ax1.axhline(float(pivot_low), linestyle="--", linewidth=0.9, alpha=0.40, color="#b45309")
        ax1.axhline(float(pivot_high), linestyle="--", linewidth=0.9, alpha=0.40, color="#b45309")

    suffix = "W" if is_weekly else "D"
    y_span = float(high.max() - low.min()) if np.isfinite(high.max()) and np.isfinite(low.min()) else 0.0
    if y_span <= 0:
        y_span = max(float(high.max()) * 0.08, 1.0)

    base_label_gap = y_span * (0.065 if is_weekly else 0.045)
    horizontal_step = 2 if is_weekly else 4
    placed = []

    def _find_label_slot(bar_idx: int, anchor_y: float):
        candidates = [(0, 0)] + [
            ((-1 if level % 2 else 1) * ((level + 1) // 2) * horizontal_step,
             (-1 if level % 2 else 1) * ((level + 1) // 2) * base_label_gap)
            for level in range(1, 7)
        ]
        best = None
        best_penalty = None
        for x_shift, y_shift in candidates:
            cand_idx = min(max(bar_idx + x_shift, 0), len(x) - 1)
            cand_y = anchor_y + y_shift
            penalty = abs(x_shift) * 0.9 + abs(y_shift) / max(base_label_gap, 1e-9)
            overlap = False
            for prev_idx, prev_y in placed:
                if abs(cand_idx - prev_idx) <= (3 if is_weekly else 6) and abs(cand_y - prev_y) < base_label_gap * 0.90:
                    overlap = True
                    penalty += 100
                    break
            if not overlap:
                return cand_idx, cand_y
            if best is None or penalty < best_penalty:
                best = (cand_idx, cand_y)
                best_penalty = penalty
        return best if best is not None else (bar_idx, anchor_y)

    for peak_i, trough_i, depth, duration in pair_seq:
        trough_price = float(low.iloc[trough_i])
        anchor_y = trough_price - y_span * 0.02
        label_idx, label_y = _find_label_slot(trough_i, anchor_y)
        placed.append((label_idx, label_y))
        rounded_depth = int(round(depth))
        ax1.annotate(
            f"(-{rounded_depth}%, {duration}{suffix})",
            xy=(x[trough_i], trough_price),
            xytext=(x[label_idx], label_y),
            textcoords="data",
            ha="center",
            va="top",
            fontsize=CHART_ANNOTATION_FONTSIZE,
            fontweight="bold",
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.18", alpha=0.10, facecolor="#e2e8f0", edgecolor="none"),
        )

    chart_suffix = "Weekly" if is_weekly else "Daily"
    ax1.set_title(f"{symbol} - {chart_suffix} - {stage}", pad=12, fontweight="bold", color="#0f172a")
    fig.text(0.5, 0.52, "StockGita", ha="center", va="center", fontsize=CHART_TITLE_FONTSIZE, alpha=0.055, rotation=24, weight="bold", color="#0f172a")
    # No grid lines: cleaner chart surface for dashboard viewing.
    ax1.grid(False)
    ax1.legend(loc="upper left", ncol=4, frameon=False, handlelength=2.6, columnspacing=1.4, borderaxespad=0.6)
    ax1.tick_params(axis="both", labelsize=CHART_TICK_FONTSIZE, pad=7)
    ax1.set_ylabel("Price", fontweight="bold", fontsize=CHART_TITLE_FONTSIZE)

    if len(x) >= 2:
        if hasattr(x, "dtype") and "datetime" in str(x.dtype):
            step = x[-1] - x[-2]
            if pd.isna(step) or step == pd.Timedelta(0):
                step = pd.Timedelta(days=7 if is_weekly else 1)
            right_pad = step * (3 if is_weekly else 6)
        else:
            right_pad = 3 if is_weekly else 6
        ax1.set_xlim(x[0], x[-1] + right_pad)

    margin_top = y_span * 0.10
    margin_bottom = y_span * 0.18
    ax1.set_ylim(max(0, float(low.min()) - margin_bottom), float(high.max()) + margin_top)

    bar_width = 4 if is_weekly else 0.9
    vol_window = 10 if is_weekly else 20
    # Keep volume visually simple: one Excel violet pastel bar color.
    ax2.bar(x, volume.values, width=bar_width, alpha=0.72, color="#8064A2")
    vol_ma = volume.rolling(vol_window).mean()
    if vol_ma.notna().sum() == len(volume):
        ax2.plot(x, vol_ma.values, linewidth=2.3, label=("10W Vol MA" if is_weekly else "20D Vol MA"), color="#8064A2")

    # No grid lines on volume panel.
    ax2.grid(False)
    ax2.set_ylabel("Volume", fontweight="bold", fontsize=CHART_TITLE_FONTSIZE)
    ax2.set_yticks([])
    ax2.tick_params(axis="y", which="both", length=0, labelleft=False)
    ax2.tick_params(axis="x", labelsize=CHART_TICK_FONTSIZE, pad=7)
    if len(x) >= 2:
        if hasattr(x, "dtype") and "datetime" in str(x.dtype):
            step = x[-1] - x[-2]
            if pd.isna(step) or step == pd.Timedelta(0):
                step = pd.Timedelta(days=7 if is_weekly else 1)
            right_pad = step * (3 if is_weekly else 6)
        else:
            right_pad = 3 if is_weekly else 6
        ax2.set_xlim(x[0], x[-1] + right_pad)
    if vol_ma.notna().sum() == len(volume):
        ax2.legend(loc="upper left", frameon=False, fontsize=CHART_ANNOTATION_FONTSIZE)

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    apply_mobile_chart_readability(fig)
    fig.savefig(outfile, dpi=dpi, facecolor=fig.get_facecolor(), pad_inches=0.12)
    plt.close(fig)


def _ticker_set_from_existing_csv(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=lambda c: c in {"ticker", "Ticker"})
        col = "ticker" if "ticker" in df.columns else "Ticker" if "Ticker" in df.columns else None
        if col is None:
            return set()
        return set(df[col].dropna().astype(str).str.upper())
    except Exception:
        return set()




def _ticker_set_from_trending_file(path: Path, limit: int = 20) -> set[str]:
    """Read first `limit` tickers from a manual Trending Stocks CSV/XLSX file."""
    if not path.exists():
        return set()
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        if df.empty:
            return set()
        normalized_cols = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
        ticker_col = None
        for cand in ["ticker", "symbol", "stock", "stock_symbol", "nse_symbol"]:
            if cand in normalized_cols:
                ticker_col = normalized_cols[cand]
                break
        if ticker_col is None:
            ticker_col = df.columns[0]
        out = []
        seen = set()
        for value in df[ticker_col].dropna().tolist():
            t = str(value).strip().upper()
            if not t or t in {"NAN", "NONE"}:
                continue
            if not t.startswith("^") and not t.endswith(".NS"):
                t = f"{t}.NS"
            if t not in seen:
                seen.add(t)
                out.append(t)
            if len(out) >= limit:
                break
        return set(out)
    except Exception:
        return set()

def build_dashboard_chart_tickers(
    combined_df: pd.DataFrame,
    price_moves: pd.DataFrame,
    stock_changes: pd.DataFrame,
    outdir: Path,
    *,
    top_rank_limit: int = 140,
) -> List[str]:
    """Return all tickers needed by the public dashboard views.

    This preserves dashboard functionality while avoiding generation of charts for
    every included stock on every run. It covers:
    - Interesting 20/top-ranked pool
    - Miscellaneous stage buckets
    - New Stage 2
    - Top/Bottom daily movers
    - Prior Interesting 20 archive/latest file
    """
    tickers: set[str] = set()
    if combined_df is not None and not combined_df.empty and "ticker" in combined_df.columns:
        ranked = combined_df.copy()
        if "current_rank" in ranked.columns:
            ranked["current_rank"] = pd.to_numeric(ranked["current_rank"], errors="coerce")
            tickers.update(ranked.sort_values("current_rank", ascending=True).head(top_rank_limit)["ticker"].dropna().astype(str).str.upper())
        else:
            tickers.update(ranked.head(top_rank_limit)["ticker"].dropna().astype(str).str.upper())

        # Make sure each stage has enough prebuilt charts for Miscellaneous 20.
        if "stage" in ranked.columns:
            for stage in ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]:
                part = ranked[ranked["stage"].astype(str).eq(stage)]
                if "current_rank" in part.columns:
                    part = part.sort_values("current_rank", ascending=True)
                tickers.update(part.head(30)["ticker"].dropna().astype(str).str.upper())

    if price_moves is not None and not price_moves.empty and "ticker" in price_moves.columns:
        move_col = "change_1d_pct" if "change_1d_pct" in price_moves.columns else None
        if move_col:
            pm = price_moves.copy()
            pm[move_col] = pd.to_numeric(pm[move_col], errors="coerce")
            tickers.update(pm.sort_values(move_col, ascending=False).head(20)["ticker"].dropna().astype(str).str.upper())
            tickers.update(pm.sort_values(move_col, ascending=True).head(20)["ticker"].dropna().astype(str).str.upper())
        else:
            tickers.update(price_moves.head(40)["ticker"].dropna().astype(str).str.upper())

    if stock_changes is not None and not stock_changes.empty and "ticker" in stock_changes.columns:
        if "entered_stage_2" in stock_changes.columns:
            entered = stock_changes[stock_changes["entered_stage_2"].astype(str).str.lower().isin(["true", "1", "yes"])]
            tickers.update(entered["ticker"].dropna().astype(str).str.upper())
        tickers.update(stock_changes.head(40)["ticker"].dropna().astype(str).str.upper())

    # Include prior Interesting 20 so Last Week Interesting charts remain available.
    tickers.update(_ticker_set_from_existing_csv(outdir / "interesting20_latest.csv"))
    archive_dir = outdir / "interesting20_archive"
    if archive_dir.exists():
        recent_archives = sorted(archive_dir.glob("*_interesting20.csv"))[-3:]
        for path in recent_archives:
            tickers.update(_ticker_set_from_existing_csv(path))

    # Include manually curated Trending Stocks so dashboard-scope chart generation
    # prebuilds daily/weekly charts for the first 20 names in trending_stocks.csv/xlsx.
    for trending_path in [
        outdir / "trending_stocks.csv",
        Path("trending_stocks.csv"),
        outdir / "trending_stocks.xlsx",
        Path("trending_stocks.xlsx"),
    ]:
        tickers.update(_ticker_set_from_trending_file(trending_path, limit=20))

    return sorted(tickers)


def export_selected_charts(
    final_report: pd.DataFrame,
    price_data: Dict[str, pd.DataFrame],
    outdir: Path,
    tickers_to_export: Optional[List[str]] = None,
    *,
    skip_existing: bool = False,
    dpi: int = 240,
) -> Dict[str, str]:
    charts_root = outdir / "charts"
    daily_dir = charts_root / "daily"
    weekly_dir = charts_root / "weekly"
    daily_dir.mkdir(parents=True, exist_ok=True)
    weekly_dir.mkdir(parents=True, exist_ok=True)
    score_map = final_report.set_index("ticker").to_dict(orient="index") if "ticker" in final_report.columns else {}

    if tickers_to_export is None:
        tickers = [t for t in price_data.keys() if t != DEFAULT_CONFIG["market_index"]]
    else:
        wanted = {str(t).upper() for t in tickers_to_export}
        tickers = [t for t in price_data.keys() if str(t).upper() in wanted and t != DEFAULT_CONFIG["market_index"]]

    print(f"Generating charts for {len(tickers):,} symbols ({'all' if tickers_to_export is None else 'dashboard-needed'} scope)...")
    t0 = time.perf_counter()
    failed_charts = 0
    for idx, ticker in enumerate(tickers, start=1):
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue
        row = score_map.get(ticker, {})
        chart_name = str(row.get("Company Name") or ticker).strip()
        safe = sanitize_filename(ticker)
        try:
            export_chart(
                df,
                chart_name,
                "Daily",
                daily_dir / f"{safe}_daily.png",
                row.get("daily_pivot"),
                row.get("daily_setup_bucket", "watchlist"),
                float(row.get("final_daily_score", row.get("daily_score", 0)) or 0),
                row.get("stage", ""),
                False,
                dpi=dpi,
                skip_existing=skip_existing,
            )
            weekly_df = resample_weekly(df)
            if not weekly_df.empty:
                export_chart(
                    weekly_df,
                    chart_name,
                    "Weekly",
                    weekly_dir / f"{safe}_weekly.png",
                    row.get("weekly_pivot"),
                    row.get("weekly_setup_bucket", "weekly_watchlist"),
                    float(row.get("final_weekly_score", row.get("weekly_score", 0)) or 0),
                    row.get("stage", ""),
                    True,
                    dpi=dpi,
                    skip_existing=skip_existing,
                )
        except Exception as exc:
            failed_charts += 1
            print(f"Warning: chart export failed for {ticker}: {exc}")
            plt.close("all")
        if idx % 25 == 0 or idx == len(tickers):
            gc.collect()
            print(f"Charts processed {idx:,}/{len(tickers):,} in {time.perf_counter()-t0:.2f}s")
    if failed_charts:
        print(f"Chart generation completed with {failed_charts:,} chart warning(s).")
    print(f"Chart generation complete in {time.perf_counter()-t0:.2f}s")
    return {"daily_charts_dir": str(daily_dir), "weekly_charts_dir": str(weekly_dir)}


def export_all_charts(final_report: pd.DataFrame, price_data: Dict[str, pd.DataFrame], outdir: Path) -> Dict[str, str]:
    return export_selected_charts(final_report, price_data, outdir, tickers_to_export=None)

def _clean_stock_snapshot(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "Company Name" not in out.columns:
        for col in ["Company Name_x", "Company Name_y"]:
            if col in out.columns:
                out["Company Name"] = out[col]
                break
    if "Industry" not in out.columns:
        for col in ["Industry_x", "Industry_y"]:
            if col in out.columns:
                out["Industry"] = out[col]
                break
    for col, fallback in {"final_daily_score": "daily_score", "final_weekly_score": "weekly_score", "final_combined_score": "combined_score"}.items():
        if col not in out.columns:
            out[col] = pd.to_numeric(out.get(fallback), errors="coerce")
    for col in ["daily_setup_bucket", "weekly_setup_bucket", "stage"]:
        if col not in out.columns:
            out[col] = pd.NA
    keep_cols = [
        "ticker", "Company Name", "Industry", "sector", "is_fo_stock", "fo_category", "Include", "stage", "stage_raw", "stage_variant", "stage_classification", "stage_display", "stage_confidence", "stage_reason", "stage_state_reason", "stage_failed_since", "last_stage2_date", "stage_pending_raw", "daily_setup_bucket", "weekly_setup_bucket", "combined_bucket",
        "daily_score", "weekly_score", "combined_score", "industry_boost", "final_daily_score", "final_weekly_score",
        "final_combined_score", "rs_3m_pct", "rs_6m_pct", "avg_turnover_inr", "notes",
    ]
    out = out[[c for c in keep_cols if c in out.columns]].copy()
    for col in ["daily_score", "weekly_score", "combined_score", "industry_boost", "final_daily_score", "final_weekly_score", "final_combined_score", "rs_3m_pct", "rs_6m_pct", "avg_turnover_inr"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

def build_stock_changes(current_df: pd.DataFrame, previous_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = _clean_stock_snapshot(current_df).sort_values("final_combined_score", ascending=False).reset_index(drop=True)
    df["current_rank"] = np.arange(1, len(df) + 1)
    if previous_df is None or previous_df.empty:
        df["prev_rank"] = np.nan
        df["rank_change"] = np.nan
        df["prev_score"] = np.nan
        df["combined_score_change"] = np.nan
        df["new_daily_breakout"] = False
        df["new_weekly_breakout"] = False
        df["entered_stage_2"] = False
        df["new_top_10"] = df["current_rank"] <= 10
        df["new_top_20"] = df["current_rank"] <= 20
        return df

    prev = _clean_stock_snapshot(previous_df).sort_values("final_combined_score", ascending=False).reset_index(drop=True)
    prev["prev_rank"] = np.arange(1, len(prev) + 1)
    prev = prev.rename(columns={
        "stage": "prev_stage",
        "daily_setup_bucket": "prev_daily_setup_bucket",
        "weekly_setup_bucket": "prev_weekly_setup_bucket",
        "final_combined_score": "prev_score",
    })
    df = df.merge(prev[["ticker", "prev_rank", "prev_stage", "prev_daily_setup_bucket", "prev_weekly_setup_bucket", "prev_score"]], on="ticker", how="left")
    df["rank_change"] = df["prev_rank"] - df["current_rank"]
    df["combined_score_change"] = df["final_combined_score"] - df["prev_score"]
    df["new_daily_breakout"] = (df["daily_setup_bucket"] == "breakout_today") & (df["prev_daily_setup_bucket"] != "breakout_today")
    df["new_weekly_breakout"] = (df["weekly_setup_bucket"] == "weekly_breakout") & (df["prev_weekly_setup_bucket"] != "weekly_breakout")
    df["entered_stage_2"] = (df["stage"] == "Stage 2") & (df["prev_stage"] != "Stage 2")
    df["new_top_10"] = (df["current_rank"] <= 10) & (~df["prev_rank"].between(1, 10, inclusive="both").fillna(False))
    df["new_top_20"] = (df["current_rank"] <= 20) & (~df["prev_rank"].between(1, 20, inclusive="both").fillna(False))
    return df.sort_values(["current_rank", "final_combined_score"], ascending=[True, False]).reset_index(drop=True)

def build_industry_changes(current_df: pd.DataFrame, previous_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = current_df.copy().sort_values(["avg_combined_score", "rs_rank", "strong_combined"], ascending=[False, False, False]).reset_index(drop=True)
    df["current_rank"] = np.arange(1, len(df) + 1)
    if previous_df is None or previous_df.empty:
        df["prev_rank"] = np.nan
        df["rank_change"] = np.nan
        df["combined_score_change"] = np.nan
        df["strong_combined_change"] = np.nan
        df["actionable_daily_change"] = np.nan
        df["actionable_weekly_change"] = np.nan
        df["new_cluster"] = df["strong_combined"].fillna(0) >= 2
        return df

    prev = previous_df.copy().sort_values(["avg_combined_score", "rs_rank", "strong_combined"], ascending=[False, False, False]).reset_index(drop=True)
    prev["prev_rank"] = np.arange(1, len(prev) + 1)
    prev = prev.rename(columns={
        "avg_combined_score": "prev_avg_combined_score",
        "rs_rank": "prev_rs_rank",
        "strong_combined": "prev_strong_combined",
        "actionable_daily": "prev_actionable_daily",
        "actionable_weekly": "prev_actionable_weekly",
    })
    cols = ["Industry", "prev_rank", "prev_avg_combined_score", "prev_rs_rank", "prev_strong_combined", "prev_actionable_daily", "prev_actionable_weekly"]
    df = df.merge(prev[cols], on="Industry", how="left")
    df["rank_change"] = df["prev_rank"] - df["current_rank"]
    df["combined_score_change"] = (df["avg_combined_score"] - df["prev_avg_combined_score"]).round(2)
    df["strong_combined_change"] = (df["strong_combined"] - df["prev_strong_combined"]).round(0)
    df["actionable_daily_change"] = (df["actionable_daily"] - df["prev_actionable_daily"]).round(0)
    df["actionable_weekly_change"] = (df["actionable_weekly"] - df["prev_actionable_weekly"]).round(0)
    df["new_cluster"] = (df["strong_combined"].fillna(0) >= 2) & (df["prev_strong_combined"].fillna(0) < 2)
    return df.sort_values(["current_rank", "avg_combined_score"], ascending=[True, False]).reset_index(drop=True)




def _read_existing_stage_history(out_path: Path) -> pd.DataFrame:
    history_file = out_path / "stage_action_history.csv"
    if not history_file.exists():
        return pd.DataFrame()
    try:
        hist = pd.read_csv(history_file, parse_dates=["snapshot_date"])
        if "snapshot_date" in hist.columns:
            hist["snapshot_date"] = pd.to_datetime(hist["snapshot_date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        return hist.dropna(subset=["snapshot_date", "ticker"])
    except Exception:
        return pd.DataFrame()


def _latest_stage_map_from_history(history_df: pd.DataFrame) -> Dict[str, str]:
    if history_df is None or history_df.empty or "ticker" not in history_df.columns or "stage" not in history_df.columns:
        return {}
    hist = history_df.copy().dropna(subset=["ticker"])
    hist["ticker"] = hist["ticker"].astype(str).str.upper().str.strip()
    hist = hist.sort_values(["snapshot_date", "ticker"])
    latest = hist.drop_duplicates("ticker", keep="last")
    return dict(zip(latest["ticker"], latest["stage"].astype(str)))


def _last_stage2_date_map(history_df: pd.DataFrame) -> Dict[str, pd.Timestamp]:
    if history_df is None or history_df.empty or "ticker" not in history_df.columns or "stage" not in history_df.columns:
        return {}
    hist = history_df.copy().dropna(subset=["ticker"])
    hist["ticker"] = hist["ticker"].astype(str).str.upper().str.strip()
    stage_text = hist["stage"].astype(str)
    raw_text = hist["stage_raw"].astype(str) if "stage_raw" in hist.columns else stage_text
    stage2_rows = hist[(stage_text == "Stage 2") | (raw_text == "Stage 2")].copy()
    if stage2_rows.empty:
        return {}
    stage2_rows = stage2_rows.sort_values(["snapshot_date", "ticker"])
    latest = stage2_rows.drop_duplicates("ticker", keep="last")
    return dict(zip(latest["ticker"], pd.to_datetime(latest["snapshot_date"], errors="coerce")))


def _previous_stage_map_from_combined(prev_combined: Optional[pd.DataFrame]) -> Dict[str, str]:
    if prev_combined is None or prev_combined.empty or "ticker" not in prev_combined.columns or "stage" not in prev_combined.columns:
        return {}
    prev = prev_combined.copy().dropna(subset=["ticker"])
    prev["ticker"] = prev["ticker"].astype(str).str.upper().str.strip()
    prev = prev.drop_duplicates("ticker", keep="last")
    return dict(zip(prev["ticker"], prev["stage"].astype(str)))


def _append_note(existing: object, note: str) -> str:
    text = "" if pd.isna(existing) else str(existing).strip()
    if not text:
        return note
    if note in text:
        return text
    return f"{text} | {note}"


def _history_stage_col(history_df: pd.DataFrame) -> str:
    if history_df is not None and not history_df.empty and "stage_raw" in history_df.columns:
        return "stage_raw"
    return "stage"


def _ticker_history(history_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if history_df is None or history_df.empty or "ticker" not in history_df.columns:
        return pd.DataFrame()
    t = str(ticker or "").upper().strip()
    hist = history_df.copy()
    hist["ticker"] = hist["ticker"].astype(str).str.upper().str.strip()
    hist = hist[hist["ticker"].eq(t)].copy()
    if hist.empty:
        return hist
    if "snapshot_date" in hist.columns:
        hist["snapshot_date"] = pd.to_datetime(hist["snapshot_date"], errors="coerce")
        hist = hist.dropna(subset=["snapshot_date"]).sort_values("snapshot_date")
    return hist


def _consecutive_raw_stage_runs(history_df: pd.DataFrame, prev_combined: Optional[pd.DataFrame], ticker: str, target_stage: str) -> int:
    """Count prior consecutive daily runs where raw stage matched target_stage.

    Current day is intentionally NOT counted here. Caller should add 1 when today's raw stage matches.
    """
    target = str(target_stage or "").strip()
    if not target:
        return 0

    records: List[Tuple[pd.Timestamp, str]] = []
    hist = _ticker_history(history_df, ticker)
    if not hist.empty:
        col = _history_stage_col(hist)
        for _, r in hist.iterrows():
            dt = pd.to_datetime(r.get("snapshot_date"), errors="coerce")
            stg = str(r.get(col, r.get("stage", "")) or "").strip()
            if pd.notna(dt) and stg:
                records.append((pd.Timestamp(dt).normalize(), stg))

    if prev_combined is not None and not prev_combined.empty and "ticker" in prev_combined.columns:
        prev = prev_combined.copy()
        prev["ticker"] = prev["ticker"].astype(str).str.upper().str.strip()
        row = prev[prev["ticker"].eq(str(ticker or "").upper().strip())]
        if not row.empty:
            r = row.iloc[-1]
            stg = str(r.get("stage_raw", r.get("stage", "")) or "").strip()
            if stg:
                # Use a synthetic timestamp after history, so it contributes if history is missing/not initialized.
                records.append((pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize() - pd.Timedelta(hours=1), stg))

    if not records:
        return 0

    # Deduplicate by timestamp and walk backward from latest.
    df = pd.DataFrame(records, columns=["dt", "stage"])
    df = df.dropna(subset=["dt"]).sort_values("dt").drop_duplicates("dt", keep="last")
    count = 0
    for stg in reversed(df["stage"].tolist()):
        if stg == target:
            count += 1
        else:
            break
    return count


def _consecutive_public_stage_runs(history_df: pd.DataFrame, prev_combined: Optional[pd.DataFrame], ticker: str, target_stage: str) -> int:
    target = str(target_stage or "").strip()
    if not target:
        return 0
    records: List[Tuple[pd.Timestamp, str]] = []
    hist = _ticker_history(history_df, ticker)
    if not hist.empty:
        for _, r in hist.iterrows():
            dt = pd.to_datetime(r.get("snapshot_date"), errors="coerce")
            stg = str(r.get("stage", "") or "").strip()
            if pd.notna(dt) and stg:
                records.append((pd.Timestamp(dt).normalize(), stg))
    if prev_combined is not None and not prev_combined.empty and "ticker" in prev_combined.columns and "stage" in prev_combined.columns:
        prev = prev_combined.copy()
        prev["ticker"] = prev["ticker"].astype(str).str.upper().str.strip()
        row = prev[prev["ticker"].eq(str(ticker or "").upper().strip())]
        if not row.empty:
            records.append((pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize() - pd.Timedelta(hours=1), str(row.iloc[-1].get("stage", "") or "").strip()))
    if not records:
        return 0
    df = pd.DataFrame(records, columns=["dt", "stage"])
    df = df.dropna(subset=["dt"]).sort_values("dt").drop_duplicates("dt", keep="last")
    count = 0
    for stg in reversed(df["stage"].tolist()):
        if stg == target:
            count += 1
        else:
            break
    return count


def _seen_public_stage_since_last(history_df: pd.DataFrame, ticker: str, last_stage: str, required_stage: str) -> bool:
    hist = _ticker_history(history_df, ticker)
    if hist.empty or "stage" not in hist.columns:
        return False
    stages = hist["stage"].astype(str).tolist()
    last_idx = None
    for i, stg in enumerate(stages):
        if stg == last_stage:
            last_idx = i
    if last_idx is None:
        return False
    return any(stg == required_stage for stg in stages[last_idx + 1:])


def _stage_number(stage: str) -> Optional[int]:
    text = str(stage or "").strip()
    if text == "Stage 1":
        return 1
    if text == "Stage 2":
        return 2
    if text == "Stage 3":
        return 3
    if text == "Stage 4":
        return 4
    return None


def _set_pending_stage(out: pd.DataFrame, idx: int, row: pd.Series, raw_stage: str, reason: str) -> None:
    out.at[idx, "stage"] = "Not Sure"
    out.at[idx, "stage_variant"] = f"Pending {raw_stage}" if raw_stage else "Pending Confirmation"
    out.at[idx, "stage_confidence"] = min(float(pd.to_numeric(row.get("stage_confidence"), errors="coerce") or 0.0), 0.55)
    out.at[idx, "stage_reason"] = reason
    out.at[idx, "stage_state_reason"] = reason
    out.at[idx, "notes"] = _append_note(row.get("notes", ""), reason)


def apply_stage_state_memory(
    current_df: pd.DataFrame,
    out_path: Path,
    prev_combined: Optional[pd.DataFrame],
    config: dict,
) -> pd.DataFrame:
    """Apply public stage-state memory after raw chart classification.

    This is the trust layer. Raw chart classification remains in `stage_raw`; public `stage`
    is smoothed using a simple state machine:
    - no unidentified fallback: uncertainty is Not Sure;
    - no direct Stage 4 -> Stage 2 public jump;
    - Stage 2 entry needs repeated confirmation runs;
    - failed Stage 2 remains visible before a new base/repair stage is accepted.
    """
    if current_df is None or current_df.empty or "ticker" not in current_df.columns or "stage" not in current_df.columns:
        return current_df

    out = current_df.copy()
    out["stage_raw"] = out["stage"].astype(str)
    for col in ["stage_state_reason", "stage_failed_since", "last_stage2_date", "stage_pending_raw"]:
        if col not in out.columns:
            out[col] = ""

    history_df = _read_existing_stage_history(out_path)
    prior_stage = _latest_stage_map_from_history(history_df)
    # If stage_action_history is absent/stale, yesterday's combined file is still useful.
    prior_stage.update({k: v for k, v in _previous_stage_map_from_combined(prev_combined).items() if k not in prior_stage})
    last_stage2 = _last_stage2_date_map(history_df)

    if prev_combined is not None and not prev_combined.empty and "ticker" in prev_combined.columns and "stage" in prev_combined.columns:
        yesterday = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None) - pd.Timedelta(days=1)
        for _, prev_row in prev_combined.iterrows():
            ticker = str(prev_row.get("ticker", "")).upper().strip()
            if ticker and str(prev_row.get("stage", "")) == "Stage 2" and ticker not in last_stage2:
                last_stage2[ticker] = yesterday

    today = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
    failed_hold_days = int(config.get("stage2_failed_hold_days", 21) or 21)
    transition_confirm_days = int(config.get("stage_transition_confirm_days", 3) or 3)
    stage2_confirm_days = int(config.get("stage2_entry_confirm_days", transition_confirm_days) or transition_confirm_days)
    stage4_to_stage2_min_stage1_days = int(config.get("stage4_to_stage2_min_stage1_days", 3) or 3)
    enforce_no_jumps = bool(config.get("enforce_no_stage_jumps", True))

    for idx, row in out.iterrows():
        ticker = str(row.get("ticker", "")).upper().strip()
        raw_stage = str(row.get("stage_raw", "Not Sure") or "Not Sure").strip()
        if raw_stage in {"Unknown", "", "nan", "None"}:
            raw_stage = "Not Sure"
            out.at[idx, "stage_raw"] = "Not Sure"
        prev_stage = str(prior_stage.get(ticker, "") or "").strip()
        last_s2 = last_stage2.get(ticker)
        days_since_s2 = None
        if pd.notna(last_s2):
            try:
                days_since_s2 = int((today - pd.Timestamp(last_s2).normalize()).days)
                out.at[idx, "last_stage2_date"] = pd.Timestamp(last_s2).date().isoformat()
            except Exception:
                days_since_s2 = None

        raw_confirm_runs = _consecutive_raw_stage_runs(history_df, prev_combined, ticker, raw_stage) + (1 if raw_stage not in {"Not Sure"} else 0)
        public_stage1_runs = _consecutive_public_stage_runs(history_df, prev_combined, ticker, "Stage 1")
        recently_stage2 = days_since_s2 is not None and days_since_s2 <= failed_hold_days

        # 1) Explicit failed Stage 2 mechanism. This takes priority over normal transitions.
        broke_from_stage2 = (
            raw_stage != "Stage 2"
            and (prev_stage == "Stage 2" or (prev_stage == "Stage 2 Failed" and recently_stage2) or recently_stage2)
        )
        if broke_from_stage2:
            out.at[idx, "stage"] = "Stage 2 Failed"
            out.at[idx, "stage_variant"] = "Failed Stage 2"
            out.at[idx, "stage_confidence"] = 0.72
            if not str(row.get("stage_failed_since", "") or "").strip():
                out.at[idx, "stage_failed_since"] = today.date().isoformat()
            out.at[idx, "stage_reason"] = (
                "This stock was recently Stage 2, but the latest structure no longer satisfies Stage 2 rules. "
                "It is kept as Stage 2 Failed until a fresh base/repair structure forms."
            )
            out.at[idx, "stage_state_reason"] = "Recent Stage 2 break; public stage held as Stage 2 Failed."
            out.at[idx, "notes"] = _append_note(row.get("notes", ""), "Stage 2 failed state applied from stage memory.")
            continue

        # 2) While in failed Stage 2, do not instantly relabel unless the repair/advance is confirmed.
        if prev_stage == "Stage 2 Failed":
            if raw_stage == "Stage 2" and raw_confirm_runs >= stage2_confirm_days:
                out.at[idx, "stage_state_reason"] = f"Stage 2 reclaimed after {raw_confirm_runs} confirmation runs."
                continue
            if recently_stage2:
                out.at[idx, "stage"] = "Stage 2 Failed"
                out.at[idx, "stage_variant"] = "Failed Stage 2"
                out.at[idx, "stage_confidence"] = 0.70
                out.at[idx, "stage_reason"] = "Failed Stage 2 cooling period is still active; waiting for a fresh base or confirmed reclaim."
                out.at[idx, "stage_state_reason"] = "Failed Stage 2 hold period active."
                out.at[idx, "notes"] = _append_note(row.get("notes", ""), "Failed Stage 2 hold period active.")
                continue
            if raw_stage in {"Stage 1", "Stage 3", "Stage 4"} and raw_confirm_runs < transition_confirm_days:
                _set_pending_stage(out, idx, row, raw_stage, f"Stage 2 Failed -> {raw_stage} needs {transition_confirm_days} confirmation runs; current count {raw_confirm_runs}.")
                out.at[idx, "stage_pending_raw"] = raw_stage
                continue

        # 3) No direct Stage 4 -> Stage 2. Must show Stage 1/base repair first.
        if prev_stage == "Stage 4" and raw_stage == "Stage 2":
            seen_stage1_since_last_stage4 = _seen_public_stage_since_last(history_df, ticker, "Stage 4", "Stage 1")
            if (not seen_stage1_since_last_stage4) or public_stage1_runs < stage4_to_stage2_min_stage1_days:
                reason = (
                    "Blocked Stage 4 -> Stage 2 jump. Public Stage 2 needs Stage 1/base repair first "
                    f"and at least {stage4_to_stage2_min_stage1_days} Stage 1 confirmation runs."
                )
                _set_pending_stage(out, idx, row, raw_stage, reason)
                out.at[idx, "stage_pending_raw"] = raw_stage
                continue

        # 4) Stage 2 entry/promotion requires repeated raw confirmation.
        if raw_stage == "Stage 2" and prev_stage != "Stage 2":
            if raw_confirm_runs < stage2_confirm_days:
                reason = f"Stage 2 pending confirmation: needs {stage2_confirm_days} raw Stage 2 runs; current count {raw_confirm_runs}."
                _set_pending_stage(out, idx, row, raw_stage, reason)
                out.at[idx, "stage_pending_raw"] = raw_stage
                continue
            out.at[idx, "stage_state_reason"] = f"Stage 2 confirmed with {raw_confirm_runs} raw confirmation runs."

        # 5) General no-stage-jump rule for public stage changes.
        prev_num = _stage_number(prev_stage)
        raw_num = _stage_number(raw_stage)
        if enforce_no_jumps and prev_num is not None and raw_num is not None and prev_stage != raw_stage:
            allowed = False
            # Normal adjacent transitions plus the cycle reset Stage 4 -> Stage 1.
            if abs(raw_num - prev_num) == 1:
                allowed = True
            if prev_stage == "Stage 4" and raw_stage == "Stage 1":
                allowed = raw_confirm_runs >= transition_confirm_days
            if prev_stage == "Stage 1" and raw_stage == "Stage 2":
                allowed = raw_confirm_runs >= stage2_confirm_days
            if not allowed:
                reason = f"Blocked public stage jump {prev_stage} -> {raw_stage}; waiting for intermediate/confirmed structure."
                _set_pending_stage(out, idx, row, raw_stage, reason)
                out.at[idx, "stage_pending_raw"] = raw_stage
                continue

        if raw_stage == "Not Sure":
            out.at[idx, "stage"] = "Not Sure"
            out.at[idx, "stage_variant"] = "Not Sure"
            out.at[idx, "stage_confidence"] = 0.0
            out.at[idx, "stage_reason"] = "Stage rules did not identify a reliable structure."
            out.at[idx, "stage_state_reason"] = "No confident stage classification."

    return out


def _ensure_rank_column_for_snapshot(df: pd.DataFrame, score_col: str = "final_combined_score") -> pd.DataFrame:
    """Ensure stable dataset rank: 1 = strongest row by score."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "current_rank" in out.columns:
        out["current_rank"] = pd.to_numeric(out["current_rank"], errors="coerce")
    else:
        if score_col in out.columns:
            out = out.sort_values([score_col], ascending=[False], na_position="last").reset_index(drop=True)
        out["current_rank"] = np.arange(1, len(out) + 1)
    return out


def _interesting_priority(row: pd.Series) -> float:
    """Prioritize top-ranked names where structure is near breakout / strong setup."""
    priority = 0.0
    stage = str(row.get("stage", ""))
    combined_bucket = str(row.get("combined_bucket", ""))
    daily_bucket = str(row.get("daily_setup_bucket", ""))
    weekly_bucket = str(row.get("weekly_setup_bucket", ""))

    priority += {"Stage 2": 30, "Stage 1": 12, "Stage 2 Failed": 5, "Stage 3": 6, "Stage 4": 0, "Not Sure": 1}.get(stage, 1)
    priority += {
        "high_conviction_breakout": 70,
        "high_conviction_near_pivot": 62,
        "building_setup": 30,
        "watchlist": 8,
    }.get(combined_bucket, 0)
    priority += {
        "breakout_today": 54,
        "near_pivot": 46,
        "building_setup": 25,
        "watchlist": 4,
    }.get(daily_bucket, 0)
    priority += {
        "weekly_breakout": 42,
        "weekly_near_pivot": 36,
        "weekly_watchlist": 4,
    }.get(weekly_bucket, 0)

    for col, points in [("volume_is_drying_up", 9), ("weekly_volume_is_drying_up", 7)]:
        val = row.get(col, False)
        if isinstance(val, str):
            val = val.strip().lower() in {"true", "1", "yes", "y"}
        if bool(val):
            priority += points

    for col, points in [("daily_breakout_distance_pct", 14), ("weekly_breakout_distance_pct", 10)]:
        dist = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(dist):
            if -5.0 <= float(dist) <= 1.5:
                priority += points
            elif 1.5 < float(dist) <= 4.0:
                priority += points * 0.45

    score = pd.to_numeric(row.get("final_combined_score", row.get("combined_score")), errors="coerce")
    if pd.notna(score):
        priority += min(float(score), 100.0) * 0.20

    current_rank = pd.to_numeric(row.get("current_rank"), errors="coerce")
    if pd.notna(current_rank):
        priority += max(0.0, 30.0 - min(float(current_rank), 30.0)) * 0.25

    return round(float(priority), 4)


def build_interesting20_snapshot(combined_df: pd.DataFrame, limit: int = 20, top_pool: int = 30) -> pd.DataFrame:
    """Build the public Interesting 20 list from the top-ranked dataset pool."""
    if combined_df is None or combined_df.empty:
        return pd.DataFrame()
    out = _ensure_rank_column_for_snapshot(combined_df, "final_combined_score")
    out["current_rank"] = pd.to_numeric(out["current_rank"], errors="coerce")
    pool = out[out["current_rank"].le(top_pool)].copy()
    if pool.empty:
        pool = out.sort_values("current_rank", ascending=True, na_position="last").head(top_pool).copy()
    pool["interesting_priority"] = pool.apply(_interesting_priority, axis=1)
    sort_cols = ["interesting_priority", "current_rank", "final_combined_score"]
    ascending = [False, True, False]
    pool = pool.sort_values(sort_cols, ascending=ascending, na_position="last").head(limit).copy()
    pool["snapshot_date"] = pd.Timestamp.now(tz="Asia/Kolkata").date().isoformat()
    keep = [c for c in [
        "snapshot_date", "ticker", "Company Name", "Industry", "stage", "stage_raw", "stage_variant", "stage_classification", "stage_display", "stage_confidence", "stage_reason", "stage_state_reason", "stage_failed_since", "last_stage2_date", "stage_pending_raw", "current_rank",
        "interesting_priority", "daily_setup_bucket", "weekly_setup_bucket", "combined_bucket",
        "final_combined_score", "daily_breakout_distance_pct", "weekly_breakout_distance_pct",
        "rs_3m_pct", "rs_6m_pct", "volume_dryup_ratio", "breakout_volume_ratio", "weekly_volume_ratio",
        "volume_is_drying_up", "weekly_volume_is_drying_up", "notes",
    ] if c in pool.columns]
    return pool[keep].reset_index(drop=True)


def save_interesting20_archive(out_path: Path, snapshot_df: pd.DataFrame, chart_paths: Dict[str, str]) -> Dict[str, str]:
    """Persist today's Interesting 20 list and copy the matching daily/weekly chart images."""
    paths: Dict[str, str] = {}
    if snapshot_df is None or snapshot_df.empty:
        return paths

    today = str(snapshot_df["snapshot_date"].iloc[0]) if "snapshot_date" in snapshot_df.columns else pd.Timestamp.now(tz="Asia/Kolkata").date().isoformat()
    archive_root = out_path / "interesting20_archive"
    dated_root = archive_root / today
    dated_daily = dated_root / "charts" / "daily"
    dated_weekly = dated_root / "charts" / "weekly"
    archive_root.mkdir(parents=True, exist_ok=True)
    dated_daily.mkdir(parents=True, exist_ok=True)
    dated_weekly.mkdir(parents=True, exist_ok=True)

    latest_file = out_path / "interesting20_latest.csv"
    dated_file = archive_root / f"{today}_interesting20.csv"
    snapshot_df.to_csv(latest_file, index=False)
    snapshot_df.to_csv(dated_file, index=False)
    paths["interesting20_latest"] = str(latest_file)
    paths["interesting20_archive_csv"] = str(dated_file)

    daily_dir = Path(chart_paths.get("daily_charts_dir", out_path / "charts" / "daily"))
    weekly_dir = Path(chart_paths.get("weekly_charts_dir", out_path / "charts" / "weekly"))
    copied = 0
    for ticker in snapshot_df.get("ticker", pd.Series(dtype=str)).dropna().astype(str):
        safe = sanitize_filename(ticker)
        for src_dir, dst_dir, suffix in [(daily_dir, dated_daily, "_daily.png"), (weekly_dir, dated_weekly, "_weekly.png")]:
            src = src_dir / f"{safe}{suffix}"
            if src.exists():
                shutil.copy2(src, dst_dir / src.name)
                copied += 1
    paths["interesting20_archive_charts_dir"] = str(dated_root / "charts")
    paths["interesting20_archive_chart_count"] = str(copied)
    return paths




def apply_mobile_chart_readability(fig) -> None:
    """Force readable chart text for mobile PNG rendering."""
    try:
        fig.set_size_inches(*CHART_FIGSIZE_DAILY, forward=True)
    except Exception:
        pass
    for ax in getattr(fig, "axes", []):
        try:
            ax.tick_params(axis="both", labelsize=CHART_TICK_FONTSIZE)
            ax.xaxis.label.set_size(CHART_AXIS_FONTSIZE)
            ax.yaxis.label.set_size(CHART_AXIS_FONTSIZE)
            ax.title.set_size(CHART_TITLE_FONTSIZE)
            legend = ax.get_legend()
            if legend:
                for item in legend.get_texts():
                    item.set_fontsize(CHART_LEGEND_FONTSIZE)
        except Exception:
            pass

def build_outputs(
    universe_path: str,
    outdir: str,
    config: Optional[dict] = None,
    export_all_ticker_charts: bool = True,
    wide_price: Optional[str] = None,
    chart_scope: str = "all",
    skip_existing_charts: bool = False,
    chart_dpi: int = 150,
) -> Dict[str, str]:
    # Charts are always regenerated. This keeps Watchlist and Charts-tab search fresh.
    skip_existing_charts = False
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.perf_counter()
    universe_df = load_nifty500_universe(universe_path)
    universe_used_file = out_path / "universe_used.csv"
    atomic_to_csv(universe_df, universe_used_file)
    tickers = universe_df["Ticker"].astype(str).str.upper().tolist()
    print(f"Universe used: {len(tickers):,} included EQ stocks")

    if wide_price:
        price_data = load_wide_price_data(wide_price, tickers, cfg["market_index"], max_rows=int(cfg.get("max_price_rows", 620) or 0) or None)
    else:
        price_data = None

    report, regime, price_data = build_vcp_universe_report(tickers, cfg, price_data=price_data)
    if report.empty:
        raise RuntimeError("No screener results produced.")

    final_report = report.merge(universe_df, left_on="ticker", right_on="Ticker", how="left")
    industry_df = build_industry_strength_table(final_report)
    final_report = apply_industry_boost(final_report, industry_df, cfg)

    prev_combined_for_stage_memory = _safe_read_csv(out_path / "vcp_combined_ranked.csv")
    final_report = apply_stage_state_memory(final_report, out_path, prev_combined_for_stage_memory, cfg)
    final_report = ensure_stage_classification_columns(final_report)

    metadata_cols = ["sector", "industry_group", "is_fo_stock", "fo_category", "Include"]
    common_cols = ["ticker", "Company Name", "Industry"] + metadata_cols + ["stage", "stage_raw", "stage_variant", "stage_classification", "stage_display", "stage_confidence", "stage_reason", "stage_state_reason", "stage_failed_since", "last_stage2_date", "stage_pending_raw", "rs_3m_pct", "rs_6m_pct", "avg_turnover_inr", "volume_dryup_ratio", "breakout_volume_ratio", "weekly_volume_ratio", "volume_is_drying_up", "weekly_volume_is_drying_up", "notes"]
    daily_cols = common_cols + ["daily_setup_bucket", "daily_score", "final_daily_score", "daily_pivot", "daily_breakout_distance_pct", "daily_contraction_depths_pct", "daily_contraction_durations", "daily_contraction_score", "daily_base_duration_days"]
    weekly_cols = common_cols + ["weekly_setup_bucket", "weekly_score", "final_weekly_score", "weekly_pivot", "weekly_breakout_distance_pct", "weekly_contraction_depths_pct", "weekly_contraction_durations", "weekly_contraction_score", "weekly_base_duration_weeks", "weekly_vcp_quality"]
    combined_cols = common_cols + ["daily_setup_bucket", "weekly_setup_bucket", "combined_bucket", "daily_score", "weekly_score", "combined_score", "industry_boost", "final_combined_score"]

    daily_df = final_report[[c for c in daily_cols if c in final_report.columns]].sort_values(["final_daily_score", "daily_score"], ascending=[False, False]).reset_index(drop=True)
    weekly_df = final_report[[c for c in weekly_cols if c in final_report.columns]].sort_values(["final_weekly_score", "weekly_score"], ascending=[False, False]).reset_index(drop=True)
    combined_df = final_report[[c for c in combined_cols if c in final_report.columns]].sort_values(["final_combined_score", "combined_score"], ascending=[False, False]).reset_index(drop=True)

    daily_df["current_rank"] = np.arange(1, len(daily_df) + 1)
    weekly_df["current_rank"] = np.arange(1, len(weekly_df) + 1)
    combined_df["current_rank"] = np.arange(1, len(combined_df) + 1)
    daily_df = ensure_stage_classification_columns(daily_df)
    weekly_df = ensure_stage_classification_columns(weekly_df)
    combined_df = ensure_stage_classification_columns(combined_df)
    industry_df = industry_df.copy().reset_index(drop=True)
    industry_df["current_rank"] = np.arange(1, len(industry_df) + 1)

    prev_combined = _safe_read_csv(out_path / "vcp_combined_ranked.csv")
    prev_industry = _safe_read_csv(out_path / "industry_strength.csv")

    stock_changes = build_stock_changes(combined_df, prev_combined)
    industry_changes = build_industry_changes(industry_df, prev_industry)
    top_movers = stock_changes.sort_values(["new_top_10", "new_top_20", "new_daily_breakout", "new_weekly_breakout", "rank_change", "combined_score_change"], ascending=[False, False, False, False, False, False]).reset_index(drop=True)

    benchmark_hist_df = price_data.get(cfg["market_index"])
    price_moves = build_price_moves(combined_df, price_data)
    history_file = update_stage_action_history(out_path, combined_df, price_data, benchmark_hist_df, universe_df, cfg)

    daily_file = out_path / "vcp_daily_ranked.csv"
    weekly_file = out_path / "vcp_weekly_ranked.csv"
    combined_file = out_path / "vcp_combined_ranked.csv"
    industry_file = out_path / "industry_strength.csv"
    regime_file = out_path / "market_regime.csv"
    stock_changes_file = out_path / "stock_changes.csv"
    industry_changes_file = out_path / "industry_changes.csv"
    top_movers_file = out_path / "top_movers.csv"
    price_moves_file = out_path / "stock_price_moves.csv"

    atomic_to_csv(daily_df, daily_file)
    atomic_to_csv(weekly_df, weekly_file)
    atomic_to_csv(combined_df, combined_file)
    atomic_to_csv(industry_df, industry_file)
    atomic_to_csv(pd.DataFrame([asdict(regime)]), regime_file)
    atomic_to_csv(ensure_stage_classification_columns(stock_changes), stock_changes_file)
    atomic_to_csv(industry_changes, industry_changes_file)
    atomic_to_csv(ensure_stage_classification_columns(top_movers), top_movers_file)
    atomic_to_csv(ensure_stage_classification_columns(price_moves), price_moves_file)

    
    if not export_all_ticker_charts or chart_scope == "none":
        chart_paths = {"daily_charts_dir": str(out_path / "charts" / "daily"), "weekly_charts_dir": str(out_path / "charts" / "weekly")}
    elif chart_scope == "all":
        chart_paths = export_selected_charts(final_report, price_data, out_path, tickers_to_export=None, skip_existing=skip_existing_charts, dpi=chart_dpi)
    else:
        chart_tickers = build_dashboard_chart_tickers(combined_df, price_moves, stock_changes, out_path)
        chart_paths = export_selected_charts(final_report, price_data, out_path, tickers_to_export=chart_tickers, skip_existing=skip_existing_charts, dpi=chart_dpi)
    interesting_snapshot = build_interesting20_snapshot(combined_df, limit=20, top_pool=30)
    interesting_paths = save_interesting20_archive(out_path, interesting_snapshot, chart_paths)
    print(f"Total engine runtime: {time.perf_counter()-overall_t0:.2f}s")
    return {"daily": str(daily_file), "weekly": str(weekly_file), "combined": str(combined_file), "industry": str(industry_file), "regime": str(regime_file), "stock_changes": str(stock_changes_file), "industry_changes": str(industry_changes_file), "top_movers": str(top_movers_file), "price_moves": str(price_moves_file), "history": str(history_file), "universe_used": str(universe_used_file), **chart_paths, **interesting_paths}

def _perf_from_close(close: pd.Series, bars_back: int) -> float:
    s = close.dropna()
    if len(s) <= bars_back:
        return np.nan
    prev = float(s.iloc[-(bars_back + 1)])
    curr = float(s.iloc[-1])
    if prev == 0:
        return np.nan
    return round((curr / prev - 1) * 100, 2)

def _perf_ytd(close: pd.Series) -> float:
    s = close.dropna()
    if s.empty:
        return np.nan
    current_year = int(s.index[-1].year)
    year_slice = s[s.index.year == current_year]
    if year_slice.empty:
        return np.nan
    first_close = float(year_slice.iloc[0])
    last_close = float(year_slice.iloc[-1])
    if first_close == 0:
        return np.nan
    return round((last_close / first_close - 1) * 100, 2)

def build_price_moves(current_df: pd.DataFrame, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = _clean_stock_snapshot(current_df).copy()
    if base.empty:
        return pd.DataFrame()
    rows = []
    for _, row in base.iterrows():
        ticker = row.get("ticker")
        df = price_data.get(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = df["Close"].dropna()
        if close.empty:
            continue
        rows.append({
            "ticker": ticker,
            "Company Name": row.get("Company Name"),
            "Industry": row.get("Industry"),
            "sector": row.get("sector"),
            "is_fo_stock": row.get("is_fo_stock"),
            "fo_category": row.get("fo_category"),
            "stage": row.get("stage"),
            "overall_setup_label": row.get("combined_bucket"),
            "final_combined_score": row.get("final_combined_score"),
            "change_1d_pct": _perf_from_close(close, 1),
            "change_1w_pct": _perf_from_close(close, 5),
            "change_1m_pct": _perf_from_close(close, 21),
            "change_ytd_pct": _perf_ytd(close),
            "last_close": round(float(close.iloc[-1]), 2),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for c in ["change_1d_pct", "change_1w_pct", "change_1m_pct", "change_ytd_pct", "final_combined_score", "last_close"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_values(["change_1d_pct", "final_combined_score"], ascending=[False, False]).reset_index(drop=True)


def derive_public_action(stage: str, combined_bucket: str, score: float) -> str:
    if stage == "Stage 2 Failed":
        return "Failed Stage 2"
    if stage == "Not Sure":
        return "Not Sure"
    if stage == "Stage 2":
        if combined_bucket in {"high_conviction_breakout", "high_conviction_near_pivot"} and score >= 70:
            return "Strong Structure"
        return "Advancing"
    if stage == "Stage 1":
        return "Base Building"
    if stage == "Stage 3":
        return "Transition"
    if stage == "Stage 4":
        return "Weak Structure"
    return "Mixed"

def derive_super_action(stage: str, combined_bucket: str, score: float) -> str:
    if stage == "Stage 2 Failed":
        return "Avoid / Wait for new base"
    if stage == "Not Sure":
        return "No Action"
    if stage == "Stage 2":
        if combined_bucket == "high_conviction_breakout" and score >= 72:
            return "Buy"
        if combined_bucket in {"high_conviction_near_pivot", "building_setup"} and score >= 62:
            return "Watch / Add on confirmation"
        return "Hold / Trend intact"
    if stage == "Stage 1":
        return "Watchlist / Early base"
    if stage == "Stage 3":
        return "Reduce / Avoid fresh longs"
    if stage == "Stage 4":
        return "Exit / Avoid"
    return "No Action"

def build_stage_action_history_snapshot(snapshot_df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    if snapshot_df is None or snapshot_df.empty:
        return pd.DataFrame()
    out = ensure_stage_classification_columns(snapshot_df)
    score_col = "final_combined_score" if "final_combined_score" in out.columns else "combined_score"
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out["snapshot_date"] = pd.Timestamp(snapshot_date).normalize()
    out["public_action"] = out.apply(lambda r: derive_public_action(str(r.get("stage", "")), str(r.get("combined_bucket", "")), float(pd.to_numeric(r.get(score_col), errors="coerce") if pd.notna(pd.to_numeric(r.get(score_col), errors="coerce")) else 0.0)), axis=1)
    out["super_action"] = out.apply(lambda r: derive_super_action(str(r.get("stage", "")), str(r.get("combined_bucket", "")), float(pd.to_numeric(r.get(score_col), errors="coerce") if pd.notna(pd.to_numeric(r.get(score_col), errors="coerce")) else 0.0)), axis=1)
    keep_cols = [c for c in [
        "snapshot_date", "ticker", "Company Name", "Industry", "sector", "is_fo_stock", "fo_category", "stage", "stage_raw", "stage_variant", "stage_classification", "stage_display", "stage_confidence", "stage_reason", "stage_state_reason", "stage_failed_since", "last_stage2_date", "stage_pending_raw", "combined_bucket", score_col,
        "volume_dryup_ratio", "breakout_volume_ratio", "weekly_volume_ratio", "volume_is_drying_up", "weekly_volume_is_drying_up",
        "public_action", "super_action"
    ] if c in out.columns]
    history = out[keep_cols].copy()
    if score_col in history.columns and score_col != "final_combined_score":
        history = history.rename(columns={score_col: "final_combined_score"})
    return history

def build_six_month_history(price_data: Dict[str, pd.DataFrame], benchmark_df: pd.DataFrame, universe_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    lookback = int(config.get("history_init_lookback_trading_days", 126))
    history_rows = []
    tickers = universe_df["Ticker"].tolist()
    benchmark_close = benchmark_df["Close"].dropna().astype(float)

    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or df.empty or len(df) < max(config.get("min_history", 300), lookback + 260):
            continue
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
        if len(df) < 260:
            continue
        snapshot_dates = df.index[-lookback:]
        company = universe_df.loc[universe_df["Ticker"] == ticker, "Company Name"].iloc[0]
        industry = universe_df.loc[universe_df["Ticker"] == ticker, "Industry"].iloc[0]

        for snap_date in snapshot_dates:
            trunc = df.loc[:snap_date].copy()
            bench_trunc = benchmark_df.loc[:snap_date].copy()
            if len(trunc) < 260 or len(bench_trunc) < 260:
                continue
            try:
                regime = market_regime(bench_trunc, config["market_index"], config["market_ma_fast"], config["market_ma_slow"], price_data=None, universe_tickers=None)
                result = analyze_symbol(ticker, trunc, bench_trunc, regime, config)
                if not result:
                    continue
                row = asdict(result)
                row["stage_classification"] = row.get("stage_variant", row.get("stage", "Not Sure"))
                row["stage_display"] = row.get("stage", "Not Sure") if row.get("stage_classification") == row.get("stage") else f"{row.get('stage', 'Not Sure')} • {row.get('stage_classification', 'Not Sure')}"
                row["snapshot_date"] = pd.Timestamp(snap_date).normalize()
                row["Company Name"] = company
                row["Industry"] = industry
                row["public_action"] = derive_public_action(row.get("stage", ""), row.get("combined_bucket", ""), float(row.get("combined_score", 0) or 0))
                row["super_action"] = derive_super_action(row.get("stage", ""), row.get("combined_bucket", ""), float(row.get("combined_score", 0) or 0))
                history_rows.append({k: row.get(k) for k in [
                    "snapshot_date", "ticker", "Company Name", "Industry", "stage", "stage_variant", "stage_classification", "stage_display", "stage_confidence", "stage_reason", "combined_bucket", "combined_score",
                    "volume_dryup_ratio", "breakout_volume_ratio", "weekly_volume_ratio", "volume_is_drying_up", "weekly_volume_is_drying_up", "public_action", "super_action"
                ]})
            except Exception:
                continue

    if not history_rows:
        return pd.DataFrame()
    history = pd.DataFrame(history_rows).rename(columns={"combined_score": "final_combined_score"})
    history = history.sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)
    return history

def update_stage_action_history(out_path: Path, current_snapshot: pd.DataFrame, price_data: Dict[str, pd.DataFrame], benchmark_df: pd.DataFrame, universe_df: pd.DataFrame, config: dict) -> Path:
    history_file = out_path / str(config.get("history_file_name", "stage_action_history.csv"))
    today = pd.Timestamp.now("UTC").normalize().tz_localize(None)
    current_history = build_stage_action_history_snapshot(current_snapshot, today)

    existing = _safe_read_csv(history_file, parse_dates=["snapshot_date"])
    if existing is None:
        existing = build_six_month_history(price_data, benchmark_df, universe_df, config) if bool(config.get("history_init_enabled", True)) else pd.DataFrame()

    if not current_history.empty:
        existing = pd.concat([existing, current_history], ignore_index=True) if not existing.empty else current_history

    if existing.empty:
        atomic_to_csv(existing, history_file)
        return history_file

    existing["snapshot_date"] = pd.to_datetime(existing["snapshot_date"], utc=True).dt.tz_convert(None).dt.normalize()
    existing = existing.drop_duplicates(subset=["snapshot_date", "ticker"], keep="last")
    existing = existing.sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)
    atomic_to_csv(existing, history_file)
    return history_file



def write_engine_run_metadata(out_path: Path) -> None:
    """Persist actual engine run time for dashboard display.

    Dashboard should not infer freshness from current time. It should read this file.
    """
    metadata = {
        "engine_ran_at_ist": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "engine_ran_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    try:
        out_path.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out_path / "engine_run_metadata.json", json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"Warning: could not write engine_run_metadata.json: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily + Weekly VCP Change-Tracking Screener with local Yahoo wide-file support")
    parser.add_argument("--universe", required=True, help="Path to universe CSV. New schema supports Include and f&o columns.")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--wide-price", default=None, help="Folder containing wide_open.csv/wide_high.csv/wide_low.csv/wide_close.csv/wide_volume.csv, or yahoo_price_data_wide.xlsx")
    parser.add_argument("--max-price-rows", type=int, default=620, help="Read only the latest N rows from the wide price files. Use 0 to load all rows.")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation for fast testing.")
    parser.add_argument("--chart-scope", choices=["dashboard", "all", "none"], default="all", help="all = generate charts for every included stock; dashboard = dashboard-needed charts only; none = no charts. Default: all.")
    parser.add_argument("--skip-existing-charts", action="store_true", help="Deprecated and ignored. Charts are regenerated every run so Watchlist and Charts search stay current.")
    parser.add_argument("--chart-dpi", type=int, default=150, help="PNG DPI for dashboard charts. 130-160 is usually enough for Streamlit.")
    parser.add_argument("--init-history", action="store_true", help="Backfill historical stage_action_history. Slow; off by default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = {
        "max_price_rows": None if args.max_price_rows == 0 else args.max_price_rows,
        "history_init_enabled": bool(args.init_history),
    }
    outputs = build_outputs(
        args.universe,
        args.outdir,
        config=cfg,
        export_all_ticker_charts=not args.no_charts,
        wide_price=args.wide_price,
        chart_scope="none" if args.no_charts else args.chart_scope,
        skip_existing_charts=False,
        chart_dpi=int(args.chart_dpi),
    )
    write_engine_run_metadata(Path(args.outdir))
    print("Saved files:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
