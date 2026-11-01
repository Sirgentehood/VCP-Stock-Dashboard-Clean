from __future__ import annotations
import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass, asdict, field
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
    "stage4_to_stage2_min_stage1_days": 20,
    "enforce_no_stage_jumps": True,
}

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

    # Clean public hero-card fields. These are meant for public dashboards and
    # avoid recommendation-style or defect-heavy language.
    public_stage_label: str = ""
    volume_pattern_display: str = ""
    nifty_3m_outperformance_pct: float = np.nan
    nifty_3m_outperformance_label: str = ""

    # Public-safe structure fields. These are safe to expose in dashboards because
    # they describe technical structure, not buy/sell/hold advice.
    structure_score: float = 0.0
    trend_template_pass: bool = False
    setup_quality_label: str = "Not Rated"
    relative_strength_label: str = "Unclear"
    volume_pattern_label: str = "Unclear"
    technical_flags: str = ""
    risk_pct: float = np.nan

    # Internal-only setup fields. Keep these out of public UI views if you are
    # launching an educational/non-advisory product.
    internal_leader_score: float = 0.0
    internal_trend_score: float = 0.0
    internal_setup_score: float = 0.0
    internal_risk_score: float = 0.0
    internal_quality_score: float = 0.0
    internal_is_true_leader: bool = False
    internal_is_proper_setup: bool = False
    internal_is_low_risk: bool = False
    internal_is_buyable_setup: bool = False
    internal_failure_reasons: str = ""


@dataclass
class StockFeatures:
    """Single evidence object for both public stage and internal setup analysis.

    The important design rule is that this class stores facts, not labels.
    Stage classification and setup/actionability scoring read from this object.
    """
    ticker: str
    close_series: pd.Series = field(repr=False)
    high_series: pd.Series = field(repr=False)
    low_series: pd.Series = field(repr=False)
    volume_series: pd.Series = field(repr=False)
    weekly_df: pd.DataFrame = field(repr=False)

    close: float = np.nan
    high_52w: float = np.nan
    low_52w: float = np.nan
    dist_from_52w_high_pct: float = np.nan
    advance_from_52w_low_pct: float = np.nan

    ma50: float = np.nan
    ma150: float = np.nan
    ma200: float = np.nan
    price_above_ma50: bool = False
    price_above_ma150: bool = False
    price_above_ma200: bool = False
    ma_stack_bullish: bool = False
    ma_stack_bearish: bool = False
    ma50_slope_pct: float = np.nan
    ma150_slope_pct: float = np.nan
    ma200_slope_pct: float = np.nan
    ma200_trend_1m_pct: float = np.nan

    weekly_close: float = np.nan
    weekly_ma10: float = np.nan
    weekly_ma30: float = np.nan
    price_above_weekly_ma10: bool = False
    price_above_weekly_ma30: bool = False
    weekly_ma10_slope_pct: float = np.nan
    weekly_ma30_slope_pct: float = np.nan
    weekly_ma30_trend_10w_pct: float = np.nan

    ret_4w_pct: float = np.nan
    ret_8w_pct: float = np.nan
    ret_13w_pct: float = np.nan
    ret_26w_pct: float = np.nan
    rs_3m_pct: float = np.nan
    rs_6m_pct: float = np.nan
    rs_line_last: float = np.nan
    rs_line_10w_ma: float = np.nan
    rs_line_30w_ma: float = np.nan
    rs_line_13w_slope_pct: float = np.nan
    rs_line_26w_high: bool = False
    rs_line_52w_high: bool = False
    rs_line_above_10w: bool = False
    rs_line_above_30w: bool = False

    avg_turnover_inr: float = np.nan
    median_turnover_inr: float = np.nan
    liquidity_ok: bool = False
    volume_dryup_ratio: float = np.nan
    weekly_volume_dryup_ratio: float = np.nan
    breakout_volume_ratio: float = np.nan
    weekly_volume_ratio: float = np.nan
    distribution_weeks_12: int = 0
    accumulation_weeks_12: int = 0

    daily_depths: List[float] = field(default_factory=list)
    daily_durations: List[int] = field(default_factory=list)
    daily_contraction_score: float = 0.0
    daily_base_duration: float = 0.0
    weekly_depths: List[float] = field(default_factory=list)
    weekly_durations: List[int] = field(default_factory=list)
    weekly_contraction_score: float = 0.0
    weekly_base_duration: float = 0.0
    weekly_quality: str = "weak"

    daily_pivot: float = np.nan
    daily_breakout_distance_pct: float = np.nan
    weekly_pivot: float = np.nan
    weekly_breakout_distance_pct: float = np.nan
    recent_range_pct: float = np.nan
    tight_range_ok: bool = False
    weekly_range_12w_pct: float = np.nan
    weekly_range_20w_pct: float = np.nan
    recent_low_6w: float = np.nan
    no_recent_breakdown: bool = False
    logical_stop_price: float = np.nan
    risk_pct: float = np.nan

    upper_circuit_like_days_20: int = 0
    lower_circuit_like_days_20: int = 0
    abnormal_gap_days_60: int = 0
    zero_volume_days_60: int = 0
    corporate_action_suspected: bool = False
    illiquidity_risk: bool = False


@dataclass
class StageResult:
    stage: str
    variant: str
    confidence: float
    reason: str


@dataclass
class SetupQualityResult:
    leader_score: float
    trend_score: float
    setup_score: float
    risk_score: float
    quality_score: float
    setup_quality_label: str
    relative_strength_label: str
    volume_pattern_label: str
    trend_template_pass: bool
    is_true_leader: bool
    is_proper_setup: bool
    is_low_risk: bool
    is_buyable_setup_internal: bool
    public_flags: List[str] = field(default_factory=list)
    internal_flags: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)


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
    """Backward-compatible stage wrapper used by breadth/history code.

    The public stage engine is now weekly-first. This wrapper keeps older call
    sites working while routing the decision through the newer weekly structure
    classifier when only a close series is available.
    """
    return classify_stage_from_close_only(close).stage


def determine_stage_details(close: pd.Series, ma50: float, ma150: float, ma200: float) -> Tuple[str, str, float, str]:
    """Backward-compatible details wrapper for close-only callers."""
    result = classify_stage_from_close_only(close)
    return result.stage, result.variant, result.confidence, result.reason


def vcp_quality_label(score: float, base_bars: float, depths: List[float], min_base_bars: int) -> str:
    if len(depths) < 2 or base_bars < min_base_bars:
        return "weak"
    return "strong" if score >= 0.66 else ("moderate" if score >= 0.5 else "weak")


def _finite(value) -> bool:
    try:
        return pd.notna(value) and np.isfinite(float(value))
    except Exception:
        return False


def _safe_float(value, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def pct_change_over_bars(series: pd.Series, bars: int) -> float:
    s = series.dropna().astype(float)
    if len(s) <= bars:
        return np.nan
    prev = float(s.iloc[-bars - 1])
    curr = float(s.iloc[-1])
    if prev == 0:
        return np.nan
    return float((curr / prev - 1) * 100)


def classify_stage_from_close_only(close: pd.Series) -> StageResult:
    """Weekly-first stage classification when only a close series is available.

    Used by legacy breadth/history callers. The full per-stock analysis uses
    `classify_public_stage(features)` because it has volume, RS and VCP evidence.
    """
    c = close.dropna().astype(float)
    if len(c) < 260:
        return StageResult("Not Sure", "Unclear", 0.0, "Not enough price history for reliable weekly stage classification.")

    weekly = c.resample("W-FRI").last().dropna()
    if len(weekly) < 60:
        return StageResult("Not Sure", "Unclear", 0.0, "Not enough weekly history for reliable stage classification.")

    last = float(c.iloc[-1])
    ma50 = float(c.rolling(50).mean().iloc[-1])
    ma150 = float(c.rolling(150).mean().iloc[-1])
    ma200 = float(c.rolling(200).mean().iloc[-1])
    ma50_slope = pct_change_over_bars(c.rolling(50).mean(), 21)
    ma200_slope = pct_change_over_bars(c.rolling(200).mean(), 21)

    weekly_ma10 = weekly.rolling(10).mean()
    weekly_ma30 = weekly.rolling(30).mean()
    w10 = _safe_float(weekly_ma10.iloc[-1])
    w30 = _safe_float(weekly_ma30.iloc[-1])
    w30_trend = pct_change_over_bars(weekly_ma30, 10)

    high_52w = float(c.iloc[-252:].max())
    low_52w = float(c.iloc[-252:].min())
    dist_high = (last / high_52w - 1) * 100 if high_52w > 0 else np.nan
    advance_low = (last / low_52w - 1) * 100 if low_52w > 0 else np.nan
    ret_13w = pct_return(c, 63)
    ret_26w = pct_return(c, 126)

    above_w30 = _finite(w30) and last > w30
    below_w30 = _finite(w30) and last < w30
    above_200 = _finite(ma200) and last > ma200
    below_200 = _finite(ma200) and last < ma200
    above_150 = _finite(ma150) and last > ma150
    above_50 = _finite(ma50) and last > ma50
    below_50 = _finite(ma50) and last < ma50

    stage4 = (
        below_w30 and below_200
        and _finite(w30_trend) and w30_trend <= -2.0
        and _finite(ma200_slope) and ma200_slope <= -1.0
        and _finite(dist_high) and dist_high <= -25
        and (_finite(ret_13w) and ret_13w <= -5)
    )
    if stage4:
        return StageResult("Stage 4", "Stage 4", 0.82, "Price is below the 30-week average and below long-term structure.")

    prior_strength = (
        (_finite(advance_low) and advance_low >= 60)
        or (_finite(ret_26w) and ret_26w >= 20)
    ) and (_finite(dist_high) and dist_high >= -35)
    stage3 = (
        prior_strength
        and (below_50 or not above_w30)
        and _finite(w30_trend) and w30_trend <= 1.0
        and ((_finite(ret_13w) and ret_13w <= 0) or (_finite(ma50_slope) and ma50_slope < 0))
    )
    if stage3:
        return StageResult("Stage 3", "Early", 0.68, "Prior strength has shifted into an early Stage 3 transition around the weekly trend area.")

    stage2 = (
        above_w30 and above_150 and above_200
        and _finite(w30_trend) and w30_trend >= -0.5
        and _finite(dist_high) and dist_high >= -25
        and _finite(advance_low) and advance_low >= 25
        and ((_finite(ret_13w) and ret_13w >= 0) or above_50)
    )
    if stage2:
        extended = (
            (_finite(ma50) and last >= ma50 * 1.18)
            or (_finite(ma200) and last >= ma200 * 1.45)
            or (_finite(advance_low) and advance_low >= 120 and _finite(ret_13w) and ret_13w >= 25)
        )
        if extended:
            return StageResult("Stage 2", "Extended", 0.74, "Weekly advancing structure remains intact, but price is stretched versus key averages.")
        return StageResult("Stage 2", "Advancing", 0.82, "Price is above long-term structure and the 30-week trend is constructive.")

    near_w30 = _finite(w30) and 0.88 * w30 <= last <= 1.12 * w30
    near_200 = _finite(ma200) and 0.88 * ma200 <= last <= 1.12 * ma200
    stage1 = (
        (near_w30 or near_200 or above_200)
        and (not _finite(w30_trend) or w30_trend >= -2.5)
        and not (_finite(ret_13w) and ret_13w <= -8 and below_200)
    )
    if stage1:
        variant = "Early Turn" if above_w30 and above_50 and _finite(ret_13w) and ret_13w > 0 else "Base Building"
        return StageResult("Stage 1", "Stage 1", 0.64, "Price is building a base around long-term weekly structure without a confirmed advancing phase.")

    if below_w30 and (_finite(w30_trend) and w30_trend < -1.0):
        return StageResult("Stage 4", "Stage 4", 0.64, "Price is below the 30-week trend.")
    return StageResult("Not Sure", "Unclear", 0.0, "Weekly structure does not meet reliable Stage 1/2/3/4 rules.")


def _compute_rs_line_features(close: pd.Series, benchmark_close: pd.Series) -> Dict[str, object]:
    aligned = pd.concat([close.rename("stock"), benchmark_close.rename("benchmark")], axis=1).dropna()
    if len(aligned) < 160:
        return {
            "rs_line_last": np.nan,
            "rs_line_10w_ma": np.nan,
            "rs_line_30w_ma": np.nan,
            "rs_line_13w_slope_pct": np.nan,
            "rs_line_26w_high": False,
            "rs_line_52w_high": False,
            "rs_line_above_10w": False,
            "rs_line_above_30w": False,
        }
    rs = aligned["stock"].astype(float) / aligned["benchmark"].astype(float)
    rs = rs.replace([np.inf, -np.inf], np.nan).dropna()
    if rs.empty:
        return {
            "rs_line_last": np.nan,
            "rs_line_10w_ma": np.nan,
            "rs_line_30w_ma": np.nan,
            "rs_line_13w_slope_pct": np.nan,
            "rs_line_26w_high": False,
            "rs_line_52w_high": False,
            "rs_line_above_10w": False,
            "rs_line_above_30w": False,
        }
    rs_w = rs.resample("W-FRI").last().dropna()
    rs10 = rs_w.rolling(10).mean()
    rs30 = rs_w.rolling(30).mean()
    last = _safe_float(rs_w.iloc[-1])
    ma10 = _safe_float(rs10.iloc[-1])
    ma30 = _safe_float(rs30.iloc[-1])
    return {
        "rs_line_last": last,
        "rs_line_10w_ma": ma10,
        "rs_line_30w_ma": ma30,
        "rs_line_13w_slope_pct": pct_change_over_bars(rs_w, 13),
        "rs_line_26w_high": bool(len(rs_w) >= 26 and _finite(last) and last >= float(rs_w.iloc[-26:].max()) * 0.995),
        "rs_line_52w_high": bool(len(rs_w) >= 52 and _finite(last) and last >= float(rs_w.iloc[-52:].max()) * 0.995),
        "rs_line_above_10w": bool(_finite(last) and _finite(ma10) and last > ma10),
        "rs_line_above_30w": bool(_finite(last) and _finite(ma30) and last > ma30),
    }


def _count_weekly_accumulation_distribution(weekly_df: pd.DataFrame, window: int = 12) -> Tuple[int, int]:
    if weekly_df is None or weekly_df.empty or len(weekly_df) < window + 10:
        return 0, 0
    close = weekly_df["Close"].astype(float)
    volume = weekly_df["Volume"].astype(float)
    ret = close.pct_change() * 100
    vol_ma = volume.rolling(10).mean()
    recent = pd.DataFrame({"ret": ret, "volume": volume, "vol_ma": vol_ma}).dropna().tail(window)
    if recent.empty:
        return 0, 0
    distribution = int(((recent["ret"] <= -2.0) & (recent["volume"] > recent["vol_ma"] * 1.05)).sum())
    accumulation = int(((recent["ret"] >= 2.0) & (recent["volume"] > recent["vol_ma"] * 1.05)).sum())
    return accumulation, distribution


def _count_indian_market_trap_flags(df: pd.DataFrame) -> Dict[str, object]:
    out = {
        "upper_circuit_like_days_20": 0,
        "lower_circuit_like_days_20": 0,
        "abnormal_gap_days_60": 0,
        "zero_volume_days_60": 0,
        "corporate_action_suspected": False,
    }
    if df is None or df.empty or not {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
        return out
    d = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    if len(d) < 2:
        return out
    close = d["Close"].astype(float)
    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    open_ = d["Open"].astype(float)
    volume = d["Volume"].astype(float)
    ret = close.pct_change() * 100
    close_near_high = close >= high * 0.995
    close_near_low = close <= low * 1.005
    out["upper_circuit_like_days_20"] = int(((ret.tail(20) >= 4.8) & close_near_high.tail(20)).sum())
    out["lower_circuit_like_days_20"] = int(((ret.tail(20) <= -4.8) & close_near_low.tail(20)).sum())
    prev_close = close.shift(1)
    gap = (open_ / prev_close - 1) * 100
    out["abnormal_gap_days_60"] = int((gap.abs().tail(60) >= 25).sum())
    out["zero_volume_days_60"] = int((volume.tail(60) <= 0).sum())
    out["corporate_action_suspected"] = bool((ret.abs().tail(90) >= 35).any() or out["abnormal_gap_days_60"] > 0)
    return out


def compute_stock_features(
    ticker: str,
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    config: dict,
) -> Optional[StockFeatures]:
    """Compute the single evidence layer used by stage + setup engines."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return None

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy().sort_index()
    if len(df) < int(config.get("min_history", 260)):
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
    ma50_series = close.rolling(50).mean()
    ma150_series = close.rolling(150).mean()
    ma200_series = close.rolling(200).mean()
    ma50 = _safe_float(ma50_series.iloc[-1])
    ma150 = _safe_float(ma150_series.iloc[-1])
    ma200 = _safe_float(ma200_series.iloc[-1])

    weekly_ma10_series = weekly_close.rolling(10).mean()
    weekly_ma30_series = weekly_close.rolling(30).mean()
    weekly_ma10 = _safe_float(weekly_ma10_series.iloc[-1])
    weekly_ma30 = _safe_float(weekly_ma30_series.iloc[-1])
    weekly_close_now = float(weekly_close.iloc[-1])

    high_52w = float(close.iloc[-252:].max())
    low_52w = float(close.iloc[-252:].min())

    stock_3m = pct_return(close, 63)
    stock_6m = pct_return(close, 126)
    bm_close = benchmark_df["Close"].dropna().astype(float)
    bm_3m = pct_return(bm_close, 63)
    bm_6m = pct_return(bm_close, 126)
    rs_3m = stock_3m - bm_3m if pd.notna(stock_3m) and pd.notna(bm_3m) else np.nan
    rs_6m = stock_6m - bm_6m if pd.notna(stock_6m) and pd.notna(bm_6m) else np.nan
    rs_features = _compute_rs_line_features(close, bm_close)

    daily_window = df.iloc[-140:]
    daily_depths, daily_durations, daily_base_duration = detect_vcp_contractions(
        daily_window["High"], daily_window["Low"], daily_window["Close"],
        config["swing_order_daily"], config["max_contractions"],
        config["min_contraction_days_daily"], config["min_contraction_depth_pct_daily"],
    )
    daily_contraction_score_val = contraction_score(daily_depths)

    weekly_window = weekly_df.iloc[-52:]
    weekly_depths, weekly_durations, weekly_base_duration = detect_vcp_contractions(
        weekly_window["High"], weekly_window["Low"], weekly_window["Close"],
        config["swing_order_weekly"], config["max_contractions"],
        config["min_contraction_days_weekly"], config["min_contraction_depth_pct_weekly"],
    )
    weekly_contraction_score_val = contraction_score(weekly_depths)
    weekly_quality = vcp_quality_label(
        weekly_contraction_score_val, weekly_base_duration, weekly_depths, config["min_base_duration_weeks"]
    )

    daily_pivot = compute_pivot(high, config["pivot_lookback_daily"], daily_base_duration)
    daily_breakout_distance = (close_now / daily_pivot - 1) * 100 if _finite(daily_pivot) and daily_pivot > 0 else np.nan
    weekly_pivot = compute_pivot(weekly_high, config["pivot_lookback_weekly"], weekly_base_duration)
    weekly_breakout_distance = (weekly_close_now / weekly_pivot - 1) * 100 if _finite(weekly_pivot) and weekly_pivot > 0 else np.nan

    recent_range_pct = (
        (close.iloc[-config["recent_range_days"]:].max() - close.iloc[-config["recent_range_days"]:].min()) /
        close.iloc[-config["recent_range_days"]:].max() * 100
    ) if len(close) >= config["recent_range_days"] and close.iloc[-config["recent_range_days"]:].max() > 0 else np.nan
    tight_range_ok = _finite(recent_range_pct) and recent_range_pct <= config["recent_range_max_pct"]

    weekly_range_12w = ((weekly_close.iloc[-12:].max() / weekly_close.iloc[-12:].min()) - 1) * 100 if len(weekly_close) >= 12 and weekly_close.iloc[-12:].min() > 0 else np.nan
    weekly_range_20w = ((weekly_close.iloc[-20:].max() / weekly_close.iloc[-20:].min()) - 1) * 100 if len(weekly_close) >= 20 and weekly_close.iloc[-20:].min() > 0 else np.nan
    recent_low_6w = float(low.iloc[-30:].min()) if len(low) >= 30 else np.nan
    no_recent_breakdown = _finite(recent_low_6w) and close_now >= recent_low_6w * 1.03

    volume_dryup_ratio_val = volume_ratio(volume, config["volume_short_window"], config["volume_long_window"])
    weekly_volume_dryup_ratio = volume_ratio(weekly_df["Volume"].astype(float), 4, 12) if len(weekly_df) >= 12 else np.nan
    breakout_volume_ratio_val = recent_breakout_volume_ratio(volume, 30)
    weekly_volume_ratio_val = current_week_volume_ratio(volume, weekly_df["Volume"].astype(float), current_days=5, weekly_window=10)
    avg_turnover_val = avg_turnover(close, volume, 20)
    median_turnover_val = float((close.iloc[-20:] * volume.iloc[-20:]).median()) if len(close) >= 20 and len(volume) >= 20 else np.nan
    min_turnover = float(config.get("min_avg_turnover_inr", 5e7) or 0)
    liquidity_ok = bool(
        _finite(avg_turnover_val) and avg_turnover_val >= min_turnover
        and (_finite(median_turnover_val) and median_turnover_val >= min_turnover * 0.50)
    )

    accumulation_weeks, distribution_weeks = _count_weekly_accumulation_distribution(weekly_df, 12)
    trap_flags = _count_indian_market_trap_flags(df)

    recent_low_3w = float(low.iloc[-15:].min()) if len(low) >= 15 else np.nan
    ma50_stop = ma50 * 0.97 if _finite(ma50) else np.nan
    stop_candidates = [x for x in [recent_low_3w, ma50_stop] if _finite(x) and x < close_now]
    logical_stop = max(stop_candidates) if stop_candidates else recent_low_3w
    risk_pct = (daily_pivot - logical_stop) / daily_pivot * 100 if _finite(daily_pivot) and _finite(logical_stop) and daily_pivot > 0 and logical_stop < daily_pivot else np.nan

    illiquidity_risk = bool(
        not liquidity_ok
        or int(trap_flags.get("zero_volume_days_60", 0)) > 0
    )

    return StockFeatures(
        ticker=ticker,
        close_series=close,
        high_series=high,
        low_series=low,
        volume_series=volume,
        weekly_df=weekly_df,
        close=close_now,
        high_52w=high_52w,
        low_52w=low_52w,
        dist_from_52w_high_pct=(close_now / high_52w - 1) * 100 if high_52w > 0 else np.nan,
        advance_from_52w_low_pct=(close_now / low_52w - 1) * 100 if low_52w > 0 else np.nan,
        ma50=ma50,
        ma150=ma150,
        ma200=ma200,
        price_above_ma50=bool(_finite(ma50) and close_now > ma50),
        price_above_ma150=bool(_finite(ma150) and close_now > ma150),
        price_above_ma200=bool(_finite(ma200) and close_now > ma200),
        ma_stack_bullish=bool(_finite(ma50) and _finite(ma150) and _finite(ma200) and close_now > ma50 > ma150 > ma200),
        ma_stack_bearish=bool(_finite(ma50) and _finite(ma150) and _finite(ma200) and close_now < ma50 < ma150 < ma200),
        ma50_slope_pct=slope_pct(ma50_series, 20),
        ma150_slope_pct=slope_pct(ma150_series, 20),
        ma200_slope_pct=slope_pct(ma200_series, 20),
        ma200_trend_1m_pct=pct_change_over_bars(ma200_series, 21),
        weekly_close=weekly_close_now,
        weekly_ma10=weekly_ma10,
        weekly_ma30=weekly_ma30,
        price_above_weekly_ma10=bool(_finite(weekly_ma10) and weekly_close_now > weekly_ma10),
        price_above_weekly_ma30=bool(_finite(weekly_ma30) and weekly_close_now > weekly_ma30),
        weekly_ma10_slope_pct=slope_pct(weekly_ma10_series, 6),
        weekly_ma30_slope_pct=slope_pct(weekly_ma30_series, 6),
        weekly_ma30_trend_10w_pct=pct_change_over_bars(weekly_ma30_series, 10),
        ret_4w_pct=pct_return(close, 21),
        ret_8w_pct=pct_return(close, 42),
        ret_13w_pct=pct_return(close, 63),
        ret_26w_pct=pct_return(close, 126),
        rs_3m_pct=rs_3m,
        rs_6m_pct=rs_6m,
        rs_line_last=_safe_float(rs_features["rs_line_last"]),
        rs_line_10w_ma=_safe_float(rs_features["rs_line_10w_ma"]),
        rs_line_30w_ma=_safe_float(rs_features["rs_line_30w_ma"]),
        rs_line_13w_slope_pct=_safe_float(rs_features["rs_line_13w_slope_pct"]),
        rs_line_26w_high=bool(rs_features["rs_line_26w_high"]),
        rs_line_52w_high=bool(rs_features["rs_line_52w_high"]),
        rs_line_above_10w=bool(rs_features["rs_line_above_10w"]),
        rs_line_above_30w=bool(rs_features["rs_line_above_30w"]),
        avg_turnover_inr=avg_turnover_val,
        median_turnover_inr=median_turnover_val,
        liquidity_ok=liquidity_ok,
        volume_dryup_ratio=volume_dryup_ratio_val,
        weekly_volume_dryup_ratio=weekly_volume_dryup_ratio,
        breakout_volume_ratio=breakout_volume_ratio_val,
        weekly_volume_ratio=weekly_volume_ratio_val,
        distribution_weeks_12=distribution_weeks,
        accumulation_weeks_12=accumulation_weeks,
        daily_depths=daily_depths,
        daily_durations=daily_durations,
        daily_contraction_score=daily_contraction_score_val,
        daily_base_duration=daily_base_duration,
        weekly_depths=weekly_depths,
        weekly_durations=weekly_durations,
        weekly_contraction_score=weekly_contraction_score_val,
        weekly_base_duration=weekly_base_duration,
        weekly_quality=weekly_quality,
        daily_pivot=daily_pivot,
        daily_breakout_distance_pct=daily_breakout_distance,
        weekly_pivot=weekly_pivot,
        weekly_breakout_distance_pct=weekly_breakout_distance,
        recent_range_pct=recent_range_pct,
        tight_range_ok=tight_range_ok,
        weekly_range_12w_pct=weekly_range_12w,
        weekly_range_20w_pct=weekly_range_20w,
        recent_low_6w=recent_low_6w,
        no_recent_breakdown=no_recent_breakdown,
        logical_stop_price=logical_stop,
        risk_pct=risk_pct,
        upper_circuit_like_days_20=int(trap_flags["upper_circuit_like_days_20"]),
        lower_circuit_like_days_20=int(trap_flags["lower_circuit_like_days_20"]),
        abnormal_gap_days_60=int(trap_flags["abnormal_gap_days_60"]),
        zero_volume_days_60=int(trap_flags["zero_volume_days_60"]),
        corporate_action_suspected=bool(trap_flags["corporate_action_suspected"]),
        illiquidity_risk=illiquidity_risk,
    )


def classify_public_stage(features: StockFeatures, config: Optional[dict] = None) -> StageResult:
    """Weekly-first public stage classifier.

    The public stage answers only: which broad structure is this stock in?
    It intentionally does not decide whether the setup is buyable/actionable.
    """
    f = features
    rs_constructive = (
        (_finite(f.rs_3m_pct) and f.rs_3m_pct >= 0)
        or (_finite(f.rs_6m_pct) and f.rs_6m_pct >= 0)
        or f.rs_line_above_30w
        or f.rs_line_26w_high
    )
    rs_weakening = (
        (_finite(f.rs_3m_pct) and f.rs_3m_pct <= -4)
        or (_finite(f.rs_line_13w_slope_pct) and f.rs_line_13w_slope_pct <= -3)
        or (not f.rs_line_above_30w and _finite(f.rs_line_30w_ma))
    )
    w30_rising_or_flat = _finite(f.weekly_ma30_trend_10w_pct) and f.weekly_ma30_trend_10w_pct >= -0.75
    w30_rising = _finite(f.weekly_ma30_trend_10w_pct) and f.weekly_ma30_trend_10w_pct >= 0.75
    w30_falling = _finite(f.weekly_ma30_trend_10w_pct) and f.weekly_ma30_trend_10w_pct <= -2.0
    ma200_not_falling_hard = (not _finite(f.ma200_trend_1m_pct)) or f.ma200_trend_1m_pct >= -1.5

    # Confirmed downtrend gets first priority because it is the clearest stage.
    stage4_confirmed = (
        not f.price_above_weekly_ma30
        and not f.price_above_ma200
        and (w30_falling or (_finite(f.ma200_trend_1m_pct) and f.ma200_trend_1m_pct <= -1.0))
        and _finite(f.dist_from_52w_high_pct) and f.dist_from_52w_high_pct <= -25
        and ((_finite(f.ret_13w_pct) and f.ret_13w_pct <= -5) or rs_weakening)
    )
    if stage4_confirmed:
        return StageResult(
            "Stage 4", "Stage 4", 0.86,
            "Price is below the 30-week trend and below long-term support structure."
        )

    # Stage 3 is former strength that is now showing damage. This prevents weak
    # daily pullbacks from being over-labelled as distribution.
    prior_strength = (
        _finite(f.dist_from_52w_high_pct) and f.dist_from_52w_high_pct >= -35
        and (
            (_finite(f.advance_from_52w_low_pct) and f.advance_from_52w_low_pct >= 60)
            or (_finite(f.ret_26w_pct) and f.ret_26w_pct >= 18)
            or (_finite(f.rs_6m_pct) and f.rs_6m_pct >= 5)
            or f.rs_line_26w_high
        )
    )
    distribution_damage_score = 0
    if not f.price_above_ma50 or not f.price_above_weekly_ma10:
        distribution_damage_score += 1
    if rs_weakening:
        distribution_damage_score += 1
    if f.distribution_weeks_12 >= 2 and f.distribution_weeks_12 > f.accumulation_weeks_12:
        distribution_damage_score += 1
    if _finite(f.ret_8w_pct) and f.ret_8w_pct <= 0:
        distribution_damage_score += 1
    if _finite(f.weekly_ma30_trend_10w_pct) and f.weekly_ma30_trend_10w_pct <= 1.0:
        distribution_damage_score += 1

    if prior_strength and distribution_damage_score >= 3 and not stage4_confirmed:
        variant = "Late" if f.distribution_weeks_12 >= 2 else "Early"
        return StageResult(
            "Stage 3", variant, 0.76 if variant == "Late" else 0.68,
            "Prior strength has shifted into a Stage 3 transition based on trend, relative strength, and volume evidence."
        )

    # Advancing phase. Stage 2 must be weekly-supported; daily trend template is
    # used for quality later, not as the sole stage source.
    stage2_core = (
        f.price_above_weekly_ma30
        and f.price_above_ma150
        and f.price_above_ma200
        and w30_rising_or_flat
        and ma200_not_falling_hard
        and _finite(f.dist_from_52w_high_pct) and f.dist_from_52w_high_pct >= -25
        and _finite(f.advance_from_52w_low_pct) and f.advance_from_52w_low_pct >= 25
        and (rs_constructive or (_finite(f.ret_13w_pct) and f.ret_13w_pct >= 5))
    )
    if stage2_core:
        extended = (
            (_finite(f.ma50) and f.close >= f.ma50 * 1.18)
            or (_finite(f.weekly_ma10) and f.weekly_close >= f.weekly_ma10 * 1.12)
            or (_finite(f.ma200) and f.close >= f.ma200 * 1.45)
            or (_finite(f.advance_from_52w_low_pct) and f.advance_from_52w_low_pct >= 120 and _finite(f.ret_13w_pct) and f.ret_13w_pct >= 25)
        )
        if extended:
            return StageResult(
                "Stage 2", "Extended", 0.78,
                "Weekly advancing structure remains intact, but price is stretched versus key moving averages."
            )
        confidence = 0.90 if (f.ma_stack_bullish and w30_rising and rs_constructive) else 0.80
        return StageResult(
            "Stage 2", "Advancing", confidence,
            "Price is above constructive weekly and long-term trend structure with improving relative strength."
        )

    # Base/repair. This should be a broad structural bucket, not an action label.
    near_weekly_ma30 = _finite(f.weekly_ma30) and 0.88 * f.weekly_ma30 <= f.close <= 1.12 * f.weekly_ma30
    near_ma200 = _finite(f.ma200) and 0.88 * f.ma200 <= f.close <= 1.12 * f.ma200
    stage1_base = (
        (near_weekly_ma30 or near_ma200 or f.price_above_ma200)
        and not stage4_confirmed
        and (not _finite(f.weekly_ma30_trend_10w_pct) or f.weekly_ma30_trend_10w_pct >= -2.5)
        and not (_finite(f.ret_13w_pct) and f.ret_13w_pct <= -8 and not f.price_above_ma200)
    )
    if stage1_base:
        early_turn = (
            f.price_above_weekly_ma30
            and f.price_above_ma50
            and ((_finite(f.ret_13w_pct) and f.ret_13w_pct > 0) or f.rs_line_above_10w)
            and not rs_weakening
        )
        return StageResult(
            "Stage 1", "Stage 1", 0.68 if early_turn else 0.64,
            "Price is building a base around long-term weekly structure without a confirmed advancing phase."
        )

    if (not f.price_above_weekly_ma30) and (w30_falling or not f.price_above_ma200):
        return StageResult("Stage 4", "Stage 4", 0.66, "Price is below weekly structure.")
    return StageResult("Not Sure", "Unclear", 0.0, "The weekly structure does not meet reliable stage criteria.")


def public_stage_label(stage: str, variant: str = "") -> str:
    """Clean public stage label for dashboard cards.

    The canonical `stage` field remains useful for filters. This label is the
    front-end friendly display value.
    """
    stg = str(stage or "").strip()
    var = str(variant or "").strip()
    if stg in {"Failed Stage 2", "Stage 2 Failed"}:
        return "Failed Stage 2"
    if stg == "Stage 1":
        return "Stage 1"
    if stg == "Stage 2":
        return f"Stage 2 - {var}" if var in {"Advancing", "Extended"} else "Stage 2"
    if stg == "Stage 3":
        return f"Stage 3 {var}" if var in {"Early", "Late"} else "Stage 3"
    if stg == "Stage 4":
        return "Stage 4"
    if stg == "Not Sure":
        return "Not Sure - Unclear"
    return stg or "Not Sure - Unclear"


def _nifty_3m_outperformance_label(rs_3m_pct: float) -> str:
    if not _finite(rs_3m_pct):
        return "3M Nifty comparison unavailable"
    value = round(float(abs(rs_3m_pct)), 2)
    if rs_3m_pct >= 0:
        return f"Outperformed Nifty by {value}% in 3 Months"
    return f"Underperformed Nifty by {value}% in 3 Months"


def _volume_pattern_display(features: StockFeatures) -> str:
    """Neutral public volume wording with the actual ratio.

    Prefer weekly current-volume ratio because the public card should read like:
    "Volume Drying Up (0.40x vs 10W Avg)".
    """
    ratio = features.weekly_volume_ratio
    basis = "10W Avg"
    if not _finite(ratio):
        ratio = features.weekly_volume_dryup_ratio
        basis = "12W Avg"
    if not _finite(ratio):
        ratio = features.volume_dryup_ratio
        basis = "50D Avg"
    if not _finite(ratio):
        return "Volume comparison unavailable"

    ratio_text = f"{float(ratio):.2f}x vs {basis}"
    if ratio <= 0.85:
        return f"Volume Drying Up ({ratio_text})"
    if ratio >= 1.50:
        return f"Volume Expanding ({ratio_text})"
    return f"Volume Normal ({ratio_text})"


def _bounded_score(value: float, low: float = 0.0, high: float = 100.0) -> float:
    if not _finite(value):
        return 0.0
    return float(max(low, min(high, value)))


def _relative_strength_label(score: float) -> str:
    if score >= 80:
        return "Very Strong"
    if score >= 60:
        return "Strong"
    if score >= 40:
        return "Neutral"
    return "Weak"


def _setup_quality_label(score: float) -> str:
    if score >= 78:
        return "High"
    if score >= 58:
        return "Medium"
    if score >= 38:
        return "Low"
    return "Not Rated"


def _volume_pattern_label(features: StockFeatures) -> str:
    # Public-neutral volume state. Selling/distribution diagnostics stay internal.
    if _finite(features.weekly_volume_ratio):
        if features.weekly_volume_ratio <= 0.85:
            return "Volume Drying Up"
        if features.weekly_volume_ratio >= 1.50:
            return "Volume Expanding"
    if _finite(features.volume_dryup_ratio) and features.volume_dryup_ratio <= 0.85:
        return "Volume Drying Up"
    if _finite(features.breakout_volume_ratio) and features.breakout_volume_ratio >= 1.8:
        return "Volume Expanding"
    return "Volume Normal"


def evaluate_internal_setup_quality(
    features: StockFeatures,
    stage_result: StageResult,
    regime: Optional[MarketRegime],
    config: dict,
) -> SetupQualityResult:
    """Internal Minervini-style setup-quality engine.

    It deliberately returns internal booleans such as `is_buyable_setup_internal`,
    but public UIs should expose only setup_quality_label, structure score and flags.
    """
    f = features
    internal_flags: List[str] = []
    public_flags: List[str] = []
    failure_reasons: List[str] = []

    # Trend template / leadership scores.
    trend_conditions = [
        f.price_above_ma50,
        f.price_above_ma150,
        f.price_above_ma200,
        f.ma_stack_bullish,
        _finite(f.ma200_trend_1m_pct) and f.ma200_trend_1m_pct >= -0.5,
        _finite(f.dist_from_52w_high_pct) and f.dist_from_52w_high_pct >= -25,
        _finite(f.advance_from_52w_low_pct) and f.advance_from_52w_low_pct >= 30,
        f.price_above_weekly_ma30,
    ]
    trend_score = round(sum(bool(x) for x in trend_conditions) / len(trend_conditions) * 100, 2)
    trend_template_pass = bool(trend_score >= 75 and stage_result.stage == "Stage 2")

    leader_points = 0.0
    if _finite(f.rs_3m_pct):
        leader_points += max(0.0, min(20.0, 10.0 + f.rs_3m_pct * 0.9))
    if _finite(f.rs_6m_pct):
        leader_points += max(0.0, min(20.0, 10.0 + f.rs_6m_pct * 0.7))
    if f.rs_line_above_10w:
        leader_points += 15
    if f.rs_line_above_30w:
        leader_points += 15
    if f.rs_line_26w_high:
        leader_points += 15
    if f.rs_line_52w_high:
        leader_points += 15
    if _finite(f.rs_line_13w_slope_pct) and f.rs_line_13w_slope_pct > 0:
        leader_points += 10
    leader_score = round(_bounded_score(leader_points), 2)

    # Proper setup score: VCP/base quality, pivot proximity, volume dry-up.
    setup_score = 0.0
    if len(f.daily_depths) >= 2:
        setup_score += min(20.0, f.daily_contraction_score * 20.0)
    if len(f.weekly_depths) >= 2:
        setup_score += min(18.0, f.weekly_contraction_score * 18.0)
    if f.daily_base_duration >= config["min_base_duration_days"]:
        setup_score += 10
    if f.weekly_base_duration >= config["min_base_duration_weeks"]:
        setup_score += 10
    if _finite(f.daily_breakout_distance_pct) and -5.0 <= f.daily_breakout_distance_pct <= 1.5:
        setup_score += 14
    elif _finite(f.daily_breakout_distance_pct) and 1.5 < f.daily_breakout_distance_pct <= 5.0:
        setup_score += 5
        internal_flags.append("slightly_extended_from_pivot")
    if _finite(f.volume_dryup_ratio) and f.volume_dryup_ratio <= 0.85:
        setup_score += 12
    elif _finite(f.volume_dryup_ratio) and f.volume_dryup_ratio > 1.05:
        internal_flags.append("no_volume_dryup")
    if f.tight_range_ok:
        setup_score += 8
    if f.weekly_quality == "strong":
        setup_score += 8
    elif f.weekly_quality == "moderate":
        setup_score += 4

    # Fault detection. These reduce setup quality but do not necessarily change stage.
    if f.daily_depths and max(f.daily_depths) > 30:
        setup_score -= 12
        internal_flags.append("base_too_deep_daily")
        public_flags.append("Wide Base")
    if f.weekly_depths and max(f.weekly_depths) > 35:
        setup_score -= 14
        internal_flags.append("base_too_deep_weekly")
        public_flags.append("Wide Weekly Base")
    if _finite(f.weekly_range_12w_pct) and f.weekly_range_12w_pct > 28:
        setup_score -= 8
        internal_flags.append("weekly_structure_loose")
        public_flags.append("Loose Structure")
    if stage_result.stage == "Stage 2" and stage_result.variant == "Extended":
        setup_score -= 10
        internal_flags.append("extended_stage2")
        public_flags.append("Extended")
    if f.distribution_weeks_12 >= 2 and f.distribution_weeks_12 > f.accumulation_weeks_12:
        setup_score -= 14
        internal_flags.append("distribution_volume")
    setup_score = round(_bounded_score(setup_score), 2)

    # Low-risk score based on distance from pivot to logical stop. This is internal
    # setup quality; the public UI may show only a generic "wide risk" flag.
    if _finite(f.risk_pct):
        if f.risk_pct <= 5:
            risk_score = 100.0
        elif f.risk_pct <= 8:
            risk_score = 82.0
        elif f.risk_pct <= 12:
            risk_score = 58.0
            public_flags.append("Wider Risk")
        elif f.risk_pct <= 15:
            risk_score = 35.0
            public_flags.append("Wide Risk")
        else:
            risk_score = 15.0
            public_flags.append("Wide Risk")
            internal_flags.append("risk_too_wide")
    else:
        risk_score = 35.0
        internal_flags.append("risk_not_measurable")

    # Indian-market technical-trap flags.
    if f.upper_circuit_like_days_20 >= 3:
        public_flags.append("Circuit Risk")
        internal_flags.append("upper_circuit_risk")
    if f.lower_circuit_like_days_20 >= 2:
        public_flags.append("Circuit Risk")
        internal_flags.append("lower_circuit_risk")
    if f.corporate_action_suspected:
        public_flags.append("Corporate Action Check")
        internal_flags.append("corporate_action_suspected")
    if f.illiquidity_risk:
        public_flags.append("Low Liquidity")
        internal_flags.append("illiquidity_risk")

    # Market regime is a quality modifier, not a stage input.
    market_score = 50.0
    if regime is not None:
        market_score = {
            "strong_risk_on": 100.0,
            "risk_on": 82.0,
            "mixed": 60.0,
            "risk_off": 35.0,
            "strong_risk_off": 15.0,
        }.get(regime.regime_label, 50.0)

    trap_penalty = 0.0
    for flag in {str(x) for x in internal_flags}:
        if flag in {"upper_circuit_risk", "lower_circuit_risk", "corporate_action_suspected", "illiquidity_risk"}:
            trap_penalty += 10.0
        elif flag in {"risk_too_wide", "distribution_volume"}:
            trap_penalty += 8.0
        elif flag in {"base_too_deep_daily", "base_too_deep_weekly", "weekly_structure_loose"}:
            trap_penalty += 6.0

    quality_score = (
        0.24 * leader_score
        + 0.24 * trend_score
        + 0.24 * setup_score
        + 0.16 * risk_score
        + 0.12 * market_score
        - trap_penalty
    )
    quality_score = round(_bounded_score(quality_score), 2)

    is_true_leader = bool(leader_score >= 70 and (f.rs_line_26w_high or f.rs_line_above_30w or _finite(f.rs_6m_pct) and f.rs_6m_pct >= 8))
    is_proper_setup = bool(setup_score >= 65 and stage_result.stage in {"Stage 1", "Stage 2"} and "distribution_volume" not in internal_flags)
    is_low_risk = bool(risk_score >= 75 and "risk_too_wide" not in internal_flags)
    is_buyable_internal = bool(
        stage_result.stage == "Stage 2"
        and stage_result.variant != "Extended"
        and trend_template_pass
        and is_true_leader
        and is_proper_setup
        and is_low_risk
        and quality_score >= 78
        and not any(flag in internal_flags for flag in ["upper_circuit_risk", "lower_circuit_risk", "corporate_action_suspected", "illiquidity_risk"])
    )

    if not is_true_leader:
        failure_reasons.append("not_true_leader")
    if not is_proper_setup:
        failure_reasons.append("setup_not_proper_or_not_tight")
    if not is_low_risk:
        failure_reasons.append("risk_not_low")
    if stage_result.stage != "Stage 2":
        failure_reasons.append("not_confirmed_stage2")
    if stage_result.variant == "Extended":
        failure_reasons.append("extended_structure")

    # De-duplicate public flags while preserving order.
    seen = set()
    public_flags_clean = []
    for flag in public_flags:
        if flag and flag not in seen:
            seen.add(flag)
            public_flags_clean.append(flag)

    return SetupQualityResult(
        leader_score=round(float(leader_score), 2),
        trend_score=round(float(trend_score), 2),
        setup_score=round(float(setup_score), 2),
        risk_score=round(float(risk_score), 2),
        quality_score=quality_score,
        setup_quality_label=_setup_quality_label(quality_score),
        relative_strength_label=_relative_strength_label(leader_score),
        volume_pattern_label=_volume_pattern_label(f),
        trend_template_pass=trend_template_pass,
        is_true_leader=is_true_leader,
        is_proper_setup=is_proper_setup,
        is_low_risk=is_low_risk,
        is_buyable_setup_internal=is_buyable_internal,
        public_flags=public_flags_clean,
        internal_flags=internal_flags,
        failure_reasons=failure_reasons,
    )


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
    """Analyze one symbol using one evidence engine and two output heads.

    Public head: weekly-first stage classification.
    Internal head: leader/setup/risk quality scoring used for ranking.
    """
    f = compute_stock_features(ticker, df, benchmark_df, config)
    if f is None:
        return None

    stage_result = classify_public_stage(f, config)
    setup_result = evaluate_internal_setup_quality(f, stage_result, regime, config)

    stage = stage_result.stage
    stage_variant = stage_result.variant
    stage_confidence = stage_result.confidence
    stage_reason = stage_result.reason

    market_regime_ok = regime.regime_label in {"strong_risk_on", "risk_on", "mixed"}
    rs_combo = np.nanmean([f.rs_3m_pct, f.rs_6m_pct])

    stage2_trend_template = bool(setup_result.trend_template_pass)

    stage1_base_ready = (
        stage == "Stage 1"
        and _finite(f.weekly_ma30_trend_10w_pct) and -2.0 <= f.weekly_ma30_trend_10w_pct <= 2.5
        and _finite(f.ma200_trend_1m_pct) and f.ma200_trend_1m_pct >= -1.5
        and f.price_above_ma150
        and f.price_above_ma200
        and _finite(f.dist_from_52w_high_pct) and -35 <= f.dist_from_52w_high_pct <= -3
        and _finite(f.weekly_range_12w_pct) and f.weekly_range_12w_pct <= 22
        and _finite(f.weekly_range_20w_pct) and f.weekly_range_20w_pct <= 38
        and _finite(rs_combo) and rs_combo >= -5
        and f.no_recent_breakdown
        and not f.ma_stack_bearish
    )

    strong_daily_vcp = (
        len(f.daily_depths) >= 2
        and f.daily_base_duration >= config["min_base_duration_days"]
        and f.daily_contraction_score >= 0.60
        and f.daily_depths[-1] <= min(config["max_latest_contraction_pct"], 8.0)
        and _finite(f.volume_dryup_ratio) and f.volume_dryup_ratio <= 0.90
    )
    strict_stage1_daily_vcp = (
        strong_daily_vcp
        and f.daily_depths[0] <= 30
        and max(f.daily_depths) <= 30
        and _finite(f.daily_breakout_distance_pct) and -4.0 <= f.daily_breakout_distance_pct <= 1.5
        and f.tight_range_ok
    )
    weekly_vcp_ok = (
        len(f.weekly_depths) >= 2
        and f.weekly_base_duration >= config["min_base_duration_weeks"]
        and f.weekly_contraction_score >= max(config["min_weekly_strength_score"], 0.55)
        and f.weekly_quality in {"strong", "moderate"}
    )

    near_pivot_stage2_ok = (
        _finite(f.daily_breakout_distance_pct)
        and -5.0 <= f.daily_breakout_distance_pct <= 1.5
        and f.tight_range_ok
        and _finite(f.breakout_volume_ratio) and f.breakout_volume_ratio >= 0.85
    )
    near_pivot_stage1_ok = (
        _finite(f.daily_breakout_distance_pct)
        and -3.0 <= f.daily_breakout_distance_pct <= 1.0
        and f.tight_range_ok
        and _finite(f.volume_dryup_ratio) and f.volume_dryup_ratio <= 0.90
        and f.no_recent_breakdown
    )
    near_pivot_ok = near_pivot_stage2_ok if stage == "Stage 2" else near_pivot_stage1_ok if stage == "Stage 1" else False

    breakout_today = bool(
        _finite(f.daily_breakout_distance_pct)
        and f.daily_breakout_distance_pct > 0
        and _finite(f.breakout_volume_ratio)
        and f.breakout_volume_ratio >= config["breakout_volume_ratio"]
        and stage2_trend_template
        and strong_daily_vcp
        and "Extended" not in stage_variant
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
            f.tight_range_ok,
            market_regime_ok,
        )

    weekly_bucket = classify_weekly_bucket(stage, weekly_vcp_ok, f.weekly_breakout_distance_pct, f.weekly_quality)
    if stage == "Stage 1" and (not stage1_base_ready or not weekly_vcp_ok):
        weekly_bucket = "weekly_watchlist"

    daily_score = score_daily(
        stage,
        trend_template_ok,
        regime.regime_label,
        f.liquidity_ok,
        near_pivot_ok,
        breakout_today,
        f.daily_contraction_score,
        f.daily_base_duration,
        f.dist_from_52w_high_pct,
        f.volume_dryup_ratio,
        f.breakout_volume_ratio,
        f.rs_3m_pct,
        f.rs_6m_pct,
    )
    weekly_score = score_weekly(
        stage,
        f.weekly_contraction_score,
        f.weekly_base_duration,
        f.weekly_breakout_distance_pct,
        f.weekly_quality,
        f.rs_3m_pct,
        f.rs_6m_pct,
    )

    # Public stage penalties are still stage-specific, but setup quality now
    # contributes to ranking separately from the public stage label.
    if stage == "Stage 1":
        if not stage1_base_ready:
            daily_score -= 12
            weekly_score -= 8
        elif not strict_stage1_daily_vcp:
            daily_score -= 8
            weekly_score -= 5
        if breakout_today:
            daily_score -= 8
        if _finite(f.daily_breakout_distance_pct) and f.daily_breakout_distance_pct > 0:
            daily_score -= 3
    elif stage == "Stage 3":
        daily_score -= 10
        weekly_score -= 8
    elif stage == "Stage 4":
        daily_score -= 14
        weekly_score -= 12
    elif stage == "Not Sure":
        daily_score -= 8
        weekly_score -= 8

    # Penalize public ranking for technical traps without converting them into advice.
    trap_penalty = 0.0
    if f.illiquidity_risk:
        trap_penalty += 6
    if f.corporate_action_suspected:
        trap_penalty += 6
    if f.upper_circuit_like_days_20 >= 3 or f.lower_circuit_like_days_20 >= 2:
        trap_penalty += 8
    if f.distribution_weeks_12 >= 2 and f.distribution_weeks_12 > f.accumulation_weeks_12:
        trap_penalty += 6

    daily_score = round(float(max(0.0, daily_score - trap_penalty * 0.40)), 2)
    weekly_score = round(float(max(0.0, weekly_score - trap_penalty * 0.30)), 2)

    combo_bucket = combined_bucket(daily_bucket, weekly_bucket)
    classic_combined = 0.55 * daily_score + 0.45 * weekly_score
    combined_score = round(max(0.0, 0.68 * classic_combined + 0.32 * setup_result.quality_score - trap_penalty), 2)
    structure_score = combined_score

    volume_is_drying_up = bool(_finite(f.volume_dryup_ratio) and f.volume_dryup_ratio <= 0.85)
    weekly_volume_is_drying_up = bool(_finite(f.weekly_volume_ratio) and f.weekly_volume_ratio <= 0.90)

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
    if f.weekly_quality == "strong":
        notes.append("weekly_strong")
    if setup_result.public_flags:
        notes.append("flags=" + ";".join(setup_result.public_flags))
    if stage == "Stage 1" and not strict_stage1_daily_vcp:
        notes.append("stage1_not_actionable")
    if stage == "Stage 1" and not stage1_base_ready:
        notes.append("stage1_needs_more_base")
    if stage == "Stage 3":
        notes.append("distribution_or_weakening_risk")
    if stage == "Stage 4":
        notes.append("downtrend")

    return VCPScoreCard(
        ticker,
        round(f.close, 2),
        round(f.ma50, 2) if _finite(f.ma50) else np.nan,
        round(f.ma150, 2) if _finite(f.ma150) else np.nan,
        round(f.ma200, 2) if _finite(f.ma200) else np.nan,
        stage,
        stage_variant,
        round(float(stage_confidence), 2),
        stage_reason,
        round(float(f.rs_3m_pct), 2) if _finite(f.rs_3m_pct) else np.nan,
        round(float(f.rs_6m_pct), 2) if _finite(f.rs_6m_pct) else np.nan,
        round(float(f.avg_turnover_inr), 2) if _finite(f.avg_turnover_inr) else np.nan,
        daily_bucket,
        daily_score,
        round(float(f.daily_pivot), 2) if _finite(f.daily_pivot) else np.nan,
        round(float(f.daily_breakout_distance_pct), 2) if _finite(f.daily_breakout_distance_pct) else np.nan,
        f.daily_depths,
        f.daily_durations,
        round(float(f.daily_contraction_score), 2),
        round(float(f.daily_base_duration), 2),
        weekly_bucket,
        weekly_score,
        round(float(f.weekly_pivot), 2) if _finite(f.weekly_pivot) else np.nan,
        round(float(f.weekly_breakout_distance_pct), 2) if _finite(f.weekly_breakout_distance_pct) else np.nan,
        f.weekly_depths,
        f.weekly_durations,
        round(float(f.weekly_contraction_score), 2),
        round(float(f.weekly_base_duration), 2),
        f.weekly_quality,
        combo_bucket,
        combined_score,
        round(float(f.volume_dryup_ratio), 2) if _finite(f.volume_dryup_ratio) else np.nan,
        round(float(f.breakout_volume_ratio), 2) if _finite(f.breakout_volume_ratio) else np.nan,
        round(float(f.weekly_volume_ratio), 2) if _finite(f.weekly_volume_ratio) else np.nan,
        volume_is_drying_up,
        weekly_volume_is_drying_up,
        ", ".join(notes),
        public_stage_label=public_stage_label(stage, stage_variant),
        volume_pattern_display=_volume_pattern_display(f),
        nifty_3m_outperformance_pct=round(float(f.rs_3m_pct), 2) if _finite(f.rs_3m_pct) else np.nan,
        nifty_3m_outperformance_label=_nifty_3m_outperformance_label(f.rs_3m_pct),
        structure_score=round(float(structure_score), 2),
        trend_template_pass=bool(trend_template_ok),
        setup_quality_label=setup_result.setup_quality_label,
        relative_strength_label=setup_result.relative_strength_label,
        volume_pattern_label=setup_result.volume_pattern_label,
        technical_flags="; ".join(setup_result.public_flags) if setup_result.public_flags else "No Flags",
        risk_pct=round(float(f.risk_pct), 2) if _finite(f.risk_pct) else np.nan,
        internal_leader_score=setup_result.leader_score,
        internal_trend_score=setup_result.trend_score,
        internal_setup_score=setup_result.setup_score,
        internal_risk_score=setup_result.risk_score,
        internal_quality_score=setup_result.quality_score,
        internal_is_true_leader=setup_result.is_true_leader,
        internal_is_proper_setup=setup_result.is_proper_setup,
        internal_is_low_risk=setup_result.is_low_risk,
        internal_is_buyable_setup=setup_result.is_buyable_setup_internal,
        internal_failure_reasons="; ".join(setup_result.failure_reasons) if setup_result.failure_reasons else "No Internal Gaps",
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
    if skip_existing and outfile.exists() and outfile.stat().st_size > 10_000:
        return

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
    top_rank_limit: int = 180,
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
    for idx, ticker in enumerate(tickers, start=1):
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue
        row = score_map.get(ticker, {})
        chart_name = str(row.get("Company Name") or ticker).strip()
        safe = sanitize_filename(ticker)
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
        if idx % 25 == 0 or idx == len(tickers):
            print(f"Charts processed {idx:,}/{len(tickers):,} in {time.perf_counter()-t0:.2f}s")
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
        "ticker", "Company Name", "Industry", "sector", "is_fo_stock", "fo_category", "Include", "stage", "stage_raw", "stage_variant", "stage_confidence", "stage_reason", "stage_state_reason", "stage_failed_since", "last_stage2_date", "stage_pending_raw", "daily_setup_bucket", "weekly_setup_bucket", "combined_bucket",
        "daily_score", "weekly_score", "combined_score", "industry_boost", "final_daily_score", "final_weekly_score",
        "final_combined_score", "public_stage_label", "structure_score", "relative_strength_label", "volume_pattern_label", "volume_pattern_display", "nifty_3m_outperformance_pct", "nifty_3m_outperformance_label", "trend_template_pass",
        "rs_3m_pct", "rs_6m_pct", "avg_turnover_inr", "notes",
    ]
    out = out[[c for c in keep_cols if c in out.columns]].copy()
    for col in ["daily_score", "weekly_score", "combined_score", "industry_boost", "final_daily_score", "final_weekly_score", "final_combined_score", "structure_score", "nifty_3m_outperformance_pct", "rs_3m_pct", "rs_6m_pct", "avg_turnover_inr"]:
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


def _run_date(config: Optional[dict] = None) -> pd.Timestamp:
    """The market date this run computes, as a naive midnight timestamp.

    MarketGita edit: build_outputs puts the benchmark's last bar in config["as_of"].
    This used to read the IST wall clock, so a run after midnight (a catch-up, or a
    missed day) stamped a session with the wrong date. Without as_of it still does.
    """
    if config and config.get("as_of") is not None:
        return pd.Timestamp(config["as_of"])
    return pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)


def _consecutive_raw_stage_runs(history_df: pd.DataFrame, prev_combined: Optional[pd.DataFrame], ticker: str, target_stage: str, as_of: pd.Timestamp) -> int:
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
                records.append((as_of - pd.Timedelta(hours=1), stg))

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


def _consecutive_public_stage_runs(history_df: pd.DataFrame, prev_combined: Optional[pd.DataFrame], ticker: str, target_stage: str, as_of: pd.Timestamp) -> int:
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
            records.append((as_of - pd.Timedelta(hours=1), str(row.iloc[-1].get("stage", "") or "").strip()))
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
    today = _run_date(config)

    if prev_combined is not None and not prev_combined.empty and "ticker" in prev_combined.columns and "stage" in prev_combined.columns:
        yesterday = today - pd.Timedelta(days=1)
        for _, prev_row in prev_combined.iterrows():
            ticker = str(prev_row.get("ticker", "")).upper().strip()
            if ticker and str(prev_row.get("stage", "")) == "Stage 2" and ticker not in last_stage2:
                last_stage2[ticker] = yesterday

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

        raw_confirm_runs = _consecutive_raw_stage_runs(history_df, prev_combined, ticker, raw_stage, today) + (1 if raw_stage not in {"Not Sure"} else 0)
        public_stage1_runs = _consecutive_public_stage_runs(history_df, prev_combined, ticker, "Stage 1", today)
        recently_stage2 = days_since_s2 is not None and days_since_s2 <= failed_hold_days

        # On a first ever run, do not turn clean raw Stage 2 names into Not Sure
        # only because the memory table has not accumulated confirmation rows yet.
        if not prev_stage:
            out.at[idx, "stage_state_reason"] = "Initial classification; no prior stage memory available."
            continue

        # 1) Explicit failed Stage 2 mechanism. This takes priority over normal transitions.
        broke_from_stage2 = (
            raw_stage != "Stage 2"
            and (prev_stage == "Stage 2" or (prev_stage in {"Failed Stage 2", "Stage 2 Failed"} and recently_stage2) or recently_stage2)
        )
        if broke_from_stage2:
            out.at[idx, "stage"] = "Failed Stage 2"
            out.at[idx, "stage_variant"] = "Failed Stage 2"
            out.at[idx, "stage_confidence"] = 0.72
            if not str(row.get("stage_failed_since", "") or "").strip():
                out.at[idx, "stage_failed_since"] = today.date().isoformat()
            out.at[idx, "stage_reason"] = (
                "This stock was recently Stage 2, but the latest structure no longer satisfies Stage 2 rules. "
                "It is kept as Failed Stage 2 until a fresh base/repair structure forms."
            )
            out.at[idx, "stage_state_reason"] = "Recent Stage 2 break; public stage held as Failed Stage 2."
            out.at[idx, "notes"] = _append_note(row.get("notes", ""), "Stage 2 failed state applied from stage memory.")
            continue

        # 2) While in failed Stage 2, do not instantly relabel unless the repair/advance is confirmed.
        if prev_stage in {"Failed Stage 2", "Stage 2 Failed"}:
            if raw_stage == "Stage 2" and raw_confirm_runs >= stage2_confirm_days:
                out.at[idx, "stage_state_reason"] = f"Stage 2 reclaimed after {raw_confirm_runs} confirmation runs."
                continue
            if recently_stage2:
                out.at[idx, "stage"] = "Failed Stage 2"
                out.at[idx, "stage_variant"] = "Failed Stage 2"
                out.at[idx, "stage_confidence"] = 0.70
                out.at[idx, "stage_reason"] = "Failed Stage 2 cooling period is still active; waiting for a fresh base or confirmed reclaim."
                out.at[idx, "stage_state_reason"] = "Failed Stage 2 hold period active."
                out.at[idx, "notes"] = _append_note(row.get("notes", ""), "Failed Stage 2 hold period active.")
                continue
            if raw_stage in {"Stage 1", "Stage 3", "Stage 4"} and raw_confirm_runs < transition_confirm_days:
                _set_pending_stage(out, idx, row, raw_stage, f"Failed Stage 2 -> {raw_stage} needs {transition_confirm_days} confirmation runs; current count {raw_confirm_runs}.")
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

    if "public_stage_label" in out.columns:
        out["public_stage_label"] = out.apply(lambda r: public_stage_label(str(r.get("stage", "")), str(r.get("stage_variant", ""))), axis=1)

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

    priority += {"Stage 2": 30, "Stage 1": 12, "Failed Stage 2": 5, "Stage 2 Failed": 5, "Stage 3": 6, "Stage 4": 0, "Not Sure": 1}.get(stage, 1)
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

    priority += {
        "High": 18,
        "Medium": 8,
        "Low": 2,
        "Not Rated": 0,
    }.get(str(row.get("setup_quality_label", "")), 0)
    priority += {
        "Very Strong": 14,
        "Strong": 8,
        "Neutral": 2,
        "Weak": -8,
    }.get(str(row.get("relative_strength_label", "")), 0)
    flags = str(row.get("technical_flags", "") or "")
    if any(x in flags for x in ["Circuit Risk", "Corporate Action Check", "Low Liquidity"]):
        priority -= 18
    if "Extended" in flags:
        priority -= 6

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


def build_interesting20_snapshot(combined_df: pd.DataFrame, limit: int = 20, top_pool: int = 30, config: Optional[dict] = None) -> pd.DataFrame:
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
    pool["snapshot_date"] = _run_date(config).date().isoformat()
    keep = [c for c in [
        "snapshot_date", "ticker", "Company Name", "Industry", "stage", "stage_raw", "stage_variant", "stage_confidence", "stage_reason", "stage_state_reason", "stage_failed_since", "last_stage2_date", "stage_pending_raw", "current_rank",
        "interesting_priority", "daily_setup_bucket", "weekly_setup_bucket", "combined_bucket",
        "final_combined_score", "public_stage_label", "structure_score", "relative_strength_label", "volume_pattern_label", "volume_pattern_display", "nifty_3m_outperformance_pct", "nifty_3m_outperformance_label",
        "daily_breakout_distance_pct", "weekly_breakout_distance_pct",
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
    chart_scope: str = "dashboard",
    skip_existing_charts: bool = False,
    chart_dpi: int = 150,
) -> Dict[str, str]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.perf_counter()
    universe_df = load_nifty500_universe(universe_path)
    universe_used_file = out_path / "universe_used.csv"
    universe_df.to_csv(universe_used_file, index=False)
    tickers = universe_df["Ticker"].astype(str).str.upper().tolist()
    print(f"Universe used: {len(tickers):,} included EQ stocks")

    if wide_price:
        price_data = load_wide_price_data(wide_price, tickers, cfg["market_index"], max_rows=int(cfg.get("max_price_rows", 620) or 0) or None)
    else:
        price_data = None

    report, regime, price_data = build_vcp_universe_report(tickers, cfg, price_data=price_data)
    if report.empty:
        raise RuntimeError("No screener results produced.")
    # MarketGita edit: every date below comes from the data, not the wall clock (see _run_date).
    last_bar = pd.Timestamp(price_data[cfg["market_index"]]["Close"].dropna().index[-1])
    cfg["as_of"] = (last_bar.tz_localize(None) if last_bar.tzinfo else last_bar).normalize()
    print(f"Market date (last {cfg['market_index']} bar): {cfg['as_of'].date()}")

    final_report = report.merge(universe_df, left_on="ticker", right_on="Ticker", how="left")
    industry_df = build_industry_strength_table(final_report)
    final_report = apply_industry_boost(final_report, industry_df, cfg)

    prev_combined_for_stage_memory = pd.read_csv(out_path / "vcp_combined_ranked.csv") if (out_path / "vcp_combined_ranked.csv").exists() else None
    final_report = apply_stage_state_memory(final_report, out_path, prev_combined_for_stage_memory, cfg)

    metadata_cols = ["sector", "industry_group", "is_fo_stock", "fo_category", "Include"]
    public_structure_cols = [
        "public_stage_label", "structure_score", "relative_strength_label",
        "volume_pattern_label", "volume_pattern_display",
        "nifty_3m_outperformance_pct", "nifty_3m_outperformance_label",
        "trend_template_pass",
    ]
    internal_setup_cols = [
        "setup_quality_label", "technical_flags", "risk_pct",
        "internal_leader_score", "internal_trend_score", "internal_setup_score", "internal_risk_score", "internal_quality_score",
        "internal_is_true_leader", "internal_is_proper_setup", "internal_is_low_risk", "internal_is_buyable_setup", "internal_failure_reasons",
    ]
    common_cols = ["ticker", "Company Name", "Industry"] + metadata_cols + ["stage", "stage_raw", "stage_variant", "stage_confidence", "stage_reason", "stage_state_reason", "stage_failed_since", "last_stage2_date", "stage_pending_raw"] + public_structure_cols + ["rs_3m_pct", "rs_6m_pct", "avg_turnover_inr", "volume_dryup_ratio", "breakout_volume_ratio", "weekly_volume_ratio", "volume_is_drying_up", "weekly_volume_is_drying_up", "notes"]
    daily_cols = common_cols + ["daily_setup_bucket", "daily_score", "final_daily_score", "daily_pivot", "daily_breakout_distance_pct", "daily_contraction_depths_pct", "daily_contraction_durations", "daily_contraction_score", "daily_base_duration_days"]
    weekly_cols = common_cols + ["weekly_setup_bucket", "weekly_score", "final_weekly_score", "weekly_pivot", "weekly_breakout_distance_pct", "weekly_contraction_depths_pct", "weekly_contraction_durations", "weekly_contraction_score", "weekly_base_duration_weeks", "weekly_vcp_quality"]
    combined_cols = common_cols + ["daily_setup_bucket", "weekly_setup_bucket", "combined_bucket", "daily_score", "weekly_score", "combined_score", "industry_boost", "final_combined_score"]
    internal_cols = ["ticker", "Company Name", "Industry"] + metadata_cols + ["stage", "stage_raw", "stage_variant", "stage_confidence", "stage_reason"] + public_structure_cols + internal_setup_cols + ["daily_setup_bucket", "weekly_setup_bucket", "combined_bucket", "final_combined_score", "notes"]

    daily_df = final_report[[c for c in daily_cols if c in final_report.columns]].sort_values(["final_daily_score", "daily_score"], ascending=[False, False]).reset_index(drop=True)
    weekly_df = final_report[[c for c in weekly_cols if c in final_report.columns]].sort_values(["final_weekly_score", "weekly_score"], ascending=[False, False]).reset_index(drop=True)
    combined_df = final_report[[c for c in combined_cols if c in final_report.columns]].sort_values(["final_combined_score", "combined_score"], ascending=[False, False]).reset_index(drop=True)
    internal_df = final_report[[c for c in internal_cols if c in final_report.columns]].sort_values(["internal_quality_score", "final_combined_score"], ascending=[False, False]).reset_index(drop=True)

    daily_df["current_rank"] = np.arange(1, len(daily_df) + 1)
    weekly_df["current_rank"] = np.arange(1, len(weekly_df) + 1)
    combined_df["current_rank"] = np.arange(1, len(combined_df) + 1)
    internal_df["current_rank"] = np.arange(1, len(internal_df) + 1)
    industry_df = industry_df.copy().reset_index(drop=True)
    industry_df["current_rank"] = np.arange(1, len(industry_df) + 1)

    prev_combined = pd.read_csv(out_path / "vcp_combined_ranked.csv") if (out_path / "vcp_combined_ranked.csv").exists() else None
    prev_industry = pd.read_csv(out_path / "industry_strength.csv") if (out_path / "industry_strength.csv").exists() else None

    stock_changes = build_stock_changes(combined_df, prev_combined)
    industry_changes = build_industry_changes(industry_df, prev_industry)
    top_movers = stock_changes.sort_values(["new_top_10", "new_top_20", "new_daily_breakout", "new_weekly_breakout", "rank_change", "combined_score_change"], ascending=[False, False, False, False, False, False]).reset_index(drop=True)

    benchmark_hist_df = price_data.get(cfg["market_index"])
    price_moves = build_price_moves(combined_df, price_data)
    history_file = update_stage_action_history(out_path, combined_df, price_data, benchmark_hist_df, universe_df, cfg)

    daily_file = out_path / "vcp_daily_ranked.csv"
    weekly_file = out_path / "vcp_weekly_ranked.csv"
    combined_file = out_path / "vcp_combined_ranked.csv"
    internal_file = out_path / "internal_setup_ranked.csv"
    industry_file = out_path / "industry_strength.csv"
    regime_file = out_path / "market_regime.csv"
    stock_changes_file = out_path / "stock_changes.csv"
    industry_changes_file = out_path / "industry_changes.csv"
    top_movers_file = out_path / "top_movers.csv"
    price_moves_file = out_path / "stock_price_moves.csv"

    daily_df.to_csv(daily_file, index=False)
    weekly_df.to_csv(weekly_file, index=False)
    combined_df.to_csv(combined_file, index=False)
    internal_df.to_csv(internal_file, index=False)
    industry_df.to_csv(industry_file, index=False)
    pd.DataFrame([asdict(regime)]).to_csv(regime_file, index=False)
    stock_changes.to_csv(stock_changes_file, index=False)
    industry_changes.to_csv(industry_changes_file, index=False)
    top_movers.to_csv(top_movers_file, index=False)
    price_moves.to_csv(price_moves_file, index=False)

    
    if not export_all_ticker_charts or chart_scope == "none":
        chart_paths = {"daily_charts_dir": str(out_path / "charts" / "daily"), "weekly_charts_dir": str(out_path / "charts" / "weekly")}
    elif chart_scope == "all":
        chart_paths = export_selected_charts(final_report, price_data, out_path, tickers_to_export=None, skip_existing=skip_existing_charts, dpi=chart_dpi)
    else:
        chart_tickers = build_dashboard_chart_tickers(combined_df, price_moves, stock_changes, out_path)
        chart_paths = export_selected_charts(final_report, price_data, out_path, tickers_to_export=chart_tickers, skip_existing=skip_existing_charts, dpi=chart_dpi)
    interesting_snapshot = build_interesting20_snapshot(combined_df, limit=20, top_pool=30, config=cfg)
    interesting_paths = save_interesting20_archive(out_path, interesting_snapshot, chart_paths)
    print(f"Total engine runtime: {time.perf_counter()-overall_t0:.2f}s")
    return {"daily": str(daily_file), "weekly": str(weekly_file), "combined": str(combined_file), "internal": str(internal_file), "industry": str(industry_file), "regime": str(regime_file), "stock_changes": str(stock_changes_file), "industry_changes": str(industry_changes_file), "top_movers": str(top_movers_file), "price_moves": str(price_moves_file), "history": str(history_file), "universe_used": str(universe_used_file), "as_of": cfg["as_of"].date().isoformat(), **chart_paths, **interesting_paths}

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


def derive_technical_status(stage: str, combined_bucket: str, score: float) -> str:
    """Public-safe structure label. No buy/sell/hold language."""
    if stage in {"Failed Stage 2", "Stage 2 Failed"}:
        return "Failed Stage 2"
    if stage == "Not Sure":
        return "Unclear Structure"
    if stage == "Stage 2":
        if combined_bucket in {"high_conviction_breakout", "high_conviction_near_pivot"} and score >= 70:
            return "Strong Stage 2 Structure"
        return "Stage 2 Structure"
    if stage == "Stage 1":
        return "Stage 1 Structure"
    if stage == "Stage 3":
        return "Stage 3 Structure"
    if stage == "Stage 4":
        return "Stage 4 Structure"
    return "Mixed Structure"


def derive_structure_status(stage: str, combined_bucket: str, score: float) -> str:
    """Secondary public-safe status for history snapshots."""
    if stage in {"Failed Stage 2", "Stage 2 Failed"}:
        return "Failed Stage 2"
    if stage == "Not Sure":
        return "Not Rated"
    if stage == "Stage 2":
        if combined_bucket == "high_conviction_breakout" and score >= 72:
            return "High Structure Quality"
        if combined_bucket in {"high_conviction_near_pivot", "building_setup"} and score >= 62:
            return "Improving Structure Quality"
        return "Stage 2 Structure"
    if stage == "Stage 1":
        return "Stage 1 Structure"
    if stage == "Stage 3":
        return "Stage 3 Structure"
    if stage == "Stage 4":
        return "Stage 4 Structure"
    return "Not Rated"


# Backward-compatible names kept for older private workflows. They now return
# public-safe structure language only.
def derive_public_action(stage: str, combined_bucket: str, score: float) -> str:
    return derive_technical_status(stage, combined_bucket, score)


def derive_super_action(stage: str, combined_bucket: str, score: float) -> str:
    return derive_structure_status(stage, combined_bucket, score)


def build_stage_action_history_snapshot(snapshot_df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    if snapshot_df is None or snapshot_df.empty:
        return pd.DataFrame()
    out = snapshot_df.copy()
    score_col = "final_combined_score" if "final_combined_score" in out.columns else "combined_score"
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out["snapshot_date"] = pd.Timestamp(snapshot_date).normalize()
    out["technical_status"] = out.apply(lambda r: derive_technical_status(str(r.get("stage", "")), str(r.get("combined_bucket", "")), float(pd.to_numeric(r.get(score_col), errors="coerce") if pd.notna(pd.to_numeric(r.get(score_col), errors="coerce")) else 0.0)), axis=1)
    out["structure_status"] = out.apply(lambda r: derive_structure_status(str(r.get("stage", "")), str(r.get("combined_bucket", "")), float(pd.to_numeric(r.get(score_col), errors="coerce") if pd.notna(pd.to_numeric(r.get(score_col), errors="coerce")) else 0.0)), axis=1)
    keep_cols = [c for c in [
        "snapshot_date", "ticker", "Company Name", "Industry", "sector", "is_fo_stock", "fo_category", "stage", "stage_raw", "stage_variant", "stage_confidence", "stage_reason", "stage_state_reason", "stage_failed_since", "last_stage2_date", "stage_pending_raw", "combined_bucket", score_col,
        "public_stage_label", "structure_score", "relative_strength_label", "volume_pattern_label", "volume_pattern_display", "nifty_3m_outperformance_pct", "nifty_3m_outperformance_label",
        "volume_dryup_ratio", "breakout_volume_ratio", "weekly_volume_ratio", "volume_is_drying_up", "weekly_volume_is_drying_up",
        "technical_status", "structure_status"
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
                row["snapshot_date"] = pd.Timestamp(snap_date).normalize()
                row["Company Name"] = company
                row["Industry"] = industry
                row["technical_status"] = derive_technical_status(row.get("stage", ""), row.get("combined_bucket", ""), float(row.get("combined_score", 0) or 0))
                row["structure_status"] = derive_structure_status(row.get("stage", ""), row.get("combined_bucket", ""), float(row.get("combined_score", 0) or 0))
                history_rows.append({k: row.get(k) for k in [
                    "snapshot_date", "ticker", "Company Name", "Industry", "stage", "combined_bucket", "combined_score",
                    "volume_dryup_ratio", "breakout_volume_ratio", "weekly_volume_ratio", "volume_is_drying_up", "weekly_volume_is_drying_up", "technical_status", "structure_status"
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
    today = _run_date(config)
    current_history = build_stage_action_history_snapshot(current_snapshot, today)

    if history_file.exists():
        existing = pd.read_csv(history_file, parse_dates=["snapshot_date"])
    else:
        existing = build_six_month_history(price_data, benchmark_df, universe_df, config) if bool(config.get("history_init_enabled", True)) else pd.DataFrame()

    if not current_history.empty:
        existing = pd.concat([existing, current_history], ignore_index=True) if not existing.empty else current_history

    if existing.empty:
        existing.to_csv(history_file, index=False)
        return history_file

    existing["snapshot_date"] = pd.to_datetime(existing["snapshot_date"], utc=True).dt.tz_convert(None).dt.normalize()
    existing = existing.drop_duplicates(subset=["snapshot_date", "ticker"], keep="last")
    existing = existing.sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)
    existing.to_csv(history_file, index=False)
    return history_file



def write_engine_run_metadata(out_path: Path, as_of: Optional[str] = None) -> None:
    """Persist actual engine run time for dashboard display.

    Dashboard should not infer freshness from current time. It should read this file.
    MarketGita edit: also records the market date computed and the code version.
    """
    metadata = {
        "engine_ran_at_ist": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "engine_ran_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "as_of": as_of,
        "code_version": os.environ.get("GITHUB_SHA", "local"),
    }
    try:
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "engine_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"Warning: could not write engine_run_metadata.json: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily + Weekly VCP Screener with local Yahoo wide-file support")
    parser.add_argument("--universe", required=True, help="Path to universe CSV. New schema supports Include and f&o columns.")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--wide-price", default=None, help="Folder containing wide_open.csv/wide_high.csv/wide_low.csv/wide_close.csv/wide_volume.csv, or yahoo_price_data_wide.xlsx")
    parser.add_argument("--max-price-rows", type=int, default=620, help="Read only the latest N rows from the wide price files. Use 0 to load all rows.")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation for fast testing.")
    parser.add_argument("--chart-scope", choices=["dashboard", "all", "none"], default="dashboard", help="dashboard = generate only charts used by dashboard views; all = every included stock; none = no charts. Default: dashboard.")
    parser.add_argument("--skip-existing-charts", action="store_true", help="Do not regenerate chart PNGs that already exist. Fastest for reruns after non-price changes.")
    parser.add_argument("--chart-dpi", type=int, default=240, help="PNG DPI for dashboard charts. 130-160 is usually enough for Streamlit.")
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
        skip_existing_charts=bool(args.skip_existing_charts),
        chart_dpi=int(args.chart_dpi),
    )
    write_engine_run_metadata(Path(args.outdir), as_of=outputs.pop("as_of", None))
    print("Saved files:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
