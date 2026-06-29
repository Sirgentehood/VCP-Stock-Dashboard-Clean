import streamlit as st
import pandas as pd
from pathlib import Path
import re

st.set_page_config(page_title="Market Structure Radar", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root {
  --card-border: rgba(128,128,128,0.16);
  --muted: rgba(255,255,255,0.72);
  --strong: #1ec977;
  --developing: #f0b429;
  --weak: #ff6b6b;
  --cautious: #ff9f43;
  --up: #1ec977;
  --down: #ff6b6b;
  --stage1-bg: rgba(55,95,220,0.14);
  --stage2-bg: rgba(0,179,179,0.12);
  --stage3-bg: rgba(212,160,23,0.12);
  --stage4-bg: rgba(170,80,180,0.14);
  --stage1-border: rgba(55,95,220,0.34);
  --stage2-border: rgba(0,179,179,0.28);
  --stage3-border: rgba(212,160,23,0.28);
  --stage4-border: rgba(170,80,180,0.34);
}
.block-container {padding-top: 0.45rem; padding-bottom: 1.2rem; padding-left: 0.7rem; padding-right: 0.7rem; max-width: 1400px;}
[data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none;}
.stTabs [data-baseweb="tab"] {font-size: 1.05rem; font-weight: 700;}
.stTabs [data-baseweb="tab-list"] {gap: 0.55rem; margin-top: 0.1rem;}
.hero-card, .stock-card, .learn-card, .info-card {
  border: 2px solid var(--card-border);
  border-radius: 18px;
  padding: 0.85rem 0.95rem;
  background: rgba(255,255,255,0.03);
}
.hero-card {padding: 1rem 1rem;}
.kicker {font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted);}
.big-number {font-size: 1.34rem; font-weight: 800; margin-top: 0.08rem; margin-bottom: 0.1rem;}
.muted {color: var(--muted);}
.status-pill {display:inline-block; font-size:0.74rem; font-weight:700; padding:0.18rem 0.5rem; border-radius:999px; white-space:nowrap;}
.status-strong {background: rgba(30,201,119,0.14); color: var(--strong); border:1px solid rgba(30,201,119,0.35);}
.status-developing {background: rgba(240,180,41,0.14); color: var(--developing); border:1px solid rgba(240,180,41,0.35);}
.status-weak {background: rgba(255,107,107,0.14); color: var(--weak); border:1px solid rgba(255,107,107,0.35);}
.status-cautious {background: rgba(255,159,67,0.14); color: var(--cautious); border:1px solid rgba(255,159,67,0.35);}
.structure-pill {display:inline-block; font-size:0.74rem; font-weight:800; padding:0.18rem 0.5rem; border-radius:999px; white-space:nowrap; margin-top:0.22rem; background: rgba(255,255,255,0.08); color:#eef3ff; border:1px solid rgba(255,255,255,0.18);}
.stock-title {font-size: 1.02rem; font-weight: 700; margin-bottom: 0.06rem; line-height: 1.2;}
.meta-line {font-size: 0.93rem; font-weight: 600; line-height: 1.18; margin-top: 0.06rem; margin-bottom: 0.02rem;}
.stock-subtitle {font-size: 0.92rem; color: var(--muted); margin-top: 0.04rem; line-height: 1.25;}
.stock-card {margin-bottom: 0.5rem; padding-top: 0.78rem; padding-bottom: 0.78rem;}
.stage-card-1 {background: var(--stage1-bg); border-color: var(--stage1-border);}
.stage-card-2 {background: var(--stage2-bg); border-color: var(--stage2-border);}
.stage-card-3 {background: var(--stage3-bg); border-color: var(--stage3-border);}
.stage-card-4 {background: var(--stage4-bg); border-color: var(--stage4-border);}
.change-badge-up {font-size: 1.12rem; font-weight: 900; margin-top: 0.1rem; color: var(--up);}
.change-badge-down {font-size: 1.12rem; font-weight: 900; margin-top: 0.1rem; color: var(--down);}
.rank-text {font-size: 0.84rem; font-weight: 700; color: var(--muted); margin-top: 0.18rem;}
.disclosure {border-left: 4px solid rgba(240,180,41,0.55); background: rgba(240,180,41,0.08); border-radius: 12px; padding: 0.75rem 0.9rem; font-size: 0.88rem; margin-bottom: 0.7rem; margin-top: 1rem;}
.list-tight {margin: 0.2rem 0 0 1rem; padding: 0;}
.change-text {font-size: 0.88rem; margin-top: 0.06rem; line-height: 1.2;}
.small-note {font-size: 0.84rem; color: var(--muted);}
.dist-row {display:flex; align-items:center; gap:0.55rem; margin:0.35rem 0;}
.dist-label {width: 90px; font-size:0.9rem; font-weight:700;}
.dist-bar-wrap {flex:1; background:rgba(255,255,255,0.08); border-radius:999px; height:12px; overflow:hidden;}
.dist-bar {height:12px; border-radius:999px;}
.dist-value {width:40px; text-align:right; font-size:0.9rem; font-weight:700;}
@media (max-width: 768px) {
  .block-container {padding-top: 0.35rem; padding-left: 0.35rem; padding-right: 0.35rem;}
  .stTabs [data-baseweb="tab"] {font-size: 0.93rem;}
  .stock-title {font-size: 0.96rem;}
  .meta-line {font-size: 0.88rem; line-height:1.28;}
}
</style>
""", unsafe_allow_html=True)

LABELS = {
    "Strong": {"css": "status-strong"},
    "Developing": {"css": "status-developing"},
    "Weak": {"css": "status-weak"},
    "Cautious": {"css": "status-cautious"},
}
MAX_PORTFOLIO_STOCKS = None

@st.cache_data(show_spinner=False)
def load_csv(path: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(path)

def safe_read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return load_csv(str(p), p.stat().st_mtime_ns)
    except Exception:
        return pd.DataFrame()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
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
    drop_cols = [c for c in ["Company Name_x", "Company Name_y", "Industry_x", "Industry_y", "Ticker"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out

def ensure_current_rank(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "current_rank" not in out.columns:
        for alt in ["rank", "rs_rank", "daily_rank", "weekly_rank", "final_rank", "combined_rank", "stock_rank"]:
            if alt in out.columns:
                out["current_rank"] = pd.to_numeric(out[alt], errors="coerce")
                break
    if "current_rank" not in out.columns:
        out["current_rank"] = float("nan")
    else:
        out["current_rank"] = pd.to_numeric(out["current_rank"], errors="coerce")
    return out

def build_rank_lookup_map(*dfs: pd.DataFrame) -> dict:
    rank_map = {}
    rank_cols = ["current_rank", "rank", "rs_rank", "daily_rank", "weekly_rank", "final_rank", "combined_rank", "stock_rank"]
    for df in dfs:
        if df is None or df.empty:
            continue
        ticker_col = None
        for cand in ["ticker", "Ticker", "symbol", "Symbol"]:
            if cand in df.columns:
                ticker_col = cand
                break
        if ticker_col is None:
            continue
        work = df.copy()
        work[ticker_col] = work[ticker_col].astype(str).str.strip()
        chosen_rank_col = None
        for col in rank_cols:
            if col in work.columns:
                chosen_rank_col = col
                break
        if chosen_rank_col is None:
            continue
        work[chosen_rank_col] = pd.to_numeric(work[chosen_rank_col], errors="coerce")
        work = work.dropna(subset=[ticker_col, chosen_rank_col])
        for _, r in work.iterrows():
            ticker = str(r[ticker_col]).strip()
            rank_val = pd.to_numeric(r[chosen_rank_col], errors="coerce")
            if pd.notna(rank_val):
                rank_map[ticker] = float(rank_val)
                rank_map[ticker.replace(".NS", "")] = float(rank_val)
    return rank_map

def backfill_current_rank(df: pd.DataFrame, rank_map: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = ensure_current_rank(df.copy())
    if "ticker" not in out.columns:
        return out
    out["ticker"] = out["ticker"].astype(str).str.strip()
    missing_mask = out["current_rank"].isna()
    if missing_mask.any():
        out.loc[missing_mask, "current_rank"] = out.loc[missing_mask, "ticker"].map(rank_map)
    missing_mask = out["current_rank"].isna()
    if missing_mask.any():
        out.loc[missing_mask, "current_rank"] = out.loc[missing_mask, "ticker"].str.replace(".NS", "", regex=False).map(rank_map)
    out["current_rank"] = pd.to_numeric(out["current_rank"], errors="coerce")
    return out

def sort_by_rank(df: pd.DataFrame, descending: bool = False, company_tiebreak: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = ensure_current_rank(df)
    sort_cols = ["current_rank"]
    ascending = [not descending]
    if company_tiebreak and "Company Name" in out.columns:
        sort_cols.append("Company Name")
        ascending.append(True)
    return safe_sort_values(out, sort_cols, ascending)


def sort_watchlist_view(df: pd.DataFrame, selected_watchlist: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = ensure_current_rank(df.copy())
    has_action = "action_confidence" in out.columns and "action" in out.columns
    if selected_watchlist == "Stage 4":
        if has_action:
            return safe_sort_values(out, ["action_confidence", "current_rank", "Company Name"], [False, False, True])
        return safe_sort_values(out, ["current_rank", "Company Name"], [False, True])
    if has_action:
        return safe_sort_values(out, ["action_confidence", "current_rank", "Company Name"], [False, True, True])
    return safe_sort_values(out, ["current_rank", "Company Name"], [True, True])

def dry_up_status(row: pd.Series) -> str:
    daily = boolish(row.get("volume_is_drying_up", False))
    weekly = boolish(row.get("weekly_volume_is_drying_up", False))
    if daily and weekly:
        return "Daily + Weekly dry-up"
    if weekly:
        return "Weekly dry-up"
    if daily:
        return "Daily dry-up"
    return ""

def build_mini_signal_text(row: pd.Series) -> str:
    parts = []
    dry_candidates = [
        ("volume_is_drying_up", "Volume drying up"),
        ("volume_drying_up", "Volume drying up"),
        ("volume_dry_up", "Volume drying up"),
        ("vol_dry_up", "Volume drying up"),
        ("drying_up", "Volume drying up"),
    ]
    for col, label in dry_candidates:
        if col in row.index and boolish(row.get(col, False)):
            parts.append(label)
            break

    if boolish(row.get("weekly_volume_is_drying_up", False)):
        parts.append("Weekly dry-up")

    for col, label in [("volume_ratio", "Vol ratio"), ("avg_volume_ratio", "Avg vol ratio"), ("vol_ratio", "Vol ratio"), ("volume_dryup_ratio", "Dry-up ratio")]:
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(val):
            parts.append(f"{label} {val:.2f}x")
            break

    for col, label in [("rs_3m_pct", "RS 3M"), ("rs_6m_pct", "RS 6M"), ("change_1w_pct", "1W"), ("change_1m_pct", "1M")]:
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(val):
            parts.append(f"{label} {val:+.1f}%")

    if boolish(row.get("entered_stage_2", False)):
        parts.append("Entered Stage 2")
    if boolish(row.get("new_weekly_breakout", False)):
        parts.append("Weekly breakout")
    elif boolish(row.get("new_daily_breakout", False)):
        parts.append("Daily breakout")

    rank_change = pd.to_numeric(row.get("rank_change"), errors="coerce")
    if pd.notna(rank_change) and rank_change != 0:
        parts.append(f"Rank {'↑' if rank_change > 0 else '↓'} {abs(int(rank_change))}")

    return " • ".join(parts[:5])

def classify_stock(row: pd.Series) -> str:
    stage = str(row.get("stage", ""))
    score = pd.to_numeric(row.get("final_combined_score", row.get("combined_score")), errors="coerce")
    rs3 = pd.to_numeric(row.get("rs_3m_pct"), errors="coerce")
    rs6 = pd.to_numeric(row.get("rs_6m_pct"), errors="coerce")
    if stage == "Stage 2":
        if pd.notna(score) and score >= 70:
            return "Strong"
        return "Developing"
    if stage == "Stage 1":
        if pd.notna(rs3) and pd.notna(rs6) and rs3 < 0 and rs6 < 0:
            return "Cautious"
        return "Developing"
    if stage == "Stage 3":
        if pd.notna(score) and score >= 65:
            return "Cautious"
        if pd.notna(rs3) and pd.notna(rs6) and rs3 > 0 and rs6 >= 0:
            return "Cautious"
        return "Weak"
    if stage == "Stage 4":
        return "Weak"
    if pd.notna(rs3) and pd.notna(rs6) and rs3 < 0 and rs6 < 0:
        return "Weak"
    return "Developing"

def ensure_label(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_current_rank(normalize_columns(df))
    if out is None or out.empty:
        return out
    if "stage" not in out.columns:
        out["stage"] = "Not Sure"
    if "stage_classification" not in out.columns:
        out["stage_classification"] = out["stage_variant"] if "stage_variant" in out.columns else out["stage"]
    if "stage_variant" not in out.columns:
        out["stage_variant"] = out["stage_classification"]
    if "stage_display" not in out.columns:
        out["stage_display"] = out.apply(stage_display_text, axis=1)
    numeric_cols = [
        "final_combined_score", "avg_combined_score", "current_rank", "prev_rank", "rank_change",
        "combined_score_change", "change_1d_pct", "change_1w_pct", "change_1m_pct", "change_ytd_pct", "rs_rank"
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if not out.empty and "label" not in out.columns and "classification" not in out.columns:
        out["label"] = out.apply(classify_stock, axis=1)
    elif "classification" in out.columns and "label" not in out.columns:
        out["label"] = out["classification"]
    return out

def structure_score(row: pd.Series) -> int:
    stage = str(row.get("stage", ""))
    label = str(row.get("label", row.get("classification", "Developing")))
    score = pd.to_numeric(row.get("final_combined_score", row.get("avg_combined_score", row.get("combined_score"))), errors="coerce")
    rank = pd.to_numeric(row.get("current_rank"), errors="coerce")
    rank_change = pd.to_numeric(row.get("rank_change"), errors="coerce")
    value = 0.0
    value += {"Stage 1": 24, "Stage 2": 48, "Stage 3": 26, "Stage 4": 10}.get(stage, 20)
    value += {"Strong": 20, "Developing": 12, "Cautious": 6, "Weak": 0}.get(label, 8)
    if pd.notna(score):
        value += min(26, max(0, score * 0.28))
    if pd.notna(rank):
        value += max(0, 14 - min(rank, 14))
    if pd.notna(rank_change) and rank_change > 0:
        value += min(8, rank_change * 1.1)
    if bool(row.get("entered_stage_2", False)):
        value += 6
    if bool(row.get("new_weekly_breakout", False)):
        value += 5
    if bool(row.get("new_daily_breakout", False)):
        value += 4
    return int(max(0, min(100, round(value))))

def structure_category(row: pd.Series) -> str:
    stage = str(row.get("stage", ""))
    label = str(row.get("label", row.get("classification", "Developing")))
    score = structure_score(row)
    if stage == "Stage 2" and label == "Strong":
        return "Strong Structure"
    if stage == "Stage 2" and score >= 55:
        return "Developing Structure"
    if stage == "Stage 1":
        return "Emerging Structure"
    if stage == "Stage 3":
        return "Transitioning Structure"
    if stage == "Stage 4" or label == "Weak":
        return "Weak Structure"
    if label == "Cautious":
        return "Cautious Structure"
    return "Mixed Structure"

def stage_primary_label(stage: str) -> str:
    return {
        "Stage 1": "Base Phase",
        "Stage 2": "Advancing Phase",
        "Stage 3": "Transition Phase",
        "Stage 4": "Declining Phase",
    }.get(stage, stage or "Unknown")

def stage_short_description(stage: str) -> str:
    return {
        "Stage 1": "Base formation or repair in this model.",
        "Stage 2": "Advancing structure in this model.",
        "Stage 3": "Trend slowing or mixed structure.",
        "Stage 4": "Declining structure in this model.",
    }.get(stage, "Mixed structure in this model.")

def stage_classification_text(row: pd.Series) -> str:
    """Detailed stage label for cards and tables.

    The engine writes `stage_variant` / `stage_classification`, for example:
    Early Stage 2, Clean Stage 2, Stage 1 - Base/Repair, Failed Stage 2.
    This fallback keeps the dashboard safe with older output files too.
    """
    stage = str(row.get("stage", "Not Sure") or "Not Sure").strip()
    for col in ["stage_classification", "stage_variant"]:
        if col in row.index:
            val = str(row.get(col, "") or "").strip()
            if val and val.lower() not in {"nan", "none", "unknown"}:
                return val
    return stage_primary_label(stage)


def stage_display_text(row: pd.Series) -> str:
    stage = str(row.get("stage", "Not Sure") or "Not Sure").strip()
    classification = stage_classification_text(row)
    if not classification or classification == stage:
        return stage
    return f"{stage} • {classification}"

def stage_condition_text(row: pd.Series) -> str:
    stage = str(row.get("stage", ""))
    label = str(row.get("label", row.get("classification", "Developing")))
    score = pd.to_numeric(row.get("final_combined_score", row.get("avg_combined_score", row.get("combined_score"))), errors="coerce")
    rank_change = pd.to_numeric(row.get("rank_change"), errors="coerce")
    if stage == "Stage 1":
        return "Improving" if pd.notna(score) and score >= 65 else "Early"
    if stage == "Stage 2":
        if label == "Strong" and pd.notna(rank_change) and rank_change > 0:
            return "Improving"
        if label == "Strong":
            return "Stable"
        return "Developing"
    if stage == "Stage 3":
        return "Under Pressure"
    if stage == "Stage 4":
        return "Weak"
    return "Mixed"

def stock_display_label(row: pd.Series) -> str:
    company = str(row.get("Company Name", row.get("ticker", "Stock"))).strip()
    ticker = str(row.get("ticker", "")).replace(".NS", "").strip()
    return f"{company} ({ticker})" if ticker else company


def industry_icon(industry: str) -> str:
    ind = str(industry or "").lower()
    mapping = [
        (["bank", "financial", "nbfc", "insurance"], "🏦"),
        (["it", "software", "technology", "tech", "internet"], "💻"),
        (["pharma", "health", "hospital"], "💊"),
        (["auto", "automobile", "tyre"], "🚗"),
        (["metal", "steel", "mining"], "⛏️"),
        (["energy", "oil", "gas", "power", "utility"], "⚡"),
        (["fmcg", "consumer", "retail", "apparel"], "🛍️"),
        (["realty", "real estate", "construction", "cement"], "🏗️"),
        (["telecom", "media"], "📡"),
        (["chemical", "fertilizer"], "🧪"),
        (["industrial", "capital goods", "engineering"], "🏭"),
    ]
    for keys, icon in mapping:
        if any(k in ind for k in keys):
            return icon
    return "🏷️"

def score_explanation_line(row: pd.Series) -> str:
    score = structure_score(row)
    stage = str(row.get("stage", ""))
    label = str(row.get("label", row.get("classification", "Developing")))
    return f"Model Score: {score}/100 • Higher means stronger structure inside this model. It does not mean higher returns and it is not a recommendation. Current classification: {label} in {stage or 'Unknown Stage'}."

def interpretation_line(row: pd.Series) -> str:
    stage = str(row.get("stage", ""))
    mapping = {
        "Stage 1": "This stock is currently in a base-formation phase in the model.",
        "Stage 2": "This stock is currently in an advancing phase in the model.",
        "Stage 3": "This stock is currently in a transition phase in the model.",
        "Stage 4": "This stock is currently in a declining phase in the model.",
    }
    return mapping.get(stage, "This reflects the current model classification.")

def signal_summary(row: pd.Series) -> str:
    parts = []
    rank_change = pd.to_numeric(row.get("rank_change"), errors="coerce")
    if pd.notna(rank_change):
        if rank_change > 0:
            parts.append(f"Dataset rank improved by {int(rank_change)}")
        elif rank_change < 0:
            parts.append(f"Dataset rank declined by {abs(int(rank_change))}")
    if bool(row.get("new_weekly_breakout", False)):
        parts.append("Weekly breakout flag")
    elif bool(row.get("new_daily_breakout", False)):
        parts.append("Daily breakout flag")
    return " • ".join(parts[:2]) if parts else "No major new structure-change flag in the latest update."

def render_disclosure():
    st.markdown("""
<div class="disclosure">
This engine converts rule-based structure data into directional trade actions. It is still a model, not certainty. Execution quality, sizing, liquidity, slippage, and stop discipline remain your responsibility.
</div>
""", unsafe_allow_html=True)

def render_summary_card(title: str, value: str, subtitle: str):
    st.markdown(f"""
<div class="hero-card">
  <div class="kicker">{title}</div>
  <div class="big-number">{value}</div>
  <div class="muted">{subtitle}</div>
</div>
""", unsafe_allow_html=True)

def _stage_card_class(stage_raw: str) -> str:
    return {
        "Stage 1": "stage-card-1",
        "Stage 2": "stage-card-2",
        "Stage 3": "stage-card-3",
        "Stage 4": "stage-card-4",
    }.get(stage_raw, "")

def company_choices(df: pd.DataFrame):
    if df.empty:
        return {}
    tmp = df.dropna(subset=["Company Name", "ticker"]).copy()
    tmp["Company Name"] = tmp["Company Name"].astype(str).str.strip()
    tmp = tmp.sort_values(["Company Name", "ticker"], ascending=[True, True])
    tmp = tmp.drop_duplicates(subset=["Company Name"], keep="first")
    return dict(zip(tmp["Company Name"], tmp["ticker"]))

def resolve_rank_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    for col in ["current_rank", "rank", "rs_rank", "daily_rank", "weekly_rank", "final_rank", "combined_rank", "stock_rank"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")


def ensure_rank_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "current_rank" not in out.columns:
        out["current_rank"] = resolve_rank_series(out)
    else:
        out["current_rank"] = pd.to_numeric(out["current_rank"], errors="coerce")
    return out


def chart_dropdown_options(df: pd.DataFrame):
    if df.empty:
        return {}
    tmp = ensure_rank_column(df.dropna(subset=["Company Name", "ticker"]).copy())
    tmp["display_label"] = tmp.apply(stock_display_label, axis=1)
    sort_cols = [c for c in ["current_rank", "display_label"] if c in tmp.columns]
    asc = [True, True][:len(sort_cols)]
    tmp = tmp.sort_values(sort_cols, ascending=asc, na_position="last").drop_duplicates(subset=["ticker"], keep="first")
    return dict(zip(tmp["display_label"], tmp["ticker"]))

def resolve_chart_path(charts_dir: str, ticker: str, suffix: str):
    chart_dir = Path(charts_dir)
    if not chart_dir.exists():
        return None
    ticker = str(ticker).strip()
    raw = ticker.replace(".NS", "")
    candidates = []
    for candidate in {
        ticker, raw,
        ticker.replace(".", "_"), raw.replace(".", "_"),
        ticker.replace("&", "_"), raw.replace("&", "_"),
        ticker.replace("&", "AND"), raw.replace("&", "AND"),
        ticker.replace("&", "and"), raw.replace("&", "and"),
        re.sub(r"[^A-Za-z0-9]+", "_", ticker),
        re.sub(r"[^A-Za-z0-9]+", "_", raw),
        re.sub(r"[^A-Za-z0-9]+", "", ticker),
        re.sub(r"[^A-Za-z0-9]+", "", raw),
    }:
        if candidate:
            candidates.append(candidate + suffix)
    for name in candidates:
        path = chart_dir / name
        if path.exists():
            return path
    raw_key = re.sub(r"[^A-Za-z0-9]+", "", raw).lower()
    for path in chart_dir.glob(f"*{suffix}"):
        stem_key = re.sub(r"[^A-Za-z0-9]+", "", path.stem).lower()
        if raw_key and raw_key in stem_key:
            return path
    return None

@st.cache_data(show_spinner=False)
def load_image_bytes(path: str, mtime_ns: int) -> bytes:
    return Path(path).read_bytes()

def safe_image_bytes(path):
    if not path or not path.exists():
        return None
    return load_image_bytes(str(path), path.stat().st_mtime_ns)

def stage_count_summary(combined_df: pd.DataFrame):
    counts = combined_df["stage"].value_counts() if "stage" in combined_df.columns else pd.Series(dtype=int)
    return {
        "Stage 1": int(counts.get("Stage 1", 0)),
        "Stage 2": int(counts.get("Stage 2", 0)),
        "Stage 3": int(counts.get("Stage 3", 0)),
        "Stage 4": int(counts.get("Stage 4", 0)),
    }

def stage2_count_by_industry(combined_df: pd.DataFrame) -> pd.DataFrame:
    if combined_df.empty or "Industry" not in combined_df.columns or "stage" not in combined_df.columns:
        return pd.DataFrame(columns=["Industry", "Stage 2 Stocks"])
    return combined_df.groupby("Industry", dropna=True)["stage"].apply(lambda s: int((s == "Stage 2").sum())).reset_index(name="Stage 2 Stocks")

def build_today_changes(changes_df: pd.DataFrame, industry_changes_df: pd.DataFrame):
    summary = {"New Strong": 0, "Entered Stage 2": 0, "New Breakouts": 0}
    if changes_df.empty:
        return pd.DataFrame(), summary

    df = changes_df.copy()
    if "label" not in df.columns:
        df["label"] = df.apply(classify_stock, axis=1)

    if "entered_stage_2" in df.columns:
        summary["Entered Stage 2"] = int(df["entered_stage_2"].fillna(False).sum())
    if "new_daily_breakout" in df.columns:
        summary["New Breakouts"] += int(df["new_daily_breakout"].fillna(False).sum())
    if "new_weekly_breakout" in df.columns:
        summary["New Breakouts"] += int(df["new_weekly_breakout"].fillna(False).sum())

    def what_changed(row):
        parts = []
        if bool(row.get("entered_stage_2", False)):
            parts.append("Moved into Stage 2")
        if bool(row.get("new_weekly_breakout", False)):
            parts.append("Weekly breakout flag")
        if bool(row.get("new_daily_breakout", False)):
            parts.append("Daily breakout flag")
        rc = pd.to_numeric(row.get("rank_change"), errors="coerce")
        if pd.notna(rc) and rc > 0:
            parts.append(f"Dataset rank improved by {int(rc)}")
        elif pd.notna(rc) and rc < 0:
            parts.append(f"Dataset rank declined by {abs(int(rc))}")
        return " • ".join(parts[:3]) if parts else "No major new structure-change flag in the latest update."

    df["what_changed"] = df.apply(what_changed, axis=1)

    df["change_priority"] = 0
    if "entered_stage_2" in df.columns:
        df["change_priority"] += df["entered_stage_2"].fillna(False).astype(int) * 100
    if "new_weekly_breakout" in df.columns:
        df["change_priority"] += df["new_weekly_breakout"].fillna(False).astype(int) * 70
    if "new_daily_breakout" in df.columns:
        df["change_priority"] += df["new_daily_breakout"].fillna(False).astype(int) * 50
    if "label" in df.columns:
        df["change_priority"] += (df["label"].astype(str) == "Strong").astype(int) * 20

    if "rank_change" in df.columns:
        rank_change_num = pd.to_numeric(df["rank_change"], errors="coerce").fillna(0)
        df["change_priority"] += rank_change_num.clip(lower=0, upper=25).astype(int) * 3
        df["rank_change_num"] = rank_change_num
    else:
        df["rank_change_num"] = 0

    sort_cols = ["change_priority", "rank_change_num"]
    ascending = [False, False]
    if "Company Name" in df.columns:
        sort_cols.append("Company Name")
        ascending.append(True)

    top_changed = df.sort_values(sort_cols, ascending=ascending, na_position="last").head(12).copy()
    return top_changed, summary

def build_alert_candidates(combined_df: pd.DataFrame, changes_df: pd.DataFrame) -> pd.DataFrame:
    if combined_df.empty:
        return pd.DataFrame()
    alerts = []
    changes_lookup = changes_df.copy() if not changes_df.empty else pd.DataFrame()
    if not changes_lookup.empty and "ticker" in changes_lookup.columns:
        changes_lookup["ticker"] = changes_lookup["ticker"].astype(str).str.strip()
        changes_lookup = changes_lookup.set_index("ticker", drop=False)

    for _, row in combined_df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        cr = changes_lookup.loc[ticker] if (not changes_lookup.empty and ticker in changes_lookup.index) else None
        alert_type = ""
        reason = ""
        if cr is not None and bool(cr.get("entered_stage_2", False)):
            alert_type = "Stage transition event"
            reason = "The stock moved into Stage 2 in this framework."
        elif cr is not None and bool(cr.get("new_weekly_breakout", False)):
            alert_type = "Weekly structure breakout event"
            reason = "A fresh weekly breakout flag was detected."
        elif cr is not None and bool(cr.get("new_daily_breakout", False)):
            alert_type = "Daily structure breakout event"
            reason = "A fresh daily breakout flag was detected."
        else:
            rank_change = pd.to_numeric(row.get("rank_change"), errors="coerce")
            if pd.notna(rank_change) and abs(rank_change) >= 5:
                alert_type = "Relative position change"
                direction = "improved" if rank_change > 0 else "declined"
                reason = f"Dataset rank {direction} by {abs(int(rank_change))} places."
        if alert_type:
            item = row.copy()
            item["alert_type"] = alert_type
            item["alert_reason"] = reason
            alerts.append(item)

    if not alerts:
        return pd.DataFrame()
    out = pd.DataFrame(alerts)
    if "Company Name" in out.columns:
        out = out.sort_values(["Company Name"], ascending=[True])
    return out.head(20)


def build_simple_rank_map(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    ticker_col = None
    for cand in ["ticker", "Ticker", "symbol", "Symbol"]:
        if cand in df.columns:
            ticker_col = cand
            break
    if ticker_col is None:
        return {}
    rank_col = None
    for cand in ["current_rank", "rank", "rs_rank", "daily_rank", "weekly_rank", "final_rank", "combined_rank", "stock_rank"]:
        if cand in df.columns:
            rank_col = cand
            break
    if rank_col is None:
        return {}
    temp = df[[ticker_col, rank_col]].copy()
    temp[ticker_col] = temp[ticker_col].astype(str).str.strip()
    temp[rank_col] = pd.to_numeric(temp[rank_col], errors="coerce")
    temp = temp.dropna(subset=[ticker_col, rank_col]).drop_duplicates(subset=[ticker_col], keep="first")
    out = {}
    for _, r in temp.iterrows():
        t = str(r[ticker_col]).strip()
        out[t] = str(int(r[rank_col]))
        out[t.replace(".NS", "")] = str(int(r[rank_col]))
    return out


def rank_lookup(df: pd.DataFrame, ticker: str, preferred_cols: list) -> str:
    if df.empty:
        return ""
    work = df.copy()
    ticker_col = None
    for cand in ["ticker", "Ticker", "symbol", "Symbol"]:
        if cand in work.columns:
            ticker_col = cand
            break
    if ticker_col is None:
        return ""
    work[ticker_col] = work[ticker_col].astype(str).str.strip()
    ticker_norm = str(ticker).strip()
    match = work[work[ticker_col] == ticker_norm]
    if match.empty:
        match = work[work[ticker_col].str.replace(".NS", "", regex=False) == ticker_norm.replace(".NS", "")]
    if match.empty:
        return ""
    row = match.iloc[0]
    for col in preferred_cols + ["current_rank", "rank", "rs_rank", "daily_rank", "weekly_rank", "final_rank", "combined_rank", "stock_rank"]:
        if col in match.columns:
            val = pd.to_numeric(row.get(col), errors="coerce")
            if pd.notna(val):
                return str(int(val))
    return ""

def get_industry_portfolio_options(industry_df: pd.DataFrame, combined_df: pd.DataFrame, limit: int = 21) -> list:
    if "Industry" in combined_df.columns:
        industries = sorted(set(combined_df["Industry"].dropna().astype(str).str.strip().tolist()))
        return industries[:limit]
    if not industry_df.empty and "Industry" in industry_df.columns:
        industries = sorted(set(industry_df["Industry"].dropna().astype(str).str.strip().tolist()))
        return industries[:limit]
    return []


def market_tone(regime_df: pd.DataFrame, combined_df: pd.DataFrame) -> str:
    if not regime_df.empty and "regime_label" in regime_df.columns:
        label = str(regime_df.iloc[0]["regime_label"])
        return {"risk_on": "Risk On", "mixed": "Mixed", "risk_off": "Risk Off"}.get(label, "Mixed")
    strong_count = int((combined_df["label"] == "Strong").sum()) if not combined_df.empty and "label" in combined_df.columns else 0
    if strong_count >= 15:
        return "Risk On"
    if strong_count >= 6:
        return "Mixed"
    return "Risk Off"


def market_tone_text(label: str) -> str:
    return {
        "Risk On": "More names are participating in advancing structures.",
        "Mixed": "Participation is selective. Some names are strong while others are weak.",
        "Risk Off": "Fewer names are in advancing structures and more are in repair or decline phases.",
    }.get(label, "This is a neutral descriptive view of the current dataset.")


def top_industry_text(industry_df: pd.DataFrame, n: int = 3) -> str:
    if industry_df.empty or "Industry" not in industry_df.columns:
        return "Not available"
    return ", ".join(industry_df.head(n)["Industry"].astype(str).tolist())


def dedupe_names(names: list, limit: int = MAX_PORTFOLIO_STOCKS) -> list:
    out, seen = [], set()
    for name in names:
        if pd.isna(name):
            continue
        name = str(name).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
        if limit is not None and len(out) >= limit:
            break
    return out


def parse_ticker_input(raw_text: str) -> list:
    if not raw_text:
        return []
    parts = re.split(r"[\s,;\n\t]+", str(raw_text).strip())
    out = []
    seen = set()
    for part in parts:
        ticker = part.strip().upper()
        if not ticker:
            continue
        ticker_ns = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
        for candidate in [ticker_ns, ticker]:
            if candidate and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
    return out


def names_from_tickers(tickers: list, source_df: pd.DataFrame) -> list:
    if source_df is None or source_df.empty or not tickers:
        return []
    work = source_df.copy()
    if "ticker" not in work.columns or "Company Name" not in work.columns:
        return []
    work["ticker"] = work["ticker"].astype(str).str.strip()
    work["_ticker_raw"] = work["ticker"].str.replace(".NS", "", regex=False).str.upper()
    out = []
    seen = set()
    for t in tickers:
        t_norm = str(t).strip().upper()
        t_raw = t_norm.replace(".NS", "")
        match = work[(work["ticker"].str.upper() == t_norm) | (work["_ticker_raw"] == t_raw)]
        if not match.empty:
            name = str(match.iloc[0]["Company Name"]).strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


def get_prebuilt_portfolio(name: str, combined: pd.DataFrame, changes: pd.DataFrame, industries: list) -> list:
    source_df = DECISION_DF if 'DECISION_DF' in globals() and DECISION_DF is not None and not DECISION_DF.empty else combined
    ranked = sort_watchlist_view(source_df, selected_watchlist=name).copy()
    names = []
    if name in {"Buy", "Tactical Buy", "Watch for Long", "Short", "Tactical Short", "Watch for Short", "No Trade"} and "action" in ranked.columns:
        names = ranked.loc[ranked["action"].astype(str) == name, "Company Name"].dropna().tolist()
    elif name in {"Stage 1", "Stage 2", "Stage 2 Failed", "Stage 3", "Stage 4", "Not Sure"}:
        names = ranked.loc[ranked["stage"].astype(str) == name, "Company Name"].dropna().tolist()
    elif str(name).startswith("Class: ") and "stage_classification" in ranked.columns:
        cls = str(name).replace("Class: ", "", 1)
        names = ranked.loc[ranked["stage_classification"].astype(str) == cls, "Company Name"].dropna().tolist()
    elif name in {"Strong", "Developing", "Cautious", "Weak"}:
        names = ranked.loc[ranked["label"] == name, "Company Name"].dropna().tolist()
    elif name in industries:
        names = ranked.loc[ranked["Industry"].astype(str).str.strip() == name, "Company Name"].dropna().tolist()
    return dedupe_names(names, limit=MAX_PORTFOLIO_STOCKS)


def render_distribution(stage_counts: dict):
    total = max(1, sum(stage_counts.values()))
    colors = {
        "Stage 1": "#4f7dff",
        "Stage 2": "#16c5c5",
        "Stage 3": "#d4a017",
        "Stage 4": "#aa50b4",
    }
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("**Market distribution**")
    for key in ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]:
        value = int(stage_counts.get(key, 0))
        pct = round((value / total) * 100)
        st.markdown(
            f"""
            <div class="dist-row">
              <div class="dist-label">{key}</div>
              <div class="dist-bar-wrap"><div class="dist-bar" style="width:{pct}%; background:{colors[key]};"></div></div>
              <div class="dist-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

def market_bias_score(stage_counts: dict, regime_df: pd.DataFrame) -> int:
    total = max(1, sum(stage_counts.values()))
    stage2 = int(stage_counts.get("Stage 2", 0))
    stage4 = int(stage_counts.get("Stage 4", 0))
    stage3 = int(stage_counts.get("Stage 3", 0))
    score = 50
    score += round((stage2 / total) * 60)
    score -= round((stage4 / total) * 55)
    score -= round((stage3 / total) * 18)
    if not regime_df.empty and "regime_label" in regime_df.columns:
        regime_label = str(regime_df.iloc[0].get("regime_label", "")).lower().strip()
        score += {"risk_on": 12, "mixed": 0, "risk_off": -12}.get(regime_label, 0)
    return int(max(0, min(100, score)))

def market_action_bias(stage_counts: dict, regime_df: pd.DataFrame) -> str:
    score = market_bias_score(stage_counts, regime_df)
    if score >= 65:
        return "Long Bias"
    if score <= 38:
        return "Short Bias"
    return "Two-Sided"

def market_action_text(stage_counts: dict, regime_df: pd.DataFrame) -> str:
    bias = market_action_bias(stage_counts, regime_df)
    score = market_bias_score(stage_counts, regime_df)
    if bias == "Long Bias":
        return f"Market internals favour long setups. Long score {score}/100. Prioritize Stage 2 strength, rising ranks, and leading industries."
    if bias == "Short Bias":
        return f"Market internals favour short setups. Short score {100-score}/100. Prioritize Stage 4 weakness, failed rebounds, and weak industries."
    return f"Market internals are mixed. Bias score {score}/100. Trade smaller, demand industry confirmation, and keep both long and short lists ready."

def build_industry_support_map(industry_df: pd.DataFrame, combined_df: pd.DataFrame) -> dict:
    support = {}
    if combined_df.empty or "Industry" not in combined_df.columns:
        return support
    stage2_counts = stage2_count_by_industry(combined_df)
    stage2_map = {}
    if not stage2_counts.empty:
        stage2_map = dict(zip(stage2_counts["Industry"].astype(str), stage2_counts["Stage 2 Stocks"]))
    for ind in combined_df["Industry"].dropna().astype(str).unique().tolist():
        support[ind] = {"industry_score": 50, "industry_rank": None, "stage2_count": int(stage2_map.get(ind, 0)), "industry_view": "Neutral"}
    if not industry_df.empty and "Industry" in industry_df.columns:
        temp = industry_df.copy()
        if "current_rank" in temp.columns:
            temp["current_rank"] = pd.to_numeric(temp["current_rank"], errors="coerce")
        if "avg_combined_score" in temp.columns:
            temp["avg_combined_score"] = pd.to_numeric(temp["avg_combined_score"], errors="coerce")
        for _, row in temp.iterrows():
            ind = str(row.get("Industry", "")).strip()
            if not ind:
                continue
            rank = pd.to_numeric(row.get("current_rank"), errors="coerce")
            avg_score = pd.to_numeric(row.get("avg_combined_score"), errors="coerce")
            base = 50
            if pd.notna(avg_score):
                base += max(-20, min(20, int(round((avg_score - 50) * 0.6))))
            if pd.notna(rank):
                base += max(-18, min(18, 22 - int(rank)))
            base += min(18, int(stage2_map.get(ind, 0)) * 2)
            base = int(max(0, min(100, base)))
            view = "Neutral"
            if base >= 68:
                view = "Strong Tailwind"
            elif base >= 56:
                view = "Positive"
            elif base <= 35:
                view = "Weak"
            elif base <= 45:
                view = "Fragile"
            support[ind] = {
                "industry_score": base,
                "industry_rank": (int(rank) if pd.notna(rank) else None),
                "stage2_count": int(stage2_map.get(ind, 0)),
                "industry_view": view,
            }
    return support

def boolish(v) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}

def compute_trade_action(row: pd.Series, market_bias: str, market_bias_score_value: int, industry_support_map: dict) -> pd.Series:
    stage = str(row.get("stage", "")).strip()
    label = str(row.get("label", row.get("classification", "Developing"))).strip()
    score = pd.to_numeric(row.get("final_combined_score", row.get("avg_combined_score", row.get("combined_score"))), errors="coerce")
    rank = pd.to_numeric(row.get("current_rank"), errors="coerce")
    rank_change = pd.to_numeric(row.get("rank_change"), errors="coerce")
    rs3 = pd.to_numeric(row.get("rs_3m_pct"), errors="coerce")
    rs6 = pd.to_numeric(row.get("rs_6m_pct"), errors="coerce")
    price_1d = pd.to_numeric(row.get("change_1d_pct"), errors="coerce")
    price_1w = pd.to_numeric(row.get("change_1w_pct"), errors="coerce")
    industry_name = str(row.get("Industry", "")).strip()
    industry_meta = industry_support_map.get(industry_name, {"industry_score": 50, "industry_rank": None, "stage2_count": 0, "industry_view": "Neutral"})

    long_score = 0
    short_score = 0
    reasons_long = []
    reasons_short = []

    long_score += {"Stage 1": 18, "Stage 2": 38, "Stage 3": 10, "Stage 4": 0}.get(stage, 8)
    short_score += {"Stage 1": 2, "Stage 2": 0, "Stage 3": 24, "Stage 4": 40}.get(stage, 8)
    long_score += {"Strong": 22, "Developing": 12, "Cautious": 5, "Weak": 0}.get(label, 8)
    short_score += {"Strong": 0, "Developing": 4, "Cautious": 14, "Weak": 22}.get(label, 8)

    if pd.notna(score):
        long_score += max(0, min(24, int(round((score - 48) * 0.75))))
        short_score += max(0, min(24, int(round((55 - score) * 0.75))))
    if pd.notna(rank):
        long_score += max(0, 16 - min(int(rank), 16))
        short_score += max(0, min(int(rank) - 18, 16))
    if pd.notna(rank_change):
        if rank_change > 0:
            long_score += min(12, int(rank_change * 1.4))
            reasons_long.append(f"rank improving by {int(rank_change)}")
        elif rank_change < 0:
            short_score += min(12, int(abs(rank_change) * 1.4))
            reasons_short.append(f"rank falling by {abs(int(rank_change))}")
    if pd.notna(rs3) and pd.notna(rs6):
        if rs3 > 0 and rs6 > 0:
            long_score += 12
            reasons_long.append("3M and 6M relative strength positive")
        if rs3 < 0 and rs6 < 0:
            short_score += 12
            reasons_short.append("3M and 6M relative strength negative")
    if pd.notna(price_1w):
        if price_1w > 0:
            long_score += min(8, int(round(price_1w)))
        elif price_1w < 0:
            short_score += min(8, int(round(abs(price_1w))))
    if pd.notna(price_1d):
        if price_1d > 0:
            long_score += min(4, int(round(price_1d)))
        elif price_1d < 0:
            short_score += min(4, int(round(abs(price_1d))))

    if boolish(row.get("entered_stage_2", False)):
        long_score += 14
        reasons_long.append("fresh move into Stage 2")
    if boolish(row.get("new_weekly_breakout", False)):
        long_score += 12
        reasons_long.append("weekly breakout")
    if boolish(row.get("new_daily_breakout", False)):
        long_score += 8
        reasons_long.append("daily breakout")

    if stage == "Stage 4":
        reasons_short.append("Stage 4 decline structure")
    elif stage == "Stage 3":
        reasons_short.append("transition structure vulnerable to breakdown")
    elif stage == "Stage 2":
        reasons_long.append("Stage 2 advancing structure")
    elif stage == "Stage 1":
        reasons_long.append("base-building structure")

    industry_score = int(industry_meta.get("industry_score", 50))
    long_score += max(-8, min(14, int(round((industry_score - 50) * 0.35))))
    short_score += max(-8, min(14, int(round((50 - industry_score) * 0.35))))

    if market_bias == "Long Bias":
        long_score += 8
        short_score -= 6
    elif market_bias == "Short Bias":
        short_score += 8
        long_score -= 6

    long_score = int(max(0, min(100, long_score)))
    short_score = int(max(0, min(100, short_score)))

    action = "No Trade"
    action_confidence = max(long_score, short_score)
    setup_quality = "Low"
    if action_confidence >= 75:
        setup_quality = "High"
    elif action_confidence >= 60:
        setup_quality = "Medium"

    if long_score >= short_score + 10 and long_score >= 58:
        action = "Buy" if market_bias != "Short Bias" else "Tactical Buy"
    elif short_score >= long_score + 10 and short_score >= 58:
        action = "Short" if market_bias != "Long Bias" else "Tactical Short"
    elif long_score >= 50 and stage in {"Stage 1", "Stage 2"}:
        action = "Watch for Long"
    elif short_score >= 50 and stage in {"Stage 3", "Stage 4"}:
        action = "Watch for Short"

    trade_side = "Long" if "Buy" in action or action == "Watch for Long" else ("Short" if "Short" in action or action == "Watch for Short" else "Neutral")
    stop_framework = "Not defined"
    if trade_side == "Long":
        stop_framework = "Below recent swing low / failed breakout / Stage change back to 1 or 3 weakness"
    elif trade_side == "Short":
        stop_framework = "Above recent swing high / failed breakdown / sharp rank recovery"

    rationale = reasons_long[:3] if trade_side == "Long" else reasons_short[:3]
    if not rationale:
        rationale = ["structure not strong enough for a clean directional trade"]

    return pd.Series({
        "market_bias": market_bias,
        "market_bias_score": market_bias_score_value,
        "industry_score": industry_score,
        "industry_view": industry_meta.get("industry_view", "Neutral"),
        "industry_rank": industry_meta.get("industry_rank"),
        "industry_stage2_count": industry_meta.get("stage2_count", 0),
        "long_score": long_score,
        "short_score": short_score,
        "action": action,
        "trade_side": trade_side,
        "action_confidence": action_confidence,
        "setup_quality": setup_quality,
        "rationale": " • ".join(rationale),
        "stop_framework": stop_framework,
    })

def build_decision_engine_table(combined_df: pd.DataFrame, changes_df: pd.DataFrame, industry_df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    if combined_df.empty:
        return pd.DataFrame()
    df = combined_df.copy()
    if "current_rank" not in df.columns:
        for alt in ["rank", "rs_rank", "daily_rank", "weekly_rank", "final_rank", "combined_rank", "stock_rank"]:
            if alt in df.columns:
                df["current_rank"] = pd.to_numeric(df[alt], errors="coerce")
                break
        if "current_rank" not in df.columns:
            df["current_rank"] = pd.NA
    if not changes_df.empty and "ticker" in changes_df.columns:
        change_cols = [c for c in ["ticker", "entered_stage_2", "new_weekly_breakout", "new_daily_breakout"] if c in changes_df.columns]
        df = df.merge(changes_df[change_cols].drop_duplicates(subset=["ticker"]), on="ticker", how="left", suffixes=("", "_chg"))
    stage_counts = stage_count_summary(df)
    bias = market_action_bias(stage_counts, regime_df)
    bias_score = market_bias_score(stage_counts, regime_df)
    industry_support_map = build_industry_support_map(industry_df, df)
    decision_cols = df.apply(lambda row: compute_trade_action(row, bias, bias_score, industry_support_map), axis=1)
    df = pd.concat([df, decision_cols], axis=1)
    for col, default in {
        "industry_score": 0,
        "action_confidence": 0,
        "long_score": 0,
        "short_score": 0,
    }.items():
        if col not in df.columns:
            df[col] = default
    df["decision_priority"] = df[["long_score", "short_score"]].max(axis=1)
    return df

def safe_sort_values(df: pd.DataFrame, by: list, ascending: list):
    usable_by, usable_asc = [], []
    for col, asc in zip(by, ascending):
        if col in df.columns:
            usable_by.append(col)
            usable_asc.append(asc)
    if not usable_by:
        return df
    return df.sort_values(usable_by, ascending=usable_asc, na_position="last")

def top_trade_candidates(decision_df: pd.DataFrame, side: str, top_n: int = 8) -> pd.DataFrame:
    if decision_df.empty:
        return pd.DataFrame()

    if side == "Long":
        candidates = decision_df[decision_df["trade_side"] == "Long"].copy()
        if candidates.empty:
            return candidates
        candidates["_rank_fallback"] = pd.to_numeric(candidates.get("current_rank"), errors="coerce") if "current_rank" in candidates.columns else float("nan")
        return safe_sort_values(
            candidates,
            ["long_score", "action_confidence", "industry_score", "_rank_fallback"],
            [False, False, False, True],
        ).drop(columns=["_rank_fallback"], errors="ignore").head(top_n)

    candidates = decision_df[decision_df["trade_side"] == "Short"].copy()
    if candidates.empty:
        return candidates
    candidates["_rank_fallback"] = pd.to_numeric(candidates.get("current_rank"), errors="coerce") if "current_rank" in candidates.columns else float("nan")
    return safe_sort_values(
        candidates,
        ["short_score", "action_confidence", "industry_score", "_rank_fallback"],
        [False, False, True, True],
    ).drop(columns=["_rank_fallback"], errors="ignore").head(top_n)

def render_trade_card(row: pd.Series):
    side = str(row.get("trade_side", "Neutral"))
    action = str(row.get("action", "No Trade"))
    score = int(row.get("long_score", 0) if side == "Long" else row.get("short_score", 0))
    title = stock_display_label(row)
    industry_name = str(row.get("Industry", "")).strip()
    industry_view = str(row.get("industry_view", "Neutral"))
    rank_val = pd.to_numeric(row.get("current_rank"), errors="coerce")
    rank_txt = f"Dataset Rank {int(rank_val)}" if pd.notna(rank_val) else f"Dataset Rank {get_stock_rank(row.get('ticker', ''))}"
    mini_text = build_mini_signal_text(row)
    mini_html = f"<div class='small-note' style='margin-top:0.15rem;'>{mini_text}</div>" if mini_text else ""
    st.markdown(f"""
<div class='stock-card {_stage_card_class(str(row.get("stage", "")))}'>
  <div style='display:flex; justify-content:space-between; align-items:flex-start; gap:0.6rem;'>
    <div style='min-width:0;'>
      <div class='stock-title'>{title}</div>
      <div class='meta-line'>{action} • {side} • Score {score}/100</div>
      <div class='stock-subtitle'>{str(row.get("rationale", ""))}</div>
      <div class='small-note' style='margin-top:0.2rem;'>{industry_icon(industry_name)} {industry_name or 'Unknown industry'} • {industry_view}</div>
      {mini_html}
      <div class='small-note' style='margin-top:0.15rem;'>Stop framework: {str(row.get("stop_framework", "Not defined"))}</div>
    </div>
    <div style='display:flex; flex-direction:column; align-items:flex-end; gap:0.05rem;'>
      <div class='status-pill {'status-strong' if side == 'Long' else 'status-weak' if side == 'Short' else 'status-cautious'}'>{action}</div>
      <div class='rank-text'>{rank_txt}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

def decision_summary_stats(decision_df: pd.DataFrame) -> dict:
    if decision_df.empty:
        return {"buy": 0, "short": 0, "watch": 0, "no_trade": 0}
    actions = decision_df["action"].astype(str)
    return {
        "buy": int(actions.str.contains("Buy", regex=False).sum()),
        "short": int(actions.str.contains("Short", regex=False).sum()),
        "watch": int(actions.str.contains("Watch", regex=False).sum()),
        "no_trade": int((actions == "No Trade").sum()),
    }


def portfolio_health_summary(current: pd.DataFrame):
    if current is None or current.empty:
        return "Empty", "No stocks are currently in this watchlist."
    stage_counts = current["stage"].value_counts() if "stage" in current.columns else pd.Series(dtype=int)
    total = len(current)
    stage2 = int(stage_counts.get("Stage 2", 0))
    stage1 = int(stage_counts.get("Stage 1", 0))
    stage3 = int(stage_counts.get("Stage 3", 0))
    stage4 = int(stage_counts.get("Stage 4", 0))
    title = "Mixed composition"
    if stage2 >= max(3, round(total * 0.45)):
        title = "Advancing-heavy mix"
    elif stage4 >= max(2, round(total * 0.30)):
        title = "Transition or decline-heavy mix"
    elif stage1 >= max(2, round(total * 0.30)):
        title = "Base-heavy mix"
    text = f"Out of {total} stocks, {stage1} are in Stage 1, {stage2} are in Stage 2, {stage3} are in Stage 3, and {stage4} are in Stage 4."
    return title, text


def render_stock_detail(row):
    stock_name = str(row.get("Company Name", row.get("stock_name", "")) or "")
    ticker = str(row.get("ticker", "") or "")
    stage = str(row.get("stage", "") or "")
    stage_class = stage_classification_text(row)
    stage_reason = str(row.get("stage_reason", "") or "").strip()
    st.markdown("#### Stock detail")
    st.markdown(f"**{stock_name} ({ticker})**")
    st.caption(f"{stage_display_text(row)} · {structure_category(row)}")
    meta = []
    rank_val = pd.to_numeric(row.get("current_rank"), errors="coerce")
    if pd.notna(rank_val):
        meta.append(f"Dataset Rank: {int(rank_val)}")
    meta.append(f"Model Score: {structure_score(row)}")
    st.write(" • ".join(meta))
    col1, col2, col3 = st.columns(3)
    with col1:
        current_desc = interpretation_line(row)
        if stage_class:
            current_desc += f"<br><br><b>Stage classification:</b> {stage_class}"
        if stage_reason:
            current_desc += f"<br><span class='small-note'>{stage_reason}</span>"
        st.markdown('<div class="info-card"><b>Current model description</b><br>' + current_desc + '</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-card"><b>Recent structure-change flags</b><br>' + signal_summary(row) + '</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="info-card"><b>Industry</b><br>' + str(row.get("Industry", "Not available")) + '</div>', unsafe_allow_html=True)

def card(row: pd.Series, pct=None, use_stage_color=False, show_change_text: str = "", stock_rank: str = "n/a", action_label: str = ""):
    label = row.get("label", row.get("classification", "Developing"))
    style = LABELS.get(label, LABELS["Developing"])
    stage_raw = str(row.get("stage", "Unknown"))
    stage_label = stage_primary_label(stage_raw)
    stage_class = stage_classification_text(row)
    stage_desc = stage_short_description(stage_raw)
    stage_condition = stage_condition_text(row)
    stage_meta = f"{stage_raw} • {stage_class}" if stage_class and stage_class != stage_raw else f"{stage_raw} • {stage_label}"
    display_name = stock_display_label(row)
    structure = structure_category(row)
    score = structure_score(row)
    interpret = interpretation_line(row)
    signals = signal_summary(row)
    signals_html = f"<div class='small-note' style='margin-top:0.15rem;'>{signals}</div>" if signals and signals != "No major new structure-change flag in the latest update." else ""
    mini_text = build_mini_signal_text(row)
    mini_html = f"<div class='small-note' style='margin-top:0.15rem;'>{mini_text}</div>" if mini_text else ""
    industry_name = str(row.get("Industry", "")).strip()
    industry_with_icon = f"{industry_icon(industry_name)} {industry_name}" if industry_name else "Not available"

    classes = []
    if use_stage_color:
        stage_cls = _stage_card_class(stage_raw)
        if stage_cls:
            classes.append(stage_cls)

    rank_val = pd.to_numeric(row.get("current_rank"), errors="coerce")
    row_ticker = str(row.get("ticker", "")).strip()
    global_rank = GLOBAL_RANK_MAP.get(row_ticker)
    if global_rank is None and row_ticker:
        global_rank = GLOBAL_RANK_MAP.get(row_ticker.replace(".NS", ""))
    resolved_rank = (
        str(int(rank_val)) if pd.notna(rank_val)
        else str(int(global_rank)) if global_rank is not None
        else (str(stock_rank).strip() if str(stock_rank).strip() else "n/a")
    )
    rank_change_val = pd.to_numeric(row.get("rank_change"), errors="coerce")
    rank_change_html = ""
    if pd.notna(rank_change_val) and rank_change_val != 0:
        direction = "↑" if rank_change_val > 0 else "↓"
        rank_change_html = f"<div class='small-note' style='margin-top:0.1rem;'>Dataset Rank Change {direction} {abs(int(rank_change_val))}</div>"

    change_html = ""
    if pct is not None:
        cls = "change-badge-up" if pct > 0 else "change-badge-down"
        change_html = f"<div class='{cls}'>{pct:+.2f}%</div>"

    extra_change = f"<div class='change-text'>{show_change_text}</div>" if show_change_text else ""
    class_attr = " ".join(classes)
    status_html = f"<div class='status-pill {style['css']}'>{label}</div>"
    action_html = f"<div class='status-pill status-cautious' style='margin-top:0.2rem;'>{action_label}</div>" if action_label else ""
    structure_html = f"<div class='structure-pill'>{structure} · Model Score {score}/100</div>"
    rank_html = f"<div class='rank-text'>Dataset Rank {resolved_rank}</div>"

    html = (
        f"<div class='stock-card {class_attr}'>"
        f"<div style='display:flex; justify-content:space-between; align-items:flex-start; gap:0.55rem;'>"
        f"<div style='min-width:0;'>"
        f"<div class='stock-title'>{display_name}</div>"
        f"<div class='meta-line'>{stage_meta} • {stage_condition}</div>"
        f"<div class='stock-subtitle'>{interpret}</div>"
        f"{structure_html}"
        f"<div class='small-note'>Higher model score means stronger structure inside this model. It is not a recommendation.</div>"
        f"<div class='small-note' style='margin-top:0.15rem;'>{stage_desc}</div>"
        f"<div class='small-note' style='margin-top:0.2rem;'>{industry_with_icon}</div>"
        f"{signals_html}{mini_html}"
        f"</div>"
        f"<div style='display:flex; flex-direction:column; align-items:flex-end; gap:0.05rem;'>"
        f"{status_html}{action_html}{rank_html}{rank_change_html}{change_html}"
        f"</div>"
        f"</div>"
        f"{extra_change}"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


FO_COLUMN_CANDIDATES = [
    "F&O", "FNO", "FO", "F_AND_O", "F and O", "F&O Stock", "F&O Stocks",
    "is_FO", "is_fo", "is_fno", "is_f_and_o", "fno", "fo", "f_o", "FnO", "NSE_FO"
]


def find_fo_column(df: pd.DataFrame) -> str | None:
    """Return the first F&O marker column found in the dataframe."""
    if df is None or df.empty:
        return None
    direct = {str(c).strip(): c for c in df.columns}
    lower = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for cand in FO_COLUMN_CANDIDATES:
        if cand in direct:
            return direct[cand]
        key = cand.strip().lower().replace(" ", "_")
        if key in lower:
            return lower[key]
    return None


def normalize_fo_value(value) -> bool:
    """Convert common F&O flag values into True/False."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t", "fo", "f&o", "fno", "f and o", "f_and_o"}


def normalize_stock_symbol(value) -> str:
    """Normalize NSE/Yahoo ticker variants to one comparable key."""
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = text.replace(".NS", "")
    return text


def add_normalized_fo_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create a stable is_fo_stock column only when an F&O column is present."""
    if df is None or df.empty:
        return df
    out = df.copy()
    fo_col = find_fo_column(out)
    if fo_col is None:
        return out
    out["is_fo_stock"] = out[fo_col].apply(normalize_fo_value)
    return out


def load_fo_symbol_set_from_universe() -> set:
    """Load F&O symbols from universe files, then use it to enrich output CSVs.

    Most dashboard output files do not carry the universe F&O column. This lookup
    prevents the dashboard from going blank when the F&O toggle is switched on.
    """
    candidates = [
        Path("universe.csv"),
        Path("universe(1).csv"),
        Path("universe_with_FO.csv"),
        Path("universe_with_full_FO.csv"),
        Path("outputs/universe.csv"),
        Path("outputs/universe_clean.csv"),
        Path("outputs/universe_with_FO.csv"),
        Path("outputs/universe_with_full_FO.csv"),
    ]
    symbols = set()
    for path in candidates:
        if not path.exists():
            continue
        try:
            uni = pd.read_csv(path)
        except Exception:
            continue
        fo_col = find_fo_column(uni)
        if fo_col is None:
            continue
        ticker_col = None
        for cand in ["ticker", "Ticker", "symbol", "Symbol", "SYMBOL"]:
            if cand in uni.columns:
                ticker_col = cand
                break
        if ticker_col is None:
            continue
        mask = uni[fo_col].apply(normalize_fo_value)
        symbols.update(uni.loc[mask, ticker_col].apply(normalize_stock_symbol).dropna().tolist())
    return {s for s in symbols if s}


def apply_fo_lookup(df: pd.DataFrame, fo_symbols: set) -> pd.DataFrame:
    """Add/repair is_fo_stock using a universe-derived F&O symbol set."""
    if df is None or df.empty or not fo_symbols:
        return df
    out = df.copy()
    ticker_col = None
    for cand in ["ticker", "Ticker", "symbol", "Symbol", "SYMBOL"]:
        if cand in out.columns:
            ticker_col = cand
            break
    if ticker_col is None:
        return out
    by_symbol = out[ticker_col].apply(normalize_stock_symbol).isin(fo_symbols)
    if "is_fo_stock" in out.columns:
        out["is_fo_stock"] = out["is_fo_stock"].fillna(False).astype(bool) | by_symbol
    else:
        out["is_fo_stock"] = by_symbol
    return out


def filter_fo_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return only F&O stocks when F&O information exists; otherwise keep data visible."""
    if df is None or df.empty:
        return df
    out = add_normalized_fo_column(df)
    if "is_fo_stock" not in out.columns:
        return out
    filtered = out[out["is_fo_stock"]].copy()
    return filtered if not filtered.empty else out.iloc[0:0].copy()


def fo_available(*dfs: pd.DataFrame) -> bool:
    return any(find_fo_column(df) is not None or (df is not None and "is_fo_stock" in df.columns) for df in dfs)


if "show_fo_only" not in st.session_state:
    st.session_state["show_fo_only"] = False

outdir = "outputs"
help_image_path = "market_phases_reference.png"
combined = ensure_label(safe_read(f"{outdir}/vcp_combined_ranked.csv"))
daily_df = ensure_label(safe_read(f"{outdir}/vcp_daily_ranked.csv"))
weekly_df = ensure_label(safe_read(f"{outdir}/vcp_weekly_ranked.csv"))
industry = ensure_label(safe_read(f"{outdir}/industry_strength.csv"))
changes = ensure_label(safe_read(f"{outdir}/stock_changes.csv"))
industry_changes = ensure_label(safe_read(f"{outdir}/industry_changes.csv"))
moves = ensure_label(safe_read(f"{outdir}/stock_price_moves.csv"))
top_movers = ensure_label(safe_read(f"{outdir}/top_movers.csv"))
if top_movers.empty:
    top_movers = moves.copy()
regime = safe_read(f"{outdir}/market_regime.csv")

# Normalize F&O marker across all loaded datasets. If the dashboard toggle is ON,
# every downstream tab/table/chart uses only F&O stocks.
combined = add_normalized_fo_column(combined)
daily_df = add_normalized_fo_column(daily_df)
weekly_df = add_normalized_fo_column(weekly_df)
changes = add_normalized_fo_column(changes)
moves = add_normalized_fo_column(moves)
top_movers = add_normalized_fo_column(top_movers)

FO_SYMBOL_SET = load_fo_symbol_set_from_universe()
combined = apply_fo_lookup(combined, FO_SYMBOL_SET)
daily_df = apply_fo_lookup(daily_df, FO_SYMBOL_SET)
weekly_df = apply_fo_lookup(weekly_df, FO_SYMBOL_SET)
changes = apply_fo_lookup(changes, FO_SYMBOL_SET)
moves = apply_fo_lookup(moves, FO_SYMBOL_SET)
top_movers = apply_fo_lookup(top_movers, FO_SYMBOL_SET)

if st.session_state.get("show_fo_only", False):
    combined = filter_fo_only(combined)
    daily_df = filter_fo_only(daily_df)
    weekly_df = filter_fo_only(weekly_df)
    changes = filter_fo_only(changes)
    moves = filter_fo_only(moves)
    top_movers = filter_fo_only(top_movers)

GLOBAL_RANK_MAP = build_rank_lookup_map(combined, daily_df, weekly_df, top_movers, moves, changes)
combined = backfill_current_rank(combined, GLOBAL_RANK_MAP)
daily_df = backfill_current_rank(daily_df, GLOBAL_RANK_MAP)
weekly_df = backfill_current_rank(weekly_df, GLOBAL_RANK_MAP)
changes = backfill_current_rank(changes, GLOBAL_RANK_MAP)
moves = backfill_current_rank(moves, GLOBAL_RANK_MAP)
top_movers = backfill_current_rank(top_movers, GLOBAL_RANK_MAP)

if combined.empty:
    st.error("No data found in the default outputs folder.")
    st.info("Create an outputs folder beside this file and keep the generated CSV files there.")
    st.stop()

for df_name in ["combined", "daily_df", "weekly_df", "changes", "moves", "top_movers"]:
    _df = locals().get(df_name)
    if _df is not None and not _df.empty:
        if "decision_score" not in _df.columns:
            _df["decision_score"] = _df.apply(structure_score, axis=1)
        if "decision_state" not in _df.columns:
            _df["decision_state"] = _df.apply(structure_category, axis=1)

daily_dir = f"{outdir}/charts/daily"
weekly_dir = f"{outdir}/charts/weekly"
chart_choice_map = chart_dropdown_options(combined)
top_changed_df, changes_summary = build_today_changes(changes, industry_changes)
alert_candidates = build_alert_candidates(combined, changes)
stage_counts = stage_count_summary(combined)
INDUSTRY_PORTFOLIOS = get_industry_portfolio_options(industry, combined, limit=21)
DECISION_DF = build_decision_engine_table(combined, changes, industry, regime)
DECISION_DF = backfill_current_rank(DECISION_DF, GLOBAL_RANK_MAP)
RANK_SOURCE_DF = DECISION_DF if not DECISION_DF.empty else combined
TOP_MOVER_RANK_MAP = build_simple_rank_map(RANK_SOURCE_DF)
DECISION_STATS = decision_summary_stats(DECISION_DF)
TOP_LONGS = top_trade_candidates(DECISION_DF, "Long", top_n=10)
TOP_SHORTS = top_trade_candidates(DECISION_DF, "Short", top_n=10)
if alert_candidates.empty and not DECISION_DF.empty:
    alert_candidates = safe_sort_values(
        DECISION_DF[DECISION_DF["action"].astype(str) != "No Trade"].copy(),
        ["action_confidence", "decision_priority", "current_rank"],
        [False, False, True],
    ).head(20)
    if not alert_candidates.empty:
        alert_candidates["alert_type"] = alert_candidates.get("action", "Action")
        alert_candidates["alert_reason"] = alert_candidates.get("rationale", "")

def get_stock_rank(ticker: str) -> str:
    t = str(ticker).strip()
    val = (
        TOP_MOVER_RANK_MAP.get(t)
        or TOP_MOVER_RANK_MAP.get(t.replace(".NS", ""))
        or (str(int(GLOBAL_RANK_MAP.get(t))) if GLOBAL_RANK_MAP.get(t) is not None else "")
        or (str(int(GLOBAL_RANK_MAP.get(t.replace(".NS", "")))) if GLOBAL_RANK_MAP.get(t.replace(".NS", "")) is not None else "")
        or rank_lookup(RANK_SOURCE_DF, ticker, ["current_rank", "rank", "rs_rank", "daily_rank", "weekly_rank", "final_rank", "combined_rank", "stock_rank"])
        or rank_lookup(combined, ticker, ["current_rank", "rank", "rs_rank", "daily_rank", "weekly_rank", "final_rank", "combined_rank", "stock_rank"])
    )
    return val if str(val).strip() else "n/a"



# ============================
# Production Mobile UI Layer
# ============================
st.markdown("""
<style>
.pm-shell {max-width: 480px; margin: 0 auto;}
.pm-sticky {position: sticky; top: 0; z-index: 999; padding: 0.55rem 0.35rem 0.7rem 0.35rem; backdrop-filter: blur(18px); background: rgba(5,7,13,0.94); border-bottom: 1px solid rgba(255,255,255,0.09);}
.pm-title {font-size: 1.35rem; font-weight: 950; line-height: 1.05; letter-spacing: -0.02em;}
.pm-subtitle {font-size: 0.79rem; color: rgba(255,255,255,0.62); margin-top: 0.22rem; line-height: 1.25;}
.pm-metrics {display:grid; grid-template-columns: repeat(2, 1fr); gap:0.45rem; margin:0.72rem 0 0.58rem 0;}
.pm-metric {border: 1px solid rgba(255,255,255,0.11); border-radius: 18px; padding: 0.58rem 0.62rem; background: linear-gradient(180deg, rgba(255,255,255,0.070), rgba(255,255,255,0.035)); box-shadow: 0 10px 24px rgba(0,0,0,0.18);}
.pm-metric-label {font-size:0.67rem; color:rgba(255,255,255,0.60); font-weight:850; text-transform:uppercase; letter-spacing:0.05em;}
.pm-metric-value {font-size:1.14rem; font-weight:950; margin-top:0.08rem; line-height:1.05;}
.pm-strip {display:flex; gap:0.45rem; overflow-x:auto; padding:0.08rem 0 0.3rem 0; margin-bottom:0.4rem; scrollbar-width:none;}
.pm-strip::-webkit-scrollbar {display:none;}
.pm-sector {flex:0 0 auto; border:1px solid rgba(255,255,255,0.12); background:rgba(255,255,255,0.055); border-radius:999px; padding:0.32rem 0.62rem; font-size:0.76rem; font-weight:850; color:#eef3ff;}
.pm-card {border:1px solid rgba(255,255,255,0.12); border-radius:26px; background:linear-gradient(180deg, rgba(255,255,255,0.078), rgba(255,255,255,0.030)); overflow:hidden; margin:0.82rem auto; box-shadow:0 16px 34px rgba(0,0,0,0.26);}
.pm-card-head {padding:0.78rem 0.82rem 0.68rem 0.82rem; border-bottom:1px solid rgba(255,255,255,0.08);}
.pm-row {display:flex; justify-content:space-between; align-items:flex-start; gap:0.7rem;}
.pm-company {font-size:1.04rem; font-weight:950; line-height:1.15; letter-spacing:-0.01em; margin-bottom:0.12rem;}
.pm-meta {font-size:0.78rem; color:rgba(255,255,255,0.63); font-weight:720; line-height:1.25;}
.pm-rank-box {text-align:right; min-width:62px;}
.pm-rank-label {font-size:0.62rem; color:rgba(255,255,255,0.54); text-transform:uppercase; font-weight:900; letter-spacing:0.05em;}
.pm-rank-value {font-size:1.22rem; font-weight:950; line-height:1; margin-top:0.08rem;}
.pm-chips {display:flex; flex-wrap:wrap; gap:0.35rem; margin-top:0.58rem;}
.pm-chip {font-size:0.70rem; font-weight:850; padding:0.20rem 0.50rem; border-radius:999px; border:1px solid rgba(255,255,255,0.14); background:rgba(255,255,255,0.07); color:#eef3ff;}
.pm-green {background:rgba(30,201,119,0.14); color:#39e99b; border-color:rgba(30,201,119,0.30);}
.pm-yellow {background:rgba(240,180,41,0.14); color:#ffd166; border-color:rgba(240,180,41,0.32);}
.pm-red {background:rgba(255,107,107,0.13); color:#ff8585; border-color:rgba(255,107,107,0.32);}
.pm-note {font-size:0.81rem; color:rgba(255,255,255,0.77); line-height:1.34; margin-top:0.56rem;}
.pm-chart-label {font-size:0.66rem; color:rgba(255,255,255,0.54); font-weight:950; text-transform:uppercase; letter-spacing:0.08em; padding:0.52rem 0.75rem 0.28rem 0.75rem; background:rgba(0,0,0,0.20);}
.pm-chart-wrap {background:#05070d; border-top:1px solid rgba(255,255,255,0.06);}
.pm-chart-missing {padding:1.35rem 0.8rem; text-align:center; color:rgba(255,255,255,0.50); font-size:0.78rem; background:#05070d; border-top:1px solid rgba(255,255,255,0.06);}
.pm-actions {display:grid; grid-template-columns: repeat(3, 1fr); gap:0.45rem; padding:0.70rem 0.75rem 0.78rem 0.75rem; border-top:1px solid rgba(255,255,255,0.08); background:rgba(255,255,255,0.025);}
.pm-progress {font-size:0.78rem; color:rgba(255,255,255,0.64); font-weight:800; margin:0.45rem 0; text-align:center;}
.pm-hint {font-size:0.75rem; color:rgba(255,255,255,0.56); text-align:center; margin:0.25rem 0 0.55rem 0;}
.pm-disclaimer {border-left: 4px solid rgba(240,180,41,0.52); background:rgba(240,180,41,0.075); border-radius:14px; padding:0.65rem 0.72rem; font-size:0.78rem; color:rgba(255,255,255,0.78); line-height:1.34; margin:0.7rem 0;}
@media (min-width: 769px) {.pm-shell {max-width: 540px;}}
</style>
""", unsafe_allow_html=True)


def _pm_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _pm_chip_class(row: pd.Series) -> str:
    stage = str(row.get("stage", ""))
    label = str(row.get("label", row.get("classification", "")))
    if stage == "Stage 2" or label == "Strong":
        return "pm-chip pm-green"
    if stage in {"Stage 3", "Stage 4"} or label == "Weak":
        return "pm-chip pm-red"
    return "pm-chip pm-yellow"


def _pm_reason(row: pd.Series) -> str:
    parts = []
    stage = str(row.get("stage", ""))
    stage_class = stage_classification_text(row)
    if stage_class and stage_class not in {stage, stage_primary_label(stage)}:
        parts.append(stage_class)
    if stage == "Stage 2":
        parts.append("Advancing structure")
    elif stage == "Stage 3":
        parts.append("Transition phase")
    elif stage == "Stage 4":
        parts.append("Declining structure")
    elif stage == "Stage 1":
        parts.append("Base formation")

    for col, label in [("rs_3m_pct", "RS 3M"), ("rs_6m_pct", "RS 6M")]:
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(val):
            parts.append(f"{label} {val:+.1f}%")

    rank_change = pd.to_numeric(row.get("rank_change"), errors="coerce")
    if pd.notna(rank_change) and rank_change != 0:
        parts.append(f"Rank {'↑' if rank_change > 0 else '↓'} {abs(int(rank_change))}")

    if boolish(row.get("new_weekly_breakout", False)):
        parts.append("Weekly breakout flag")
    elif boolish(row.get("new_daily_breakout", False)):
        parts.append("Daily breakout flag")

    mini = build_mini_signal_text(row)
    if mini:
        parts.append(mini)

    return " • ".join([p for p in parts if p][:4]) or "No major new structure-change flag in the latest update."


def _pm_prepare_view(feed_df: pd.DataFrame, filter_choice: str, limit: int) -> pd.DataFrame:
    view = ensure_label(ensure_current_rank(feed_df.copy()))
    view = sort_by_rank(view).copy()
    if filter_choice == "Improving" and "rank_change" in view.columns:
        view["_rank_change_num"] = pd.to_numeric(view["rank_change"], errors="coerce").fillna(0)
        view = view[view["_rank_change_num"] > 0].sort_values("_rank_change_num", ascending=False)
    elif filter_choice == "Stage 2":
        view = view[view["stage"].astype(str).eq("Stage 2")]
    elif filter_choice == "Weakening":
        view = view[view["stage"].astype(str).isin(["Stage 3", "Stage 4"])]
    elif filter_choice == "New Flags":
        masks = []
        for col in ["entered_stage_2", "new_weekly_breakout", "new_daily_breakout"]:
            if col in view.columns:
                masks.append(view[col].fillna(False).astype(bool))
        if masks:
            mask = masks[0]
            for m in masks[1:]:
                mask = mask | m
            view = view[mask]
    return view.head(limit).reset_index(drop=True)


def _pm_chart_bytes(chart_dir: str, ticker: str, suffix: str):
    path = resolve_chart_path(chart_dir, ticker, suffix) or resolve_chart_path(chart_dir, ticker, ".png")
    return safe_image_bytes(path) if path is not None else None


def _pm_render_charts(ticker: str, daily_chart_dir: str, weekly_chart_dir: str):
    daily_img = _pm_chart_bytes(daily_chart_dir, ticker, "_daily.png")
    weekly_img = _pm_chart_bytes(weekly_chart_dir, ticker, "_weekly.png")
    st.markdown("<div class='pm-chart-label'>Daily chart</div>", unsafe_allow_html=True)
    if daily_img:
        st.markdown("<div class='pm-chart-wrap'>", unsafe_allow_html=True)
        st.image(daily_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='pm-chart-missing'>Daily chart not available</div>", unsafe_allow_html=True)
    st.markdown("<div class='pm-chart-label'>Weekly chart</div>", unsafe_allow_html=True)
    if weekly_img:
        st.markdown("<div class='pm-chart-wrap'>", unsafe_allow_html=True)
        st.image(weekly_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='pm-chart-missing'>Weekly chart not available</div>", unsafe_allow_html=True)


def _pm_render_card(row: pd.Series, daily_chart_dir: str, weekly_chart_dir: str, show_actions: bool = False):
    ticker = str(row.get("ticker", "")).strip()
    rank_val = pd.to_numeric(row.get("current_rank"), errors="coerce")
    rank_text = f"#{int(rank_val)}" if pd.notna(rank_val) else "-"
    company = stock_display_label(row)
    industry_name = str(row.get("Industry", "-") or "-")
    stage = str(row.get("stage", "-") or "-")
    stage_class = stage_classification_text(row)
    structure = structure_category(row)
    score = structure_score(row)
    chip_class = _pm_chip_class(row)
    reason = _pm_reason(row)
    rank_change = pd.to_numeric(row.get("rank_change"), errors="coerce")
    rank_chip = ""
    if pd.notna(rank_change) and rank_change != 0:
        rank_chip = f"<span class='pm-chip {'pm-green' if rank_change > 0 else 'pm-red'}'>Rank {'↑' if rank_change > 0 else '↓'} {abs(int(rank_change))}</span>"
    st.markdown("<div class='pm-card'>", unsafe_allow_html=True)
    st.markdown(f"""
<div class='pm-card-head'>
  <div class='pm-row'>
    <div style='min-width:0;'>
      <div class='pm-company'>{company}</div>
      <div class='pm-meta'>{industry_icon(industry_name)} {industry_name} • {stage} • {stage_class}</div>
    </div>
    <div class='pm-rank-box'>
      <div class='pm-rank-label'>Rank</div>
      <div class='pm-rank-value'>{rank_text}</div>
    </div>
  </div>
  <div class='pm-chips'>
    <span class='{chip_class}'>{structure}</span>
    <span class='pm-chip'>Score {score}/100</span>
    {rank_chip}
  </div>
  <div class='pm-note'>{reason}</div>
</div>
""", unsafe_allow_html=True)
    _pm_render_charts(ticker, daily_chart_dir, weekly_chart_dir)
    if show_actions:
        st.markdown("<div class='pm-actions'>", unsafe_allow_html=True)
        a, b, c = st.columns(3)
        with a:
            if st.button("👎 Skip", use_container_width=True, key=f"pm_skip_{ticker}"):
                st.session_state["pm_swipe_index"] = st.session_state.get("pm_swipe_index", 0) + 1
                _pm_rerun()
        with b:
            if st.button("⭐ Save", use_container_width=True, key=f"pm_save_{ticker}"):
                saved = st.session_state.setdefault("pm_saved_tickers", [])
                if ticker and ticker not in saved:
                    saved.append(ticker)
                st.session_state["pm_swipe_index"] = st.session_state.get("pm_swipe_index", 0) + 1
                _pm_rerun()
        with c:
            if st.button("➡️ Next", use_container_width=True, key=f"pm_next_{ticker}"):
                st.session_state["pm_swipe_index"] = st.session_state.get("pm_swipe_index", 0) + 1
                _pm_rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_production_mobile_ui(feed_df: pd.DataFrame, daily_chart_dir: str, weekly_chart_dir: str):
    st.markdown("<div class='pm-shell'>", unsafe_allow_html=True)
    st.markdown("""
<div class='pm-sticky'>
  <div class='pm-title'>Post-Close Market Reset</div>
  <div class='pm-subtitle'>Fast structure scan after market close. Daily + weekly charts stacked. Data view only, not investment advice.</div>
</div>
""", unsafe_allow_html=True)
    if feed_df is None or feed_df.empty:
        st.info("No data available for mobile feed.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    prepared = ensure_label(ensure_current_rank(feed_df.copy()))
    prepared = sort_by_rank(prepared).copy()
    counts = stage_count_summary(prepared)
    stage2 = counts.get("Stage 2", 0)
    weakening = counts.get("Stage 3", 0) + counts.get("Stage 4", 0)
    leader_sector = "-"
    if "Industry" in prepared.columns and "stage" in prepared.columns:
        sector_counts = prepared[prepared["stage"].eq("Stage 2")].groupby("Industry").size().sort_values(ascending=False)
        if not sector_counts.empty:
            leader_sector = str(sector_counts.index[0])
    st.markdown(f"""
<div class='pm-metrics'>
  <div class='pm-metric'><div class='pm-metric-label'>Stage 2</div><div class='pm-metric-value'>{stage2}</div></div>
  <div class='pm-metric'><div class='pm-metric-label'>Weakening</div><div class='pm-metric-value'>{weakening}</div></div>
  <div class='pm-metric'><div class='pm-metric-label'>Leader Sector</div><div class='pm-metric-value' style='font-size:0.92rem;'>{leader_sector}</div></div>
  <div class='pm-metric'><div class='pm-metric-label'>Updated</div><div class='pm-metric-value' style='font-size:0.92rem;'>Post Close</div></div>
</div>
""", unsafe_allow_html=True)
    if "Industry" in prepared.columns:
        top_sectors = prepared[prepared["stage"].astype(str).eq("Stage 2")].groupby("Industry").size().sort_values(ascending=False).head(8)
        if not top_sectors.empty:
            chips = "".join([f"<div class='pm-sector'>{industry_icon(str(k))} {k} · {int(v)}</div>" for k, v in top_sectors.items()])
            st.markdown(f"<div class='pm-strip'>{chips}</div>", unsafe_allow_html=True)
    mode = st.radio("Mode", ["Feed", "Swipe"], horizontal=True, label_visibility="collapsed")
    filter_choice = st.radio("Filter", ["Top", "Improving", "Stage 2", "Weakening", "New Flags"], horizontal=True, label_visibility="collapsed")
    max_cards = st.slider("Cards", min_value=5, max_value=60, value=25, step=5, label_visibility="collapsed")
    view = _pm_prepare_view(prepared, filter_choice, max_cards)
    st.markdown("<div class='pm-disclaimer'>This screen is a public-view prototype: structure labels, ranks, sectors and charts only. No buy/sell/short recommendation is shown.</div>", unsafe_allow_html=True)
    if view.empty:
        st.info("No stocks match this filter.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    if mode == "Swipe":
        if "pm_swipe_index" not in st.session_state:
            st.session_state["pm_swipe_index"] = 0
        if st.session_state["pm_swipe_index"] >= len(view):
            st.success("Done for this filtered set.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Restart", use_container_width=True):
                    st.session_state["pm_swipe_index"] = 0
                    _pm_rerun()
            with c2:
                saved = st.session_state.get("pm_saved_tickers", [])
                st.write(f"Saved: {len(saved)}")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        idx = int(st.session_state["pm_swipe_index"])
        st.markdown(f"<div class='pm-progress'>{idx + 1} / {len(view)}</div>", unsafe_allow_html=True)
        st.markdown("<div class='pm-hint'>Use Skip / Save / Next like a Tinder-style review flow.</div>", unsafe_allow_html=True)
        _pm_render_card(view.iloc[idx], daily_chart_dir, weekly_chart_dir, show_actions=True)
    else:
        for _, row in view.iterrows():
            _pm_render_card(row, daily_chart_dir, weekly_chart_dir, show_actions=False)
    st.markdown("</div>", unsafe_allow_html=True)

if "watchlist_names" not in st.session_state:
    st.session_state["watchlist_names"] = []
if "custom_watchlist_names" not in st.session_state:
    st.session_state["custom_watchlist_names"] = []
if "watchlist_selection_prev" not in st.session_state:
    st.session_state["watchlist_selection_prev"] = "Custom"
if "watchlist_chart_index" not in st.session_state:
    st.session_state["watchlist_chart_index"] = 0
if "chart_selected_ticker" not in st.session_state:
    first_ticker = combined["ticker"].dropna().astype(str).head(1).tolist()
    st.session_state["chart_selected_ticker"] = first_ticker[0] if first_ticker else None

st.title("Market Decision Engine")
st.caption("Structure-led trade engine with long, short, and watchlist actions")

fo_ui_cols = st.columns([1, 3])
with fo_ui_cols[0]:
    st.toggle("F&O only", key="show_fo_only", help="Show only stocks marked as F&O in your data file.")
with fo_ui_cols[1]:
    if st.session_state.get("show_fo_only", False):
        st.caption(f"F&O filter active • {len(combined)} stocks shown")
        if combined.empty:
            st.warning("No matching F&O stocks found. Check that your universe file is beside this app or inside outputs/ and has a ticker/symbol column plus an F&O column.")
    elif fo_available(combined, daily_df, weekly_df, moves, top_movers, DECISION_DF):
        st.caption("F&O filter available from your data column.")
    else:
        st.caption("F&O column not found yet. Add a column like F&O, FO, FNO, or is_FO to enable the filter.")

# --- Private publishing control ---
# This creates outputs/public_daily.json for your future public mobile website.
# It removes Buy/Sell/Short language and exports only structure analytics.
with st.expander("🚀 Private Publishing Controls", expanded=False):
    st.write("Generate a SEBI-safer public JSON feed from the current processed dashboard data.")
    public_count = st.slider("Number of public stocks to export", min_value=5, max_value=100, value=30, step=5)
    if st.button("Export Public Data", type="primary"):
        try:
            from export_public_data import export_json
            exported_path = export_json(combined, max_items=public_count)
            st.success(f"Public data exported successfully: {exported_path}")
            st.caption("This export contains structure labels, ranks, sectors, RS metrics, and chart paths only — no Buy/Sell/Short actions.")
        except Exception as exc:
            st.error(f"Export failed: {exc}")

tab_names = ["Today", "Trade Board", "Watchlist", "Charts", "Market", "Structure Changes", "Learn", "Disclaimer"]
tabs = st.tabs(tab_names)

with tabs[0]:
    current_market_tone = market_tone(regime, combined)
    current_bias = market_action_bias(stage_counts, regime)
    current_bias_score = market_bias_score(stage_counts, regime)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_summary_card("Market mode", current_market_tone, market_tone_text(current_market_tone))
    with c2:
        render_summary_card("Action bias", current_bias, market_action_text(stage_counts, regime))
    with c3:
        render_summary_card("Buy candidates", str(DECISION_STATS["buy"]), "Names currently marked as Buy or Tactical Buy")
    with c4:
        render_summary_card("Short candidates", str(DECISION_STATS["short"]), "Names currently marked as Short or Tactical Short")

    st.markdown("### Top trade actions for today")
    lcol, scol = st.columns(2)
    with lcol:
        st.markdown("#### Top longs")
        if TOP_LONGS.empty:
            st.info("No strong long candidate today.")
        else:
            for _, r in TOP_LONGS.head(5).iterrows():
                render_trade_card(r)
    with scol:
        st.markdown("#### Top shorts")
        if TOP_SHORTS.empty:
            st.info("No strong short candidate today.")
        else:
            for _, r in TOP_SHORTS.head(5).iterrows():
                render_trade_card(r)

    st.markdown("### What changed today")
    if top_changed_df.empty:
        st.info("No recent change rows are available.")
    else:
        cols = st.columns(3)
        for i, (_, r) in enumerate(top_changed_df.head(6).iterrows()):
            with cols[i % 3]:
                card(r, use_stage_color=True, show_change_text=str(r.get("what_changed", "")), stock_rank=get_stock_rank(r["ticker"]))

    st.markdown("### Market snapshot")
    left, right = st.columns([1.15, 0.85])
    with left:
        render_distribution(stage_counts)
    with right:
        st.markdown(f'<div class="info-card"><b>Execution note</b><ul class="list-tight"><li>Bias: <b>{current_bias}</b> ({current_bias_score}/100).</li><li>Top longs should ideally come from industries with positive support.</li><li>Top shorts should ideally come from weak or fragile industries.</li><li>Trade Board shows the full ranked action list.</li></ul></div>', unsafe_allow_html=True)

    st.markdown("### Sample structures from the dataset")
    for title, stage_key in [("Advancing structures (sample)", "Stage 2"), ("Base structures (sample)", "Stage 1"), ("Transition or decline structures (sample)", "Stage 3_4")]:
        st.markdown(f"#### {title}")
        if stage_key == "Stage 3_4":
            sample_df = combined[combined["stage"].isin(["Stage 3", "Stage 4"])].sort_values(["Company Name", "ticker"]).head(3)
        else:
            sample_df = combined[combined["stage"] == stage_key].sort_values(["Company Name", "ticker"]).head(3)
        if sample_df.empty:
            st.info("No rows available.")
        else:
            cols = st.columns(3)
            for i, (_, r) in enumerate(sample_df.iterrows()):
                with cols[i % 3]:
                    card(r, use_stage_color=True, stock_rank=get_stock_rank(r["ticker"]))

with tabs[1]:
    st.markdown("### Trade Board")
    if DECISION_DF.empty:
        st.info("Decision engine table is not available.")
    else:
        side_filter = st.selectbox("Trade side", ["All", "Long", "Short", "Neutral"], key="trade_board_side")
        action_filter = st.selectbox("Action", ["All", "Buy", "Tactical Buy", "Watch for Long", "Short", "Tactical Short", "Watch for Short", "No Trade"], key="trade_board_action")
        min_score = st.slider("Minimum action confidence", 0, 100, 55, 1, key="trade_board_confidence")
        board = DECISION_DF.copy()
        if side_filter != "All":
            board = board[board["trade_side"] == side_filter]
        if action_filter != "All":
            board = board[board["action"] == action_filter]
        board = board[board["action_confidence"] >= min_score]
        board = safe_sort_values(board, ["decision_priority", "industry_score", "current_rank"], [False, False, True])

        t1, t2, t3, t4 = st.columns(4)
        with t1:
            render_summary_card("Buy", str(int(board["action"].astype(str).str.contains("Buy", regex=False).sum())), "Rows on current filter")
        with t2:
            render_summary_card("Short", str(int(board["action"].astype(str).str.contains("Short", regex=False).sum())), "Rows on current filter")
        with t3:
            render_summary_card("Avg industry support", str(int(board["industry_score"].fillna(0).mean())) if not board.empty else "0", "Higher helps longs, lower helps shorts")
        with t4:
            render_summary_card("Rows", str(len(board)), "Filtered trade rows")

        board = board.copy()
        board["dry_up_status"] = board.apply(dry_up_status, axis=1)
        board["mini_signals"] = board.apply(build_mini_signal_text, axis=1)
        show_cols = [c for c in [
            "Company Name", "ticker", "Industry", "stage", "stage_classification", "label", "action", "trade_side",
            "long_score", "short_score", "action_confidence", "industry_score", "industry_view",
            "current_rank", "rank_change", "is_fo_stock", "dry_up_status", "mini_signals", "rationale"
        ] if c in board.columns]
        st.dataframe(board[show_cols], use_container_width=True, hide_index=True, height=460)

        st.markdown("#### Trade cards")
        for _, r in board.head(20).iterrows():
            render_trade_card(r)


with tabs[2]:
    st.markdown("### Watchlist")
    portfolio_options = [
        "Custom",
        "Buy",
        "Tactical Buy",
        "Watch for Long",
        "Short",
        "Tactical Short",
        "Watch for Short",
        "No Trade",
        "Strong",
        "Developing",
        "Cautious",
        "Weak",
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "Stage 4",
    ] + sorted([f"Class: {x}" for x in combined.get("stage_classification", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if x and x not in {"nan", "None"}]) + INDUSTRY_PORTFOLIOS
    selected_watchlist = st.selectbox("Watchlist view", portfolio_options, key="watchlist_view")
    previous_watchlist = st.session_state.get("watchlist_selection_prev", "Custom")
    if selected_watchlist != previous_watchlist:
        if previous_watchlist == "Custom":
            st.session_state["custom_watchlist_names"] = dedupe_names(st.session_state["watchlist_names"], limit=MAX_PORTFOLIO_STOCKS)
        if selected_watchlist == "Custom":
            st.session_state["watchlist_names"] = dedupe_names(st.session_state.get("custom_watchlist_names", []), limit=MAX_PORTFOLIO_STOCKS)
        else:
            st.session_state["watchlist_names"] = get_prebuilt_portfolio(selected_watchlist, combined, changes, INDUSTRY_PORTFOLIOS)
        st.session_state["watchlist_selection_prev"] = selected_watchlist
    st.session_state["watchlist_names"] = dedupe_names(st.session_state["watchlist_names"], limit=MAX_PORTFOLIO_STOCKS)
    selected_to_add = st.selectbox("Add stock", options=[None] + list(chart_choice_map.keys()), index=0, placeholder="Type stock name or ticker", key="watchlist_add_name")
    add_cols = st.columns([1.0, 1.2])
    with add_cols[0]:
        if st.button("Add selected stock", use_container_width=True, key="watchlist_add_btn") and selected_to_add:
            selected_name = selected_to_add.rsplit(" (", 1)[0]
            st.session_state["watchlist_names"] = dedupe_names(st.session_state["watchlist_names"] + [selected_name], limit=MAX_PORTFOLIO_STOCKS)
            st.session_state["custom_watchlist_names"] = dedupe_names(st.session_state["watchlist_names"], limit=MAX_PORTFOLIO_STOCKS)
            st.rerun()
    with add_cols[1]:
        ticker_text = st.text_area("Add by tickers", value="", height=90, placeholder="RELIANCE, TCS, INFY or RELIANCE.NS, TCS.NS", key="watchlist_add_tickers_text")
        if st.button("Add ticker list", use_container_width=True, key="watchlist_add_tickers_btn"):
            parsed_tickers = parse_ticker_input(ticker_text)
            source_for_lookup = DECISION_DF if not DECISION_DF.empty else combined
            matched_names = names_from_tickers(parsed_tickers, source_for_lookup)
            st.session_state["watchlist_names"] = dedupe_names(st.session_state["watchlist_names"] + matched_names, limit=MAX_PORTFOLIO_STOCKS)
            st.session_state["custom_watchlist_names"] = dedupe_names(st.session_state["watchlist_names"], limit=MAX_PORTFOLIO_STOCKS)
            all_known = {str(x).replace(".NS", "").upper() for x in (combined["ticker"].dropna().astype(str).tolist() if "ticker" in combined.columns else [])}
            missing = [t for t in parsed_tickers if t.replace(".NS", "").upper() not in all_known]
            if matched_names:
                st.success(f"Added {len(matched_names)} stock(s) from ticker list.")
            if missing:
                st.warning("Not found: " + ", ".join(missing[:50]))
            st.rerun()
    if not st.session_state["watchlist_names"]:
        st.info("No stocks added yet. You can add from the dropdown or paste a ticker list.")
    else:
        source_df = DECISION_DF if not DECISION_DF.empty else combined
        current = source_df[source_df["Company Name"].isin(st.session_state["watchlist_names"])].copy()
        if current.empty:
            current = combined[combined["Company Name"].isin(st.session_state["watchlist_names"])].copy()
        current = sort_watchlist_view(current, selected_watchlist=selected_watchlist)
        watch_counts = stage_count_summary(current)
        render_distribution(watch_counts)

        if not current.empty:
            current["portfolio_action"] = current.get("action", "No Trade")
            current["dry_up_status"] = current.apply(dry_up_status, axis=1)
            summary_cols = [c for c in ["Company Name", "ticker", "stage", "stage_classification", "label", "is_fo_stock", "portfolio_action", "dry_up_status", "current_rank", "industry_score", "action_confidence"] if c in current.columns]
            st.markdown("#### Portfolio action table")
            st.dataframe(current[summary_cols], use_container_width=True, hide_index=True, height=260)

        portfolio_ordered = current.reset_index(drop=True)
        if not portfolio_ordered.empty:
            st.divider()
            st.markdown("### Watchlist charts")
            chart_limit = min(60, len(portfolio_ordered))
            st.caption(f"Showing charts for the first {chart_limit} watchlist stocks by rank. Use the Charts tab for the full list.")
            for idx, (_, prow) in enumerate(portfolio_ordered.head(chart_limit).iterrows(), start=1):
                pticker_short = str(prow["ticker"]).replace(".NS", "")
                st.markdown(f"#### {idx}. {stock_display_label(prow)} • Dataset Rank {get_stock_rank(prow['ticker'])}")
                pc1, pc2 = st.columns(2)
                with pc1:
                    st.markdown(f"**Daily chart • {pticker_short}**")
                    pdpath = resolve_chart_path(daily_dir, prow["ticker"], "_daily.png")
                    if pdpath:
                        st.image(safe_image_bytes(pdpath), use_container_width=True)
                    else:
                        st.info("Daily chart not available.")
                with pc2:
                    st.markdown(f"**Weekly chart • {pticker_short}**")
                    pwpath = resolve_chart_path(weekly_dir, prow["ticker"], "_weekly.png")
                    if pwpath:
                        st.image(safe_image_bytes(pwpath), use_container_width=True)
                    else:
                        st.info("Weekly chart not available.")

                card(
                    prow,
                    use_stage_color=True,
                    stock_rank=get_stock_rank(prow["ticker"]),
                    action_label=str(prow.get("action", "No Trade")),
                    show_change_text=(f"Portfolio Action • {str(prow.get('action', 'No Trade'))} • {str(prow.get('rationale', ''))}" if "action" in prow.index else ""),
                )
                # render_stock_detail(prow)

        removable = [""] + sorted(st.session_state["watchlist_names"])
        selected_remove = st.selectbox("Remove stock", removable, key="watchlist_remove_name")
        if st.button("Remove from watchlist", use_container_width=True, key="watchlist_remove_btn") and selected_remove:
            st.session_state["watchlist_names"] = [x for x in st.session_state["watchlist_names"] if x != selected_remove]
            st.session_state["custom_watchlist_names"] = dedupe_names(st.session_state["watchlist_names"], limit=MAX_PORTFOLIO_STOCKS)
            st.rerun()
    render_disclosure()

with tabs[3]:
    st.markdown("### Charts")
    ranked_alpha = sort_by_rank(DECISION_DF if not DECISION_DF.empty else combined, descending=False, company_tiebreak=True).reset_index(drop=True).copy()
    ticker_list = ranked_alpha["ticker"].dropna().astype(str).tolist()
    options = list(chart_choice_map.keys())

    if "chart_index" not in st.session_state:
        st.session_state["chart_index"] = 0

    if ticker_list:
        st.session_state["chart_index"] = max(0, min(st.session_state["chart_index"], len(ticker_list) - 1))

    def _label_for_ticker(ticker):
        if not ticker:
            return None
        for label, tick in chart_choice_map.items():
            if tick == ticker:
                return label
        return options[0] if options else None

    def _sync_chart_selectbox_from_index():
        if ticker_list:
            st.session_state["charts_selectbox_live"] = _label_for_ticker(ticker_list[st.session_state["chart_index"]])

    def _go_prev_chart():
        if ticker_list and st.session_state["chart_index"] > 0:
            st.session_state["chart_index"] -= 1
            _sync_chart_selectbox_from_index()

    def _go_next_chart():
        if ticker_list and st.session_state["chart_index"] < len(ticker_list) - 1:
            st.session_state["chart_index"] += 1
            _sync_chart_selectbox_from_index()

    if "charts_selectbox_live" not in st.session_state:
        _sync_chart_selectbox_from_index()

    if ticker_list and st.session_state.get("charts_selectbox_live") in chart_choice_map:
        selected_ticker = chart_choice_map[st.session_state["charts_selectbox_live"]]
        if selected_ticker in ticker_list:
            st.session_state["chart_index"] = ticker_list.index(selected_ticker)

    selected_display = st.selectbox(
        "Select stock",
        options=options,
        index=(options.index(st.session_state["charts_selectbox_live"]) if options and st.session_state.get("charts_selectbox_live") in options else 0),
        placeholder="Type stock name or ticker",
        key="charts_selectbox_live",
    )

    if selected_display and ticker_list:
        chosen_ticker = chart_choice_map[selected_display]
        if chosen_ticker in ticker_list:
            st.session_state["chart_index"] = ticker_list.index(chosen_ticker)

    if not ticker_list:
        st.info("No chart rows are available.")
    else:
        idx = st.session_state["chart_index"]
        row = ranked_alpha.iloc[idx]
        dpath = resolve_chart_path(daily_dir, row["ticker"], "_daily.png")
        wpath = resolve_chart_path(weekly_dir, row["ticker"], "_weekly.png")

        st.markdown(f"**Selected:** {stock_display_label(row)}")
        st.caption(interpretation_line(row))

        a, b = st.columns(2)
        with a:
            st.markdown(f"#### Daily chart • Dataset Rank {get_stock_rank(row['ticker'])}")
            if dpath:
                st.image(safe_image_bytes(dpath), use_container_width=True)
            else:
                st.info("Daily chart not available.")
        with b:
            st.markdown(f"#### Weekly chart • Dataset Rank {get_stock_rank(row['ticker'])}")
            if wpath:
                st.image(safe_image_bytes(wpath), use_container_width=True)
            else:
                st.info("Weekly chart not available.")

        nav1, nav2 = st.columns(2)
        with nav1:
            st.button(
                "Previous",
                use_container_width=True,
                disabled=(idx == 0),
                key="charts_prev_btn",
                on_click=_go_prev_chart,
            )
        with nav2:
            st.button(
                "Next",
                use_container_width=True,
                disabled=(idx >= len(ticker_list) - 1),
                key="charts_next_btn",
                on_click=_go_next_chart,
            )

        card(row, use_stage_color=True, stock_rank=get_stock_rank(row["ticker"]))
        render_stock_detail(row)
    render_disclosure()
with tabs[4]:
    st.markdown("### Market")
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_summary_card("Stage 1", str(stage_counts["Stage 1"]), "Base / repair")
    with c2: render_summary_card("Stage 2", str(stage_counts["Stage 2"]), "Advancing")
    with c3: render_summary_card("Stage 3", str(stage_counts["Stage 3"]), "Transition")
    with c4: render_summary_card("Stage 4", str(stage_counts["Stage 4"]), "Declining")
    left, right = st.columns(2)
    with left:
        view = industry.copy()
        if not view.empty and "Industry" in view.columns:
            stage2 = stage2_count_by_industry(combined)
            view = view.merge(stage2, on="Industry", how="left")
            st.dataframe(view[[c for c in ["Industry", "avg_combined_score", "current_rank", "Stage 2 Stocks"] if c in view.columns]], use_container_width=True, hide_index=True, height=520)
        else:
            st.info("Industry data not available.")
    with right:
        if not combined.empty and "stage_classification" in combined.columns:
            st.markdown("#### Stage classification mix")
            class_mix = combined["stage_classification"].fillna("Not Sure").astype(str).value_counts().reset_index()
            class_mix.columns = ["Stage Classification", "Stocks"]
            st.dataframe(class_mix, use_container_width=True, hide_index=True, height=230)
        if industry_changes.empty:
            st.info("Industry changes data not available.")
        else:
            cols = [c for c in ["Industry", "current_rank", "prev_rank", "rank_change"] if c in industry_changes.columns]
            st.dataframe(industry_changes[cols], use_container_width=True, hide_index=True, height=520)
    render_disclosure()
with tabs[5]:
    st.markdown("### Structure Changes")
    if alert_candidates.empty:
        st.info("No structure-change rows were found in the latest data.")
    else:
        for _, r in alert_candidates.iterrows():
            card(r, use_stage_color=True, show_change_text=f"{r['alert_type']} • {r['alert_reason']}", stock_rank=get_stock_rank(r["ticker"]))
    render_disclosure()

with tabs[6]:
    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown("""
<div class="learn-card">
  <div class="stock-title">How to use this app</div>
  <ul class="list-tight">
    <li>Start with <b>Today</b> to understand the market mode and recent changes.</li>
    <li>Use <b>Watchlist</b> to track your own basket.</li>
    <li>Use <b>Charts</b> for daily and weekly context.</li>
  </ul>
</div>
<div class="learn-card" style="margin-top:0.7rem;">
  <div class="stock-title">How the stage model should be read</div>
  <ul class="list-tight">
    <li><b>Stage 1</b>: base formation or repair.</li>
    <li><b>Stage 2</b>: advancing structure.</li>
    <li><b>Stage 3</b>: slowing or transition phase.</li>
    <li><b>Stage 4</b>: declining structure.</li>
  </ul>
</div>
<div class="learn-card" style="margin-top:0.7rem;">
  <div class="stock-title">Important note</div>
  This app explains structure and trends inside a rule-based model. It does not provide investment advice.
</div>
""", unsafe_allow_html=True)
    with right:
        img = Path(help_image_path)
        if img.exists():
            st.image(str(img), caption="Reference image for the four market phases", use_container_width=True)
        else:
            st.markdown('<div class="info-card"><b>Onboarding note</b><br>This tool helps users understand market structure. It does not tell users what to buy.</div>', unsafe_allow_html=True)
    render_disclosure()

with tabs[7]:
    st.markdown("### Disclaimer")
    st.write("This tool is for informational purposes only. It presents rule-based stage classifications and market summaries. It does not provide investment advice, recommendations, or opinions on buying, selling, or holding securities. It does not rank, recommend, prioritize, or suggest any securities for investment purposes, and it does not provide model portfolios, suitability analysis, or allocation recommendations.")
    render_disclosure()

