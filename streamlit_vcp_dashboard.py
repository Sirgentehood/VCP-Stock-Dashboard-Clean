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
MAX_PORTFOLIO_STOCKS = 25

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
    out = normalize_columns(df)
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
This platform provides rule-based market structure analytics and visualization only. It does not provide investment advice, recommendations, or opinions on buying, selling, or holding securities. It does not rank, recommend, prioritize, or suggest any securities for investment purposes. No output on this dashboard should be treated as a personalized recommendation, model portfolio, suitability assessment, or allocation advice. Users remain solely responsible for any investment decisions.
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

def chart_dropdown_options(df: pd.DataFrame):
    if df.empty:
        return {}
    tmp = df.dropna(subset=["Company Name", "ticker"]).copy()
    tmp["display_label"] = tmp.apply(stock_display_label, axis=1)
    tmp = tmp.sort_values(["display_label"], ascending=[True]).drop_duplicates(subset=["ticker"], keep="first")
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
        return "n/a"
    work = df.copy()
    ticker_col = None
    for cand in ["ticker", "Ticker", "symbol", "Symbol"]:
        if cand in work.columns:
            ticker_col = cand
            break
    if ticker_col is None:
        return "n/a"
    work[ticker_col] = work[ticker_col].astype(str).str.strip()
    ticker_norm = str(ticker).strip()
    match = work[work[ticker_col] == ticker_norm]
    if match.empty:
        match = work[work[ticker_col].str.replace(".NS", "", regex=False) == ticker_norm.replace(".NS", "")]
    if match.empty:
        return "n/a"
    row = match.iloc[0]
    for col in preferred_cols + ["current_rank", "rank", "rs_rank", "daily_rank", "weekly_rank", "final_rank", "combined_rank", "stock_rank"]:
        if col in match.columns:
            val = pd.to_numeric(row.get(col), errors="coerce")
            if pd.notna(val):
                return str(int(val))
    return "n/a"

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
        if len(out) >= limit:
            break
    return out


def get_prebuilt_portfolio(name: str, combined: pd.DataFrame, changes: pd.DataFrame, industries: list) -> list:
    ranked = combined.sort_values(["Company Name", "ticker"], ascending=[True, True], na_position="last").copy()
    names = []
    if name in {"Stage 1", "Stage 2", "Stage 3", "Stage 4"}:
        names = ranked.loc[ranked["stage"] == name, "Company Name"].dropna().tolist()
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
    st.markdown("#### Stock detail")
    st.markdown(f"**{stock_name} ({ticker})**")
    st.caption(f"{stage_primary_label(stage)} · {structure_category(row)}")
    meta = []
    rank_val = pd.to_numeric(row.get("current_rank"), errors="coerce")
    if pd.notna(rank_val):
        meta.append(f"Dataset Rank: {int(rank_val)}")
    meta.append(f"Model Score: {structure_score(row)}")
    st.write(" • ".join(meta))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="info-card"><b>Current model description</b><br>' + interpretation_line(row) + '</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-card"><b>Recent structure-change flags</b><br>' + signal_summary(row) + '</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="info-card"><b>Industry</b><br>' + str(row.get("Industry", "Not available")) + '</div>', unsafe_allow_html=True)

def card(row: pd.Series, pct=None, use_stage_color=False, show_change_text: str = "", stock_rank: str = "n/a"):
    label = row.get("label", row.get("classification", "Developing"))
    style = LABELS.get(label, LABELS["Developing"])
    stage_raw = str(row.get("stage", "Unknown"))
    stage_label = stage_primary_label(stage_raw)
    stage_desc = stage_short_description(stage_raw)
    stage_condition = stage_condition_text(row)
    display_name = stock_display_label(row)
    structure = structure_category(row)
    score = structure_score(row)
    interpret = interpretation_line(row)
    signals = signal_summary(row)
    signals_html = f"<div class='small-note' style='margin-top:0.15rem;'>{signals}</div>" if signals and signals != "No major new structure-change flag in the latest update." else ""
    industry_name = str(row.get("Industry", "")).strip()
    industry_with_icon = f"{industry_icon(industry_name)} {industry_name}" if industry_name else "Not available"

    classes = []
    if use_stage_color:
        stage_cls = _stage_card_class(stage_raw)
        if stage_cls:
            classes.append(stage_cls)

    change_html = ""
    if pct is not None:
        cls = "change-badge-up" if pct > 0 else "change-badge-down"
        change_html = f"<div class='{cls}'>{pct:+.2f}%</div>"

    extra_change = f"<div class='change-text'>{show_change_text}</div>" if show_change_text else ""
    class_attr = " ".join(classes)
    status_html = f"<div class='status-pill {style['css']}'>{label}</div>"
    structure_html = f"<div class='structure-pill'>{structure} · Model Score {score}/100</div>"
    rank_html = f"<div class='rank-text'>Dataset Rank {stock_rank}</div>"

    html = (
        f"<div class='stock-card {class_attr}'>"
        f"<div style='display:flex; justify-content:space-between; align-items:flex-start; gap:0.55rem;'>"
        f"<div style='min-width:0;'>"
        f"<div class='stock-title'>{display_name}</div>"
        f"<div class='meta-line'>{stage_raw} • {stage_label} • {stage_condition}</div>"
        f"<div class='stock-subtitle'>{interpret}</div>"
        f"{structure_html}"
        f"<div class='small-note'>Higher model score means stronger structure inside this model. It is not a recommendation.</div>"
        f"<div class='small-note' style='margin-top:0.15rem;'>{stage_desc}</div>"
        f"<div class='small-note' style='margin-top:0.2rem;'>{industry_with_icon}</div>"
        f"{signals_html}"
        f"</div>"
        f"<div style='display:flex; flex-direction:column; align-items:flex-end; gap:0.05rem;'>"
        f"{status_html}{rank_html}{change_html}"
        f"</div>"
        f"</div>"
        f"{extra_change}"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)

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
TOP_MOVER_RANK_MAP = build_simple_rank_map(top_movers)
INDUSTRY_PORTFOLIOS = get_industry_portfolio_options(industry, combined, limit=21)

def get_stock_rank(ticker: str) -> str:
    t = str(ticker).strip()
    return TOP_MOVER_RANK_MAP.get(t) or TOP_MOVER_RANK_MAP.get(t.replace(".NS","")) or rank_lookup(top_movers, ticker, ["current_rank"])

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

st.title("Market Structure Radar")
st.caption("Rule-based market structure analytics and visualization")
view_mode = st.radio("View mode", ["Beginner", "Pro"], horizontal=True, index=0)
tab_names = ["Today", "Explore", "Movers", "Watchlist", "Charts", "Learn", "Disclaimer"] if view_mode == "Beginner" else ["Today", "Explore", "Movers", "Watchlist", "Charts", "Market", "Structure Changes", "Learn", "Disclaimer"]
tabs = st.tabs(tab_names)

with tabs[0]:
    current_market_tone = market_tone(regime, combined)
    c1, c2, c3 = st.columns(3)
    with c1:
        render_summary_card("Market mode", current_market_tone, market_tone_text(current_market_tone))
    with c2:
        render_summary_card("Stage 2 count", str(stage_counts["Stage 2"]), "Stocks currently in advancing phase")
    with c3:
        render_summary_card("Top industries in view", top_industry_text(industry), "Industries at the top of the current industry table")
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
        st.markdown('<div class="info-card"><b>How to read today’s view</b><ul class="list-tight"><li>Start with market mode.</li><li>See what changed today.</li><li>Use Explore for filters and Watchlist for your own basket.</li><li>This app explains structure. It does not tell users what to buy.</li></ul></div>', unsafe_allow_html=True)
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
    render_disclosure()

with tabs[1]:
    st.markdown("### Explore")
    filt1, filt2, filt3 = st.columns(3)
    with filt1:
        stage_filter = st.selectbox("Stage", ["All", "Stage 1", "Stage 2", "Stage 3", "Stage 4"])
    with filt2:
        label_filter = st.selectbox("Label", ["All", "Strong", "Developing", "Cautious", "Weak"])
    with filt3:
        industry_options = ["All"] + sorted([x for x in combined["Industry"].dropna().astype(str).unique().tolist()])
        industry_filter = st.selectbox("Industry", industry_options)
    explore_df = combined.copy()
    if stage_filter != "All":
        explore_df = explore_df[explore_df["stage"] == stage_filter]
    if label_filter != "All":
        explore_df = explore_df[explore_df["label"] == label_filter]
    if industry_filter != "All":
        explore_df = explore_df[explore_df["Industry"].astype(str) == industry_filter]
    explore_df = explore_df.sort_values(["Company Name", "ticker"], ascending=[True, True])
    st.caption("Alphabetical listing with filters. This is a descriptive dataset view, not a recommendation view.")
    for _, r in explore_df.head(40).iterrows():
        card(r, use_stage_color=True, stock_rank=get_stock_rank(r["ticker"]))
    render_disclosure()

with tabs[2]:
    st.markdown("### Movers")
    if moves.empty:
        st.info("Price move data not found yet.")
    else:
        window_map = {"1 Day":"change_1d_pct","1 Week":"change_1w_pct","1 Month":"change_1m_pct","YTD":"change_ytd_pct"}
        selected = st.radio("Move window", list(window_map.keys()), horizontal=True, index=0, key="movers_window")
        col = window_map[selected]
        mv = moves.copy()
        mv[col] = pd.to_numeric(mv[col], errors="coerce")
        mv = mv.dropna(subset=[col])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### Biggest upward moves • {selected}")
            for _, r in mv.sort_values([col, "Company Name"], ascending=[False, True]).head(10).iterrows():
                stock_rank = get_stock_rank(r["ticker"])
                card(r, pct=float(r[col]), use_stage_color=True, stock_rank=stock_rank)
        with c2:
            st.markdown(f"#### Major downward moves • {selected}")
            for _, r in mv.sort_values([col, "Company Name"], ascending=[True, True]).head(10).iterrows():
                stock_rank = get_stock_rank(r["ticker"])
                card(r, pct=float(r[col]), use_stage_color=True, stock_rank=stock_rank)
    render_disclosure()

with tabs[3]:
    st.markdown("### Watchlist")
    portfolio_options = ["Custom", "Strong", "Developing", "Cautious", "Weak", "Stage 1", "Stage 2", "Stage 3", "Stage 4"] + INDUSTRY_PORTFOLIOS
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
    if st.button("Add to watchlist", use_container_width=True, key="watchlist_add_btn") and selected_to_add:
        selected_name = selected_to_add.rsplit(" (", 1)[0]
        st.session_state["watchlist_names"] = dedupe_names(st.session_state["watchlist_names"] + [selected_name], limit=MAX_PORTFOLIO_STOCKS)
        st.session_state["custom_watchlist_names"] = dedupe_names(st.session_state["watchlist_names"], limit=MAX_PORTFOLIO_STOCKS)
        st.rerun()
    if not st.session_state["watchlist_names"]:
        st.info("No stocks added yet.")
    else:
        current = combined[combined["Company Name"].isin(st.session_state["watchlist_names"])].copy().sort_values(["Company Name", "ticker"])
        watch_counts = stage_count_summary(current)
        render_distribution(watch_counts)
        st.markdown("#### Watchlist cards")
        for _, r in current.iterrows():
            card(r, use_stage_color=True, stock_rank=get_stock_rank(r["ticker"]))
        portfolio_ordered = current.sort_values(["Company Name", "ticker"], ascending=[True, True]).reset_index(drop=True)
        if not portfolio_ordered.empty:
            st.divider()
            st.markdown("### Watchlist charts")
            st.session_state["watchlist_chart_index"] = max(0, min(st.session_state["watchlist_chart_index"], len(portfolio_ordered) - 1))
            prow = portfolio_ordered.iloc[st.session_state["watchlist_chart_index"]]
            pticker_short = str(prow["ticker"]).replace(".NS", "")

            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown(f"#### Daily chart • {pticker_short} • Dataset Rank {get_stock_rank(prow['ticker'])}")
                pdpath = resolve_chart_path(daily_dir, prow["ticker"], "_daily.png")
                if pdpath:
                    st.image(safe_image_bytes(pdpath), use_container_width=True)
                else:
                    st.info("Daily chart not available.")
            with pc2:
                st.markdown(f"#### Weekly chart • {pticker_short} • Dataset Rank {get_stock_rank(prow['ticker'])}")
                pwpath = resolve_chart_path(weekly_dir, prow["ticker"], "_weekly.png")
                if pwpath:
                    st.image(safe_image_bytes(pwpath), use_container_width=True)
                else:
                    st.info("Weekly chart not available.")

            nav1, nav2 = st.columns(2)
            with nav1:
                wprev = st.button("Previous", use_container_width=True, disabled=(st.session_state["watchlist_chart_index"] == 0), key="watchlist_prev_btn")
            with nav2:
                wnext = st.button("Next", use_container_width=True, disabled=(st.session_state["watchlist_chart_index"] >= len(portfolio_ordered) - 1), key="watchlist_next_btn")

            if wprev and st.session_state["watchlist_chart_index"] > 0:
                st.session_state["watchlist_chart_index"] -= 1
                st.rerun()
            if wnext and st.session_state["watchlist_chart_index"] < len(portfolio_ordered) - 1:
                st.session_state["watchlist_chart_index"] += 1
                st.rerun()

            card(prow, use_stage_color=True, stock_rank=get_stock_rank(prow["ticker"]))
            render_stock_detail(prow)

        removable = [""] + sorted(st.session_state["watchlist_names"])
        selected_remove = st.selectbox("Remove stock", removable, key="watchlist_remove_name")
        if st.button("Remove from watchlist", use_container_width=True, key="watchlist_remove_btn") and selected_remove:
            st.session_state["watchlist_names"] = [x for x in st.session_state["watchlist_names"] if x != selected_remove]
            st.session_state["custom_watchlist_names"] = dedupe_names(st.session_state["watchlist_names"], limit=MAX_PORTFOLIO_STOCKS)
            st.rerun()
    render_disclosure()

with tabs[4]:
    st.markdown("### Charts")
    ranked_alpha = combined.sort_values(["Company Name", "ticker"], ascending=[True, True]).reset_index(drop=True).copy()
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
tab_offset = 5
if view_mode == "Pro":
    with tabs[5]:
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
            if industry_changes.empty:
                st.info("Industry changes data not available.")
            else:
                cols = [c for c in ["Industry", "current_rank", "prev_rank", "rank_change"] if c in industry_changes.columns]
                st.dataframe(industry_changes[cols], use_container_width=True, hide_index=True, height=520)
        render_disclosure()
    with tabs[6]:
        st.markdown("### Structure Changes")
        if alert_candidates.empty:
            st.info("No structure-change rows were found in the latest data.")
        else:
            for _, r in alert_candidates.iterrows():
                card(r, use_stage_color=True, show_change_text=f"{r['alert_type']} • {r['alert_reason']}", stock_rank=get_stock_rank(r["ticker"]))
        render_disclosure()
    tab_offset = 7

with tabs[tab_offset]:
    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown("""
<div class="learn-card">
  <div class="stock-title">How to use this app</div>
  <ul class="list-tight">
    <li>Start with <b>Today</b> to understand the market mode and recent changes.</li>
    <li>Use <b>Explore</b> to filter the dataset by stage, label, and industry.</li>
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

with tabs[tab_offset + 1]:
    st.markdown("### Disclaimer")
    st.write("This tool is for informational purposes only. It presents rule-based stage classifications and market summaries. It does not provide investment advice, recommendations, or opinions on buying, selling, or holding securities. It does not rank, recommend, prioritize, or suggest any securities for investment purposes, and it does not provide model portfolios, suitability analysis, or allocation recommendations.")
    render_disclosure()
