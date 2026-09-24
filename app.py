import streamlit as st
import pandas as pd
import os
import sys
import subprocess
from datetime import datetime
import re
from functionlist import metrics
import ast
import plotly.express as px
import numpy as np
import json
import scipy.stats as stats

st.set_page_config(
    page_title="Japan Buybacks Stock Performance Prediction Programme",
    page_icon="🗾",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("modeldata/confidence_error.json","r") as f:
    ERROR_LOOKUP = json.load(f)

PREDICTIONS_FILE = "modeldata/predictions.json"

def load_predictions():

    if os.path.exists(PREDICTIONS_FILE):

        with open(PREDICTIONS_FILE, "r") as f:
            return json.load(f)

    return {}
def save_predictions(data):
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(data, f, indent=4)

DASH_TABLE_CONFIG = {
              "bbg_ticker": st.column_config.TextColumn("Bloomberg Ticker"),
              "alpha1": st.column_config.NumberColumn("Alpha after Day 1",format="%.2f%%"),
              "alpha5": st.column_config.NumberColumn("Alpha after Day 5",format="%.2f%%"),
              "continuation": st.column_config.NumberColumn("Continuation",format="%.2f%%"),
              "date": st.column_config.DateColumn("Date"),
            }
BUYBACK_TABLE_CONFIG = {
              "# id": st.column_config.NumberColumn("Trade ID"),
              "date": st.column_config.DateColumn("Date"),
              "bbg_ticker": st.column_config.TextColumn("Bloomberg Ticker"),
              "name": st.column_config.TextColumn("Company Name"),
              "shares": st.column_config.NumberColumn("Desired Shares",format="%,.0f"),
              "purchased_shares": st.column_config.NumberColumn("Purchased Shares",format="%,.0f"),
              "price": st.column_config.NumberColumn("Price",format="¥%,.0f"),
              "notional_usd_m": st.column_config.NumberColumn("Buyback Size (m USD)", format="$%,.2f"),
              "total_cap_m_usd": st.column_config.NumberColumn("Total Cap (m USD)", format="$%,.2f"),
              "shs_cap_m_usd": st.column_config.NumberColumn("Shares Cap (m USD)", format="$%,.2f"),
              "eqy_float_usd": st.column_config.NumberColumn("Equity Float (m USD)", format="$%,.2f"),
              "beta": st.column_config.NumberColumn("Beta"),
              "vol": st.column_config.NumberColumn("Volatility"),
              "adv_1m_usd":st.column_config.NumberColumn("1-month ADV (m USD)", format="$%,.2f"),
              "adv_5d_usd":st.column_config.NumberColumn("5-day ADV (m USD)", format="$%,.2f"),
              "dtc": st.column_config.NumberColumn("Days to Cover"),
              "indicative_fee_pct": st.column_config.NumberColumn("Indicative Fee Pct",format="%,.2f%%"),
              "adj_vwap": st.column_config.NumberColumn("VWAP (m USD)", format="$%,.2f"),
              "perf1_pct": st.column_config.NumberColumn("1-day stock performance", format="%,.2f%%"),
              "bmk_perf1_pct": st.column_config.NumberColumn("1-day index Performance", format="%,.2f%%"),
              "perf5_pct": st.column_config.NumberColumn("5-day stock performance", format="%,.2f%%"),
              "bmk_perf5_pct": st.column_config.NumberColumn("5-day index Performance", format="%,.2f%%"),
              "px_book_ratio": st.column_config.NumberColumn("Price-Book Ratio", format="%,.2f"),
              "cash_usd_m": st.column_config.NumberColumn("Cash Balance (m USD)", format="$%,.2f"),
              "total_assets": st.column_config.NumberColumn("Total Assets (m USD)", format="$%,.2f"),
              "fcf": st.column_config.NumberColumn("Free Cash Flow (m USD)", format="$%,.2f"),
              "ebit_interest_ratio": st.column_config.NumberColumn("EBIT-Interest Ratio", format="%,.2f"),
              "fixed_assets": st.column_config.NumberColumn("Fixed Assets (m USD)", format="$%,.2f"),
              "operating_income": st.column_config.NumberColumn("Operating Income (m USD)", format="$%,.2f"),
              "debt": st.column_config.NumberColumn("Debt (m USD)", format="$%,.2f"),
              "tobin_q_ratio": st.column_config.NumberColumn("Tobin Q Ratio", format="%,.2f"),
              "perf20_pct": st.column_config.NumberColumn("Previous 20-day stock performance", format="%.2f%%"),
              "bmk_perf20_pct": st.column_config.NumberColumn("Previous 20-day index performance", format="%.2f%%"),
              "alpha1": st.column_config.NumberColumn("1-day Alpha", format="%,.2f%%"),
              "alpha5": st.column_config.NumberColumn("5-day Alpha", format="%,.2f%%"),
              "buyback_adv": st.column_config.NumberColumn("Buyback / ADV", format="%,.2f"),
              "buyback_cap": st.column_config.NumberColumn("Buyback / Cap", format="%,.2f"),
              "buyback_float": st.column_config.NumberColumn("Buyback / Float", format="%,.2f"),
              "premium": st.column_config.NumberColumn("Premium", format="%,.2f%%"),
              "volume_shock": st.column_config.NumberColumn("Volume Shock", format="%,.2f"),
              "stress": st.column_config.NumberColumn("Stress", format="%,.2f"),
              "abs_alpha1": st.column_config.NumberColumn("Absolute Alpha1", format="%,.2f"),
              "attention": st.column_config.NumberColumn("Attention", format="%,.2f"),
              "premium_significance": st.column_config.NumberColumn("Premium Significance", format="%,.2f"),
              "alpha1_x_attention": st.column_config.NumberColumn("Alpha1 x Attention", format="%,.2f"),
              "alpha1_x_premium": st.column_config.NumberColumn("Alpha1 x Premium", format="%,.2f"),
              "alpha1_x_buyback_adv": st.column_config.NumberColumn("Alpha1 x Buyback / ADV", format="%,.2f"),
              "alpha1_x_vol": st.column_config.NumberColumn("Alpha1 x Volatility", format="%,.2f"),
              "alpha1_z": st.column_config.NumberColumn("Alpha1 / Volatility", format="%,.2f"),
              "alpha1_x_short": st.column_config.NumberColumn("Alpha1 * DTC", format="%,.2f"),
              "alpha1_x_buyback_cap": st.column_config.NumberColumn("Alpha1 * Buyback / Market Cap", format="%,.2f"),
              "reaction_efficiency": st.column_config.NumberColumn("Reaction Efficiency", format="%,.2f"),
              "buyback_to_vol": st.column_config.NumberColumn("Buyback / Volatility", format="%,.2f"),
              "buyback_x_volume": st.column_config.NumberColumn("Buyback * Volume Shock", format="%,.2f"),
              "buyback_x_stress": st.column_config.NumberColumn("Buyback * Stress", format="%,.2f"),
              "squeeze_score": st.column_config.NumberColumn("Squeeze Score", format="%,.2f"),
              "buyback_short_cover": st.column_config.NumberColumn("Buyback / DTC", format="%,.2f"),
              "turnover": st.column_config.NumberColumn("Turnover", format="%,.2f"),
              "illiquidity": st.column_config.NumberColumn("Illiquidity", format="%,.2f"),
              "premium_abs": st.column_config.NumberColumn("Absolute Premium", format="%,.2f"),
              "premium_x_volume": st.column_config.NumberColumn("Premium * Volume", format="%,.2f"),
              "cash_percent": st.column_config.NumberColumn("Deal Size / Cash Balance", format="%,.2f"),
              "cash_by_assets": st.column_config.NumberColumn("Cash Balance / Total Assets", format="%,.2f"),
              "fcf_by_assets": st.column_config.NumberColumn("FCF / Total Assets", format="%,.2f"),
              "log_assets": st.column_config.NumberColumn("Log10 Total Assets", format="%,.2f"),
              "asset_ratio": st.column_config.NumberColumn("Fixed / Total Assets", format="%,.2f"),
              "debt_ratio": st.column_config.NumberColumn("Debt / Total Assets", format="%,.2f"),
              "operating_roa": st.column_config.NumberColumn("Operating Income / Total Assets", format="%,.2f"),
              "net_debt_by_assets": st.column_config.NumberColumn("Net Debt / Total Assets", format="%,.2f"),
              "buyback_x_coverage_stress": st.column_config.NumberColumn("Buyback * EBIT-Interest Ratio", format="%,.2f"),
              "cash_stress": st.column_config.NumberColumn("Deal Size / Total Assets", format="%,.2f"),
              "valuation_divergence": st.column_config.NumberColumn("Price-Book - Tobin Q", format="%,.2f"),
              "alpha20": st.column_config.NumberColumn("Previous 20-day alpha", format="%,.2f%%"),
              "distressed_value_buyback": st.column_config.NumberColumn("Buyback / -ve Alpha20", format="%,.2f"),
              "momentum_chasing_buyback": st.column_config.NumberColumn("Buyback / +ve Alpha20", format="%,.2f"),
              "squeeze_setup": st.column_config.NumberColumn("Squeeze Setup", format="%,.2f"),
              "alpha1_vs_trend": st.column_config.NumberColumn("Alpha1 vs Momentum", format="%,.2f"),
              "contains_weekend": st.column_config.CheckboxColumn("Weekend during 5-day"),
              "weekend_before": st.column_config.CheckboxColumn("Weekend before Buyback"),
            }
UNIVERSE_TABLE_CONFIG = {
              "# id": st.column_config.NumberColumn("Trade ID"),
              "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
              "bbg_ticker": st.column_config.TextColumn("Bloomberg Ticker"),
              "name": st.column_config.TextColumn("Company Name"),
              "shares": st.column_config.NumberColumn("Desired Shares",format="%,.0f"),
              "purchased_shares": st.column_config.NumberColumn("Purchased Shares",format="%,.0f"),
              "Predicted Continuation": st.column_config.NumberColumn("Predicted Continuation",format="%,.2f"),
              "Effective Continuation": st.column_config.NumberColumn("Effective Continuation",format="%,.2f"),
              "price": st.column_config.NumberColumn("Price",format="¥%,.0f"),
              "notional_usd_m": st.column_config.NumberColumn("Buyback Size (m USD)", format="$%,.2f"),
              "total_cap_m_usd": st.column_config.NumberColumn("Total Cap (m USD)", format="$%,.2f"),
              "shs_cap_m_usd": st.column_config.NumberColumn("Shares Cap (m USD)", format="$%,.2f"),
              "eqy_float_usd": st.column_config.NumberColumn("Equity Float (m USD)", format="$%,.2f"),
              "beta": st.column_config.NumberColumn("Beta"),
              "vol": st.column_config.NumberColumn("Volatility"),
              "adv_1m_usd":st.column_config.NumberColumn("1-month ADV (m USD)", format="$%,.2f"),
              "adv_5d_usd":st.column_config.NumberColumn("5-day ADV (m USD)", format="$%,.2f"),
              "dtc": st.column_config.NumberColumn("Days to Cover"),
              "indicative_fee_pct": st.column_config.NumberColumn("Indicative Fee Pct",format="%,.2f%%"),
              "adj_vwap": st.column_config.NumberColumn("VWAP (m USD)", format="$%,.2f"),
              "perf1_pct": st.column_config.NumberColumn("1-day stock performance", format="%,.2f%%"),
              "bmk_perf1_pct": st.column_config.NumberColumn("1-day index Performance", format="%,.2f%%"),
              "perf5_pct": st.column_config.NumberColumn("5-day stock performance", format="%,.2f%%"),
              "bmk_perf5_pct": st.column_config.NumberColumn("5-day index Performance", format="%,.2f%%"),
              "px_book_ratio": st.column_config.NumberColumn("Price-Book Ratio", format="%,.2f"),
              "cash_usd_m": st.column_config.NumberColumn("Cash Balance (m USD)", format="$%,.2f"),
              "total_assets": st.column_config.NumberColumn("Total Assets (m USD)", format="$%,.2f"),
              "fcf": st.column_config.NumberColumn("Free Cash Flow (m USD)", format="$%,.2f"),
              "ebit_interest_ratio": st.column_config.NumberColumn("EBIT-Interest Ratio", format="%,.2f"),
              "fixed_assets": st.column_config.NumberColumn("Fixed Assets (m USD)", format="$%,.2f"),
              "operating_income": st.column_config.NumberColumn("Operating Income (m USD)", format="$%,.2f"),
              "debt": st.column_config.NumberColumn("Debt (m USD)", format="$%,.2f"),
              "tobin_q_ratio": st.column_config.NumberColumn("Tobin Q Ratio", format="%,.2f"),
              "perf20_pct": st.column_config.NumberColumn("Previous 20-day stock performance", format="%.2f%%"),
              "bmk_perf20_pct": st.column_config.NumberColumn("Previous 20-day index performance", format="%.2f%%"),
              "alpha1": st.column_config.NumberColumn("1-day Alpha", format="%,.2f%%"),
              "alpha5": st.column_config.NumberColumn("5-day Alpha", format="%,.2f%%"),
              "buyback_adv": st.column_config.NumberColumn("Buyback / ADV", format="%,.2f"),
              "buyback_cap": st.column_config.NumberColumn("Buyback / Cap", format="%,.2f"),
              "buyback_float": st.column_config.NumberColumn("Buyback / Float", format="%,.2f"),
              "premium": st.column_config.NumberColumn("Premium", format="%,.2f%%"),
              "volume_shock": st.column_config.NumberColumn("Volume Shock", format="%,.2f"),
              "stress": st.column_config.NumberColumn("Stress", format="%,.2f"),
              "abs_alpha1": st.column_config.NumberColumn("Absolute Alpha1", format="%,.2f"),
              "attention": st.column_config.NumberColumn("Attention", format="%,.2f"),
              "premium_significance": st.column_config.NumberColumn("Premium Significance", format="%,.2f"),
              "alpha1_x_attention": st.column_config.NumberColumn("Alpha1 x Attention", format="%,.2f"),
              "alpha1_x_premium": st.column_config.NumberColumn("Alpha1 x Premium", format="%,.2f"),
              "alpha1_x_buyback_adv": st.column_config.NumberColumn("Alpha1 x Buyback / ADV", format="%,.2f"),
              "alpha1_x_vol": st.column_config.NumberColumn("Alpha1 x Volatility", format="%,.2f"),
              "alpha1_z": st.column_config.NumberColumn("Alpha1 / Volatility", format="%,.2f"),
              "alpha1_x_short": st.column_config.NumberColumn("Alpha1 * DTC", format="%,.2f"),
              "alpha1_x_buyback_cap": st.column_config.NumberColumn("Alpha1 * Buyback / Market Cap", format="%,.2f"),
              "reaction_efficiency": st.column_config.NumberColumn("Reaction Efficiency", format="%,.2f"),
              "buyback_to_vol": st.column_config.NumberColumn("Buyback / Volatility", format="%,.2f"),
              "buyback_x_volume": st.column_config.NumberColumn("Buyback * Volume Shock", format="%,.2f"),
              "buyback_x_stress": st.column_config.NumberColumn("Buyback * Stress", format="%,.2f"),
              "squeeze_score": st.column_config.NumberColumn("Squeeze Score", format="%,.2f"),
              "buyback_short_cover": st.column_config.NumberColumn("Buyback / DTC", format="%,.2f"),
              "turnover": st.column_config.NumberColumn("Turnover", format="%,.2f"),
              "illiquidity": st.column_config.NumberColumn("Illiquidity", format="%,.2f"),
              "premium_abs": st.column_config.NumberColumn("Absolute Premium", format="%,.2f"),
              "premium_x_volume": st.column_config.NumberColumn("Premium * Volume", format="%,.2f"),
              "cash_percent": st.column_config.NumberColumn("Deal Size / Cash Balance", format="%,.2f"),
              "cash_by_assets": st.column_config.NumberColumn("Cash Balance / Total Assets", format="%,.2f"),
              "fcf_by_assets": st.column_config.NumberColumn("FCF / Total Assets", format="%,.2f"),
              "log_assets": st.column_config.NumberColumn("Log10 Total Assets", format="%,.2f"),
              "asset_ratio": st.column_config.NumberColumn("Fixed / Total Assets", format="%,.2f"),
              "debt_ratio": st.column_config.NumberColumn("Debt / Total Assets", format="%,.2f"),
              "operating_roa": st.column_config.NumberColumn("Operating Income / Total Assets", format="%,.2f"),
              "net_debt_by_assets": st.column_config.NumberColumn("Net Debt / Total Assets", format="%,.2f"),
              "buyback_x_coverage_stress": st.column_config.NumberColumn("Buyback * EBIT-Interest Ratio", format="%,.2f"),
              "cash_stress": st.column_config.NumberColumn("Deal Size / Total Assets", format="%,.2f"),
              "valuation_divergence": st.column_config.NumberColumn("Price-Book - Tobin Q", format="%,.2f"),
              "alpha20": st.column_config.NumberColumn("Previous 20-day alpha", format="%,.2f%%"),
              "distressed_value_buyback": st.column_config.NumberColumn("Buyback / -ve Alpha20", format="%,.2f"),
              "momentum_chasing_buyback": st.column_config.NumberColumn("Buyback / +ve Alpha20", format="%,.2f"),
              "squeeze_setup": st.column_config.NumberColumn("Squeeze Setup", format="%,.2f"),
              "alpha1_vs_trend": st.column_config.NumberColumn("Alpha1 vs Momentum", format="%,.2f"),
              "contains_weekend": st.column_config.CheckboxColumn("Weekend during 5-day"),
              "weekend_before": st.column_config.CheckboxColumn("Weekend before Buyback"),
              "continuation": st.column_config.NumberColumn("Continuation", format="%,.2f%%"),
            }

st.markdown("""
<style>

.kpi-card {
    background: white;
    padding: 18px;
    border-radius: 10px;
    border-left: 5px solid #064147;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 10px;
}

.kpi-title {
    color: #64748B;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi-value {
    color: #064147;
    font-size: 30px;
    font-weight: 700;
    margin-top: 5px;
}

.kpi-sub {
    color: #94A3B8;
    font-size: 12px;
    margin-top: 3px;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* Sidebar width */
section[data-testid="stSidebar"] {
    width: 320px !important;
}

/* Sidebar content scroll */
section[data-testid="stSidebar"] > div {
    overflow-y: auto !important;
}

/* When collapsed, reclaim space */
[data-testid="collapsedControl"] {
    z-index: 9999;
}

/* Remove reserved width when sidebar collapsed */
section[data-testid="stSidebar"][aria-expanded="false"] {
    margin-left: -0px !important;
}

/* Ensure main page expands */
.main .block-container {
    max-width: 100%;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Main container background */
    .stApp {
        background-color: #F4F6F8;
        color: #1E293B;
    }

    /* --- ICON BUTTON STYLING --- */
    div[role="radiogroup"] > label > div:nth-child(2) {
        font-size: 22px !important;
        display: flex !important;
        justify-content: left !important;
        align-items: left !important;
        cursor: pointer;
        width: 150px;
        height: 25px;
        margin: 0 auto 12px auto !important; 
        border-radius: 8px !important;
        border: 2px solid transparent !important; 
        background-color: transparent !important;
        color: #064147 !important;
        transition: all 0.2s ease;
    }

    div[role="radiogroup"] > label:has(input:checked) > div:nth-child(2) {
        background-color: #CFD8D7 !important; 
        border: 2px solid #004D40 !important; 
        color: #004D40 !important;
    }

    div[role="radiogroup"] > label:not(:has(input:checked)):hover > div:nth-child(2) {
        background-color: #BACBCA !important; 
    }

    /* --- FLOATING TOOLTIPS --- */
    div[role="radiogroup"] > label { position: relative !important; }
    div[role="radiogroup"] > label::after {
        position: absolute;
        left: 60px; 
        top: 50%;
        transform: translateY(-50%);
        background-color: #BACBCA; 
        color: #00332C; 
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 14px;
        font-weight: 500;
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s ease;
        z-index: 999999 !important;
        box-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Standard Header blocks (The Dark Teal Title Cards) */
    .dark-teal-header {
        background-color: #064147;
        color: #FFFFFF;
        padding: 10px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 16px;
    }

    /* Global Button Styling */
    .stButton>button {
        background-color: #064147;
        color: white !important;
        border-radius: 6px;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover { background-color: #00332C; }

    /* OVERRIDE STREAMLIT ORANGE FOCUS RINGS TO DARK TEAL */
    div[data-baseweb="input"]:focus-within {
        border-color: #064147 !important;
    }

    div[data-baseweb="input"]:focus-within {
        box-shadow: inset 0 0 0 1px #064147 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to run scripts safely
def run_python_script(script_name, args=None):
    cmd = [sys.executable, script_name]
    if args: cmd.extend(args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)
def get_buyback_data():
    file_path = "jp_buyback.xlsx"
    if os.path.exists(file_path):
        try:
            return pd.read_excel(file_path)

        except Exception:
            return None

    return None


# Sidebar Navigation
# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:

    st.markdown("## Alpha5 Platform")

    latest_date = "N/A"
    queue_size = 0

    try:

        sidebar_df = get_buyback_data()

        if sidebar_df is not None:
            date_col_sidebar = next((c for c in sidebar_df.columns if "date" in c.lower()), None)

            if date_col_sidebar:
                latest_date = (pd.to_datetime(sidebar_df[date_col_sidebar],errors="coerce").max().strftime("%Y-%m-%d"))

            try:
                sidebar_df = metrics(sidebar_df,"a")
                alpha5_col_sidebar = next((c for c in sidebar_df.columns if c.lower().replace("_", "") == "alpha5"), None)
                alpha1_col_sidebar = next((c for c in sidebar_df.columns if c.lower().replace("_", "") == "alpha1"), None)

                if (alpha5_col_sidebar and alpha1_col_sidebar):
                    continuation = (pd.to_numeric(sidebar_df[alpha5_col_sidebar],errors="coerce") - pd.to_numeric(sidebar_df[alpha1_col_sidebar],errors="coerce"))
                    queue_size = (continuation.isna().sum())

            except Exception:
                pass

    except Exception:
        pass

    st.markdown("---")
    st.caption("Platform Status")
    st.markdown(
        f"""
        **Last Data:** {latest_date}
        
        **Prediction Queue:** {queue_size:,}
        """
    )

    st.markdown("---")
    st.caption("Research")

    selected_icon = st.radio("",["📊 Dashboard","📋 Universe","🔍 Stock Checker","⚙️ Alpha5 Engine","📈 Plots"],)

    st.markdown("---")
    st.caption("System")

    if st.button("🔄 Data Sync", width="stretch"):
        with st.spinner("Updating buyback data..."):
            code, stdout, stderr = run_python_script("datapull.py")

            if code == 0:
                st.success("Data sync completed.")
                run_python_script("alpha5model.py")
            else:
                st.error("Data sync failed.")



    st.markdown("---")
    st.caption("Version")
    st.text("Alpha5 v1.0")
# ---------------------------------------------------------
# MODULE 0: Home / Summary
# ---------------------------------------------------------
if selected_icon == "📊 Dashboard":
    st.markdown('<div class="dark-teal-header">JP Buyback Dashboard</div>',unsafe_allow_html=True)
    df = get_buyback_data()

    if df is not None:

        try:
            df = metrics(df, "a")
        except Exception:
            pass

        # --------------------------------------------------
        # COLUMN DETECTION
        # --------------------------------------------------

        date_col = next((c for c in df.columns if "date" in c.lower()),None)
        ticker_col = next((
                c for c in df.columns
                if "ticker" in c.lower()
                or "code" in c.lower()
                or "bbg" in c.lower()
            ),None)

        alpha5_col = next((c for c in df.columns if "alpha5" in c.lower()),None)
        alpha1_col = next((c for c in df.columns if "alpha1" in c.lower()),None)

        # --------------------------------------------------
        # DATA CLEANING
        # --------------------------------------------------

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col],errors="coerce",format="%Y-%m-%d")

        if alpha5_col:
            df[alpha5_col] = pd.to_numeric(df[alpha5_col],errors="coerce")

        if alpha1_col:
            df[alpha1_col] = pd.to_numeric(df[alpha1_col],errors="coerce")

        # --------------------------------------------------
        # CONTINUATION
        # --------------------------------------------------

        if alpha1_col and alpha5_col:
            df["continuation"] = (df[alpha5_col] - df[alpha1_col])

        else:
            st.warning("Alpha1 and Alpha5 columns are required to calculate continuation.")
            st.stop()

        # --------------------------------------------------
        # DATA FRESHNESS
        # --------------------------------------------------

        if date_col and not df[date_col].isna().all():
            latest_update = df[date_col].max()
            st.info(f"📅 Data Freshness: {latest_update:%Y-%m-%d}")

        # --------------------------------------------------
        # KPI CALCULATIONS
        # --------------------------------------------------

        avg_cont = df["continuation"].mean()
        median_cont = df["continuation"].median()
        max_cont = df["continuation"].max()

        unique_stocks = (df[ticker_col].nunique()
            if ticker_col
            else 0
        )

        recent_buybacks = 0

        if date_col:
            cutoff = pd.Timestamp.today() - pd.Timedelta(days=30)
            recent_buybacks = (pd.to_datetime(df[date_col],errors="coerce",format="%Y-%m-%d") >= cutoff).sum()

        cont_p90 = df["continuation"].quantile(0.90)
        missing_cont = (df["continuation"].isna().sum())
        avg_colour = ("#16A34A" if avg_cont >= 0 else "#DC2626")

        # --------------------------------------------------
        # KPI ROW 1
        # --------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">BUYBACK EVENTS</div>
                <div class="kpi-value">{len(df):,}</div>
                <div class="kpi-sub">Total observations</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">AVG CONTINUATION</div>
                <div class="kpi-value" style="color:{avg_colour};">
                    {avg_cont:.2f}
                </div>
                <div class="kpi-sub">Alpha5 - Alpha1</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">BEST CONTINUATION</div>
                <div class="kpi-value">
                    {max_cont:.2f}
                </div>
                <div class="kpi-sub">Highest signal observed</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">MEDIAN CONTINUATION</div>
                <div class="kpi-value">
                    {median_cont:.2f}
                </div>
                <div class="kpi-sub">Central tendency</div>
            </div>
            """, unsafe_allow_html=True)

        # --------------------------------------------------
        # KPI ROW 2
        # --------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">UNIQUE STOCKS</div>
                <div class="kpi-value">
                    {unique_stocks}
                </div>
                <div class="kpi-sub">Coverage universe</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">RECENT BUYBACKS</div>
                <div class="kpi-value">
                    {recent_buybacks}
                </div>
                <div class="kpi-sub">Last 30 days</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">90TH %ILE CONT</div>
                <div class="kpi-value">
                    {cont_p90:.2f}
                </div>
                <div class="kpi-sub">Top decile threshold</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">MISSING VALUES</div>
                <div class="kpi-value">
                    {missing_cont}
                </div>
                <div class="kpi-sub">Continuation unavailable</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # --------------------------------------------------
        # BUYBACK ACTIVITY CHART
        # --------------------------------------------------

        if date_col and not df[date_col].isna().all():
            # activity_df = (df.groupby(df[date_col].dt.date)
            #       .size()
            #       .reset_index(name="Buybacks")
            # )
            activity_df = (df.groupby(pd.Grouper(key=date_col, freq="W-FRI")).size().reset_index(name="Buybacks"))

            activity_df.columns = ["Date","Buybacks"]
            # full_dates = pd.date_range(start=activity_df["Date"].min(),end=activity_df["Date"].max(),freq="D")
            # activity_df = (activity_df.set_index("Date").reindex(full_dates.date,fill_value=0).reset_index())
            activity_df.columns = ["Date", "Buybacks"]
            fig = px.line(activity_df,x="Date",y="Buybacks",markers=True)
            fig.update_traces(line=dict(color="#064147",width=2),marker=dict(color="#064147",size=5))

            # Add markers only for non-zero points

            # marker_df = activity_df[activity_df["Buybacks"] > 0]
            # fig.add_scatter(
            #     x=marker_df["Date"],
            #     y=marker_df["Buybacks"],
            #     mode="markers",
            #     marker=dict(
            #         color="#064147",
            #         size=5
            #     ),
            #     showlegend=False,
            #     hoverinfo="skip")

            fig.update_layout(
                title="Buyback Activity Trend",
                template="plotly_white",
                height=450,
                showlegend=False,
                title_x=0,
                xaxis_title="",
                yaxis_title="Number of Buybacks"
            )

            st.plotly_chart(fig,width="stretch")

        st.markdown("---")

        # --------------------------------------------------
        # TABLES
        # --------------------------------------------------
        def colour_cont(v):
            if pd.isna(v):
                return ""
            if v > 0:
                return "background-color: #D4EDDA"
            return "background-color: #F8D7DA"

        left_col, right_col = st.columns(2)

        display_cols = [c for c in [ticker_col,alpha1_col,alpha5_col,"continuation",date_col]if c in df.columns]

        with left_col:
            st.subheader("📈 Best Continuations")
            top_df = (df.sort_values("continuation",ascending=False).head(15))
            top_df_display = (top_df[display_cols].style.map(colour_cont,subset=["continuation"]))
            st.dataframe(top_df_display,hide_index=True,height=400,width="stretch",column_config=DASH_TABLE_CONFIG)

        with right_col:
            st.subheader("📉 Worst Continuations")
            bottom_df = (df.sort_values("continuation",ascending=True).head(15))
            bottom_df_display = (bottom_df[display_cols].style.map(colour_cont,subset=["continuation"]))
            st.dataframe(bottom_df_display,hide_index=True,height=400,width="stretch",column_config=DASH_TABLE_CONFIG)

        st.markdown("---")
        st.subheader("📋 Most Recent Buybacks")
        recent_df = (df.sort_values(date_col,ascending=False).head(20))
        st.dataframe(recent_df[display_cols],hide_index=True,width="stretch",height=350,column_config=DASH_TABLE_CONFIG)
        st.markdown("---")

        fig = px.histogram(df,x="continuation",nbins=40)
        fig.update_traces(marker_color="#064147")
        fig.update_layout(title="Continuation Distribution",template="plotly_white",height=350)
        st.plotly_chart(fig,width="stretch")

        # --------------------------------------------------
        # DOWNLOAD
        # --------------------------------------------------
        st.markdown("---")
        st.download_button("⬇ Download Full Dataset",df.to_csv(index=False),"jp_buybacks.csv","text/csv")

    else:
        st.warning("'jp_buyback.xlsx' not found. Please ensure it is in the project root folder.")
# ---------------------------------------------------------
# MODULE 1: JP Buyback Data Viewer & Realtime Update
# ---------------------------------------------------------
elif selected_icon == "📋 Universe":
    st.markdown('<div class="dark-teal-header">JP Buyback Data</div>', unsafe_allow_html=True)

    df = get_buyback_data()
    df = metrics(df, "a")
    min_date, max_date = None, None
    date_col = None
    if df is not None:
        date_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'day' in c.lower()), None)
        if date_col:
            try:
                valid_dates = pd.to_datetime(df[date_col],format="%Y-%m-%d").dt.date.dropna()
                if not valid_dates.empty:
                    min_date = valid_dates.min()
                    max_date = valid_dates.max()
            except Exception:
                pass

    latest_date = df[date_col].max()
    st.caption(f"Last update: {latest_date:%Y-%m-%d}")

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce',format="%Y-%m-%d")
        df = df.sort_values(by=date_col, ascending=False)

    col_date, col_search, col_spacer, col_cols = st.columns([1.5, 2.5, 5, 1.5])

    with col_date:
        filter_date = st.date_input("Date Filter", value=None, min_value=min_date, max_value=max_date, label_visibility="collapsed")
    with col_search:
        search_query = st.text_input("Search", placeholder="Ticker code...", label_visibility="collapsed")

    if df is not None:
        all_cols = df.columns.tolist()
        with col_cols:
            with st.popover("Columns", width="stretch"):
                selected_cols = []
                for col in all_cols:
                    if st.checkbox(col, value=True):
                        selected_cols.append(col)

        df_display = df.copy()

        # 1. Date Filter Logic
        if filter_date and date_col:
            try:
                temp_dates = pd.to_datetime(df_display[date_col],format="%Y-%m-%d").dt.date
                df_display = df_display[temp_dates == filter_date]
            except Exception:
                pass

        # 2. Strict Ticker/Code Search Logic
        if search_query:
            clean_query = str(search_query).strip()

            ticker_col = next((c for c in df_display.columns
                               if 'ticker' in c.lower() or 'code' in c.lower() or 'symbol' in c.lower()), None)

            if ticker_col:
                mask = df_display[ticker_col].astype(str).str.contains(clean_query, case=False, na=False)
                df_display = df_display[mask]
            else:
                mask = df_display.apply(
                    lambda col: col.astype(str).str.contains(clean_query, case=False, na=False)).any(axis=1)
                df_display = df_display[mask]

        # 3. Apply Column Selections
        if selected_cols:
            df_display = df_display[selected_cols]

        # =====================================================
        # BUYBACK DATA PANEL
        # =====================================================
        df_display.drop("bottom_alpha1", axis=1, inplace=True)
        df_display.drop("top_alpha1", axis=1, inplace=True)

        st.markdown("### 📋 Buyback Data")
        st.dataframe(df_display,width="stretch",height=450,hide_index=True,use_container_width=True, column_config=BUYBACK_TABLE_CONFIG)

        # =====================================================
        # CONTINUATION CALCULATION
        # =====================================================

        alpha5_col = next((c for c in df_display.columns if c.lower().replace("_", "") == "alpha5"),None)
        alpha1_col = next((c for c in df_display.columns if c.lower().replace("_", "") == "alpha1"),None)
        ticker_col = next((c for c in df_display.columns if "ticker" in c.lower() or "code" in c.lower()or "bbg" in c.lower()),None)

        if alpha5_col and alpha1_col:
            df_display["continuation"] = (pd.to_numeric(df_display[alpha5_col],errors="coerce") - pd.to_numeric(df_display[alpha1_col],errors="coerce"))

        # =====================================================
        # PREDICTION QUEUE
        # =====================================================

        prediction_df = (df_display[df_display["continuation"].isna()])

        st.markdown("---")
        st.markdown("### ⚡ Prediction Queue")
        st.info(f"{len(prediction_df)} stocks require Alpha5 prediction.")

        if len(prediction_df):
            queue_cols = [c for c in [ticker_col,date_col] if c]
            st.dataframe(prediction_df[queue_cols], width="stretch",hide_index=True,height=200, column_config={
                "bbg_ticker":st.column_config.TextColumn("Bloomberg Ticker"),
                "date":st.column_config.DateColumn("Date"),
            })

        # =====================================================
        # SESSION STATE
        # =====================================================
        if "prediction_results" not in st.session_state:
            st.session_state.prediction_results = (load_predictions())

        # =====================================================
        # RUN PREDICTIONS
        # =====================================================

        if (len(prediction_df) and ticker_col):
            if st.button("⚡ Run Predictions For Missing Names"):

                progress = st.progress(0)
                tickers = (prediction_df[ticker_col]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                total = len(tickers)

                for i, ticker in enumerate(tickers):
                    progress.progress(int(((i + 1) / total) * 100))
                    code, stdout, stderr = run_python_script("alpha5check.py",["--ticker",str(ticker)])

                    expected_cont = None
                    predicted_alpha5 = None
                    percentile = None
                    confidence = None
                    expected_error = ERROR_LOOKUP.get(confidence,np.nan)
                    signal = None

                    cont_match = re.search(r"Expected Continuation\s*:\s*(-?\d+\.?\d*)%",stdout)
                    alpha5_match = re.search(r"Predicted alpha5\s*:\s*(-?\d+\.?\d*)%",stdout)
                    percentile_match = re.search(r"Percentile Rank\s*:\s*(-?\d+\.?\d*)%",stdout)
                    confidence_match = re.search(r"Confidence\s*:\s*([A-Z]+)",stdout)
                    signal_match = re.search(r"Signal\s*:\s*([A-Z]+)",stdout)

                    if cont_match:
                        expected_cont = float(cont_match.group(1))

                    if alpha5_match:
                        predicted_alpha5 = float(alpha5_match.group(1))

                    if percentile_match:
                        percentile = float(percentile_match.group(1))

                    if confidence_match:
                        confidence = (confidence_match.group(1))

                    if signal_match:
                        signal = (signal_match.group(1))

                    st.session_state.prediction_results[str(ticker)] = {
                        "Predicted Continuation": expected_cont,
                        "Predicted Alpha5": predicted_alpha5,
                        "Signal": signal,
                        "Percentile Rank": percentile,
                        "Confidence": confidence,
                    }

                    save_predictions(st.session_state.prediction_results)
                progress.progress(100)

                st.success("Prediction run complete.")

        # =====================================================
        # MERGE PREDICTIONS
        # =====================================================

        df_display["Predicted Continuation"] = np.nan
        df_display["Predicted Alpha5"] = np.nan
        df_display["Percentile Rank"] = np.nan
        df_display["Confidence"] = ""
        df_display["Signal"] = ""

        for ticker, result in (st.session_state.prediction_results.items()):
            mask = (df_display[ticker_col].astype(str) == str(ticker))

            df_display.loc[mask,"Predicted Continuation"] = result.get("Predicted Continuation")
            df_display.loc[mask,"Predicted Alpha5"] = result.get("Predicted Alpha5")
            df_display.loc[mask,"Percentile Rank"] = result.get("Percentile Rank")
            df_display.loc[mask,"Confidence"] = result.get("Confidence")
            df_display.loc[mask,"Signal"] = result.get("Signal")

        signal_map = {
            "BUY": "🟢 BUY",
            "NEUTRAL": "🟡 NEUTRAL",
            "SHORT": "🔴 SHORT"
        }

        df_display["Signal"] = (df_display["Signal"]
            .map(signal_map)
            .fillna(df_display["Signal"])
        )

        # =====================================================
        # EFFECTIVE CONTINUATION
        # =====================================================

        df_display["Effective Continuation"] = (df_display["continuation"].fillna(df_display["Predicted Continuation"]))
        df_display["Prediction Source"] = np.where(df_display["continuation"].isna(),"Model","Observed")

        # =====================================================
        # RANKING
        # =====================================================

        df_display["Rank"] = (df_display["Effective Continuation"].rank(ascending=False,method="dense"))

        # =====================================================
        # PREDICTION SUMMARY
        # =====================================================

        prediction_results = pd.DataFrame(st.session_state.prediction_results).T

        if not prediction_results.empty:
            st.markdown("---")
            st.markdown("### 📊 Prediction Summary")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Predictions Generated",len(prediction_results))

            with c2:
                try:
                    st.metric("Avg Predicted Continuation",round(prediction_results["Predicted Continuation"].mean(),2))
                except:
                    pass

            with c3:
                try:
                    st.metric("High Confidence Signals",(prediction_results["Confidence"] == "HIGH").sum())
                except:
                    pass

        # =====================================================
        # UNIVERSE VIEW
        # =====================================================

        priority_cols = [
            "# id",
            "date",
            ticker_col,
            "name",
            "shares",
            "purchased_shares",
            "Predicted Continuation",
            "Effective Continuation",
            "Signal",
            "Confidence",
            "Prediction Source",
            "price"
        ]

        remaining_cols = [c for c in df_display.columns if c not in priority_cols]
        final_cols = ([c for c in priority_cols if c in df_display.columns] + remaining_cols)
        universe_df = (df_display[final_cols].sort_values(ticker_col))

        # =====================================================
        # PREDICTED UNIVERSE VIEW
        # =====================================================
        predicted_universe = (universe_df[universe_df["Prediction Source"] == "Model"].sort_values("Effective Continuation",ascending=False))
        st.markdown("---")
        st.markdown("### Alpha5 Predicted Universe")


        def colour_cont(v):
            if pd.isna(v):
                return "background-color:#FFF3CD"

            if v > 0:
                return "background-color:#D4EDDA"

            return "background-color:#F8D7DA"


        styled_df = (predicted_universe.style)
        st.dataframe(styled_df,width="stretch",hide_index=True,height=450, column_config=UNIVERSE_TABLE_CONFIG)
        # =====================================================
        # STOCK DRILLDOWN
        # =====================================================

        if ticker_col:
            ticker_list = sorted(universe_df[ticker_col]
                .dropna()
                .astype(str)
                .unique()
            )

            if ticker_list:
                selected_ticker = st.selectbox("Inspect Ticker",ticker_list)
                stock_df = (universe_df[universe_df[ticker_col].astype(str) == selected_ticker])

                with st.expander(f"📄 Tearsheet - {selected_ticker}"):
                    st.dataframe(stock_df.T,width="stretch")

        # =====================================================
        # DOWNLOAD CURRENT VIEW
        # =====================================================

        st.download_button("⬇ Download Current View",universe_df.to_csv(index=False),"filtered_buybacks.csv","text/csv")
        st.markdown("---")
        st.markdown("### Realtime Data Sync")
        col_sync_btn, _ = st.columns([2, 6])
        with col_sync_btn:
            update_clicked = st.button("🔄 Update Buybacks", width="stretch")

        if update_clicked:
            with st.spinner("Connecting to live feeds..."):
                code, stdout, stderr = run_python_script("datapull.py")
                if code == 0:
                    st.success("Live feeds updated successfully!")
                    st.success(f"Data synchronized at {datetime.now():%H:%M:%S}")
                    with st.expander("View Sync Logs:"):
                        st.code(stdout, language="bash")
                else:
                    st.error("Live feeds failed to update")
                    st.code(stderr, language="bash")
    else:
        st.warning("'jp_buyback.xlsx not found. Please ensure it is in the root folder.'")
# ---------------------------------------------------------
# MODULE 2: Alpha5 Model Execution (alpha5model.py)
# ---------------------------------------------------------
elif selected_icon == "⚙️ Alpha5 Engine":
    st.markdown('<div class="dark-teal-header">Alpha5 Quantitative Model </div>',
                unsafe_allow_html=True)

    if st.button("⚡ Run Alpha5 Model"):
        with st.spinner("Processing Alpha5 Model..."):
            code, stdout, stderr = run_python_script("alpha5model.py")
            if code == 0:
                st.success("Alpha5 Model execution finished!")

                # --- METRIC PARSER (Kept your existing logic) ---
                target_metrics = ["Total R-squared value"]
                metrics_found = {}
                for line in stdout.split('\n'):
                    if ':' in line and len(line.split(':', 1)) == 2:
                        key, val = line.split(':', 1)
                        clean_key = key.strip()
                        if clean_key in target_metrics:
                            metrics_found[clean_key] = val.strip()

                import plotly.graph_objects as go

                r2 = float(metrics_found.get("Total R-squared value", 0))

                # --- TABLE PARSER ---
                # --- DICTIONARY/JSON TABLE PARSER ---
                table_data_parsed = None
                table_match = re.search(r"===TABLE_START===\s*(.*?)\s*===TABLE_END===", stdout, re.DOTALL)

                if table_match:
                    raw_str = table_match.group(1).strip()
                    parsed_obj = None

                    try:
                        clean_py_str = raw_str.replace("nan", "None").replace("NaN", "None").replace("null","None")
                        parsed_obj = ast.literal_eval(clean_py_str)
                    except Exception:
                        pass


                    if parsed_obj is None:
                        try:
                            import json

                            parsed_obj = json.loads(raw_str)
                        except Exception:
                            pass

                    # 3. Fix Pandas "scalar values" error by wrapping single dicts in a list
                    if parsed_obj is not None:
                        try:
                            if isinstance(parsed_obj, dict):
                                # If all dict values are single items/scalars, wrap in a list to make a 1-row table
                                if not any(isinstance(v, (list, dict, tuple)) for v in parsed_obj.values()):
                                    parsed_obj = [parsed_obj]

                            table_data_parsed = pd.DataFrame(parsed_obj)
                        except Exception as e:
                            st.warning(f"Found table markers but failed to build DataFrame: {e}")

                # --- RENDER DASHBOARD ---
                st.markdown("### Output Dashboard")
                metric_cols = st.columns(3)

                # if metrics_found:
                #     for i, (key, val) in enumerate(metrics_found.items()):
                #         metric_cols[i % 3].metric(label=key, value=val)

                fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=r2,
                        title={"text": "Model R²"},
                        number={"valueformat": ".3f"},
                        gauge={
                            "axis": {"range": [0, 1]},
                            "bar": {"color": "#064147"},
                            "steps": [
                                {"range": [0, 0.3], "color": "#ffcccc"},
                                {"range": [0.3, 0.6], "color": "#ffe5b4"},
                                {"range": [0.6, 0.8], "color": "#d4edda"},
                                {"range": [0.8, 1.0], "color": "#8fd19e"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": r2,
                            },
                        },
                    )
                )

                fig.update_layout(height=300,margin=dict(l=20, r=20, t=50, b=20),)
                st.plotly_chart(fig, use_container_width=True)

                table_data_parsed["learning_rate"] = table_data_parsed["learning_rate"] * 100
                table_data_parsed["subsample"] = table_data_parsed["subsample"] * 100
                table_data_parsed["colsample_bytree"] = table_data_parsed["colsample_bytree"] * 100

                # Render the stylized table if found
                if table_data_parsed is not None:
                    st.markdown("#### Model Top Results")
                    st.dataframe(table_data_parsed, width="stretch", hide_index=True,column_config=
                    {"max_depth": st.column_config.NumberColumn("Maximum Depth"),
                     "n_estimators": st.column_config.NumberColumn("Number of Estimators"),
                     "learning_rate": st.column_config.NumberColumn("Learning Rate",format="%.2f%%"),
                     "subsample": st.column_config.NumberColumn("Observation Sampling Rate",format="%.2f%%"),
                     "colsample_bytree": st.column_config.NumberColumn("Random Sampling Rate",format="%.2f%%"),
                     "min_child_weight": st.column_config.NumberColumn("Minimum Child Weight",format="%.2f"),
                     "reg_alpha": st.column_config.NumberColumn("Alpha Regularisation",format="%.2f"),
                     "reg_lambda": st.column_config.NumberColumn("Lambda Regularisation",format="%.2f"),
                    })

                # Hide the raw console behind an expander
                with st.expander("Show Raw Console Output"):
                    # We can clean the JSON block out of the raw view so it doesn't look messy
                    clean_stdout = re.sub(r"===TABLE_START===.*?===TABLE_END===\n", "", stdout, flags=re.DOTALL)
                    st.code(clean_stdout, language="bash")
            else:
                st.error("Failed to update.")
                st.code(stderr, language="bash")


            # --- METRIC PARSER ---
                # Looks for output formatted like "Some Metric: 45.2" in the console text
            target_metrics = ["XGB"]

            metrics_found = {}
            for line in stdout.split('\n'):
                if ':' in line and len(line.split(':', 1)) == 2:
                    key, val = line.split(':', 1)
                    clean_key = key.strip()

                    # Only add it to the boxes if it is in your target list
                    if clean_key in target_metrics:
                        metrics_found[clean_key] = val.strip()
# ---------------------------------------------------------
# MODULE 3: Stock Checker (alpha5check.py)
# ---------------------------------------------------------
elif selected_icon == "🔍 Stock Checker":
    st.markdown('<div class="dark-teal-header">Stock Checker & Diagnostics</div>',
                unsafe_allow_html=True)

    df = get_buyback_data()
    df = metrics(df, "a")
    missing_alpha5_tickers = []

    def score_to_bin(score):
        if score < 20:
            return "0-20"
        elif score < 40:
            return "20-40"
        elif score < 60:
            return "40-60"
        elif score < 80:
            return "60-80"

        return "80-100"

    if df is not None:
        ticker_col = next((c for c in df.columns if 'bbg_ticker' in c.lower()), None)
        alpha5_col = next((c for c in df.columns if 'alpha5' in c.lower()), None)
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)

        if date_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col],format="%Y-%m-%d")
                cutoff_date = datetime.now() - pd.tseries.offsets.BusinessDay(n=5)
                df_recent = df[df[date_col] >= cutoff_date]
            except:
                df_recent = df
        else:
            df_recent = df

        if ticker_col and alpha5_col:
            missing_mask = df_recent[alpha5_col].isna() | (df_recent[alpha5_col] == "") | (df_recent[alpha5_col] == 0)
            missing_alpha5_tickers = df_recent[missing_mask][ticker_col].dropna().unique().tolist()
        elif ticker_col:
            missing_alpha5_tickers = df_recent[ticker_col].dropna().unique().tolist()

    if missing_alpha5_tickers:
        selected_ticker = st.selectbox("JP Equity", options=missing_alpha5_tickers)
    else:
        selected_ticker = st.text_input("Enter Ticker Symbol Manually:", "JP Equity")

    if st.button("🔎 Run Stock Prediction"):

        if selected_ticker:
            with st.spinner(f"Running checks for {selected_ticker}..."):
                code, stdout, stderr = run_python_script("alpha5check.py", ["--ticker", str(selected_ticker)])

                if code == 0:
                    if code == 0:
                        st.success(f"Predictions completed for {selected_ticker}!")

                        # ------------------------------------------------
                        # PARSE REPORT
                        # ------------------------------------------------

                        alpha1_match = re.search(r"Observed alpha1\s*:\s*(-?\d+\.?\d*)%",stdout)
                        alpha5_match = re.search(r"Predicted alpha5\s*:\s*(-?\d+\.?\d*)%",stdout)
                        continuation_match = re.search(r"Expected Continuation\s*:\s*(-?\d+\.?\d*)%",stdout)
                        percentile_match = re.search(r"Percentile Rank\s*:\s*(-?\d+\.?\d*)%",stdout)
                        confidence_match = re.search(r"Confidence\s*:\s*([A-Z]+)",stdout)
                        decile_match = re.search(r"Historical Decile\s*:\s*(\d+)",stdout)
                        signal_match = re.search(r"Signal\s*:\s*([A-Z]+)",stdout)


                        alpha1 = (float(alpha1_match.group(1))if alpha1_match else None)
                        alpha5 = (float(alpha5_match.group(1)) if alpha5_match else None)
                        continuation = (float(continuation_match.group(1)) if continuation_match else None)
                        percentile = (float(percentile_match.group(1)) if percentile_match else None)
                        confidence = (confidence_match.group(1) if confidence_match else "N/A")
                        decile = (decile_match.group(1) if decile_match else "N/A")
                        signal = (signal_match.group(1) if signal_match else "N/A")

                        factor_rows = []

                        positive_section = re.search(r"TOP POSITIVE DRIVERS.*?feature\s+impact(.*?)TOP NEGATIVE DRIVERS",stdout,re.DOTALL)
                        negative_section = re.search(r"TOP NEGATIVE DRIVERS.*?feature\s+impact(.*)",stdout,re.DOTALL)

                        if positive_section:
                            for line in positive_section.group(1).splitlines():
                                line = line.strip()

                                if not line:
                                    continue

                                pieces = line.rsplit(maxsplit=1)

                                if len(pieces) == 2:
                                    factor_rows.append({"Feature": pieces[0], "Impact": float(pieces[1])})

                        if negative_section:
                            for line in negative_section.group(1).splitlines():
                                line = line.strip()

                                if not line:
                                    continue

                                pieces = line.rsplit(maxsplit=1)

                                if len(pieces) == 2:
                                    factor_rows.append({"Feature": pieces[0], "Impact": float(pieces[1])})

                        factor_df = pd.DataFrame(factor_rows)
                        if not factor_df.empty:
                            factor_df["Feature"] = (factor_df["Feature"].str.replace("_", " ").str.title())
                            factor_df = factor_df.sort_values("Impact", ascending=True)
                            factor_df["Direction"] = np.where(factor_df["Impact"] > 0, "Positive", "Negative")

                        # ------------------------------------------------
                        # HISTORY
                        # ------------------------------------------------
                        if "checker_history" not in st.session_state:
                            st.session_state.checker_history = []

                        st.session_state.checker_history.append(
                            {
                                "Ticker": selected_ticker,
                                "Signal": signal,
                                "Continuation": continuation,
                                "Confidence": confidence
                            }
                        )

                        st.session_state.prediction_results[str(selected_ticker)] = {
                            "Predicted Continuation": continuation,
                            "Predicted Alpha5": alpha5,
                            "Percentile Rank": percentile,
                            "Confidence": confidence,
                            "Signal": signal,
                        }

                        save_predictions(
                            st.session_state.prediction_results
                        )

                        # ------------------------------------------------
                        # SIGNAL STYLING
                        # ------------------------------------------------

                        signal_colour = {
                            "BUY": "#16A34A",
                            "NEUTRAL": "#D97706",
                            "SHORT": "#DC2626"
                        }

                        signal_icon = {
                            "BUY": "🟢",
                            "NEUTRAL": "🟡",
                            "SHORT": "🔴"
                        }

                        confidence_colour = {
                            "HIGH": "#16A34A",
                            "MEDIUM": "#D97706",
                            "LOW": "#DC2626"
                        }

                        colour = signal_colour.get(signal)

                        # ------------------------------------------------
                        # SIGNAL CARD
                        # ------------------------------------------------
                        st.markdown(
                            f"""
                            <div style="
                                background:white;
                                border-left:6px solid #064147;
                                border-radius:10px;
                                padding:20px;
                                margin-bottom:20px;
                                box-shadow:0 2px 8px rgba(0,0,0,0.08);
                            ">
                                <h2 style="
                                    margin:0;
                                    color:{colour};
                                ">
                                    {signal_icon.get(signal, '')}
                                    {signal}
                                </h2>
                            
                            <div style="
                                margin-top:10px;
                                background:{confidence_colour.get(confidence.upper(), '#64748B')};
                                color:white;
                                padding:6px 12px;
                                border-radius:6px;
                                font-weight:bold;
                                ">
                                    {confidence} CONFIDENCE
                            </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                        # ------------------------------------------------
                        # DRIVER ANALYTICS
                        # ------------------------------------------------
                        strongest_positive = "N/A"
                        strongest_negative = "N/A"
                        top_driver_impact = np.nan
                        driver_concentration = np.nan
                        top3_pct = np.nan

                        if not factor_df.empty:
                            positive_drivers = (factor_df[factor_df["Impact"] > 0].sort_values("Impact", ascending=False))
                            negative_drivers = (factor_df[factor_df["Impact"] < 0].sort_values("Impact"))

                            if len(positive_drivers):
                                strongest_positive = (positive_drivers.iloc[0]["Feature"])

                            if len(negative_drivers):
                                strongest_negative = (negative_drivers.iloc[0]["Feature"])

                            top_driver_impact = (factor_df["Impact"].abs().max())
                            total_impact = (factor_df["Impact"].abs().sum())

                            if total_impact > 0:
                                top3_pct = (factor_df["Impact"].abs().nlargest(3).sum() / total_impact * 100)
                                driver_concentration = (top_driver_impact / total_impact * 100)

                        confidence_score = np.nan

                        if pd.notna(driver_concentration):
                            confidence_score = max(0, min(100, (1 - driver_concentration / 100) * 100))

                        score_bin = score_to_bin(confidence_score)
                        expected_error = (ERROR_LOOKUP.get(score_bin, np.nan))

                        # ------------------------------------------------
                        # KPI ROW
                        # ------------------------------------------------
                        c1, c2, c3, c4, c5, c6 = st.columns(6)

                        with c1:
                            st.metric("Observed Alpha1",(f"{alpha1:.2f}%" if alpha1 is not None else "N/A"))

                        with c2:
                            st.metric("Predicted Alpha5",(f"{alpha5:.2f}%" if alpha5 is not None else "N/A"))

                        with c3:
                            st.metric("Expected Continuation",(f"{continuation:.2f}%" if continuation is not None else "N/A"))

                        with c4:
                            st.metric("Percentile Rank",(f"{percentile:.1f}%" if percentile is not None else "N/A"))

                        with c5:
                            st.metric("Confidence Score",(f"{confidence_score:.0f}/100" if pd.notna(confidence_score) else "N/A"))

                        with c6:
                            st.metric("Expected Error",(f"±{expected_error:.1f}%" if pd.notna(expected_error) else "N/A"))

                        st.markdown("---")

                        # ------------------------------------------------
                        # SUMMARY + CHART
                        # ------------------------------------------------
                        left_col, right_col = st.columns(2)

                        with right_col:

                            st.markdown("### Factor Contributions")

                            fig = px.bar(
                                factor_df,
                                x="Impact",
                                y="Feature",
                                orientation="h",
                                color="Direction",
                                color_discrete_map={"Positive": "#16A34A", "Negative": "#DC2626"}
                            )

                            fig.update_traces(
                                texttemplate="%{x:+.2f}",
                                textposition="outside",
                                hovertemplate="<b>%{y}</b><br>" + "Contribution: %{x:+.2f}<extra></extra>"
                            )

                            fig.update_layout(
                                template="plotly_white",
                                height=500,
                                title="Alpha5 Factor Contribution Analysis",
                                xaxis_title="Contribution to Alpha5 Prediction",
                                yaxis_title="Feature",
                                hovermode="y unified",
                                showlegend=False,
                                margin=dict(l=20,r=90,t=60,b=40)
                            )

                            fig.add_vline(x=0,line_width=1,line_color="black")
                            st.plotly_chart(fig,width="stretch")

                        with left_col:
                            st.markdown("### Prediction Summary")

                            # ------------------------------------------------
                            # SIGNAL STRENGTH
                            # ------------------------------------------------
                            if percentile is not None:

                                if percentile >= 90:
                                    signal_strength = "EXTREME"

                                elif percentile >= 75:
                                    signal_strength = "STRONG"

                                elif percentile >= 50:
                                    signal_strength = "MODERATE"

                                else:
                                    signal_strength = "WEAK"

                            else:

                                signal_strength = "N/A"

                            # ------------------------------------------------
                            # SUMMARY TABLE
                            # ------------------------------------------------
                            summary_df = pd.DataFrame(
                                {
                                    "Metric": [
                                        "Signal Strength",
                                        "Historical Decile",
                                        "Strongest Positive Driver",
                                        "Strongest Negative Driver",
                                        "Driver Concentration",
                                        "Top Driver Impact",
                                        "Top 3 Drivers % of Output"
                                    ],
                                    "Value": [
                                        signal_strength,
                                        decile,
                                        strongest_positive,
                                        strongest_negative,
                                        (
                                            f"{driver_concentration:.1f}%"
                                            if pd.notna(driver_concentration)
                                            else "N/A"
                                        ),
                                        (
                                            f"{top_driver_impact:.3f}"
                                            if pd.notna(top_driver_impact)
                                            else "N/A"
                                        ),
                                        (
                                            f"{top3_pct:.1f}%"
                                            if pd.notna(top3_pct)
                                            else "N/A"
                                        )
                                    ]
                                }
                            )

                            st.dataframe(summary_df, width="stretch", hide_index=True)

                        st.markdown("---")


                        # ------------------------------------------------
                        # RECENT CHECKS
                        # ------------------------------------------------
                        with st.expander("🕒 Recent Checks",expanded=False):
                            st.dataframe(pd.DataFrame(st.session_state.checker_history), width="stretch", hide_index=True)

                        # ------------------------------------------------
                        # RAW REPORT
                        # ------------------------------------------------
                        with st.expander("Show Raw Report"):
                            st.code(stdout, language="text")
                else:
                    st.error("Prediction returned errors:")
                    st.code(stderr, language="bash")
        else:
            st.warning("Please select or enter a ticker first.")
# ---------------------------------------------------------
# MODULE 4: Feature Decile vs Continuation Plots
# ---------------------------------------------------------
elif selected_icon == "📈 Plots":
    st.markdown('<div class="dark-teal-header">Feature Decile vs. Continuation Plots</div>', unsafe_allow_html=True)

    df = get_buyback_data()
    df = metrics(df,"a")

    if df is not None:
        df_plot = df.copy()

        # Automatically locate alpha5 and alpha1 columns (case-insensitive)
        alpha5_col = next((c for c in df_plot.columns if c.lower().replace('_', '') == 'alpha5'), None)
        alpha1_col = next((c for c in df_plot.columns if c.lower().replace('_', '') == 'alpha1'), None)

        # Compute Continuation (alpha5 - alpha1) dynamically
        continuation_label = "continuation"
        if alpha5_col and alpha1_col:
            df_plot[continuation_label] = (
                    pd.to_numeric(df_plot[alpha5_col], errors='coerce') -
                    pd.to_numeric(df_plot[alpha1_col], errors='coerce')
            )

        numeric_cols = df_plot.select_dtypes(include=['float64', 'int64', 'float', 'int']).columns.tolist()

        if len(numeric_cols) >= 2:
            col_feat, col_target = st.columns(2)

            target_default_idx = 0
            if continuation_label in numeric_cols:
                target_default_idx = numeric_cols.index(continuation_label)
            else:
                for idx, c in enumerate(numeric_cols):
                    if any(term in c.lower() for term in ['continuation', 'return', 'ret', 'change', 'pnl', 'target']):
                        target_default_idx = idx
                        break

            feature_options = [c for c in numeric_cols if c != continuation_label]

            with col_feat:
                selected_feature = st.selectbox("Select Feature (X-Axis Deciles):",
                                                options=feature_options if feature_options else numeric_cols, index=0)
            with col_target:
                selected_target = "continuation"

            if selected_feature and selected_target:
                df_clean = df_plot[[selected_feature, selected_target]].dropna().copy()
                df_colour = df_plot[[selected_feature, "continuation","alpha1"]].dropna().copy()

                if len(df_clean) >= 10:
                    try:
                        df_clean['Decile'] = pd.qcut(df_clean[selected_feature],q=10,labels=[f"D{i}" for i in range(1, 11)],duplicates='drop')

                        # Aggregate mean target per decile
                        decile_df = df_clean.groupby('Decile', observed=False)[selected_target].median().reset_index()
                        decile_df.columns = ['Decile', f'Median {selected_target}']

                        st.markdown(f"#### 📊 Mean {selected_target} by {selected_feature} Decile")

                        # Interactive Bar Chart
                        import plotly.express as px

                        fig = px.line(decile_df,x="Decile",y=f"Median {selected_target}",markers=True)
                        fig.update_layout(height=500,template="plotly_white")
                        fig.update_traces(line = dict(color="#064147", width=3), marker = dict(color="#064147", size=8))
                        st.plotly_chart(fig, width="stretch")

                        # Summary Table
                        with st.expander("View Decile Breakdown Table"):
                            summary_df = df_clean.groupby('Decile', observed=False)[selected_target].agg(
                                Mean='mean',
                                Median='median',
                                Count='count',
                                ContRange= 'min',
                                Max='max'
                            ).reset_index()
                            st.dataframe(summary_df, width="stretch", hide_index=True)
                    except Exception as e:
                        st.error(f"Unable to bin feature data into 10 deciles: {e}")

                    if len(df_colour) > 30:
                        unique_vals = df_colour[selected_feature].nunique()
                        if unique_vals > 2:
                            z_scores = stats.zscore(df_colour[selected_feature])
                            df_colour[selected_feature] = df_colour[selected_feature][(z_scores > -3) & (z_scores < 3)]
                            fig = px.scatter(df_colour,x=selected_feature,y="continuation",color="alpha1",color_continuous_scale="Turbo",opacity=0.7,title=f"{selected_feature} vs Continuation (colored by Alpha1)",)
                            fig.update_layout(coloraxis_colorbar_title="Alpha1")
                            st.plotly_chart(fig, width="stretch")
                        else:
                            print(f"Skipping {selected_feature}: only {unique_vals} unique values")
                    else:
                        print(f"Skipping {selected_feature}: only {len(df_colour)} rows")
                else:
                    st.warning("Not enough valid data points (minimum 10 required) to compute deciles.")
        else:
            st.warning("The dataset needs at least 2 numerical columns to run feature decile analysis.")
    else:
        st.warning("`jp_buyback.xlsx` not found. Please ensure it is in the root project folder.")