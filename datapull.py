import sys
import pandas as pd
from pandas.tseries.offsets import BusinessDay
import xbbg
from bs4 import BeautifulSoup

xbbg.configure(host='localhost',port=8194,)

df = pd.read_excel("./jp_buyback.xlsx").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])
df["ref_date"] = df["date"] - BusinessDay(1)

hist_fields = [
    "PX_LAST",
    "TURNOVER_AVG_1M",
    "TURNOVER_AVG_5D",
]

ref_fields = [
    "CUR_MKT_CAP",
    "EQY_FLOAT",
    "BETA",
    "VOLATILITY_30D",
    "DAYS_TO_COVER",
    "INDICATIVE_FEE",
    "EQY_WEIGHTED_AVG_PX_ADJ",
    "BS_TOT_ASSET",
    "BS_CASH_NEAR_CASH_ITEM",
    "CF_FREE_CASH_FLOW",
    "FIXED_ASSETS_AND_LT_INVESTMENTS",
    "EBIT_TO_INT_EXP",
    "TOBIN_Q_RATIO",
    "SHORT_AND_LONG_TERM_DEBT",
    "SALES_REV_TURN",
    "PX_TO_BOOK_RATIO",
]

fx_rate = xbbg.blp.bdp("JPYUSD Curncy", "PX_LAST").iloc[0, 0]
fetched_rows = []

for idx, row in df.iterrows():
    ticker = row["bbg_ticker"]
    ref_dt = row["ref_date"].strftime("%Y-%m-%d")
    ref_dt_bbg = row["ref_date"].strftime("%Y%m%d")

    print(f"Fetching {idx + 1}/{len(df)}: {ticker} @ {ref_dt}")
    row_data = {"orig_idx": idx}

    try:
        h_data = xbbg.blp.bdh(tickers=ticker,flds=hist_fields,start_date=ref_dt,end_date=ref_dt)
        if not h_data.empty:
            for fld in hist_fields:
                row_data[fld] = h_data.xs(fld, axis=1, level=1).iloc[0, 0]
    except Exception as e:
        print(f"Warning: Failed hist data for {ticker}: {e}")

    try:
        r_data = xbbg.blp.bdp(tickers=ticker,flds=ref_fields,REFERENCE_DATE=ref_dt_bbg)
        if not r_data.empty:
            for fld in ref_fields:
                if fld in r_data.columns:
                    row_data[fld] = r_data[fld].iloc[0]
    except Exception as e:
        print(f"Warning: Failed ref data for {ticker}: {e}")

    fetched_rows.append(row_data)

res_df = pd.DataFrame(fetched_rows).set_index("orig_idx")
full_df = df.join(res_df)

results = pd.DataFrame({
    "date": full_df["date"],
    "bbg_ticker": full_df["bbg_ticker"],
    "shares": full_df.get("shares"),
    "purchased_shares": full_df.get("purchased_shares"),
    "price": full_df.get("price"),
    "total_cap_m_usd": full_df.get("CUR_MKT_CAP") * fx_rate,
    "shs_cap_m_usd": full_df.get("CUR_MKT_CAP") * fx_rate,
    "eqy_float_usd": full_df.get("EQY_FLOAT") * full_df.get("PX_LAST") * fx_rate / 1000000,
    "beta": full_df.get("BETA"),
    "vol": full_df.get("VOLATILITY_30D"),
    "adv_1m_usd": full_df.get("TURNOVER_AVG_1M") * fx_rate / 1000000,
    "adv_5d_usd": full_df.get("TURNOVER_AVG_5D") * fx_rate / 1000000,
    "dtc": full_df.get("DAYS_TO_COVER"),
    "indicative_fee_pct": full_df.get("INDICATIVE_FEE"),
    "adj_vwap": full_df.get("EQY_WEIGHTED_AVG_PX_ADJ"),
    "cash_usd_m": full_df.get("BS_CASH_NEAR_CASH_ITEM") * fx_rate,
    "px_book_ratio": full_df.get("PX_TO_BOOK_RATIO"),
    "total_assets": full_df.get("BS_TOT_ASSET"),
    "fcf": full_df.get("CF_FREE_CASH_FLOW"),
    "ebit_interest_ratio": full_df.get("EBIT_TO_INT_EXP"),
    "fixed_assets": full_df.get("FIXED_ASSETS_AND_LT_INVESTMENTS"),
    "operating_income": full_df.get("SALES_REV_TURN"),
    "debt": full_df.get("SHORT_AND_LONG_TERM_DEBT"),
    "tobin_q_ratio": full_df.get("TOBIN_Q_RATIO"),
})

if __name__ == "__main__":
    try:
        results.to_excel("./jp_buyback_output.xlsx", index=False)
        print("Saved successfully to jp_buyback_output.xlsx")
    except Exception as e:
        print(f"Error saving file: {e}", file=sys.stderr)