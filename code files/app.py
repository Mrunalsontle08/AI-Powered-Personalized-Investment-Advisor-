# app.py
import os
import glob
import pickle
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# ---------- Utils --------
# =========================

st.set_page_config(page_title="Smart Investment Advisor", layout="wide")

def rupee(x: float) -> str:
    try:
        return f"₹{x:,.2f}"
    except Exception:
        return "₹0.00"

def get_latest_file(folder: str, pattern: str = "*.csv") -> Optional[str]:
    try:
        files = glob.glob(os.path.join(folder, pattern))
        if not files:
            return None
        return max(files, key=os.path.getmtime)
    except Exception:
        return None

def safe_read_csv(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read CSV '{os.path.basename(path)}': {e}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_model_parts():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        return model, scaler, encoder, None
    except Exception as e:
        return None, None, None, str(e)

# =========================
# ----- Risk & Goals ------
# =========================

F30_QUESTIONS = [
    "I prefer safer investments even if returns are low",
    "I like taking financial risks if the rewards are high",
    "I track my expenses and budget regularly",
    "I feel confident managing my finances",
    "I get anxious when markets fluctuate",
    "I prefer guaranteed returns over variable ones",
    "I invest for long-term goals",
    "I panic sell during downturns",
    "I consult a financial advisor before investing",
    "I review my portfolio frequently",
    "I use multiple investment channels",
    "I like reading about finance/investing"
]

FEATURE_ORDER = ["S_Age", "S_Income", "S_Education"] + [f"F30_{i}" for i in range(1, 13)]

GOAL_CHOICES = {
    "Retirement": "Retirement",
    "Buying a House": "House",
    "Travel": "Travel",
    "Child’s Education": "Education",
    "Wealth Growth": "Growth"
}

# Annual return assumptions (compounded monthly)
ASSET_RETURNS = {
    "Stocks": 0.12,
    "Mutual Funds": 0.10,
    "Gold": 0.08,
    "PPF": 0.07,
    "Savings": 0.04,
    "Emergency Fund": 0.04
}

def base_allocation(risk_label: str) -> Dict[str, float]:
    if risk_label == "Aggressive":
        return {"Stocks": 60, "Mutual Funds": 25, "Gold": 5, "PPF": 5, "Savings": 3, "Emergency Fund": 2}
    if risk_label == "Moderate":
        return {"Stocks": 40, "Mutual Funds": 35, "Gold": 10, "PPF": 10, "Savings": 3, "Emergency Fund": 2}
    return {"Stocks": 10, "Mutual Funds": 25, "Gold": 15, "PPF": 30, "Savings": 15, "Emergency Fund": 5}

def adjust_for_goal(allocation: Dict[str, float], goal: str) -> Dict[str, float]:
    alloc = allocation.copy()

    if goal == "Retirement":
        alloc["Stocks"] += 5; alloc["Mutual Funds"] += 5
        alloc["Savings"] -= 5; alloc["Emergency Fund"] -= 5
    elif goal == "House":
        alloc["PPF"] += 5; alloc["Mutual Funds"] += 5
        alloc["Stocks"] -= 5; alloc["Gold"] -= 5
    elif goal == "Travel":
        alloc["Savings"] += 10; alloc["Mutual Funds"] += 5
        alloc["Stocks"] -= 10; alloc["PPF"] -= 5
    elif goal == "Education":
        alloc["PPF"] += 5; alloc["Mutual Funds"] += 5
        alloc["Gold"] -= 5; alloc["Savings"] -= 5
    elif goal == "Growth":
        alloc["Stocks"] += 10
        alloc["Savings"] -= 5; alloc["Emergency Fund"] -= 5

    for k in list(alloc.keys()):
        if alloc[k] < 0:
            alloc[k] = 0.0
    s = sum(alloc.values())
    if s <= 0:
        n = len(alloc)
        return {k: 100.0 / n for k in alloc}
    if abs(s - 100.0) < 1e-6:
        return alloc

    if s < 100:
        rem = 100 - s
        alloc["Savings"] += rem * 0.7
        alloc["Emergency Fund"] += rem * 0.3
    else:
        for k in alloc:
            alloc[k] = alloc[k] * 100.0 / s
    return alloc

# =========================
# ---- Recommenders -------
# =========================

def pick_stocks(stock_df: pd.DataFrame, goal: str, risk: str, seed: int, n: int = 5) -> pd.DataFrame:
    df = stock_df.copy()
    if "STOCK" not in df.columns:
        return pd.DataFrame(columns=["STOCK", "avg_return_30d", "volatility_30d"])

    for col in ["avg_return_30d", "volatility_30d"]:
        if col not in df.columns:
            df[col] = np.nan
    if "risk_level" not in df.columns:
        df["risk_level"] = "Medium"

    df = df.drop_duplicates("STOCK")

    if risk == "Conservative":
        df = df.sort_values(["volatility_30d", "avg_return_30d"], ascending=[True, False])
        df = df[df["volatility_30d"].notna()]
    elif risk == "Moderate":
        df = df.sort_values(["avg_return_30d"], ascending=False)
    else:
        df = df.sort_values(["avg_return_30d"], ascending=False)

    if goal in ("House", "Education", "Travel"):
        df = df.sort_values(["volatility_30d", "avg_return_30d"], ascending=[True, False])

    head = min(len(df), 30)
    if head == 0:
        return df.head(0)
    rng = np.random.RandomState(seed)
    subset = df.head(head)
    take = min(n, len(subset))
    return subset.sample(take, random_state=rng)

def pick_mutual_funds(mf_df: pd.DataFrame, goal: str, risk: str, seed: int, n: int = 5) -> pd.DataFrame:
    df = mf_df.copy()

    name_col = None
    for c in df.columns:
        if c.strip().lower() in ("scheme name", "scheme_name", "scheme"):
            name_col = c
            break
    if name_col is None:
        str_cols = [c for c in df.columns if df[c].dtype == object]
        name_col = str_cols[0] if str_cols else None
    if name_col is None:
        return pd.DataFrame(columns=["Scheme Name", "Volatility (90d)"])

    vol_col = None
    for c in df.columns:
        if c.strip().lower() in ("volatility (90d)", "volatility", "vol_90d"):
            vol_col = c
            break
    if vol_col is None:
        df["Volatility (90d)"] = np.nan
        vol_col = "Volatility (90d)"

    df = df.rename(columns={name_col: "Scheme Name", vol_col: "Volatility (90d)"})
    df["Scheme Name"] = df["Scheme Name"].astype(str)
    df = df.drop_duplicates("Scheme Name")

    name = df["Scheme Name"].str.lower()
    liquid_mask = name.str.contains("liquid|overnight")
    hybrid_mask = name.str.contains("balanced|hybrid|advantage|asset allocation")
    equity_mask = name.str.contains("equity|multicap|multi[- ]?cap|large cap|mid cap|flexi|index|nifty|sensex")

    if risk == "Conservative" or goal == "Travel":
        cand = df[liquid_mask].copy()
        if cand.empty: cand = df.sort_values("Volatility (90d)").copy()
    elif goal in ("House", "Education"):
        cand = df[hybrid_mask].copy()
        if cand.empty: cand = df.sort_values("Volatility (90d)").copy()
    else:
        cand = df[equity_mask].copy()
        if cand.empty: cand = df.sort_values("Volatility (90d)").copy()

    cand = cand.sort_values("Volatility (90d)", ascending=True)
    head = min(len(cand), 30)
    rng = np.random.RandomState(seed)
    subset = cand.head(head)
    take = min(n, len(subset))
    return subset.sample(take, random_state=rng)

# =========================
# ---- Projections --------
# =========================

def invested_principal_stepup(monthly: float, years: int, growth: float = 0.05) -> float:
    total = 0.0
    for y in range(years):
        total += (monthly * ((1 + growth) ** y)) * 12.0
    return total

def fv_stepup_monthly(monthly: float, years: int, annual_return: float, growth: float = 0.05) -> float:
    r = annual_return / 12.0
    value = 0.0
    for y in range(1, years + 1):
        m = monthly * ((1 + growth) ** (y - 1))
        for _ in range(12):
            value = (value + m) * (1 + r)
    return value

def project_portfolio(monthly_total: float, allocation: Dict[str, float], years_list=(3, 5, 10), stepup=0.05):
    per_asset = {a: {} for a in allocation.keys()}
    total_value, total_principal, total_profit = {}, {}, {}

    for yrs in years_list:
        tv = 0.0
        tp = 0.0
        for asset, pct in allocation.items():
            monthly_asset = monthly_total * (pct / 100.0)
            r = ASSET_RETURNS.get(asset, 0.06)
            fv = fv_stepup_monthly(monthly_asset, yrs, r, growth=stepup)
            per_asset[asset][f"{yrs}y"] = fv
            tv += fv
            tp += invested_principal_stepup(monthly_asset, yrs, growth=stepup)
        total_value[yrs] = tv
        total_principal[yrs] = tp
        total_profit[yrs] = tv - tp
    return per_asset, total_value, total_principal, total_profit

def make_breakdown_table(monthly_total: float, allocation: Dict[str, float], years: int = 10, stepup=0.05) -> pd.DataFrame:
    rows = []
    cum_inv = 0.0
    for yr in range(1, years + 1):
        monthly_this_year = monthly_total * ((1 + stepup) ** (yr - 1))
        annual_invested = monthly_this_year * 12.0
        cum_inv += annual_invested

        per_asset, total_value, _, _ = project_portfolio(monthly_total, allocation, years_list=(yr,), stepup=stepup)
        val = total_value[yr]
        profit = val - cum_inv
        rows.append({
            "Year": yr,
            "Monthly": monthly_this_year,
            "Annual Invested": annual_invested,
            "Cumulative Invested": cum_inv,
            "Portfolio Value": val,
            "Profit": profit
        })
    return pd.DataFrame(rows)

# =========================
# --------- UI ------------
# =========================

st.title(" Smart Investment Advisor")
st.caption("Uses your trained model to classify risk; suggests goal-aware allocations; "
           "recommends stocks & mutual funds; projects returns with **5% annual growth**.")

# Fixed folders (silent auto-pick of latest files)
stock_folder = "live_stocks_data_final"
mf_folder = "mutualfund_data_final"
stock_file = get_latest_file(stock_folder)
mf_file = get_latest_file(mf_folder)

with st.sidebar:
    st.header("Projection Settings")
    stepup = st.number_input(
        "Annual Growth (e.g., 0.05 for 5%)",
        min_value=0.0, max_value=0.5, value=0.5, step=0.01, format="%.2f"
    )
    st.caption("Default = 5% yearly increase.")
    # Collapsed debug only (no paths shown in main UI)
   
# Load data silently
stock_df = safe_read_csv(stock_file)
mf_df = safe_read_csv(mf_file)

# Gentle warnings (no paths)
if stock_df.empty:
    st.warning("Stock data not loaded or empty. Stock picks will be skipped.")
if mf_df.empty:
    st.warning("Mutual fund data not loaded or empty. MF picks will be skipped.")

st.divider()
st.subheader("Enter your financial profile")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Enter your Age", min_value=18, max_value=100, value=30, step=1)
with col2:
    income = st.number_input("Enter your Monthly Income (₹)", min_value=0, value=50000, step=1000)
with col3:
    edu = st.selectbox(
        "Education Level (1–5)",
        options=[1,2,3,4,5],
        format_func=lambda x: {
            1:"1 = No Formal Education",
            2:"2 = High School",
            3:"3 = Bachelor’s Degree",
            4:"4 = Master’s Degree",
            5:"5 = Doctorate/Professional"
        }[x],
        index=3
    )

st.markdown("Risk Questions(1 = Strongly Disagree | 2 = Disagree | 3 = Neutral | 4 = Agree | 5 = Strongly Agree)")
risk_cols = st.columns(3)
f30_answers: List[int] = []
for i, q in enumerate(F30_QUESTIONS, start=1):
    with risk_cols[(i-1) % 3]:
        val = st.slider(f"{q} (1–5)", min_value=1, max_value=5, value=3, key=f"f30_{i}")
        f30_answers.append(val)

goal = st.selectbox(" Primary Goal", list(GOAL_CHOICES.keys()), index=2)
monthly_invest = st.number_input(" How much do you want to invest this month (₹)?", min_value=500, value=10000, step=500)

st.divider()
run = st.button("▶ Run Advisor")

if run:
    # ----------------- Load model
    model, scaler, encoder, err = load_model_parts()
    if err or model is None or scaler is None or encoder is None:
        st.error(f"Could not load model/scaler/encoder: {err or 'Unknown error'}")
        st.stop()

    # ----------------- Predict risk
    feature_row = pd.DataFrame([[age, income, edu] + f30_answers], columns=FEATURE_ORDER)
    try:
        X_scaled = scaler.transform(feature_row)
        pred_idx = model.predict(X_scaled)[0]
        risk_label = encoder.inverse_transform([pred_idx])[0]
        if risk_label not in ("Conservative", "Moderate", "Aggressive"):
            risk_label = "Moderate"
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.markdown(f"###  Predicted Risk Profile: **{risk_label}**")

    # ----------------- Allocation
    alloc0 = base_allocation(risk_label)
    goal_key = GOAL_CHOICES[goal]
    alloc = adjust_for_goal(alloc0, goal_key)

    s_alloc = sum(alloc.values())
    if abs(s_alloc - 100) > 1e-6:
        alloc = {k: v * 100.0 / s_alloc for k, v in alloc.items()}

    amounts = {k: monthly_invest * (v/100.0) for k, v in alloc.items()}
    diff = monthly_invest - sum(amounts.values())
    if abs(diff) > 1e-6:
        if "Savings" in amounts:
            amounts["Savings"] += diff * 0.7
        if "Emergency Fund" in amounts:
            amounts["Emergency Fund"] += diff * 0.3

    st.markdown("###  Final Suggested Allocation")
    alloc_df = pd.DataFrame({
        "Asset": list(alloc.keys()),
        "Percent": [f"{v:.2f}%" for v in alloc.values()],
        "Monthly Amount (₹)": [rupee(amounts[a]) for a in alloc.keys()]
    })
    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

    # ----------------- Recommendations
    seed = int((age * 1000003 + monthly_invest * 7 + len(goal_key) * 101) % (2**32 - 1))

    # Stocks
    if not stock_df.empty and amounts.get("Stocks", 0) > 0.01:
        try:
            stock_reco = pick_stocks(stock_df, goal_key, risk_label, seed=seed, n=5).copy()
            if not stock_reco.empty:
                per_stock = amounts["Stocks"] / len(stock_reco)
                stock_reco["Invest (₹)"] = per_stock
                if "avg_return_30d" in stock_reco.columns:
                    stock_reco["Avg Return 30d"] = stock_reco["avg_return_30d"].map(
                        lambda x: f"{x:.2%}" if pd.notna(x) else "—"
                    )
                else:
                    stock_reco["Avg Return 30d"] = "—"
                st.markdown("### Recommended Stocks")
                st.dataframe(
                    stock_reco.rename(columns={"STOCK":"Stock"})[["Stock", "Invest (₹)", "Avg Return 30d"]],
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("No suitable stocks found after filtering.")
        except Exception as e:
            st.warning(f"Skipping stock picks due to error: {e}")
    elif amounts.get("Stocks", 0) <= 0.01:
        st.info("Stocks allocation is ₹0; skipping stock picks.")
    else:
        st.warning("No stock data found or required columns missing; skipping stock picks.")

    # Mutual Funds
    if not mf_df.empty and amounts.get("Mutual Funds", 0) > 0.01:
        try:
            mf_reco = pick_mutual_funds(mf_df, goal_key, risk_label, seed=seed+13, n=5).copy()
            if not mf_reco.empty:
                per_mf = amounts["Mutual Funds"] / len(mf_reco)
                mf_reco["Invest (₹)"] = per_mf
                if "Volatility (90d)" not in mf_reco.columns:
                    mf_reco["Volatility (90d)"] = np.nan
                st.markdown("###  Recommended Mutual Funds")
                st.dataframe(mf_reco[["Scheme Name", "Invest (₹)", "Volatility (90d)"]],
                             use_container_width=True, hide_index=True)
            else:
                st.info("No suitable mutual funds found after filtering.")
        except Exception as e:
            st.warning(f"Skipping mutual fund picks due to error: {e}")
    elif amounts.get("Mutual Funds", 0) <= 0.01:
        st.info("Mutual Funds allocation is ₹0; skipping MF picks.")
    else:
        st.warning("No mutual fund data found or required columns missing; skipping MF picks.")

    # ----------------- Projections 
    st.markdown("### Projections by Asset *(assumed annual returns)*")

    per_asset_vals, total_val, total_principal, total_profit = project_portfolio(
        monthly_invest, alloc, years_list=(3,5,10), stepup=stepup
    )

    proj_rows = []
    for asset in alloc.keys():
        proj_rows.append({
            "Asset": asset,
            "Monthly (₹)": amounts[asset],
            "3y": per_asset_vals[asset]["3y"],
            "5y": per_asset_vals[asset]["5y"],
            "10y": per_asset_vals[asset]["10y"]
        })
    proj_df = pd.DataFrame(proj_rows)
    for c in ["Monthly (₹)", "3y", "5y", "10y"]:
        proj_df[c] = proj_df[c].map(lambda v: rupee(v))

    st.dataframe(proj_df, use_container_width=True, hide_index=True)

    st.markdown("### Total Projected Portfolio Value (All Assets Combined)")
    st.write(f"**3 years:** {rupee(total_val[3])}")
    st.write(f"**5 years:** {rupee(total_val[5])}")
    st.write(f"**10 years:** {rupee(total_val[10])}")

    st.markdown("###  Total Principal Contributed (All Assets)")
    st.write(f"**3 years:** {rupee(total_principal[3])}")
    st.write(f"**5 years:** {rupee(total_principal[5])}")
    st.write(f"**10 years:** {rupee(total_principal[10])}")

    st.markdown("###  Total Profit (All Assets)")
    st.write(f"**3 years:** {rupee(total_profit[3])}")
    st.write(f"**5 years:** {rupee(total_profit[5])}")
    st.write(f"**10 years:** {rupee(total_profit[10])}")

    st.markdown("### Summary (All Assets Combined)")
    sip_df = pd.DataFrame({
        "Period": ["3y", "5y", "10y"],
        "Invested Amount": [total_principal[3], total_principal[5], total_principal[10]],
        "Profit (Returns)": [total_profit[3], total_profit[5], total_profit[10]],
        "Total Value": [total_val[3], total_val[5], total_val[10]],
    })
    for c in ["Invested Amount", "Profit (Returns)", "Total Value"]:
        sip_df[c] = sip_df[c].map(lambda v: rupee(v))
    st.dataframe(sip_df, use_container_width=True, hide_index=True)

    st.caption(
        " Projections assume monthly contributions increase by **5% each year** "
        " and annual compounding with these category returns: "
        f"{', '.join([f'{k} {int(v*100)}%' for k,v in ASSET_RETURNS.items()])}."
    )

    with st.expander("Show Year-by-Year Breakdown (up to 10 years)"):
        years_break = st.slider("Years to display", 1, 10, 10)
        bdf = make_breakdown_table(monthly_invest, alloc, years=years_break, stepup=stepup).copy()
        for c in ["Monthly", "Annual Invested", "Cumulative Invested", "Portfolio Value", "Profit"]:
            bdf[c] = bdf[c].map(lambda v: rupee(v))
        st.dataframe(bdf, use_container_width=True, hide_index=True)

# ============== END APP ==============
