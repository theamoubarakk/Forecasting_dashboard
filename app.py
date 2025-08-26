# app.py
import os, logging, warnings, itertools, contextlib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet, StanBackendEnum  # <-- backend enum

# --------------------------- PAGE & THEME ---------------------------
st.set_page_config(page_title="Forecast Dashboard", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 1.1rem; padding-bottom: 0.6rem;}
  .card, .kpi {
      background: #ffffff; border: 1px solid rgba(0,0,0,0.06);
      border-radius: 14px; padding: 16px;
  }
  h1, h2, h3 { color:#1f4bd8; font-weight:700; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = "(3) BABA JINA SALES DATA.xlsx"  # keep your filename

# ---------------------- UTILITIES & QUIET LOGGING -------------------
for name in ["cmdstanpy", "prophet"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).propagate = False
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def _quiet(fn, *args, **kwargs):
    with open(os.devnull, "w") as devnull, \
         contextlib.redirect_stdout(devnull), \
         contextlib.redirect_stderr(devnull):
        return fn(*args, **kwargs)

def _to_MS(s):
    """Normalize dates (month-end or any date) to month-start for alignment."""
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp("MS")

# Ensure CmdStan is available once (cached across sessions)
@st.cache_resource
def ensure_cmdstan():
    from cmdstanpy import cmdstan_path, install_cmdstan
    try:
        _ = cmdstan_path()
    except Exception:
        install_cmdstan()
    return True

ensure_cmdstan()

# ========================= FORECAST HELPERS =========================
@st.cache_data
def get_costume_forecast_df() -> pd.DataFrame:
    """Halloween → Costume (SARIMA, ×25 scaling; 2025 forecast with 80% CI)."""
    df = pd.read_excel(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    costume_df = df[(df["Category"] == "Halloween") & (df["Subcategory"] == "Costume")].copy()
    costume_df = costume_df[["Date", "Quantity"]].rename(columns={"Date": "ds", "Quantity": "y"})

    # Monthly aggregation (month-start index)
    monthly_df = (costume_df.groupby(pd.Grouper(key="ds", freq="MS")).sum()
                  .reset_index().set_index("ds").asfreq("MS"))
    # Differencing (your script)
    monthly_df["y_diff"] = monthly_df["y"].diff()
    monthly_df = monthly_df.dropna()

    # Train / test
    train_df = monthly_df[monthly_df.index.year <= 2023].copy()
    test_df  = monthly_df[monthly_df.index.year == 2024].copy()

    # Grid (kept as your code; errors are skipped)
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x, y, z, 12) for x, y, z in pdq]
    for order in pdq:
        for s_order in seasonal_pdq:
            try:
                m = SARIMAX(train_df["y"], order=order, seasonal_order=s_order,
                            enforce_stationarity=False, enforce_invertibility=False)
                res = m.fit(disp=False)
                fc = res.get_forecast(steps=len(test_df))
                _ = root_mean_squared_error(test_df["y"], fc.predicted_mean)
            except Exception:
                continue

    # Your chosen best orders
    best_order = (0, 1, 0)
    best_seasonal_order = (0, 1, 0, 12)

    # Fit full history (on non-differenced y)
    full_df = costume_df.set_index("ds").resample("MS").sum().asfreq("MS")[["y"]].copy()
    model = SARIMAX(full_df["y"], order=best_order, seasonal_order=best_seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    # Forecast 12 months (2025), 80% CI
    fc_res = results.get_forecast(steps=12)
    fc_mean = fc_res.predicted_mean
    fc_ci   = fc_res.conf_int(alpha=0.2)

    last_date = full_df.index[-1]
    fc_idx = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=12, freq="MS")

    # Scaling ×25 (history + future)
    SCALE = 25.0
    hist = full_df.copy()
    hist["y"] = hist["y"] * SCALE
    hist = hist.reset_index().rename(columns={"ds": "ds"})
    hist["ds"] = _to_MS(hist["ds"])
    hist["yhat"] = np.nan; hist["yhat_lower"] = np.nan; hist["yhat_upper"] = np.nan

    fut = pd.DataFrame({
        "ds": _to_MS(fc_idx),
        "yhat": fc_mean.values * SCALE,
        "yhat_lower": fc_ci.iloc[:, 0].values * SCALE,
        "yhat_upper": fc_ci.iloc[:, 1].values * SCALE,
    })

    out = pd.concat([hist[["ds","y","yhat","yhat_lower","yhat_upper"]], fut], ignore_index=True)
    out["category"] = "Costume"
    return out


@st.cache_data
def get_toys_forecast_df() -> pd.DataFrame:
    """Toys (Prophet multiplicative + Oct/Dec regressors; 80% CI)."""
    df = pd.read_excel(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    toy_df = df[df["Category"] == "Toys"].copy()
    monthly_toy_df = (toy_df.resample("ME", on="Date").agg({"Quantity":"sum"})
                      .reset_index().rename(columns={"Date":"ds","Quantity":"y"})
                      .sort_values("ds").dropna(subset=["ds","y"]).reset_index(drop=True))

    # Split 2024 holdout (kept from your code)
    train_df = monthly_toy_df[monthly_toy_df["ds"].dt.year <= 2023].copy()
    test_df  = monthly_toy_df[monthly_toy_df["ds"].dt.year == 2024].copy()

    def add_month_flags(dfx):
        out = dfx.copy()
        out["month"] = out["ds"].dt.month
        out["oct_bump"] = (out["month"] == 10).astype(int)
        out["dec_peak"] = (out["month"] == 12).astype(int)
        return out.drop(columns=["month"])

    train_reg = add_month_flags(train_df)
    test_reg  = add_month_flags(test_df)

    cps_grid = [0.05, 0.1, 0.3]
    sps_grid = [1.0, 5.0, 10.0]
    best_rmse, best_model, best_params = float("inf"), None, None

    for cps in cps_grid:
        for sps in sps_grid:
            m = Prophet(
                interval_width=0.8,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode="multiplicative",
                changepoint_prior_scale=cps,
                seasonality_prior_scale=sps,
                stan_backend=StanBackendEnum.CMDSTANPY,  # <-- force CmdStanPy
            )
            m.add_regressor("oct_bump", mode="multiplicative")
            m.add_regressor("dec_peak", mode="multiplicative")
            m.fit(train_reg)

            future_test = add_month_flags(pd.DataFrame({
                "ds": pd.date_range("2024-01-31", "2024-12-31", freq="ME")
            }))
            fc_test = m.predict(future_test)[["ds","yhat"]]
            eval_df = test_reg.merge(fc_test, on="ds", how="left").dropna(subset=["yhat"])
            rmse = np.sqrt(((eval_df["y"]-eval_df["yhat"])**2).mean())
            if rmse < best_rmse:
                best_rmse, best_model, best_params = rmse, m, (cps, sps)

    # Refit on full data with best params
    full_reg = add_month_flags(monthly_toy_df)
    m_full = Prophet(
        interval_width=0.8,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=best_params[0],
        seasonality_prior_scale=best_params[1],
        stan_backend=StanBackendEnum.CMDSTANPY,  # <-- force CmdStanPy
    )
    m_full.add_regressor("oct_bump", mode="multiplicative")
    m_full.add_regressor("dec_peak", mode="multiplicative")
    m_full.fit(full_reg)

    future_2025 = add_month_flags(pd.DataFrame({
        "ds": pd.date_range("2025-01-31", "2025-12-31", freq="ME")
    }))
    fc_2025 = m_full.predict(future_2025)[["ds","yhat","yhat_lower","yhat_upper"]]

    # Normalize to MS
    hist = monthly_toy_df.copy()
    hist["ds"] = _to_MS(hist["ds"])
    hist["yhat"] = np.nan; hist["yhat_lower"] = np.nan; hist["yhat_upper"] = np.nan

    fut = fc_2025.copy()
    fut["ds"] = _to_MS(fut["ds"])
    fut[["yhat","yhat_lower","yhat_upper"]] = fut[["yhat","yhat_lower","yhat_upper"]].clip(lower=0)

    out = pd.concat([hist[["ds","y","yhat","yhat_lower","yhat_upper"]], fut], ignore_index=True)
    out["category"] = "Toys"
    return out


@st.cache_data
def get_bicycles_forecast_df() -> pd.DataFrame:
    """Bicycles (Prophet 80% CI; ×10 scaling; 2025 future)."""
    df = pd.read_excel(DATA_PATH)
    bicycles_df = df[df["Subcategory"] == "Bicycles"].copy()
    bicycles_df.rename(columns={"Date":"ds","Quantity":"y"}, inplace=True)
    bicycles_df["ds"] = pd.to_datetime(bicycles_df["ds"])
    bicycles_df = bicycles_df.sort_values("ds")

    monthly_bicycles = (bicycles_df.groupby(pd.Grouper(key="ds", freq="ME"))
                        .agg({"y":"sum"}).reset_index().dropna(subset=["ds"])
                        .sort_values("ds").reset_index(drop=True))

    model = Prophet(
        interval_width=0.8,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        stan_backend=StanBackendEnum.CMDSTANPY,  # <-- force CmdStanPy
    )
    _quiet(model.fit, monthly_bicycles)

    future_df = model.make_future_dataframe(periods=24, freq="ME")
    forecast  = _quiet(model.predict, future_df)[["ds","yhat","yhat_lower","yhat_upper"]]
    fc_2025 = forecast[(forecast["ds"] >= "2025-01-01") & (forecast["ds"] <= "2025-12-31")].copy()

    # Normalize to MS & scale ×10 (history + future)
    SCALE = 10.0
    monthly_bicycles["ds"] = _to_MS(monthly_bicycles["ds"])
    fc_2025["ds"] = _to_MS(fc_2025["ds"])

    hist = monthly_bicycles.copy()
    hist["y"] *= SCALE
    hist["yhat"] = np.nan; hist["yhat_lower"] = np.nan; hist["yhat_upper"] = np.nan

    fut = fc_2025.copy()
    fut[["yhat","yhat_lower","yhat_upper"]] *= SCALE

    out = pd.concat([hist[["ds","y","yhat","yhat_lower","yhat_upper"]],
                     fut[["ds","yhat","yhat_lower","yhat_upper"]]], ignore_index=True)
    out["category"] = "Bicycles"
    return out


@st.cache_data
def get_forecasts() -> pd.DataFrame:
    parts = []
    try: parts.append(get_costume_forecast_df())
    except Exception as e: st.warning(f"Costume forecast failed: {e}")
    try: parts.append(get_toys_forecast_df())
    except Exception as e: st.warning(f"Toys forecast failed: {e}")
    try: parts.append(get_bicycles_forecast_df())
    except Exception as e: st.warning(f"Bicycles forecast failed: {e}")

    if not parts: st.stop()
    return pd.concat(parts, ignore_index=True)

# ============================== UI / LOGIC ==============================
st.markdown("<h2>Forecast Dashboard</h2>", unsafe_allow_html=True)

df_all = get_forecasts()

# Sidebar filters
st.sidebar.header("Filters")
all_cats = sorted(df_all["category"].unique().tolist())
cats_sel = st.sidebar.multiselect("Categories", all_cats, default=all_cats)

min_d, max_d = df_all["ds"].min(), df_all["ds"].max()
date_range = st.sidebar.date_input("Date range", (min_d, max_d))
freq = st.sidebar.selectbox("Aggregation", ["Monthly","Quarterly","Yearly"], index=0)

# Apply filters
mask_cat = df_all["category"].isin(cats_sel)
mask_date = (df_all["ds"] >= pd.to_datetime(date_range[0])) & (df_all["ds"] <= pd.to_datetime(date_range[-1]))
dff = df_all.loc[mask_cat & mask_date].copy()

def resample_df(dfx, freq_name):
    rule = {"Monthly":"M","Quarterly":"Q","Yearly":"Y"}[freq_name]
    g = (dfx.set_index("ds")
           .groupby("category")[["y","yhat","yhat_lower","yhat_upper"]]
           .resample(rule).sum()
           .reset_index())
    return g

def kpis(dfx):
    total_actual = dfx["y"].dropna().sum()
    total_forecast = dfx["yhat"].sum()
    common = dfx.dropna(subset=["y"]).copy()
    mape = (np.abs(common["y"]-common["yhat"]) / np.maximum(1e-9, np.abs(common["y"]))).mean()*100 if len(common) else np.nan
    return total_actual, total_forecast, mape

def line_chart(dfx, title):
    base = alt.Chart(dfx).encode(x="ds:T")
    band = base.mark_area(opacity=0.15).encode(
        y="yhat_lower:Q", y2="yhat_upper:Q", color=alt.Color("category:N", legend=None)
    )
    fc = base.mark_line(strokeDash=[4,3]).encode(y="yhat:Q", color="category:N")
    act = base.mark_line().encode(y="y:Q", color="category:N")
    return alt.layer(band, fc, act).properties(height=260, title=title).interactive()

def bar_chart(dfx, title):
    return (alt.Chart(dfx)
            .mark_bar()
            .encode(x="ds:T", y="y:Q", color="category:N", tooltip=["ds:T","category:N","y:Q"])
            .properties(height=220, title=title)
            .interactive())

agg = resample_df(dff, freq)

# KPIs
k1, k2, k3 = st.columns(3)
act_sum, fc_sum, mape = kpis(agg)
with k1:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.metric("Actual (selected period)", f"{act_sum:,.0f}")
    st.markdown("</div>", unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.metric("Forecast (selected period)", f"{fc_sum:,.0f}")
    st.markdown("</div>", unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.metric("MAPE (overlap)", f"{mape:0.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

# 3-column layout
left, mid, right = st.columns([1,1,1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.altair_chart(line_chart(agg, "Actual vs Forecast (Bands = CI)"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with mid:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.altair_chart(bar_chart(agg.dropna(subset=["y"]), "Historical Actuals"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    tabs = st.tabs(cats_sel if cats_sel else ["No category selected"])
    for i, c in enumerate(cats_sel):
        with tabs[i]:
            st.altair_chart(line_chart(agg[agg["category"]==c], f"{c}: Actual vs Forecast"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Tip: choose categories & date range from the sidebar; switch aggregation Monthly/Quarterly/Yearly.")
