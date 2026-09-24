# =============================================================================
# Bitcoin Data Analytics Dashboard
# Internship Project | Kaggle Dataset: BTC/USD 1-Min OHLCV
# Author: Intern
# =============================================================================

import gc
from io import StringIO
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Bitcoin Analytics Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS — mirrors prior project card / dashboard style
# =============================================================================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #262d3f);
        border: 1px solid #3a3f5c;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-label { color: #9ca3af; font-size: 13px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
    .metric-value { color: #f9fafb; font-size: 28px; font-weight: 700; margin: 6px 0; }
    .metric-delta-pos { color: #22c55e; font-size: 13px; font-weight: 600; }
    .metric-delta-neg { color: #ef4444; font-size: 13px; font-weight: 600; }
    .section-header {
        font-size: 20px; font-weight: 700; color: #f9fafb;
        border-left: 4px solid #f7931a;
        padding-left: 12px; margin: 24px 0 14px 0;
    }
    .sidebar-title { color: #f7931a; font-size: 16px; font-weight: 700; }
    .stPlotlyChart { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LAYER — Low-memory chunked loading + Unix → DateTime + Daily resample
# =============================================================================
DATA_PATH = "btcusd_1-min_data.csv"
CHUNK_SIZE = 200_000


@st.cache_data(show_spinner="⏳ Loading & downsampling Bitcoin data…")
def load_and_resample(filepath: str) -> pd.DataFrame:
    """
    Chunked low-memory read of 1-min OHLCV data.
    Immediately downsamples to Daily ('D') to keep memory manageable.
    """
    chunks = []
    try:
        reader = pd.read_csv(
            filepath,
            chunksize=CHUNK_SIZE,
            dtype={
                "Open":   "float32",
                "High":   "float32",
                "Low":    "float32",
                "Close":  "float32",
                "Volume": "float32",
            },
        )
        for chunk in reader:
            # Parse Unix timestamp
            chunk["Date"] = pd.to_datetime(chunk["Timestamp"], unit="s", utc=True)
            chunk.set_index("Date", inplace=True)
            chunk.drop(columns=["Timestamp"], inplace=True)

            # Aggressive daily resample — shrinks millions of rows → thousands
            daily = chunk.resample("D").agg(
                Open=("Open", "first"),
                High=("High", "max"),
                Low=("Low", "min"),
                Close=("Close", "last"),
                Volume=("Volume", "sum"),
            )
            chunks.append(daily)
            del chunk
            gc.collect()

        df = pd.concat(chunks).resample("D").agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
        )

        # Forward-fill NaN gaps (continuous asset price representation)
        df = df.ffill()
        # Drop rows where all OHLC are zero / still NaN after ffill
        df = df[df["Close"] > 0].dropna()

        # Downcast to float32 to reduce memory footprint
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype("float32")

        df.index = df.index.tz_localize(None)  # timezone-naive for cleaner display
        gc.collect()
        return df

    except FileNotFoundError:
        st.error(f"Dataset not found at '{filepath}'. Place `btcusd_1-min_data.csv` in the same directory.")
        st.stop()


# =============================================================================
# FEATURE ENGINEERING — purely on daily downsampled data
# =============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Return_1D"]   = d["Close"].pct_change(1)
    d["Return_7D"]   = d["Close"].pct_change(7)
    d["Return_30D"]  = d["Close"].pct_change(30)
    d["MA_7"]        = d["Close"].rolling(7).mean()
    d["MA_30"]       = d["Close"].rolling(30).mean()
    d["MA_90"]       = d["Close"].rolling(90).mean()
    d["Volatility"]  = d["Close"].rolling(14).std()
    d["RSI"]         = _compute_rsi(d["Close"], 14)
    d["MACD"]        = d["Close"].ewm(span=12).mean() - d["Close"].ewm(span=26).mean()
    d["Signal"]      = d["MACD"].ewm(span=9).mean()
    d["BB_Upper"]    = d["MA_30"] + 2 * d["Volatility"]
    d["BB_Lower"]    = d["MA_30"] - 2 * d["Volatility"]
    d["High_Low_Pct"] = (d["High"] - d["Low"]) / d["Close"]
    d["Volume_MA7"]  = d["Volume"].rolling(7).mean()
    d = d.dropna()
    return d


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


# =============================================================================
# ML PIPELINE — Random Forest price forecast
# =============================================================================
FEATURE_COLS = [
    "Open", "High", "Low", "Volume",
    "Return_1D", "Return_7D", "MA_7", "MA_30", "MA_90",
    "Volatility", "RSI", "MACD", "Signal",
    "BB_Upper", "BB_Lower", "High_Low_Pct", "Volume_MA7",
]


def build_ml_dataset(df: pd.DataFrame, horizon: int):
    """Target = Close price `horizon` days ahead."""
    d = df.copy()
    d["Target"] = d["Close"].shift(-horizon)
    d = d.dropna()
    X = d[FEATURE_COLS].astype("float32")
    y = d["Target"].astype("float32")
    return X, y, d.index


@st.cache_data(show_spinner="🤖 Training Random Forest model…")
def train_model(df_json: str, horizon: int):
    df = pd.read_json(StringIO(df_json))
    df.index = pd.to_datetime(df.index)
    X, y, idx = build_ml_dataset(df, horizon)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    metrics = {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2":   r2_score(y_test, y_pred),
        "MAPE": float(np.mean(np.abs((y_test.values - y_pred) / (y_test.values + 1e-9))) * 100),
    }

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    return y_test, y_pred, idx_test, metrics, importances, scaler, model, X_test_s


# =============================================================================
# CHART HELPERS
# =============================================================================
BITCOIN_ORANGE = "#f7931a"
CHART_BG       = "#0e1117"
GRID_COLOR     = "#1e2535"
TEXT_COLOR      = "#d1d5db"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CHART_BG,
    plot_bgcolor=CHART_BG,
    font=dict(color=TEXT_COLOR, size=12),
    xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False),
    margin=dict(l=50, r=30, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    hovermode="x unified",
)


def _apply_layout(fig, title=""):
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=15, color="#f9fafb")))
    return fig


def plot_price_history(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04)
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"],  close=df["Close"],
        name="OHLC",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
    ), row=1, col=1)
    # MA overlays
    for ma, col in [("MA_7", "#60a5fa"), ("MA_30", BITCOIN_ORANGE), ("MA_90", "#a78bfa")]:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma,
                                     line=dict(color=col, width=1.5)), row=1, col=1)
    # Volume bars
    colours = ["#22c55e" if c >= o else "#ef4444"
               for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                         marker_color=colours, name="Volume", opacity=0.7), row=2, col=1)

    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text="Bitcoin Price History (Daily)", font=dict(size=15, color="#f9fafb")),
                      xaxis_rangeslider_visible=False)
    return fig


def plot_bollinger(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                             line=dict(color="#6366f1", dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                             line=dict(color="#6366f1", dash="dot", width=1),
                             fill="tonexty", fillcolor="rgba(99,102,241,0.08)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color=BITCOIN_ORANGE, width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_30"], name="MA-30",
                             line=dict(color="#94a3b8", width=1, dash="dash")))
    return _apply_layout(fig, "Bollinger Bands (30-Day Window)")


def plot_macd(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.45])
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color=BITCOIN_ORANGE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],   name="MACD",
                             line=dict(color="#60a5fa", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal",
                             line=dict(color="#f43f5e", width=1.5)), row=2, col=1)
    hist = df["MACD"] - df["Signal"]
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in hist]
    fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogram",
                         marker_color=colors, opacity=0.7), row=2, col=1)
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text="MACD Indicator", font=dict(size=15, color="#f9fafb")))
    return fig


def plot_rsi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#a78bfa", width=2)))
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought 70")
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", annotation_text="Oversold 30")
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.03)", line_width=0)
    return _apply_layout(fig, "Relative Strength Index (RSI-14)")


def plot_annual_returns(df: pd.DataFrame) -> go.Figure:
    yearly = df["Close"].resample("YE").last().pct_change().dropna() * 100
    colors = [("#22c55e" if v >= 0 else "#ef4444") for v in yearly.values]
    fig = go.Figure(go.Bar(
        x=[str(y.year) for y in yearly.index],
        y=yearly.values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in yearly.values],
        textposition="outside",
    ))
    return _apply_layout(fig, "Annual Returns (%)")


def plot_forecast(y_test, y_pred, idx_test) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(idx_test), y=list(y_test),
                             name="Actual", line=dict(color=BITCOIN_ORANGE, width=2)))
    fig.add_trace(go.Scatter(x=list(idx_test), y=list(y_pred),
                             name="Predicted", line=dict(color="#60a5fa", width=2, dash="dash")))
    return _apply_layout(fig, "Random Forest: Actual vs Predicted Close Price")


def plot_feature_importance(importances: pd.Series) -> go.Figure:
    top = importances.head(12)
    fig = go.Figure(go.Bar(
        x=top.values[::-1], y=top.index[::-1],
        orientation="h", marker_color=BITCOIN_ORANGE,
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text="Top Feature Importances", font=dict(size=15, color="#f9fafb")),
                      xaxis_title="Importance Score")
    return fig


# =============================================================================
# METRIC CARD HELPER
# =============================================================================
def metric_card(label: str, value: str, delta: str = "", positive: bool = True):
    delta_class = "metric-delta-pos" if positive else "metric-delta-neg"
    delta_html  = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-title">₿ Bitcoin Analytics</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**📂 Data Controls**")
        data_path = st.text_input("CSV File Path", value=DATA_PATH)

        st.markdown("---")
        st.markdown("**📅 Date Range Filter**")
        date_start = st.date_input("From", value=pd.Timestamp("2017-01-01"))
        date_end   = st.date_input("To",   value=pd.Timestamp("2023-12-31"))

        st.markdown("---")
        st.markdown("**🤖 ML Prediction Settings**")
        horizon = st.slider("Forecast Horizon (Days Ahead)", min_value=1, max_value=30, value=7, step=1)
        run_ml  = st.button("▶ Train & Predict", use_container_width=True)

        st.markdown("---")
        st.markdown("**📊 Chart Settings**")
        show_bb    = st.checkbox("Show Bollinger Bands", value=True)
        show_macd  = st.checkbox("Show MACD",            value=True)
        show_rsi   = st.checkbox("Show RSI",             value=True)
        show_yret  = st.checkbox("Show Annual Returns",  value=True)

        st.markdown("---")
        st.caption("Dataset: Kaggle — BTC/USD 1-Min OHLCV\n\nDashboard: Bitcoin Analytics")

    # ── Load Data ─────────────────────────────────────────────────────────────
    raw_df = load_and_resample(data_path)

    # Date filter
    df = raw_df.loc[str(date_start):str(date_end)].copy()
    if df.empty:
        st.error("No data in selected date range. Adjust the sidebar date filters.")
        st.stop()

    # Feature engineering on filtered slice
    df_feat = engineer_features(df)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## ₿ Bitcoin Data Analytics Dashboard")
    st.markdown(f"Showing **{len(df_feat):,} trading days** | "
                f"`{df_feat.index[0].date()}` → `{df_feat.index[-1].date()}`")
    st.markdown("---")

    # ── Metric Cards ──────────────────────────────────────────────────────────
    latest        = df_feat["Close"].iloc[-1]
    prev          = df_feat["Close"].iloc[-2]
    price_chg     = ((latest - prev) / prev) * 100
    all_time_high = df_feat["High"].max()
    all_time_low  = df_feat["Low"].min()
    total_vol     = df_feat["Volume"].sum()
    avg_vol_7d    = df_feat["Volume"].tail(7).mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Latest Close Price", f"${latest:,.2f}",
                          f"{'▲' if price_chg>=0 else '▼'} {abs(price_chg):.2f}% (1D)",
                          positive=(price_chg >= 0))
    with c2: metric_card("All-Time High (Period)", f"${all_time_high:,.2f}")
    with c3: metric_card("All-Time Low (Period)",  f"${all_time_low:,.2f}")
    with c4: metric_card("7-Day Avg Volume", f"{avg_vol_7d:,.1f} BTC")

    st.markdown("---")

    # ── Price History Chart ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Historical Price & Volume</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_price_history(df_feat), use_container_width=True)

    # ── Technical Indicators ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">📐 Technical Indicators</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    if show_bb:
        with col_l:
            st.plotly_chart(plot_bollinger(df_feat), use_container_width=True)
    if show_macd:
        with col_r:
            st.plotly_chart(plot_macd(df_feat), use_container_width=True)

    if show_rsi or show_yret:
        col_a, col_b = st.columns(2)
        if show_rsi:
            with col_a:
                st.plotly_chart(plot_rsi(df_feat), use_container_width=True)
        if show_yret:
            with col_b:
                st.plotly_chart(plot_annual_returns(df_feat), use_container_width=True)

    # ── Volume Trend ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📦 Volume Trend Analysis</div>', unsafe_allow_html=True)
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["Volume"],
                                 fill="tozeroy", name="Daily Volume",
                                 line=dict(color="#60a5fa", width=1.5)))
    vol_fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["Volume_MA7"],
                                 name="7-Day MA Volume",
                                 line=dict(color=BITCOIN_ORANGE, width=2, dash="dash")))
    vol_fig.update_layout(**PLOTLY_LAYOUT,
                           title=dict(text="Daily Trading Volume (BTC)", font=dict(size=15, color="#f9fafb")))
    st.plotly_chart(vol_fig, use_container_width=True)

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)
    corr_cols = ["Close", "Volume", "Return_1D", "Return_7D", "MA_7", "MA_30", "RSI", "MACD", "Volatility"]
    corr = df_feat[corr_cols].corr()
    fig_corr, ax = plt.subplots(figsize=(9, 6))
    fig_corr.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=45, ha="right", color=TEXT_COLOR, fontsize=9)
    ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(corr_cols, color=TEXT_COLOR, fontsize=9)
    plt.colorbar(im, ax=ax)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=7)
    plt.tight_layout()
    st.pyplot(fig_corr)
    plt.close(fig_corr)
    gc.collect()

    # ── ML Section ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 Machine Learning — Price Forecast</div>', unsafe_allow_html=True)

    if run_ml:
        with st.spinner(f"Training Random Forest to predict {horizon}-day-ahead Close price…"):
            df_json = df_feat[FEATURE_COLS + ["Close"]].to_json()
            y_test, y_pred, idx_test, metrics, importances, scaler, model, _ = train_model(df_json, horizon)

        # Model metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric_card("MAE  (USD)",  f"${metrics['MAE']:,.2f}")
        with m2: metric_card("RMSE (USD)",  f"${metrics['RMSE']:,.2f}")
        with m3: metric_card("R² Score",    f"{metrics['R2']:.4f}", positive=(metrics['R2'] >= 0.8))
        with m4: metric_card("MAPE (%)",    f"{metrics['MAPE']:.2f}%", positive=(metrics['MAPE'] <= 10))

        st.plotly_chart(plot_forecast(y_test, y_pred, idx_test), use_container_width=True)

        col_fi, col_err = st.columns(2)
        with col_fi:
            st.plotly_chart(plot_feature_importance(importances), use_container_width=True)
        with col_err:
            errors = np.array(y_test) - np.array(y_pred)
            fig_err = go.Figure(go.Histogram(x=errors, nbinsx=60,
                                             marker_color=BITCOIN_ORANGE, opacity=0.85))
            _apply_layout(fig_err, "Prediction Error Distribution")
            fig_err.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
            st.plotly_chart(fig_err, use_container_width=True)

        gc.collect()
    else:
        st.info("👈 Configure settings in the sidebar and click **▶ Train & Predict** to run the ML pipeline.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption("Bitcoin Analytics Dashboard | Kaggle Dataset: BTC/USD 1-Min OHLCV | Built with Streamlit + Plotly + scikit-learn")


if __name__ == "__main__":
    main()
