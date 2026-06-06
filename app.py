import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="GS Stock Predictor",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def load_models():
    lr     = joblib.load('stock-models/model_linear_regression.pkl')
    rf_clf = joblib.load('stock-models/model_random_forest_clf.pkl.gz')
    lgb    = joblib.load('stock-models/model_lightgbm.pkl')
    scaler = joblib.load('stock-models/scaler.pkl')
    return lr, rf_clf, lgb, scaler

lr, rf_clf, lgb, scaler = load_models()

@st.cache_data
def load_data():
    df = pd.read_csv('GS.csv', parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['Return_1d']  = df['Close'].pct_change(1)
    df['Return_5d']  = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Return_20d'] = df['Close'].pct_change(20)

    for w in [5, 10, 20, 50, 100, 200]:
        df[f'MA_{w}']  = df['Close'].rolling(w).mean()
        df[f'EMA_{w}'] = df['Close'].ewm(span=w).mean()

    df['Volatility_10'] = df['Return_1d'].rolling(10).std()
    df['Volatility_20'] = df['Return_1d'].rolling(20).std()

    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    df['BB_Mid']   = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['Close'].rolling(20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['Close'].rolling(20).std()
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

    df['Volume_MA10']      = df['Volume'].rolling(10).mean()
    df['Volume_Ratio']     = df['Volume'] / df['Volume_MA10']
    df['High_Low_Range']   = df['High'] - df['Low']
    df['Open_Close_Range'] = df['Close'] - df['Open']

    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)

    df['Momentum_5']  = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Momentum_20'] = df['Close'] - df['Close'].shift(20)
    df['ROC_5']       = df['Close'].pct_change(5)
    df['ROC_10']      = df['Close'].pct_change(10)

    df['OBV']        = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    df['OBV_MA10']   = df['OBV'].rolling(10).mean()
    df['OBV_Signal'] = df['OBV'] - df['OBV_MA10']

    low14  = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K']        = 100 * (df['Close'] - low14) / (high14 - low14)
    df['Stoch_D']        = df['Stoch_K'].rolling(3).mean()
    df['ATR']            = (df['High'] - df['Low']).rolling(14).mean()
    df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

    df['Target_Price']     = df['Close'].shift(-1)
    df['Target_Direction'] = (df['Target_Price'] > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

df = load_data()
feature_cols = ['Return_1d', 'Return_5d', 'Return_10d', 'Return_20d', 
                'MA_5', 'EMA_5', 'MA_10', 'EMA_10', 'MA_20', 'EMA_20', 
                'MA_50', 'EMA_50', 'MA_100', 'EMA_100', 'MA_200', 'EMA_200', 
                'Volatility_10', 'Volatility_20', 'RSI', 'MACD', 'MACD_Signal', 
                'BB_Mid', 'BB_Upper', 'BB_Lower', 'BB_Width', 'Volume_MA10', 
                'Volume_Ratio', 'High_Low_Range', 'Open_Close_Range', 
                'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5', 
                'Close_Lag_10']

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("GS Stock Predictor")
st.sidebar.markdown("Goldman Sachs ML Dashboard")
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Price Prediction",
    "30-Day Forecast",
    "Backtest",
    "Technical Indicators"
])

# ══════════════════════════════════════════════════════════
# PAGE 1 - OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Goldman Sachs (GS) -- ML Stock Predictor")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",   f"{len(df):,}")
    col2.metric("Date Range",      "1999 - 2022")
    col3.metric("Latest Close",    f"${df['Close'].iloc[-1]:.2f}")
    col4.metric("Best Model RMSE", "6.1 (Linear Reg)")

    st.markdown("---")
    st.subheader("Historical Price Chart")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['Date'], df['Close'], color='gold', linewidth=1)
    ax.fill_between(df['Date'], df['Close'], alpha=0.1, color='gold')
    ax.set_facecolor('#111')
    fig.patch.set_facecolor('#111')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333')
    ax.set_title('GS Closing Price 1999-2022', color='white')
    ax.set_ylabel('Price ($)', color='white')
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Model Performance Summary")
    results = pd.DataFrame({
        'Model': ['Linear Regression','Ridge','Lasso',
                  'LightGBM','Random Forest','KNN','LSTM','SVR'],
        'RMSE':  [6.1, 6.4, 7.0, 78.7, 80.1, 80.0, 90.8, 157.5],
        'Grade': ['Best','Excellent','Excellent',
                  'Poor','Poor','Poor','Poor','Worst']
    })
    st.dataframe(results, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 2 - PRICE PREDICTION
# ══════════════════════════════════════════════════════════
elif page == "Price Prediction":
    st.title("Next-Day Price Prediction")
    st.markdown("---")

    last_row    = df[feature_cols].iloc[[-1]]
    last_sc     = scaler.transform(last_row)
    pred_lr     = lr.predict(last_sc)[0]
    pred_lgb    = lgb.predict(last_row)[0]
    actual      = df['Close'].iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price",      f"${actual:.2f}")
    col2.metric("LR Predicted",       f"${pred_lr:.2f}",
                f"{((pred_lr-actual)/actual)*100:+.2f}%")
    col3.metric("LightGBM Predicted", f"${pred_lgb:.2f}",
                f"{((pred_lgb-actual)/actual)*100:+.2f}%")

    st.markdown("---")
    st.subheader("Prediction vs Actual (Test Set)")

    test      = df[df['Date'] >= '2020-01-01'].copy()
    X_test    = test[feature_cols]
    X_test_sc = scaler.transform(X_test)
    test['LR_Pred'] = lr.predict(X_test_sc)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test['Date'], test['Close'],   label='Actual',    color='white', linewidth=1.5)
    ax.plot(test['Date'], test['LR_Pred'], label='Predicted', color='gold',  linewidth=1.5, linestyle='--')
    ax.set_facecolor('#111')
    fig.patch.set_facecolor('#111')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333')
    ax.legend(facecolor='#222', labelcolor='white')
    ax.set_title('Linear Regression: Actual vs Predicted', color='white')
    ax.set_ylabel('Price ($)', color='white')
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════
# PAGE 3 - 30-DAY FORECAST
# ══════════════════════════════════════════════════════════
elif page == "30-Day Forecast":
    st.title("30-Day Price Forecast")
    st.markdown("---")

    days = st.slider("Forecast days", 7, 60, 30)

    last_row      = df[feature_cols].iloc[[-1]].copy()
    future_prices = []

    for _ in range(days):
        pred = lr.predict(scaler.transform(last_row))[0]
        future_prices.append(pred)
        last_row['Close_Lag_10'] = last_row['Close_Lag_5'].values[0]
        last_row['Close_Lag_5']  = last_row['Close_Lag_3'].values[0]
        last_row['Close_Lag_3']  = last_row['Close_Lag_2'].values[0]
        last_row['Close_Lag_2']  = last_row['Close_Lag_1'].values[0]
        last_row['Close_Lag_1']  = pred

    recent = df['Close'].tail(60).values

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(60), recent,
            color='white', linewidth=2, label='Actual (last 60 days)')
    ax.plot(range(60, 60 + days), future_prices,
            color='gold', linewidth=2.5, marker='o', markersize=3,
            label=f'Forecast ({days} days)')
    ax.axvline(x=60, color='gray', linestyle='--', linewidth=1.5)
    ax.fill_between(range(60, 60 + days),
                    [p * 0.97 for p in future_prices],
                    [p * 1.03 for p in future_prices],
                    alpha=0.2, color='gold', label='3% confidence band')
    ax.set_facecolor('#111')
    fig.patch.set_facecolor('#111')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333')
    ax.legend(facecolor='#222', labelcolor='white')
    ax.set_title(f'GS {days}-Day Price Forecast', color='white')
    ax.set_ylabel('Price ($)', color='white')
    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast Start",  f"${future_prices[0]:.2f}")
    col2.metric("Forecast End",    f"${future_prices[-1]:.2f}")
    change = ((future_prices[-1] - future_prices[0]) / future_prices[0]) * 100
    col3.metric("Expected Change", f"{change:+.2f}%")

    st.subheader("Daily Forecast Table")
    forecast_df = pd.DataFrame({
        'Day':   range(1, days + 1),
        'Predicted Price':    [f"${p:.2f}" for p in future_prices],
        'Change from Day 1':  [f"{((p-future_prices[0])/future_prices[0])*100:+.2f}%"
                                for p in future_prices]
    })
    st.dataframe(forecast_df, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 4 - BACKTEST
# ══════════════════════════════════════════════════════════
elif page == "Backtest":
    st.title("Strategy Backtest")
    st.markdown("---")

    threshold = st.slider("Confidence Threshold", 0.50, 0.70, 0.60, 0.01)

    test   = df[df['Date'] >= '2020-01-01'].copy()
    X_test = test[feature_cols]

    proba = rf_clf.predict_proba(X_test[feature_cols])[:, 1]
    test['Signal']           = (proba > threshold).astype(int)
    test['Strategy_Return']  = test['Signal'].shift(1) * test['Return_1d']
    test['BuyHold_Return']   = test['Return_1d']
    test['Strategy_Cum']     = (1 + test['Strategy_Return']).cumprod()
    test['BuyHold_Cum']      = (1 + test['BuyHold_Return']).cumprod()

    returns   = test['Strategy_Return'].dropna()
    sharpe    = returns.mean() / returns.std() * np.sqrt(252)
    max_dd    = (test['Strategy_Cum'] / test['Strategy_Cum'].cummax() - 1).min()
    win_rate  = (returns[returns != 0] > 0).mean() * 100
    trades    = (test['Signal'] == 1).sum()
    strat_ret = (test['Strategy_Cum'].iloc[-1] - 1) * 100
    bh_ret    = (test['BuyHold_Cum'].iloc[-1] - 1) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Strategy Return", f"{strat_ret:.1f}%")
    col2.metric("Buy & Hold",      f"{bh_ret:.1f}%")
    col3.metric("Sharpe Ratio",    f"{sharpe:.2f}")
    col4.metric("Max Drawdown",    f"{max_dd*100:.1f}%")
    col5.metric("Win Rate",        f"{win_rate:.1f}%")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test['Date'], test['Strategy_Cum'],
            label='ML Strategy', color='gold',  linewidth=2)
    ax.plot(test['Date'], test['BuyHold_Cum'],
            label='Buy & Hold',  color='white', linewidth=2)
    ax.set_facecolor('#111')
    fig.patch.set_facecolor('#111')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333')
    ax.legend(facecolor='#222', labelcolor='white')
    ax.set_title(f'ML Strategy vs Buy & Hold (Threshold={threshold})', color='white')
    ax.set_ylabel('Cumulative Return', color='white')
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════
# PAGE 5 - TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════
elif page == "Technical Indicators":
    st.title("Technical Indicators")
    st.markdown("---")

    recent = df.tail(200)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.patch.set_facecolor('#111')

    axes[0].plot(recent['Date'], recent['Close'],   color='white', linewidth=1.5, label='Close')
    axes[0].plot(recent['Date'], recent['MA_20'],   color='gold',  linewidth=1,   label='MA20')
    axes[0].plot(recent['Date'], recent['MA_50'],   color='cyan',  linewidth=1,   label='MA50')
    axes[0].plot(recent['Date'], recent['BB_Upper'],color='gray',  linewidth=0.8, linestyle='--', label='BB Upper')
    axes[0].plot(recent['Date'], recent['BB_Lower'],color='gray',  linewidth=0.8, linestyle='--', label='BB Lower')
    axes[0].set_facecolor('#111')
    axes[0].tick_params(colors='white')
    axes[0].spines[:].set_color('#333')
    axes[0].legend(facecolor='#222', labelcolor='white', fontsize=8)
    axes[0].set_title('Price + Bollinger Bands', color='white')

    axes[1].plot(recent['Date'], recent['RSI'], color='gold', linewidth=1.5)
    axes[1].axhline(70, color='red',   linestyle='--', linewidth=1)
    axes[1].axhline(30, color='green', linestyle='--', linewidth=1)
    axes[1].set_facecolor('#111')
    axes[1].tick_params(colors='white')
    axes[1].spines[:].set_color('#333')
    axes[1].set_title('RSI (14)', color='white')
    axes[1].set_ylim(0, 100)

    axes[2].plot(recent['Date'], recent['MACD'],        color='gold',  linewidth=1.5, label='MACD')
    axes[2].plot(recent['Date'], recent['MACD_Signal'], color='white', linewidth=1,   label='Signal')
    axes[2].set_facecolor('#111')
    axes[2].tick_params(colors='white')
    axes[2].spines[:].set_color('#333')
    axes[2].legend(facecolor='#222', labelcolor='white')
    axes[2].set_title('MACD', color='white')

    plt.tight_layout()
    st.pyplot(fig)