import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Webアプリの基本設定 ---
st.set_page_config(page_title="Quant Engine v2.6", layout="wide")
st.title("モンテカルロシミュレーション")

# --- 予測変換用の主要銘柄リスト ---
TICKER_LIST = [
    "三菱UFJフィナンシャル・グループ [8306.T]", "SUBARU [7270.T]", "NEC [6701.T]",
    "トヨタ自動車 [7203.T]", "ソニーグループ [6758.T]", "ソフトバンクグループ [9984.T]",
    "任天堂 [7974.T]", "ダイキン工業 [6367.T]", "日本電信電話 (NTT) [9432.T]",
    "日経平均株価 [^N225]", "S&P 500 [^GSPC]", "Apple [AAPL]", "NVIDIA [NVDA]"
]

# --- サイドバー設定 ---
st.sidebar.header("⚙️ システムパラメータ")

# 銘柄検索バー（入力すると候補が出る形式）
search_input = st.sidebar.selectbox("銘柄検索・コード入力:", ["直接入力する"] + TICKER_LIST)

if search_input == "直接入力する":
    ticker_input = st.sidebar.text_input("コードを直接入力 (例: 6367.T):", value="8306.T")
else:
    # 候補からコード部分 [xxxx] だけを抽出
    ticker_input = search_input.split("[")[-1].replace("]", "")

ticker = ticker_input.strip().upper()

days = st.sidebar.slider("予測日数 (Days):", 10, 252, 60)
simulations = st.sidebar.slider("試行回数:", 100, 3000, 1000)
ma_period = st.sidebar.slider("MA期間:", 5, 75, 25)
bb_sigma = st.sidebar.slider("ボリンジャーバンド(σ):", 1.0, 3.0, 2.0, 0.5)

st.sidebar.header("🔄 バックテスト")
backtest = st.sidebar.checkbox("有効化", value=False)
backtest_offset = st.sidebar.slider("開始地点 (日前):", 10, 150, 30) if backtest else 0

@st.cache_data(ttl=3600) 
def load_data(t):
    return yf.download(t, period="3y", progress=False)

@st.cache_data(ttl=3600)
def get_name(t):
    try:
        info = yf.Ticker(t).info
        return info.get('longName') or info.get('shortName') or t
    except: return t

if ticker:
    with st.spinner("Analyzing..."):
        full_data = load_data(ticker)
        if full_data.empty:
            st.error(f"Error: {ticker}")
        else:
            display_name = get_name(ticker)
            data = full_data.iloc[:-backtest_offset] if (backtest and backtest_offset > 0) else full_data
            actual_future = full_data.iloc[-backtest_offset:] if (backtest and backtest_offset > 0) else None
            
            prices = data['Close'].dropna().values.flatten()
            returns = np.log(prices[1:] / prices[:-1])
            mu, sigma, S0 = np.mean(returns), np.std(returns), prices[-1]

            # シミュレーション
            Z = np.random.normal(0, 1, (days, simulations))
            daily_returns = np.exp((mu - 0.5 * sigma**2) + sigma * Z)
            price_paths = np.zeros((days + 1, simulations))
            price_paths[0] = S0
            for t in range(1, days + 1):
                price_paths[t] = price_paths[t-1] * daily_returns[t-1]

            # MA/BB計算
            ma_paths = np.zeros((days + 1, simulations))
            bb_upper = np.zeros((days + 1, simulations))
            bb_lower = np.zeros((days + 1, simulations))
            for t in range(days + 1):
                combined = np.append(prices[-(ma_period-t):], price_paths[1:t+1, 0]) if t < ma_period else price_paths[t-ma_period+1:t+1, 0]
                m, s = np.mean(combined), np.std(combined)
                ma_paths[t], bb_upper[t], bb_lower[t] = m, m + (bb_sigma * s), m - (bb_sigma * s)

            # --- UI表示 ---
            st.subheader(f"📊 {display_name} ({ticker})")
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Base Price", f"{S0:.1f}")
            col_m2.metric("Return (Annual)", f"{mu*252*100:.1f}%")
            col_m3.metric("Volatility", f"{sigma*np.sqrt(252)*100:.1f}%")
            
            # --- 2画面グラフ (復活版) ---
            plt.style.use("default")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})
            
            # 左: パス推移
            ax1.plot(price_paths, color='royalblue', alpha=0.015)
            ax1.plot(ma_paths[:, 0], color='darkorange', linewidth=2, label=f"{ma_period}-Day MA")
            ax1.fill_between(range(days+1), bb_lower[:, 0], bb_upper[:, 0], color='darkorange', alpha=0.15, label=f"BB ±{bb_sigma}σ")
            
            accuracy_val = 0
            if backtest and actual_future is not None:
                actual_vals = actual_future['Close'].values[:days+1]
                ax1.plot(range(len(actual_vals)), actual_vals, color='red', linewidth=3, label="Actual Price")
                # 正しい的中率計算
                in_range = (actual_vals >= np.percentile(price_paths[:len(actual_vals)], 5, axis=1)) & \
                           (actual_vals <= np.percentile(price_paths[:len(actual_vals)], 95, axis=1))
                accuracy_val = np.mean(in_range) * 100
                col_m4.metric("Accuracy", f"{accuracy_val:.1f}%")

            ax1.set_title("Price Path Forecast (English)", fontsize=14)
            ax1.set_xlabel("Days"); ax1.set_ylabel("Price")
            ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)

            # 右: ヒストグラム (最終価格分布)
            final_prices = price_paths[-1]
            ax2.hist(final_prices, bins=40, color='royalblue', alpha=0.6, edgecolor='white')
            ax2.axvline(S0, color='red', linestyle='--', label="Current")
            ax2.set_title("Price Distribution")
            ax2.set_xlabel("Final Price"); ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)

            # --- 自動分析レポート ---
            st.markdown("---")
            st.subheader("🤖 クオンツ・モデルによる自動分析")
            prob_bullish = (np.sum(final_prices > ma_paths[-1]) / simulations) * 100
            var_95 = S0 - np.percentile(final_prices, 5)

            report = f"""
            現在、**{display_name}** は {S0:.1f} 円を基準に推移しています。
            
            * **トレンド予測**: {days}日後の強気確率は **{prob_bullish:.1f}%** です。
            * **リスク管理 (VaR)**: 95%信頼区間での最大損失額は **{var_95:.1f} 円** と予測されます。
            """
            if backtest:
                report += f"\n* **バックテスト結果**: 過去のデータに対するモデルの適合率は **{accuracy_val:.1f}%** でした。"
            
            st.info(report)
