import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Webアプリの基本設定 ---
st.set_page_config(page_title="Quant Engine", layout="wide")
st.title("モンテカルロシミュレーション")

# --- サイドバー設定 ---
st.sidebar.header("⚙️ システムパラメータ")

# シンプルで使いやすいテキスト入力バー
ticker_input = st.sidebar.text_input("🔍 銘柄コードを入力 (例: 8306.T, 6367.T, AAPL):", value="8306.T")
ticker = ticker_input.strip().upper() # 小文字を大文字に自動変換

days = st.sidebar.slider("予測日数 (Days):", 10, 252, 60)
simulations = st.sidebar.slider("試行回数:", 100, 3000, 1000)
ma_period = st.sidebar.slider("MA期間:", 5, 75, 25)
bb_sigma = st.sidebar.slider("ボリンジャーバンド(σ):", 1.0, 3.0, 2.0, 0.5)

st.sidebar.header("🔄 バックテスト設定")
backtest = st.sidebar.checkbox("有効化", value=False)
backtest_offset = st.sidebar.slider("開始地点 (日前):", 10, 150, 30) if backtest else 0

# --- データ取得関数 ---
@st.cache_data(ttl=3600) 
def load_data(t):
    return yf.download(t, period="3y", progress=False)

@st.cache_data(ttl=3600)
def get_name(t):
    try:
        info = yf.Ticker(t).info
        return info.get('longName') or info.get('shortName') or t
    except: return t

# --- メイン処理 ---
if ticker:
    with st.spinner("データを取得・分析中..."):
        full_data = load_data(ticker)
        
        if full_data.empty:
            st.error(f"「{ticker}」のデータを取得できませんでした。コードが正しいか確認してください。")
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

            # 移動平均線とボリンジャーバンドの計算
            ma_paths = np.zeros((days + 1, simulations))
            bb_upper = np.zeros((days + 1, simulations))
            bb_lower = np.zeros((days + 1, simulations))
            for t in range(days + 1):
                combined = np.append(prices[-(ma_period-t):], price_paths[1:t+1, 0]) if t < ma_period else price_paths[t-ma_period+1:t+1, 0]
                m, s = np.mean(combined), np.std(combined)
                ma_paths[t], bb_upper[t], bb_lower[t] = m, m + (bb_sigma * s), m - (bb_sigma * s)

            # --- 結果表示とUI構築 ---
            st.subheader(f"📊 {display_name} ({ticker})")
            
            # メトリクスは文字化けしないので日本語に戻しました
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("基準価格", f"{S0:.1f} 円")
            col_m2.metric("年率期待リターン", f"{mu*252*100:.1f}%")
            col_m3.metric("年率ボラティリティ", f"{sigma*np.sqrt(252)*100:.1f}%")
            
            # --- グラフの描画 (2画面 & 英語ラベル) ---
            plt.style.use("default")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})
            
            # 左: パス推移
            ax1.plot(price_paths, color='royalblue', alpha=0.015)
            ax1.plot(ma_paths[:, 0], color='darkorange', linewidth=2, label=f"{ma_period}-Day MA")
            ax1.fill_between(range(days+1), bb_lower[:, 0], bb_upper[:, 0], color='darkorange', alpha=0.15, label=f"BB ±{bb_sigma}σ")
            
            accuracy_val = 0
            if backtest and actual_future is not None:
                # 実際の株価が予測期間より短い場合のエラーを防止
                actual_vals = actual_future['Close'].values[:min(days+1, len(actual_future))]
                ax1.plot(range(len(actual_vals)), actual_vals, color='red', linewidth=3, label="Actual Price")
                
                # 的中率の計算
                in_range = (actual_vals >= np.percentile(price_paths[:len(actual_vals)], 5, axis=1)) & \
                           (actual_vals <= np.percentile(price_paths[:len(actual_vals)], 95, axis=1))
                accuracy_val = np.mean(in_range) * 100
                col_m4.metric("モデル的中率", f"{accuracy_val:.1f}%")

            ax1.set_title("Price Path Forecast & Bollinger Bands", fontsize=14)
            ax1.set_xlabel("Days"); ax1.set_ylabel("Price (JPY)")
            ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)

            # 右: ヒストグラム
            final_prices = price_paths[-1]
            ax2.hist(final_prices, bins=40, color='royalblue', alpha=0.6, edgecolor='white')
            ax2.axvline(S0, color='red', linestyle='--', label="Current")
            ax2.set_title("Final Price Distribution")
            ax2.set_xlabel("Price"); ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)

            # --- AIアナリスト詳細レポート ---
            st.markdown("---")
            st.subheader("🤖 クオンツ・モデルによる自動分析")
            
            prob_bullish = (np.sum(final_prices > ma_paths[-1]) / simulations) * 100
            percentile_5 = np.percentile(final_prices, 5)
            var_95 = S0 - percentile_5

            if prob_bullish >= 60:
                trend_eval = "統計的に**上昇トレンドが継続しやすい状態**です。順張りの戦略が機能しやすい局面と言えます。"
                status_icon = "🟢"
            elif prob_bullish <= 40:
                trend_eval = "下落リスクが高く、**トレンド転換（デッドクロス）に警戒が必要**です。新規の買いは慎重に行うべき局面です。"
                status_icon = "🔴"
            else:
                trend_eval = "強弱が拮抗しており、**方向感が定まりにくい（もみ合い）状態**です。ボラティリティによるノイズに注意してください。"
                status_icon = "🟡"

            # 【修正】バックテスト時のテキストインデント（文字の開始位置）の崩れを完全に修正しました
            backtest_str = ""
            if backtest:
                backtest_str = f"\n            * **バックテスト検証**: 過去データに基づく当モデルの信頼区間適合率は **{accuracy_val:.1f}%** でした。"

            report = f"""
            {status_icon} 現在、**{display_name}** は {S0:.1f} 円を基準に推移しています。
            年率ボラティリティ {sigma*np.sqrt(252)*100:.1f}% の環境下において、シミュレーションを実行した結果、以下のインサイトが得られました。

            * **トレンド予測**: {days}日後に株価が{ma_period}日移動平均線を上回っている確率は **{prob_bullish:.1f}%** です。{trend_eval}
            * **リスク管理 (VaR)**: 95%の信頼区間において、最悪のシナリオ（下位5%）を想定した場合の最大想定損失額は1株あたり **{var_95:.1f} 円**（予想到達価格: {percentile_5:.1f} 円）です。{backtest_str}
            
            > **💡 運用アドバイス**: 上記の最大想定損失額（VaR）を、現在の建玉のロスカットライン設定や、信用維持率のストレステストの目安としてご活用ください。
            """
            
            st.info(report)
