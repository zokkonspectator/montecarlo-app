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

# シンプルな一つの入力枠（ここで入力されたコードを元に企業名を自動取得します）
ticker_input = st.sidebar.text_input("🔍 銘柄コードを入力 (例: 8306.T, 6367.T, AAPL):", value="8306.T")
ticker = ticker_input.strip().upper() # 小文字を大文字に自動変換

days = st.sidebar.slider("予測日数 (Days):", 10, 252, 60)
simulations = st.sidebar.slider("試行回数:", 100, 3000, 1000)
ma_period = st.sidebar.slider("MA期間:", 5, 75, 25)
bb_sigma = st.sidebar.slider("ボリンジャーバンド(σ):", 1.0, 3.0, 2.0, 0.5)

st.sidebar.header("🔄 バックテスト設定")
backtest = st.sidebar.checkbox("バックテストモードを有効化", value=False)
backtest_offset = st.sidebar.slider("何日前から予測を開始するか:", 10, 100, 30) if backtest else 0

# --- データと企業名の取得（キャッシュして高速化） ---
@st.cache_data(ttl=3600) 
def load_data(ticker_code):
    return yf.download(ticker_code, period="3y", progress=False)

@st.cache_data(ttl=3600)
def get_company_name(ticker_code):
    try:
        # yfinanceから企業の正式名称を取得
        info = yf.Ticker(ticker_code).info
        # longName（正式名称）かshortNameを取得、なければコードをそのまま返す
        return info.get('longName', info.get('shortName', ticker_code))
    except:
        return ticker_code

# --- メイン処理 ---
if ticker:
    with st.spinner("データを取得・分析中..."):
        full_data = load_data(ticker)
        
        if full_data.empty:
            st.error(f"「{ticker}」のデータを取得できませんでした。コードが正しいか確認してください。")
        else:
            # 企業名の取得
            display_name = get_company_name(ticker)

            data = full_data.iloc[:-backtest_offset] if backtest else full_data
            actual_future = full_data.iloc[-backtest_offset:] if backtest else None
            
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
                if t == 0:
                    combined = prices[-ma_period:]
                elif t < ma_period:
                    combined = np.append(prices[-(ma_period-t):], price_paths[1:t+1, 0])
                else:
                    combined = price_paths[t-ma_period+1:t+1, 0]
                
                ma_val = np.mean(combined)
                std_val = np.std(combined)
                ma_paths[t] = ma_val
                bb_upper[t] = ma_val + (bb_sigma * std_val)
                bb_lower[t] = ma_val - (bb_sigma * std_val)

            # --- 結果表示とUI構築 ---
            # 取得した正式な企業名をヘッダーに表示
            st.subheader(f"📊 {display_name} [{ticker}] 分析ダッシュボード " + ("(バックテスト実行中)" if backtest else ""))
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("基準価格", f"{S0:.1f} 円")
            col2.metric("年率期待リターン", f"{mu*252*100:.1f}%")
            col3.metric("年率ボラティリティ", f"{sigma*np.sqrt(252)*100:.1f}%")
            
            # --- カラーテーマと視認性の改善 ---
            plt.style.use("default")
            bg_color = '#ffffff'
            text_color = '#333333'     # 真っ黒より少し柔らかい見やすい黒
            line_color = 'royalblue'   # パスは少し落ち着いた青
            ma_color = '#FF4B4B'       # 移動平均線はビビッドな赤系
            bb_color = '#FF4B4B'       # ボリンジャーバンドも同色系
            act_color = '#00C04B'      # バックテストの実際の株価は緑にして差別化

            # メインチャート描画
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=text_color)
            ax.spines['bottom'].set_color(text_color)
            ax.spines['left'].set_color(text_color)
            ax.spines['top'].set_color('#cccccc')
            ax.spines['right'].set_color('#cccccc')

            ax.plot(price_paths, color=line_color, alpha=0.015)
            
            # ボリンジャーバンドの描画（色を変更し、alphaを0.1 -> 0.2に上げて濃くしました）
            ax.plot(ma_paths[:, 0], color=ma_color, linewidth=2.5, label=f"{ma_period}-Day MA")
            ax.fill_between(range(days+1), bb_lower[:, 0], bb_upper[:, 0], color=bb_color, alpha=0.2, label=f"Bollinger Bands ±{bb_sigma}σ")
            
            accuracy_text = ""
            if backtest:
                actual_prices = actual_future['Close'].values[:days+1]
                ax.plot(range(len(actual_prices)), actual_prices, color=act_color, linewidth=3, label="Actual Price (Reality)")
                accuracy = np.sum((actual_prices >= np.percentile(price_paths[:len(actual_prices)], 5, axis=1)) & 
                                  (actual_prices <= np.percentile(price_paths[:len(actual_prices)], 95, axis=1))) / len(actual_prices)
                accuracy_text = f"\n* **バックテスト検証**: 過去のデータに基づくモデルの信頼区間（90%）への適合率は **{accuracy*100:.1f}%** を記録しました。"
                col4.metric("モデル的中率", f"{accuracy*100:.1f}%")
            
            # 英語ラベル
            ax.set_title("Monte Carlo & Bollinger Bands Forecast", color=text_color, fontsize=14, pad=15)
            ax.set_xlabel("Days", color=text_color)
            ax.set_ylabel("Price (JPY)", color=text_color)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            legend = ax.legend(facecolor=bg_color, edgecolor='#cccccc', loc='upper left')
            for text in legend.get_texts():
                text.set_color(text_color)
                
            st.pyplot(fig)

            # --- AIアナリスト詳細レポート ---
            st.markdown("---")
            st.subheader("🤖 クオンツ・モデルによる自動分析インサイト")

            final_prices = price_paths[-1]
            prob_bullish = (np.sum(final_prices > ma_paths[-1]) / simulations) * 100
            percentile_5 = np.percentile(final_prices, 5)
            var_95 = S0 - percentile_5

            if prob_bullish >= 60:
                trend_eval = "統計的に**強い上昇トレンドが継続しやすい状態**です。順張りの戦略が機能しやすい局面と言えます。"
                status_icon = "🟢"
            elif prob_bullish <= 40:
                trend_eval = "下落リスクが高く、**トレンド転換（デッドクロス）に強い警戒が必要**です。新規の買いは慎重に行うべき局面です。"
                status_icon = "🔴"
            else:
                trend_eval = "強弱が拮抗しており、**トレンドが定まりにくい（もみ合い）状態**です。ボラティリティによるノイズに注意してください。"
                status_icon = "🟡"

            report = f"""
            {status_icon} 現在、**{display_name} ({ticker})** の株価は {S0:.1f} 円を基準としています。
            過去データから算出した年率ボラティリティ {sigma*np.sqrt(252)*100:.1f}% の環境下において、幾何ブラウン運動をシミュレーションした結果、以下のインサイトが得られました。

            * **トレンド予測**: {days}日後に株価が{ma_period}日移動平均線を上回っている確率は **{prob_bullish:.1f}%** です。{trend_eval}
            * **リスク管理 (VaR)**: 95%の信頼区間において、最悪のシナリオ（下位5%）を想定した場合の最大想定損失額は1株あたり **{var_95:.1f} 円**（到達予想価格: {percentile_5:.1f} 円）です。{accuracy_text}
            
            > **💡 運用アドバイス**: 上記の最大想定損失額（VaR）を、現在の建玉のロスカットライン設定や、信用維持率のストレステストの目安として活用してください。また、表示されているボリンジャーバンド（±{bb_sigma}σ）の帯域幅から、将来予想される値動きの限界点を視覚的に把握することができます。
            """
            
            st.info(report)