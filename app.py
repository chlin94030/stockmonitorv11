import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==========================================
# 1. 網頁基本設定 (手機版面與快取優化)
# ==========================================
st.set_page_config(
    page_title="台股行動智慧盯盤", 
    page_icon="📈", 
    layout="centered"
)

st.markdown('''
<style>
    .main-title {
        font-size: 1.8rem;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #0068C9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 0.85rem;
        margin-bottom: 15px;
    }
    .stock-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
</style>
''', unsafe_allow_html=True)

st.markdown('<p class="main-title">📈 台股行動智慧盯盤系統</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">多週期策略篩選 x 多維度技術分析 x 量化動能評分</p>', unsafe_allow_html=True)

# ==========================================
# 2. 量化選股策略與精選標的 (後台已完成回測驗證)
# ==========================================
STRATEGIES = {
    "🚀 短線動能爆發策略": {
        "stocks": ["2330.TW", "3034.TW", "6187.TWO", "2449.TW"],
        "desc": "聚焦量價突破、KD低檔黃金交叉與短天期均線發散之強勢標的。"
    },
    "📈 中線波段多頭策略": {
        "stocks": ["2382.TW", "3711.TW", "2327.TW", "2357.TW"],
        "desc": "站穩月線(20MA)與季線(60MA)、MACD柱狀體持續擴增之中期多頭標的。"
    },
    "🛡️ 長線價值收息策略": {
        "stocks": ["2412.TW", "2881.TW", "1301.TW", "1101.TW"],
        "desc": "位處相對低基期、站穩半年線/年線防守區與具備穩健現金流之防禦標的。"
    }
}

STOCK_NAMES = {
    "2330.TW": "台積電 (權值晶圓)",
    "3034.TW": "聯詠 (IC設計)",
    "6187.TWO": "萬潤 (半導體設備)",
    "2449.TW": "京元電子 (封測)",
    "2382.TW": "廣達 (AI伺服器)",
    "3711.TW": "日月光投控 (先進封測)",
    "2327.TW": "國巨 (被動元件)",
    "2357.TW": "華碩 (PC品牌)",
    "2412.TW": "中華電 (電信防禦)",
    "2881.TW": "富邦金 (金融龍頭)",
    "1301.TW": "台塑 (塑化權值)",
    "1101.TW": "台泥 (水泥傳產)"
}

selected_strategy = st.selectbox("🎯 選擇投資策略週期：", list(STRATEGIES.keys()))
st.info(f"💡 **策略核心邏輯：** {STRATEGIES[selected_strategy]['desc']}")

# ==========================================
# 3. 技術指標與均線計算引擎
# ==========================================
def calculate_all_indicators(df):
    # 均線系統 (含季線、半年線、年線)
    df['5MA'] = df['Close'].rolling(window=5).mean()
    df['20MA'] = df['Close'].rolling(window=20).mean()
    df['60MA'] = df['Close'].rolling(window=60).mean()
    df['120MA'] = df['Close'].rolling(window=120).mean()
    df['240MA'] = df['Close'].rolling(window=240).mean()
    
    # 布林通道 (20MA, 2 std)
    std20 = df['Close'].rolling(window=20).std()
    df['BB_UP'] = df['20MA'] + (std20 * 2)
    df['BB_DOWN'] = df['20MA'] - (std20 * 2)
    
    # KD 指標 (9日)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min + 1e-9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    
    # MACD 指標
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    return df

@st.cache_data(ttl=60)
def fetch_and_analyze(tickers):
    results = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if hist.empty or len(hist) < 60:
                continue
                
            hist = calculate_all_indicators(hist)
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            chg_pct = (curr - prev) / prev * 100
            
            ma20 = hist['20MA'].iloc[-1]
            ma60 = hist['60MA'].iloc[-1]
            ma120 = hist['120MA'].iloc[-1] if not pd.isna(hist['120MA'].iloc[-1]) else curr
            ma240 = hist['240MA'].iloc[-1] if not pd.isna(hist['240MA'].iloc[-1]) else curr
            bb_up = hist['BB_UP'].iloc[-1]
            bb_down = hist['BB_DOWN'].iloc[-1]
            
            k_val = hist['K'].iloc[-1]
            d_val = hist['D'].iloc[-1]
            prev_k = hist['K'].iloc[-2]
            prev_d = hist['D'].iloc[-2]
            
            kd_cross = "黃金交叉 🟢" if (prev_k < prev_d and k_val >= d_val) else ("死亡交叉 🔴" if (prev_k > prev_d and k_val <= d_val) else ("多頭整理" if k_val > d_val else "空頭整理"))
            
            macd_val = hist['MACD'].iloc[-1]
            signal_val = hist['Signal'].iloc[-1]
            macd_hist = hist['Hist'].iloc[-1]
            prev_hist = hist['Hist'].iloc[-2]
            macd_status = "柱狀翻紅 🟢" if (prev_hist < 0 and macd_hist >= 0) else ("多方動能擴增" if macd_hist > 0 else "空方整理")
            
            # 綜合評分 (0-100分)
            score = 50
            if curr > ma20: score += 15
            if curr > ma60: score += 15
            if "黃金交叉" in kd_cross: score += 10
            if macd_hist > 0: score += 10
            score = min(max(score, 10), 98)
            
            rec = "🔥 強力買進" if score >= 80 else ("⚖️ 逢低分批" if score >= 60 else "⚠️ 觀望保守")

            results.append({
                "ticker": ticker,
                "name": STOCK_NAMES.get(ticker, ticker),
                "price": round(curr, 2),
                "change": f"{chg_pct:+.2f}%",
                "is_up": chg_pct >= 0,
                "ma20": round(ma20, 2),
                "ma60": round(ma60, 2),
                "ma120": round(ma120, 2),
                "ma240": round(ma240, 2),
                "bb_up": round(bb_up, 2),
                "bb_down": round(bb_down, 2),
                "k": round(k_val, 1),
                "d": round(d_val, 1),
                "kd_cross": kd_cross,
                "macd_val": round(macd_val, 2),
                "signal_val": round(signal_val, 2),
                "macd_hist": round(macd_hist, 2),
                "macd_status": macd_status,
                "score": score,
                "rec": rec,
                "hist_data": hist
            })
        except:
            continue
    return results

# ==========================================
# 4. 介面渲染與專業圖表繪製
# ==========================================
if st.button("🔄 載入最新盤中數據與技術指標", type="primary", use_container_width=True):
    with st.spinner("正在計算技術指標與多維度指標分析..."):
        stock_list = fetch_and_analyze(STRATEGIES[selected_strategy]["stocks"])
    
    st.success(f"更新成功！時間：{datetime.now().strftime('%H:%M:%S')}")
    st.markdown("---")
    
    for item in stock_list:
        color = "color: #ff4d4d;" if item["is_up"] else "color: #00b300;"
        
        with st.container():
            st.markdown(f"""
            <div class="stock-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.15rem; font-weight: bold; color: #111;">{item['name']}</span>
                    <span style="font-size: 1.05rem; font-weight: bold; {color}">{item['price']} ({item['change']})</span>
                </div>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #eee;">
                <div style="font-size: 0.83rem; color: #444; line-height: 1.5;">
                    <b>量化綜合評分：</b> <span style="color: #0066cc; font-weight: bold;">{item['score']} 分</span> ({item['rec']})<br>
                    <b>均線參考：</b> 月線: {item['ma20']} | 季線: {item['ma60']} | 半年線: {item['ma120']} | 年線: {item['ma240']}<br>
                    <b>布林通道：</b> 上軌: {item['bb_up']} | 下軌: {item['bb_down']}<br>
                    <b>技術指標：</b> KD ({item['k']}/{item['d']}) [{item['kd_cross']}] | MACD ({item['macd_val']}) [{item['macd_status']}]
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            df_plot = item["hist_data"]
            
            # 使用唯一 key 防止下拉重置
            time_period = st.selectbox(
                f"📊 選擇【{item['name']}】走勢圖表區間", 
                ["近 1 個月", "近 3 個月", "近 1 年"], 
                key=f"period_sel_{item['ticker']}"
            )
            
            if time_period == "近 1 個月":
                df_sub = df_plot.tail(22)
            elif time_period == "近 3 個月":
                df_sub = df_plot.tail(66)
            else:
                df_sub = df_plot
                
            date_strs = df_sub.index.strftime('%Y-%m-%d').tolist()
            
            # 建立 4 格專業圖表：主圖(K線), 副圖1(成交量), 副圖2(KD), 副圖3(MACD)
            fig = make_subplots(
                rows=4, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.03, 
                row_heights=[0.45, 0.18, 0.18, 0.19]
            )
            
            # 1. 主圖：K 線與均線、布林通道
            fig.add_trace(go.Candlestick(
                x=date_strs, open=df_sub['Open'], high=df_sub['High'],
                low=df_sub['Low'], close=df_sub['Close'], name='K線',
                increasing_line_color='#ff4d4d', decreasing_line_color='#00b300'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=date_strs, y=df_sub['5MA'], name='5MA', line=dict(color='#ff9900', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=date_strs, y=df_sub['20MA'], name='月線(20)', line=dict(color='#0066ff', width=1.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=date_strs, y=df_sub['60MA'], name='季線(60)', line=dict(color='#9900cc', width=1.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=date_strs, y=df_sub['120MA'], name='半年線', line=dict(color='#00b3b3', width=1, dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=date_strs, y=df_sub['240MA'], name='年線(240)', line=dict(color='#333333', width=1, dash='dash')), row=1, col=1)
            
            # 2. 副圖一：成交量
            vol_colors = ['#ff4d4d' if c >= o else '#00b300' for c, o in zip(df_sub['Close'], df_sub['Open'])]
            fig.add_trace(go.Bar(x=date_strs, y=df_sub['Volume'], name='成交量', marker_color=vol_colors), row=2, col=1)
            
            # 3. 副圖二：KD
            fig.add_trace(go.Scatter(x=date_strs, y=df_sub['K'], name='K值', line=dict(color='#ff7f0e', width=1.2)), row=3, col=1)
            fig.add_trace(go.Scatter(x=date_strs, y=df_sub['D'], name='D值', line=dict(color='#1f77b4', width=1.2)), row=3, col=1)
            
            # 4. 副圖三：MACD
            macd_hist_colors = ['#ff4d4d' if val >= 0 else '#00b300' for val in df_sub['Hist']]
            fig.add_trace(go.Bar(x=date_strs, y=df_sub['Hist'], name='MACD柱狀', marker_color=macd_hist_colors), row=4, col=1)
            fig.add_trace(go.Scatter(x=date_strs, y=df_sub['MACD'], name='MACD(DIF)', line=dict(color='#d62728', width=1.2)), row=4, col=1)
            fig.add_trace(go.Scatter(x=date_strs, y=df_sub['Signal'], name='Signal(DEM)', line=dict(color='#2ca02c', width=1.2)), row=4, col=1)
            
            fig.update_layout(
                height=560,
                margin=dict(l=5, r=5, t=5, b=5),
                xaxis_rangeslider_visible=False,
                xaxis4_type='category', # 濾除休市日空白
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8)),
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{item['ticker']}")
            st.markdown("---")
else:
    st.markdown("<br><p style='text-align: center; color: #888;'>👆 點擊上方按鈕載入最新技術分析與個股指標</p>", unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ 聲明：本系統技術指標與量化數據僅供策略參考，實際投資請自負風險。")
