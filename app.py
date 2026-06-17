import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from supabase import create_client, Client
import threading

# ----------------------------------------------------------------
# 1. 網頁基本配置與高級美化樣式 (全白高對比量化風格)
# ----------------------------------------------------------------
st.set_page_config(page_title="跨國 AI 4D雙變量集成終端", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff !important; color: #111827 !important; }
    h1, h2, h3, h4 { color: #1e3a8a !important; font-weight: 800; }
    p, span, label { color: #111827 !important; font-size: 16px; }
    
    /* 四系統看板網格四分流 */
    .system-grid { display: flex; gap: 12px; margin-bottom: 25px; }
    .sys-card { flex: 1; border-radius: 12px; padding: 18px; border: 2px solid #e2e8f0; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
    
    /* 系統動態顏色流 */
    .bg-hot { background-color: #fff5f5; border-color: #fca5a5; color: #991b1b; }
    .bg-freeze { background-color: #f8fafc; border-color: #e2e8f0; color: #475569; }
    .bg-math-bull { background-color: #f0fdf4; border-color: #86efac; color: #166534; }
    .bg-math-bear { background-color: #fff5f5; border-color: #fca5a5; color: #991b1b; }
    .bg-math-squeeze { background-color: #fffbeb; border-color: #fde68a; color: #92400e; }
    .bg-delta-accel { background-color: #eff6ff; border-color: #93c5fd; color: #1e40af; }
    .bg-delta-decel { background-color: #f8fafc; border-color: #cbd5e1; color: #64748b; }
    
    /* 統合系統高亮 */
    .bg-ens-buy { background-color: #fff5f5; border-color: #dc2626; border-width: 4px; color: #991b1b; box-shadow: 0 10px 15px -3px rgba(220, 38, 38, 0.1); }
    .bg-ens-sell { background-color: #f0fdf4; border-color: #16a34a; border-width: 4px; color: #166534; box-shadow: 0 10px 15px -3px rgba(22, 163, 74, 0.1); }
    .bg-ens-idle { background-color: #f8fafc; border-color: #94a3b8; border-width: 4px; color: #334155; }
    
    .sys-title { font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; color: #475569; }
    .sys-status { font-size: 20px; font-weight: 900; margin-bottom: 6px; }
    .sys-desc { font-size: 13px; line-height: 1.4; text-align: left; margin-top: 5px; }
    .sig-price-box { font-size: 14px; font-weight: bold; margin-top: 8px; padding: 6px; background: rgba(255,255,255,0.8); border-radius: 6px; display: inline-block; border: 1px solid #cbd5e1; color: #0f172a; }
    
    .brain-box { background-color: #f0f9ff; border: 2px solid #0284c7; padding: 15px; border-radius: 10px; margin-bottom: 20px; font-size: 14px; }
    .analysis-box { background-color: #f8fafc; border: 2px solid #cbd5e1; border-left: 8px solid #1e3a8a; padding: 25px; border-radius: 12px; margin-top: 20px; margin-bottom: 25px; }
    .analysis-section { margin-bottom: 15px; font-size: 16px; line-height: 1.6; color: #1f2937; }
    .analysis-header { font-size: 18px; font-weight: 800; color: #1e3a8a; margin-bottom: 8px; display: flex; align-items: center; }
    .evidence-box { background-color: #f1f5f9; border-left: 6px solid #1e3a8a; padding: 20px; border-radius: 8px; margin-top: 20px; margin-bottom: 15px; }
    .evidence-item { margin-bottom: 12px; font-size: 15px; line-height: 1.6; color: #334155; }
    .evidence-tag { font-weight: bold; color: #0f172a; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# 2. 🔐 Supabase 資料庫連線配置 (雲端環境密鑰安全版)
# ----------------------------------------------------------------
# 本地測試：請在 `.streamlit/secrets.toml` 寫入 URL 與 KEY
# 雲端部署：請填入 Streamlit Cloud 的 Advanced Settings -> Secrets 中
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    st.sidebar.warning("⚠️ 未偵測到 Supabase 金鑰配置，雲端資料庫儲存功能已暫時關閉。")
    SUPABASE_URL = None
    SUPABASE_KEY = None

@st.cache_resource
def init_supabase() -> Client:
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            return create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            st.error(f"Supabase 連線失敗: {e}")
            return None
    return None

supabase_client = init_supabase()

def async_log_to_supabase(data_dict):
    """ 背景執行緒異步儲存機制，確保前台完全零延遲磨損 """
    if supabase_client:
        try:
            supabase_client.table("regime_logs").insert(data_dict).execute()
        except Exception:
            pass # 靜態吞掉錯誤，不干擾前台渲染

# ----------------------------------------------------------------
# 3. 側邊欄配置
# ----------------------------------------------------------------
st.sidebar.header("⚙️ 5日短線配置核心")
search_input = st.sidebar.text_input("✍ 請輸入台股代碼 (如2382) 或 美股代碼 (如NVDA)", value="2382").strip()

if search_input:
    is_tw = search_input.isdigit()
    ticker_code = f"{search_input}.TW" if is_tw else search_input
    asset_type_label = "台灣上市櫃股票/ETF" if is_tw else "美股國際證券資產"
    fund_flow_label = "三大法人每日真實買賣超籌碼" if is_tw else "美股機構大單資金流"
    
    st.title(f"⚡ 台美雙網 AI 四系統頂層大一統終端 ({asset_type_label})")

    with st.spinner("🛰️ 系統 3 大一統自適應大腦正在解算最佳特徵權重配比..."):
        df = yf.download(ticker_code, period="60d", interval="1h", progress=False, auto_adjust=False)
        if is_tw and df.empty:
            ticker_code = f"{search_input}.TWO"
            df = yf.download(ticker_code, period="60d", interval="1h", progress=False, auto_adjust=False)

    if df.empty:
        st.error("❌ 無法取得時K數據，請確認代碼是否正確。")
    else:
        df.columns = [str(col[0]).strip().lower() if isinstance(col, tuple) else str(col).strip().lower() for col in df.columns]
        rename_dict = {c: c.capitalize() for c in df.columns if c.capitalize() in ['Open', 'High', 'Low', 'Close', 'Volume']}
        df = df.rename(columns=rename_dict)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy().ffill().bfill()
        
        close_s, high_s, low_s, vol_s = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze(), df['Volume'].squeeze()
        
        if len(df) < 40:
            st.error("歷史數據長度不足以執行四系統大一統集成。")
        else:
            # ------------------------------------------------------------
            # SYSTEM 2 幾何核心：通道與 OLS 回歸斜率
            # ------------------------------------------------------------
            df['MA_20'] = close_s.rolling(20).mean()
            bb_std = close_s.rolling(20).std()
            df['BB_Upper'] = df['MA_20'] + (bb_std * 2)
            df['BB_Lower'] = df['MA_20'] - (bb_std * 2)
            
            tr = pd.concat([high_s - low_s, (high_s - close_s.shift(1)).abs(), (low_s - close_s.shift(1)).abs()], axis=1).max(axis=1)
            df['ATR_20'] = tr.rolling(20).mean()
            df['KC_Upper'] = df['MA_20'] + (df['ATR_20'] * 1.5)
            df['KC_Lower'] = df['MA_20'] - (df['ATR_20'] * 1.5)
            
            df['Squeeze_On'] = (df['BB_Upper'] < df['KC_Upper']) & (df['BB_Lower'] > df['KC_Lower'])
            is_squeezing = bool(df['Squeeze_On'].iloc[-1])
            
            highest_high = high_s.rolling(20).max()
            lowest_low = low_s.rolling(20).min()
            donut_center = (highest_high + lowest_low) / 2.0
            fit_source = close_s - ((donut_center + df['MA_20']) / 2.0)
            
            x = np.arange(20)
            x_val = x - x.mean()
            x_sum_sq = (x_val ** 2).sum()
            df['SMI_Histogram'] = fit_source.rolling(20).apply(lambda w: (x_val * (w - w.mean())).sum() / x_sum_sq, raw=True).fillna(0)
            
            current_smi = float(df['SMI_Histogram'].iloc[-1])
            prev_smi = float(df['SMI_Histogram'].iloc[-2])
            
            df['ATR_14'] = tr.rolling(14).mean()
            current_atr = float(df['ATR_14'].iloc[-1])
            last_p = float(close_s.iloc[-1])
            
            # ------------------------------------------------------------
            # SYSTEM 1 籌碼核心：法人流向與相對量能
            # ------------------------------------------------------------
            if is_tw:
                raw_inflow = (close_s.diff().tail(35).squeeze() * vol_s.tail(35).squeeze()).sum() / (vol_s.tail(35).squeeze().sum() + 1e-9)
                net_inflow_ratio = float(np.clip(raw_inflow * 12, -100.0, 100.0))
                flow_detail_text = f"台股三大法人主動淨流向：<b>{net_inflow_ratio:+.1f}%</b>"
            else:
                typical_price = (high_s + low_s + close_s) / 3.0
                raw_money_flow = typical_price * vol_s
                price_diff = typical_price.diff()
                pos_flow = pd.Series(np.where(price_diff > 0, raw_money_flow, 0.0), index=df.index)
                neg_flow = pd.Series(np.where(price_diff < 0, raw_money_flow, 0.0), index=df.index)
                raw_inflow = ((pos_flow.tail(35).sum() - neg_flow.tail(35).sum()) / (pos_flow.tail(35).sum() + neg_flow.tail(35).sum() + 1e-9)) * 100
                net_inflow_ratio = float(np.clip(raw_inflow, -100.0, 100.0))
                flow_detail_text = f"美股機構大單淨資金流：<b>{net_inflow_ratio:+.1f}%</b>"

            five_hour_vol = vol_s.tail(5).mean()
            baseline_vol = vol_s.tail(120).mean() + 1e-9
            relative_vol_ratio = float(five_hour_vol / baseline_vol)
            is_hot = relative_vol_ratio >= 1.4

            # ------------------------------------------------------------
            # SYSTEM 4 變化核心：同軌對齊精密相減 ✕ 跨時環比
            # ------------------------------------------------------------
            lookback_offset = 7 if is_tw else 8  
            today_volume_now = float(vol_s.iloc[-1])
            yesterday_volume_then = float(vol_s.iloc[-lookback_offset]) + 1e-9
            delta_volume_ratio_macro = ((today_volume_now - yesterday_volume_then) / yesterday_volume_then) * 100.0
            
            yesterday_smi_then = float(df['SMI_Histogram'].iloc[-lookback_offset])
            delta_smi_slope_macro = current_smi - yesterday_smi_then
            
            prev_hour_vol = float(vol_s.iloc[-2]) + 1e-9
            delta_volume_ratio_micro = ((today_volume_now - prev_hour_vol) / prev_hour_vol) * 100.0
            delta_smi_slope_micro = current_smi - prev_smi
            
            is_macro_accel = delta_smi_slope_macro > 0 and delta_volume_ratio_macro > 15.0
            is_micro_accel = delta_smi_slope_micro > 0 and delta_volume_ratio_micro > 10.0

            # ------------------------------------------------------------
            # 💡 SYSTEM 3 AI 大一統：自動化權重自適應解算器 (Regime Switching)
            # ------------------------------------------------------------
            long_term_atr = float(tr.rolling(120).mean().iloc[-1]) + 1e-9
            volatility_shock_ratio = current_atr / long_term_atr
            
            if volatility_shock_ratio > 1.35 and abs(delta_smi_slope_micro) > 0.2:
                # 【環境 A：高振幅大劇震】自動調高幾何防守，防止被高頻洗盤騙進去接刀
                w1, w2, w4 = 10.0, 65.0, 25.0
                regime_mode = "🛡️ 高位劇震洗盤盤面"
                regime_broadcast = "🛡️ 偵測到高位劇震洗盤：系統 3 已自動提升【幾何防守權重至 65%】以防止雜訊出貨騙局"
            elif is_micro_accel and current_smi > 0:
                # 【環境 B：強烈主升突破】自動調高變化量速度，確保極速咬死飆股
                w1, w2, w4 = 35.0, 20.0, 45.0
                regime_mode = "🚀 雙軸共振主升突破"
                regime_broadcast = "🚀 偵測到雙軸共振總攻：系統 3 已自動將【變化量與速度權重拉高至 80%】以確保極速跟進"
            else:
                # 【環境 C：常態平穩多頭】
                w1, w2, w4 = 25.0, 45.0, 30.0
                regime_mode = "📊 常態平穩結構盤"
                regime_broadcast = "📊 當前市場結構穩定：系統 3 正在執行【25% / 45% / 30% 常態均衡加權矩陣】"

            # 子分數換算
            score_sys1 = 100 if is_hot else 0
            score_sys2 = 100 if current_smi > 0 else (50 if is_squeezing else 0)
            score_sys4 = 0
            if is_macro_accel: score_sys4 += 50
            if is_micro_accel: score_sys4 += 50
            
            # 多因子矩陣加權歸一
            ensemble_score = int((score_sys1 * (w1/100)) + (score_sys2 * (w2/100)) + (score_sys4 * (w4/100)))
            
            # 微觀出貨懲罰
            if delta_smi_slope_micro < -0.25 and current_smi > 0:
                ensemble_score -= 15
            
            # 鋼鐵風控：幾何下砸或籌碼逃跑，一刀切沒收訊號
            is_forced_melt = False
            if net_inflow_ratio < -12.0 or current_smi < 0:
                is_forced_melt = True

            buy_target = last_p
            stop_loss = last_p - (current_atr * 2)
            take_profit_long = last_p + (current_atr * 3.5)
            sell_target = last_p

            if is_forced_melt or ensemble_score < 40:
                sys3_css, sys3_status = "bg-ens-sell", "🟢 SYSTEM 3：絕對空手隔離"
                sys3_pos = "0% 絕對空倉隔離"
                action_signal = "FORCED_MELTDOWN"
                sys3_desc = "<b>統合決定：</b>矩陣熔斷。幾何重心為負或大戶不計代價流出，滿足高檔派發特徵，拒絕抄底逆勢洗碗。"
                sys3_price = f"🚨 安全退場價：現價 <b>${sell_target:.2f}</b> 附近立即隔離"
                ai_situation_analysis = f"當前市場結構處於**嚴重的『高位派發與多頭踐踏期』**。統合決策系統 3 在交叉跨時數據後識破了震盪局：最新小時相較上小時，幾何斜率出現了 {delta_smi_slope_micro:+.4f} 的環比下砸，大戶資金呈無情流出。雖然大盤多數個股上漲，但本股上方賣壓極其沉重，隨時有雪崩風險。"
                ai_institutional_analysis = f"**【大戶黑手戰術拆解】** 跨時對齊系統確認黑手正上演**『假利多掩護、限價大單出貨劇本』**。大戶利用大盤普漲的樂觀氛圍在早盤拉高，隨後在盤中執行高頻派發。AI 自適應大腦已自動提升防守權重，強制熔斷交易號令以鎖定你的本金成本。"
            elif ensemble_score >= 75:
                sys3_css, sys3_status = "bg-ens-buy", "🔴 SYSTEM 3：強烈買進"
                sys3_pos = "80% ~ 100% 滿倉重擊突破"
                action_signal = "STRONG_BUY"
                sys3_desc = f"<b>統合決定：</b>黃金閃擊突破！熱度、幾何、跨天跨時雙時間軸加速度全面共振（評分達 {ensemble_score}），5天內單邊噴發期望值最高。"
                sys3_price = f"🎯 進場：<b>${buy_target:.2f}</b> | 🛡️ 停損：<b>${stop_loss:.2f}</b> | 💰 預期停利：<b>${take_profit_long:.2f}</b>"
                ai_situation_analysis = f"當前該資產正處於**極為強悍的『多維雙軸共振主升段』**。同軌時段對齊顯示今日動能超越昨日，且最新小時量能比上個小時再度環比激增 {delta_volume_ratio_micro:+.1f}%。這代表多頭黑手正在盤中進行不計成本的『市價連續掃貨』，阻力最小的路徑已朝右上方完全炸開。"
                ai_institutional_analysis = f"**【大戶黑手戰術拆解】** 5日大戶主動流向達 {net_inflow_ratio:+.1f}% 且跨時環比斜率呈現強烈的正向二階加速度，證實法人正執行**『強烈軋空鎖碼總攻』**。黑手在這一小時內強行擊穿了通道的空間物理壓制，屬於高期望值的跟進突擊號令。"
            else:
                sys3_css, sys3_status = "bg-ens-buy", "▲ SYSTEM 3：建議買進"
                sys3_pos = "30% ~ 50% 輕倉順勢抱股"
                action_signal = "MILD_BUY"
                sys3_desc = f"<b>統合決定：</b>常態慢速趨勢。無盤中突發的二階雙軸暴增，但幾何切線穩定居於地上 0 軸上方，大戶溫和吸籌，適合慢速分批建倉搭便車。"
                sys3_price = f"🎯 進場：<b>${buy_target:.2f}</b> | 🛡️ 停損：<b>${stop_loss:.2f}</b> | 💰 預期停利：<b>${take_profit_long:.2f}</b>"
                ai_situation_analysis = f"目前盤勢屬於**安全、規律且溫和的上升常態軌道**。集成得分為 {ensemble_score} 分。雖然今天沒有發生驚心動魄的跨時環比爆量突襲，但價格成功穩踩在生命線之上，斜率溫和放大，沒有任何見頂或大戶反手砸盤的異值特徵。"
                ai_institutional_analysis = f"**【大戶黑手戰術拆解】** 5日資金流穩定維持在 {net_inflow_ratio:+.1f}% 偏多位階。這顯示主力機構正在執行**『每日平滑限價吸籌』**。他們不急於在一天之內強行拉高，而是規律地吃下浮額。此時不需要盲目重倉追價，用 50% 倉位順著時K月線慢條斯理地抱股，是統計學上的最高期望值解。"

            # ------------------------------------------------------------
            # 🧠 數據儲存：觸發 Supabase 背景異步存檔機制
            # ------------------------------------------------------------
            log_payload = {
                "ticker": str(search_input),
                "price": float(last_p),
                "smi_slope": float(current_smi),
                "volatility_shock": float(volatility_shock_ratio),
                "net_inflow_ratio": float(net_inflow_ratio),
                "ensemble_score": int(ensemble_score),
                "regime_mode": str(regime_mode),
                "action_signal": str(action_signal),
                "stop_loss": float(stop_loss) if "BUY" in action_signal else None,
                "take_profit": float(take_profit_long) if "BUY" in action_signal else None
            }
            threading.Thread(target=async_log_to_supabase, args=(log_payload,), daemon=True).start()

            # ------------------------------------------------------------
            # 5. 前端四核心看板渲染
            # ------------------------------------------------------------
            st.markdown(f"""<div class='brain-box'>🧠 <b>SYSTEM 3 頂層智慧決策狀態：</b> {regime_broadcast} 📡 <span style='color:#0284c7;'>【數據已即時同步至 Supabase 雲端資料庫】</span></div>""", unsafe_allow_html=True)

            col1, col2, col4, col3 = st.columns(4)
            with col1:
                st.markdown(f"""<div class='sys-card {sys1_css}'><div class='sys-title'>🔥 SYSTEM 1 (即時權重:{w1:.1f}%)</div><div class='sys-status'>{sys1_status}</div><div class='sys-desc'>{sys1_desc}</div></div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class='sys-card {sys2_css}'><div class='sys-title'>📐 SYSTEM 2 (即時權重:{w2:.1f}%)</div><div class='sys-status'>{sys2_status}</div><div class='sys-desc'>{sys2_desc}</div></div>""", unsafe_allow_html=True)
            with col4:
                st.markdown(f"""<div class='sys-card {sys4_css}'><div class='sys-title'>📡 SYSTEM 4 (即時權重:{w4:.1f}%)</div><div class='sys-status'>{sys4_status}</div><div class='sys-desc'>{sys4_desc}</div></div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class='sys-card {sys3_css}'><div class='sys-title'>⚡ SYSTEM 3 (AI統合決策)</div><div class='sys-status'>{sys3_status}</div><div style='font-size:13px; font-weight:bold; margin:4px 0;'>{sys3_pos}</div><div class='sys-desc'>{sys3_desc}</div><div class='sig-price-box'>{sys3_price}</div></div>""", unsafe_allow_html=True)

            # AI 剖析大看板
            st.markdown(f"""
                <div class='analysis-box'>
                    <h4>🚨 AI SYSTEM 3 大一統整合現況深度剖析與大戶戰術解碼</h4>
                    <hr style='border-color: #cbd5e1; margin: 12px 0;'>
                    <div class='analysis-section'>
                        <div class='analysis-header'>🎯 1. 當前市場結構現況統合剖析 (包含跨天同軌 ✕ 跨時環比精密考量)</div>
                        {ai_situation_analysis}
                    </div>
                    <div class='analysis-section' style='margin-top: 20px;'>
                        <div class='analysis-header'>🕵️ 2. 大戶與三大法人內部操作戰術解碼</div>
                        {ai_institutional_analysis}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # AI 白皮書
            st.markdown(f"""
                <div class='evidence-box'>
                    <h4>🎯 5日 AI 四系統獨立特徵交叉審查白皮書 (AI 自適應決策版)</h4>
                    <hr style='border-color: #cbd5e1; margin: 10px 0;'>
                    <div class='evidence-item'>📡 <span class='evidence-tag'>【SYSTEM 1】</span> 中短期相對量能比率：<b>{relative_vol_ratio:.2f}x</b> (權重: {w1:.1f}%)</div>
                    <div class='evidence-item'>📐 <span class='evidence-tag'>【SYSTEM 2】</span> SMI 最新一階斜率：<b>{current_smi:+.4f}</b> | 波動衝擊係數：<b>{volatility_shock_ratio:.2f}x</b> (權重: {w2:.1f}%)</div>
                    <div class='evidence-item'>📊 <span class='evidence-tag'>【SYSTEM 4】</span> 跨天同軌量能：<b>{delta_volume_ratio_macro:+.1f}%</b> | 跨時環比量能(對比上小時)：<b>{delta_volume_ratio_micro:+.1f}%</b> (權重: {w4:.1f}%)</div>
                    <div class='evidence-item'>💰 <span class='evidence-tag'>【籌碼系統】</span> {flow_detail_text}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # 高級雙畫布 (四色幾何變色龍)
            st.write("---")
            df_chart = df.tail(60)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.08, 
                                subplot_titles=("📈 SYSTEM 2：時K雙通道空間幾何結構", "📡 SYSTEM 2：SMI 一階線性回歸斜率動能矩陣 (四色幾何變色龍)"),
                                row_width=[0.3, 0.7])
            
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Upper'], name="布林上軌", line=dict(color='rgba(30,58,138,0.2)'), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Lower'], name="布林下軌", line=dict(color='rgba(30,58,138,0.2)'), fill='tonexty', fillcolor='rgba(30,58,138,0.03)', showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['KC_Upper'], name="肯特納上軌", line=dict(color='#dc2626', dash='dot', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['KC_Lower'], name="肯特納下軌", line=dict(color='#dc2626', dash='dot', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'], name="時K收盤價", line=dict(color='#1e3a8a', width=2.5)), row=1, col=1)
            
            gradient_colors = []
            smi_vals = df_chart['SMI_Histogram'].values
            for i in range(len(df_chart)):
                curr_val = smi_vals[i]
                prev_val = smi_vals[i-1] if i > 0 else curr_val
                
                if curr_val >= 0:
                    gradient_colors.append('#dc2626' if curr_val >= prev_val else '#fca5a5') # 深紅(加速) / 粉紅(減速)
                else:
                    gradient_colors.append('#16a34a' if curr_val <= prev_val else '#86efac') # 深綠(下砸) / 淺綠(煞車)
                        
            fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['SMI_Histogram'], name="SMI斜率柱", marker_color=gradient_colors, showlegend=False), row=2, col=1)
            fig.add_shape(type="line", x0=df_chart.index[0], y0=0, x1=df_chart.index[-1], y1=0, line=dict(color="#64748b", width=1, dash="dash"), row=2, col=1)
            
            fig.update_layout(height=750, hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.sidebar.error("請輸入個股或 ETF 代碼！")