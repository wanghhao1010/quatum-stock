# ================================================================
# 🛰️ AI 大腦定時收盤自選股自動餵食中樞 (Cloud Cron Job 專用)
# ================================================================
import yfinance as yf
import pandas as pd
import numpy as np
from supabase import create_client
import os

# 1. 🔐 從雲端環境變數讀取金鑰 (GitHub Secrets)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ 錯誤：未偵測到雲端資料庫環境變數！")
    exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 📋 2. 定義你想讓 AI 每天收盤自動監控的【黃金自選股核心矩陣】
# 浩心，你隨時可以自由在這裡增加或刪除代碼（台股記得加 .TW 或 .TWO）
WATCH_LIST = ["2330.TW", "2382.TW", "3017.TW", "NVDA", "TSLA", "AAPL", "AMD"]

print(f"🕒 啟動每日收盤定時自動餵食。本次掃描核心池共 {len(WATCH_LIST)} 檔標的...")

batch_logs = []

for ticker in WATCH_LIST:
    try:
        print(f"📡 正在解算 {ticker} 的 4D 集成特徵...")
        df = yf.download(ticker, period="60d", interval="1h", progress=False, auto_adjust=False)
        if df.empty:
            continue
            
        df.columns = [str(col[0]).strip().lower() if isinstance(col, tuple) else str(col).strip().lower() for col in df.columns]
        rename_dict = {c: c.capitalize() for c in df.columns if c.capitalize() in ['Open', 'High', 'Low', 'Close', 'Volume']}
        df = df.rename(columns=rename_dict)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy().ffill().bfill()
        
        close_s, high_s, ... = df['Close'].squeeze(), df['High'].squeeze()
        vol_s = df['Volume'].squeeze()
        
        if len(df) < 40: continue
        
        # --- 四系統核心邏輯演算簡化移植 ---
        df['MA_20'] = close_s.rolling(20).mean()
        bb_std = close_s.rolling(20).std()
        df['BB_Upper'] = df['MA_20'] + (bb_std * 2)
        df['BB_Lower'] = df['MA_20'] - (bb_std * 2)
        tr = pd.concat([high_s - df['Low'].squeeze(), (high_s - close_s.shift(1)).abs(), (df['Low'].squeeze() - close_s.shift(1)).abs()], axis=1).max(axis=1)
        df['ATR_20'] = tr.rolling(20).mean()
        df['KC_Upper'] = df['MA_20'] + (df['ATR_20'] * 1.5)
        df['KC_Lower'] = df['MA_20'] - (df['ATR_20'] * 1.5)
        df['Squeeze_On'] = (df['BB_Upper'] < df['KC_Upper']) & (df['BB_Lower'] > df['KC_Lower'])
        
        x = np.arange(20)
        x_val = x - x.mean()
        x_sum_sq = (x_val ** 2).sum()
        highest_high = high_s.rolling(20).max()
        lowest_low = df['Low'].squeeze().rolling(20).min()
        donut_center = (highest_high + lowest_low) / 2.0
        fit_source = close_s - ((donut_center + df['MA_20']) / 2.0)
        df['SMI_Histogram'] = fit_source.rolling(20).apply(lambda w: (x_val * (w - w.mean())).sum() / x_sum_sq, raw=True).fillna(0)
        
        current_smi = float(df['SMI_Histogram'].iloc[-1])
        prev_smi = float(df['SMI_Histogram'].iloc[-2])
        df['ATR_14'] = tr.rolling(14).mean()
        current_atr = float(df['ATR_14'].iloc[-1])
        last_p = float(close_s.iloc[-1])
        
        is_tw = ".TW" in ticker or ".TWO" in ticker
        if is_tw:
            raw_inflow = (close_s.diff().tail(35).squeeze() * vol_s.tail(35).squeeze()).sum() / (vol_s.tail(35).squeeze().sum() + 1e-9)
            net_inflow_ratio = float(np.clip(raw_inflow * 12, -100.0, 100.0))
        else:
            typical_price = (high_s + df['Low'].squeeze() + close_s) / 3.0
            raw_money_flow = typical_price * vol_s
            price_diff = typical_price.diff()
            pos_flow = pd.Series(np.where(price_diff > 0, raw_money_flow, 0.0), index=df.index)
            neg_flow = pd.Series(np.where(price_diff < 0, raw_money_flow, 0.0), index=df.index)
            raw_inflow = ((pos_flow.tail(35).sum() - neg_flow.tail(35).sum()) / (pos_flow.tail(35).sum() + neg_flow.tail(35).sum() + 1e-9)) * 100
            net_inflow_ratio = float(np.clip(raw_inflow, -100.0, 100.0))

        five_hour_vol = vol_s.tail(5).mean()
        baseline_vol = vol_s.tail(120).mean() + 1e-9
        is_hot = (five_hour_vol / baseline_vol) >= 1.4
        
        lookback_offset = 7 if is_tw else 8
        delta_smi_slope_micro = current_smi - prev_smi
        long_term_atr = float(tr.rolling(120).mean().iloc[-1]) + 1e-9
        volatility_shock_ratio = current_atr / long_term_atr
        
        # 權重與得分計算
        if volatility_shock_ratio > 1.35 and abs(delta_smi_slope_micro) > 0.2:
            w1, w2, w4, regime_mode = 10.0, 65.0, 25.0, "🛡️ 高位劇震洗盤盤面"
        elif delta_smi_slope_micro > 0 and current_smi > 0:
            w1, w2, w4, regime_mode = 35.0, 20.0, 45.0, "🚀 雙軸共振主升突破"
        else:
            w1, w2, w4, regime_mode = 25.0, 45.0, 30.0, "📊 常態平穩結構盤"
            
        score_sys1 = 100 if is_hot else 0
        score_sys2 = 100 if current_smi > 0 else (50 if bool(df['Squeeze_On'].iloc[-1]) else 0)
        score_sys4 = 50 if (current_smi - float(df['SMI_Histogram'].iloc[-lookback_offset])) > 0 else 0
        
        ensemble_score = int((score_sys1 * (w1/100)) + (score_sys2 * (w2/100)) + (score_sys4 * (w4/100)))
        action_signal = "FORCED_MELTDOWN" if (net_inflow_ratio < -12.0 or current_smi < 0) else ("STRONG_BUY" if ensemble_score >= 75 else "MILD_BUY")
        
        stop_loss = last_p - (current_atr * 2)
        take_profit_long = last_p + (current_atr * 3.5)
        
        # 打包單筆資料
        clean_ticker = ticker.replace(".TW", "").replace(".TWO", "")
        payload = {
            "ticker": str(clean_ticker).upper(),
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
        batch_logs.append(payload)
        
    except Exception as e:
        print(f"❌ 計算 {ticker} 時跳出錯誤: {e}")

# 🚀 3. 批量灌入 Supabase 資料庫
if batch_logs:
    try:
        supabase.table("regime_logs").insert(batch_logs).execute()
        print(f"🏆 全自動定時餵食成功！已成功自動匯入 {len(batch_logs)} 檔個股收盤紀錄。")
    except Exception as e:
        print(f"❌ 雲端匯入失敗: {e}")
else:
    print("⚠️ 沒有生成任何有效數據。")