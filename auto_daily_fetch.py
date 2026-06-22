# ================================================================
# 🛰️ AI 自適應大腦：定時收盤特徵抓取、殘差對帳與勝率自我學習中樞
# ================================================================
import yfinance as yf
import pandas as pd
import numpy as np
from supabase import create_client
import os
from datetime import datetime

# 🔐 從 GitHub Actions 雲端環境變數中自動讀取金鑰
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ 錯誤：未偵測到雲端資料庫環境變數！請確認 GitHub Secrets 設定。")
    exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------------------------------------------
# 🎯 核心一：歷史預測 Range 區間命中率與殘差審計反饋學習
# ----------------------------------------------------------------
print("🕵️ 啟動 AI 價格殘差與區間命中率審計程序...")
try:
    # 撈出所有「尚未結算」的單據（is_settled 為 False 的歷史紀錄）
    res = supabase.table("regime_logs").select("*").eq("is_settled", False).execute()
    unsettled_logs = res.data
    
    print(f"📊 偵測到目前有 {len(unsettled_logs)} 筆未結算單據，開始進行區間核實...")
    
    for log in unsettled_logs:
        log_id = log["id"]
        ticker = log["ticker"]
        # 還原 Yahoo Finance 的台美股代碼格式
        yf_ticker = f"{ticker}.TW" if ticker.isdigit() else ticker
        created_at = pd.to_datetime(log["created_at"])
        
        # 💡 量化風控：檢查這筆預測是否已經過去了至少 3-5 天
        days_passed = (datetime.now() - created_at.replace(tzinfo=None)).days
        if days_passed < 3: 
            continue  # 時間尚短，讓子彈再飛一會兒，暫不對帳
            
        # 抓取該單據建立到今天為止的日K走勢
        df_check = yf.download(yf_ticker, start=created_at.strftime('%Y-%m-%d'), progress=False)
        if df_check.empty: 
            continue
        df_check.columns = [str(col[0]).strip().lower() if isinstance(col, tuple) else str(col).strip().lower() for col in df_check.columns]
        
        # 1. 找出未來的「真實市場價格靶心」：這段時間最高價與最低價的幾何均值
        actual_target = float((df_check['high'].max() + df_check['low'].min()) / 2)
        predicted_target = float(log["predicted_target_price"])
        
        # 2. 抓取當時相似盤面的誤差半徑 (若無則先以 5 元作為常態緩衝)
        error_radius = float(log["price_distance_error"]) if log["price_distance_error"] and float(log["price_distance_error"]) > 0 else 5.0
        
        # 3. 計算當初預測的動態 Range 上界與下界
        range_upper = predicted_target + (error_radius * 0.5)
        range_lower = predicted_target - (error_radius * 0.5)
        
        # 4. 💡 嚴謹對帳：判定真實價格重心有沒有成功掉進當初預測的 Range 緩衝網內
        if range_lower <= actual_target <= range_upper:
            real_outcome = 1  # 🎯 精準命中靶心（Hit）
            print(f"🟢 審計成功：單據 #{log_id} ({ticker}) 真實價 ${actual_target:.2f} 完美落在預測區間內！")
        else:
            real_outcome = 0  # ❌ 脫靶（Miss）
            print(f"🔴 審計脫靶：單據 #{log_id} ({ticker}) 真實價 ${actual_target:.2f} 逸出預測區間。")
            
        # 5. 計算絕對幾何殘差距離（Range 核心進化燃料）
        price_distance_error = float(abs(actual_target - predicted_target))
        
        # 將核實對帳數據、殘差與命中標籤 (1/0) 全自動回填 Supabase
        supabase.table("regime_logs").update({
            "is_settled": True,
            "actual_best_price": actual_target,
            "price_distance_error": price_distance_error,
            "real_outcome": int(real_outcome)
        }).eq("id", log_id).execute()

except Exception as e:
    print(f"⚠️ 殘差與勝率對帳審計發生異常: {e}")

# ----------------------------------------------------------------
# 📊 核心二：每日收盤 4D 特徵計算與自適應動態目標價預測
# ----------------------------------------------------------------
WATCH_LIST = ["2330.TW", "2382.TW", "3017.TW", "NVDA", "TSLA", "AAPL", "AMD"]
print(f"\n🕒 啟動每日收盤自選股數據入庫。核心池共 {len(WATCH_LIST)} 檔標的...")

batch_logs = []
for ticker in WATCH_LIST:
    try:
        df = yf.download(ticker, period="60d", interval="1h", progress=False, auto_adjust=False)
        if df.empty: 
            continue
        
        df.columns = [str(col[0]).strip().lower() if isinstance(col, tuple) else str(col).strip().lower() for col in df.columns]
        rename_dict = {c: c.capitalize() for c in df.columns if c.capitalize() in ['Open', 'High', 'Low', 'Close', 'Volume']}
        df = df.rename(columns=rename_dict)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy().ffill().bfill()
        
        close_s, high_s, low_s, vol_s = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze(), df['Volume'].squeeze()
        
        # 幾何核心特徵計算
        df['MA_20'] = close_s.rolling(20).mean()
        bb_std = close_s.rolling(20).std()
        tr = pd.concat([high_s - low_s, (high_s - close_s.shift(1)).abs(), (low_s - close_s.shift(1)).abs()], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(14).mean()
        
        current_atr = float(df['ATR_14'].iloc[-1])
        last_p = float(close_s.iloc[-1])
        
        # SMI 幾何斜率柱計算
        highest_high = high_s.rolling(20).max()
        lowest_low = low_s.rolling(20).min()
        donut_center = (highest_high + lowest_low) / 2.0
        fit_source = close_s - ((donut_center + df['MA_20']) / 2.0)
        x = np.arange(20)
        x_val = x - x.mean()
        x_sum_sq = (x_val ** 2).sum()
        df['SMI_Histogram'] = fit_source.rolling(20).apply(lambda w: (x_val * (w - w.mean())).sum() / x_sum_sq, raw=True).fillna(0)
        
        current_smi = float(df['SMI_Histogram'].iloc[-1])
        
        # 籌碼流向主動比率
        is_tw = ".TW" in ticker or ".TWO" in ticker
        if is_tw:
            raw_inflow = (close_s.diff().tail(35).squeeze() * vol_s.tail(35).squeeze()).sum() / (vol_s.tail(35).squeeze().sum() + 1e-9)
            net_inflow_ratio = float(np.clip(raw_inflow * 12, -100.0, 100.0))
        else:
            typical_price = (high_s + low_s + close_s) / 3.0
            price_diff = typical_price.diff()
            pos_flow = pd.Series(np.where(price_diff > 0, typical_price * vol_s, 0.0), index=df.index)
            neg_flow = pd.Series(np.where(price_diff < 0, typical_price * vol_s, 0.0), index=df.index)
            net_inflow_ratio = float(((pos_flow.tail(35).sum() - neg_flow.tail(35).sum()) / (pos_flow.tail(35).sum() + neg_flow.tail(35).sum() + 1e-9)) * 100)

        # 🚀 根據特徵共振，粗算動態預測目標靶心
        if net_inflow_ratio > 15 and current_smi > 0:
            predicted_target = last_p + (current_atr * 1.8)
            regime_mode = "🚀 雙軸共振主升突破"
        elif net_inflow_ratio < -10 or current_smi < 0:
            predicted_target = last_p - (current_atr * 1.5)
            regime_mode = "🛡️ 高位劇震洗盤盤面"
        else:
            predicted_target = last_p + (current_atr * 0.3)
            regime_mode = "📊 常態平穩結構盤"

        # 清洗代碼字串符號
        clean_ticker = ticker.replace(".TW", "").replace(".TWO", "")
        
        # 封裝大一統日誌 payload (加入預設 false 狀態)
        payload = {
            "ticker": str(clean_ticker).upper(),
            "price": float(last_p),
            "smi_slope": float(current_smi),
            "volatility_shock": float(current_atr / (tr.rolling(120).mean().iloc[-1] + 1e-9)),
            "net_inflow_ratio": float(net_inflow_ratio),
            "ensemble_score": int(np.clip(50 + net_inflow_ratio, 0, 100)),
            "regime_mode": str(regime_mode),
            "action_signal": "STRONG_BUY" if net_inflow_ratio > 15 else "MILD_BUY",
            "predicted_target_price": float(predicted_target),
            "is_settled": False  # 💡 每次新入庫預設為 False，等待未來結算對帳
        }
        batch_logs.append(payload)
        
    except Exception as e:
        print(f"❌ 計算自選股 {ticker} 異常: {e}")

# 批量寫入 Supabase 雲端
if batch_logs:
    try:
        supabase.table("regime_logs").insert(batch_logs).execute()
        print(f"\n🏆 歷史與未來特徵環大一統！本日已成功自動錄入 {len(batch_logs)} 檔高精準預測單據。")
    except Exception as e:
        print(f"❌ 批量寫入 Supabase 資料庫失敗: {e}")