import os
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
from binance.client import Client
from threading import Thread
import csv

# === Cấu hình ===
DATA_DIR = r'C:\Users\Admin\Desktop\Bot\data_futures\cleaned'
MODEL_DIR = r'C:\Users\Admin\Desktop\Bot\models\xgboost'
LOG_PATH = r'C:\Users\Admin\Desktop\Bot\logs\trade_log.csv'

TICKERS = ['1000PEPEUSDT', '1000BONKUSDT', 'DOGEUSDT', 'ENAUSDT', 'SUIUSDT',
           'SOLUSDT', 'ADAUSDT', 'TRXUSDT', 'LINKUSDT', 'UNIUSDT', 'DOTUSDT']

CAPITAL = 300
MAX_PER_TRADE = 50
CONFIDENCE_THRESHOLD = 0.6
FEE_RATE = 0.0004

open_positions = []
stop_signal = False
balance = CAPITAL
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# === API Binance ===
API_KEY = 'XIoJIvQbNCbHziFAmlj0Cxc2c2Firtn0ejPkv4OqeNtA2xRbMZSPG3PhiE3trbFj'
API_SECRET = 'YOUR_SECRET_HERE'
client = Client(API_KEY, API_SECRET)

# === Hàm load mô hình ===
def load_model(ticker):
    path = os.path.join(MODEL_DIR, f'{ticker}_xgb.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

# === Hàm load dữ liệu mới nhất ===
def load_latest_data(ticker):
    path = os.path.join(DATA_DIR, f'{ticker}_1h_cleaned.csv')
    df = pd.read_csv(path)
    return df.drop(columns=['label']).iloc[-1:]

# === Hàm lấy giá thực từ Binance ===
def get_real_price(symbol):
    try:
        price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
        return round(price, 4)
    except Exception as e:
        print(f"❌ Lỗi lấy giá từ Binance cho {symbol}: {e}")
        return None

# === Hàm tạo lệnh ===
def create_trade(token, direction, confidence):
    entry_price = get_real_price(token)
    if entry_price is None:
        return None

    return {
        'token': token,
        'direction': 'LONG' if direction == 2 else 'SHORT',
        'confidence': round(confidence, 3),
        'amount': MAX_PER_TRADE,
        'entry_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'entry_price': entry_price,
        'sl_price': '',
        'tp_price': '',
        'exit_price': '',
        'exit_time': '',
        'result': '',
        'pnl_usd': '',
        'balance_after': ''
    }

# === Hàm ghi log ===
def log_trade(trade):
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=trade.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade)

# === Hàm lấy giá đóng để giả lập ===
def simulate_current_price(symbol):
    return get_real_price(symbol)

# === Kiểm tra và đóng lệnh ===
def check_and_close_trades():
    global open_positions, balance
    updated_positions = []
    for trade in open_positions:
        current_price = simulate_current_price(trade['token'])
        if current_price is None:
            updated_positions.append(trade)
            continue

        pnl_percent = (current_price - trade['entry_price']) / trade['entry_price']
        if trade['direction'] == 'SHORT':
            pnl_percent = -pnl_percent

        if pnl_percent >= 0.025 or pnl_percent <= -0.015:
            trade['exit_price'] = current_price
            trade['exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            trade['result'] = 'Win' if pnl_percent >= 0.025 else 'Loss'

            gain = pnl_percent * trade['amount']
            fee = trade['amount'] * FEE_RATE * 2
            gain -= fee
            balance += gain

            trade['pnl_usd'] = round(gain, 2)
            trade['balance_after'] = round(balance, 2)

            log_trade(trade)
            print(f"🏁 Đóng lệnh: {trade['token']} | {trade['direction']} | Giá vào: {trade['entry_price']} | Giá đóng: {current_price} | {trade['result']} | PnL: {gain:.2f} USD (sau phí) | Vào: {trade['entry_time']} | Ra: {trade['exit_time']}")
        else:
            updated_positions.append(trade)

    open_positions = updated_positions

# === Quét mô hình & vào lệnh ===
def scan_and_trade():
    global open_positions
    for token in TICKERS:
        if len(open_positions) >= CAPITAL // MAX_PER_TRADE:
            break

        model = load_model(token)
        x = load_latest_data(token)
        proba = model.predict_proba(x)[0]
        pred = np.argmax(proba)
        confidence = proba[pred]

        if pred in [1, 2] and confidence >= CONFIDENCE_THRESHOLD:
            trade = create_trade(token, pred, confidence)
            if trade:
                open_positions.append(trade)
                log_trade(trade)
                print(f"✅ Vào lệnh: {trade['token']} | {trade['direction']} | Giá vào: {trade['entry_price']} | {confidence:.2f}")

# === Vòng lặp bot ===
def run_loop():
    print("🤖 Bot bắt đầu chạy. Gõ 'exit' để dừng.")
    while not stop_signal:
        print(f"\n🔁 [{datetime.now().strftime('%H:%M:%S')}] Kiểm tra đóng lệnh...")
        check_and_close_trades()
        print(f"📊 Lệnh đang mở ({len(open_positions)}): {[t['token'] for t in open_positions]}")
        print(f"💰 Số dư hiện tại: {balance:.2f} USD")

        print("⏳ Đợi 1 phút trước khi quét lệnh mới...")
        for _ in range(12):
            if stop_signal: break
            time.sleep(5)

        print(f"🔍 [{datetime.now().strftime('%H:%M:%S')}] Quét tín hiệu vào lệnh...")
        scan_and_trade()

# === Dừng bot ===
def listen_for_exit():
    global stop_signal
    while True:
        cmd = input()
        if cmd.strip().lower() == 'exit':
            stop_signal = True
            print("🛑 Dừng bot theo yêu cầu người dùng.")
            break

if __name__ == '__main__':
    Thread(target=listen_for_exit, daemon=True).start()
    run_loop()

